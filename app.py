import sys
import os
import shutil
import zipfile
import tempfile
import gradio as gr
import numpy as np
import soundfile as sf
from multiprocessing import cpu_count

# Suppress noisy Windows ProactorEventLoop connection-reset errors.
try:
    from asyncio.proactor_events import _ProactorBasePipeTransport
    _original = _ProactorBasePipeTransport._call_connection_lost
    def _patched(self, exc):
        try: _original(self, exc)
        except ConnectionResetError: pass
    _ProactorBasePipeTransport._call_connection_lost = _patched
except Exception:
    pass

# ─── Path Setup ──────────────────────────────────────────────────────
root_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(root_dir, "core")
sys.path.append(core_dir)
sys.path.append(root_dir)
os.chdir(core_dir)

# ─── Core Imports ────────────────────────────────────────────────────
from core import (
    run_infer_script,
    run_prerequisites_script,
    run_preprocess_script,
    run_extract_script,
    run_train_script,
    stop_train_script,
    run_index_script,
)
from rvc.lib.tools.model_download import download_from_url
from tabs.download.download import (
    get_pretrained_list,
    get_pretrained_sample_rates,
    download_pretrained_model as core_download_pretrained_model,
    save_drop_model,
)

# Prerequisites
run_prerequisites_script(pretraineds_hifigan=True, models=True, exe=True, smartcutter=True)

# CSS & Theme (Ultimate Red)
theme_path = os.path.join(root_dir, "assets", "themes", "ultimate_red.json")
theme = gr.Theme.load(theme_path) if os.path.exists(theme_path) else "Default"

css = """
h1 { text-align: center; margin-top: 20px; margin-bottom: 20px; }
#generate-tab-button { font-weight: bold !important;}
#manage-tab-button { font-weight: bold !important;}
#train-tab-button { font-weight: bold !important;}
#settings-tab-button { font-weight: bold !important;}
"""

# Workflow imports
from workflow.youtube_dl import get_youtube_audio
from workflow.audio_separator import separate_audio

# ─── Helpers ─────────────────────────────────────────────────────────
model_root = os.path.join(core_dir, "logs")

def get_models():
    if not os.path.exists(model_root): return []
    return sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(model_root)
        for f in files
        if f.endswith((".pth", ".uvmp")) and not (f.startswith("G_") or f.startswith("D_"))
    ])

def get_indexes():
    if not os.path.exists(model_root): return []
    return sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(model_root)
        for f in files
        if f.endswith(".index") and "trained" not in f
    ])

def match_index(model_path):
    if not model_path: return ""
    model_dir = os.path.dirname(model_path)
    model_base = os.path.splitext(os.path.basename(model_path))[0].split("_")[0]
    try:
        for f in os.listdir(model_dir):
            if f.endswith(".index") and model_base.lower() in f.lower():
                return os.path.join(model_dir, f)
    except Exception: pass
    return ""

def refresh_models():
    return gr.update(choices=get_models()), gr.update(choices=get_indexes())

def mix_audio(vocals_path, instrumental_path, vocals_volume=1.0, instrumental_volume=1.0):
    voc_data, voc_sr = sf.read(vocals_path)
    inst_data, inst_sr = sf.read(instrumental_path)

    if inst_sr != voc_sr:
        import librosa
        inst_data = librosa.resample(inst_data.T if inst_data.ndim > 1 else inst_data,
                                      orig_sr=inst_sr, target_sr=voc_sr)
        if inst_data.ndim > 1: inst_data = inst_data.T

    min_len = min(len(voc_data), len(inst_data))
    voc_data = voc_data[:min_len]
    inst_data = inst_data[:min_len]

    if voc_data.ndim == 1 and inst_data.ndim > 1:
        voc_data = np.stack([voc_data, voc_data], axis=-1)
    elif voc_data.ndim > 1 and inst_data.ndim == 1:
        inst_data = np.stack([inst_data, inst_data], axis=-1)

    mixed = (voc_data * vocals_volume) + (inst_data * instrumental_volume)
    peak = np.max(np.abs(mixed))
    if peak > 1.0: mixed = mixed / peak

    out_path = os.path.join(root_dir, "workflow_output", "covers", "cover_output.wav")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, mixed, voc_sr)
    return out_path

# ─── Custom Downloader ──────────────────────────────────────────────
def download_custom_voice_model(url, model_name):
    if not url or not model_name:
        raise gr.Error("Please provide both a URL and a Model Name.")

    model_name = "".join(c for c in model_name if c.isalnum() or c in (" ", "-", "_")).strip()
    if not model_name:
        raise gr.Error("Invalid model name.")

    zips_folder = os.path.join(model_root, "zips")
    os.makedirs(zips_folder, exist_ok=True)

    for f in os.listdir(zips_folder):
        os.remove(os.path.join(zips_folder, f))

    gr.Info(f"Downloading model from {url}...")
    res = download_from_url(url)
    if res != "downloaded":
        raise gr.Error("Failed to download the model from the provided URL.")

    downloaded_zip = None
    for f in os.listdir(zips_folder):
        if f.endswith(".zip"):
            downloaded_zip = os.path.join(zips_folder, f)
            break

    if not downloaded_zip:
        raise gr.Error("No ZIP file was found after download.")

    temp_extract = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(downloaded_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)
    except Exception as e:
        shutil.rmtree(temp_extract)
        raise gr.Error(f"Failed to extract zip: {e}")

    pth_file = None
    index_file = None

    for root_dir_walk, _, files in os.walk(temp_extract):
        for f in files:
            if f.endswith(".pth") and "G_" not in f and "D_" not in f:
                pth_file = os.path.join(root_dir_walk, f)
            elif f.endswith(".index") and "trained" not in f:
                index_file = os.path.join(root_dir_walk, f)

    if not pth_file:
        shutil.rmtree(temp_extract)
        raise gr.Error("No valid .pth model found in the archive.")

    final_folder = os.path.join(model_root, model_name)
    os.makedirs(final_folder, exist_ok=True)

    shutil.move(pth_file, os.path.join(final_folder, f"{model_name}.pth"))
    if index_file:
        shutil.move(index_file, os.path.join(final_folder, f"{model_name}.index"))

    shutil.rmtree(temp_extract)
    os.remove(downloaded_zip)
    return f"[+] Successfully set up Voice Model: {model_name}!"

def update_pretrained_dropdown(model):
    choices = get_pretrained_sample_rates(model)
    return gr.update(choices=choices, value=choices[0] if choices else None)

def download_custom_pretrained(model, sr):
    core_download_pretrained_model(model, sr)
    return f"[+] Downloaded pretrained {model} ({sr})!"


# ═══════════════════════════════════════════════════════════════════
#  GENERATE TAB  — Accordion Style (Ultimate RVC Layout)
# ═══════════════════════════════════════════════════════════════════
def render_generate_tab():
    with gr.Tab("Song covers"):
        models = get_models()
        indexes = get_indexes()
        default_model = models[0] if models else None

        # ── Step 0: Song Retrieval ───────────────────────────────
        with gr.Accordion("Step 0: song retrieval", open=True):
            with gr.Row():
                with gr.Column():
                    source_type = gr.Radio(
                        choices=["YouTube URL", "Local file"],
                        value="YouTube URL",
                        label="Source type",
                    )
                with gr.Column():
                    yt_url = gr.Textbox(
                        label="Source",
                        placeholder="https://youtube.com/watch?v=...",
                        info="Paste a YouTube link or song URL.",
                    )
                    upload_audio = gr.Audio(
                        label="Source",
                        type="filepath",
                        visible=False,
                        waveform_options=gr.WaveformOptions(show_recording_waveform=False),
                    )

            source_type.input(
                lambda t: (gr.update(visible=t == "YouTube URL"), gr.update(visible=t == "Local file")),
                inputs=source_type,
                outputs=[yt_url, upload_audio],
                show_progress="hidden",
            )

            with gr.Row():
                retrieve_song_btn = gr.Button("Retrieve song", variant="primary")
            input_audio = gr.Audio(
                label="Song",
                type="filepath",
                interactive=False,
                waveform_options=gr.WaveformOptions(show_recording_waveform=False),
            )

        # ── Step 1: Vocal Separation ─────────────────────────────
        with gr.Accordion("Step 1: vocal separation", open=False):
            with gr.Accordion("Options", open=False):
                with gr.Row():
                    sep_model = gr.Dropdown(
                        choices=["UVR-MDX-NET-Voc_FT", "htdemucs_ft"],
                        value="UVR-MDX-NET-Voc_FT",
                        label="Separation model",
                        info="UVR-MDX is fast and clean. htdemucs_ft for complex mixes.",
                    )
            with gr.Row():
                separate_vocals_btn = gr.Button("Separate vocals", variant="primary")
            with gr.Row():
                vocals_preview = gr.Audio(
                    label="Primary stem (Vocals)",
                    type="filepath", interactive=False,
                    waveform_options=gr.WaveformOptions(show_recording_waveform=False),
                )
                inst_preview = gr.Audio(
                    label="Secondary stem (Instrumental)",
                    type="filepath", interactive=False,
                    waveform_options=gr.WaveformOptions(show_recording_waveform=False),
                )

        # ── Step 2: Vocal Conversion ─────────────────────────────
        with gr.Accordion("Step 2: vocal conversion", open=False):
            model_file = gr.Dropdown(
                label="Voice model", choices=models, value=default_model,
                interactive=True, allow_custom_value=True,
            )
            index_file = gr.Dropdown(
                label="Index file", choices=indexes,
                value=match_index(default_model) if default_model else "",
                interactive=True, allow_custom_value=True,
            )
            refresh_btn = gr.Button("🔄 Refresh models", size="sm")

            with gr.Accordion("Options", open=False):
                with gr.Row():
                    pitch = gr.Slider(-24, 24, value=0, step=1, label="Pitch (semitones)",
                                      info="↑ higher, ↓ lower. +12 male→female, -12 female→male")
                    f0_method = gr.Dropdown(
                        choices=["rmvpe", "crepe", "crepe-tiny", "fcpe", "hybrid[rmvpe+fcpe]"],
                        value="rmvpe", label="F0 method",
                        info="rmvpe: fast & accurate. crepe: singing. fcpe: newest."
                    )
                with gr.Accordion("Advanced", open=False):
                    with gr.Accordion("Voice synthesis", open=False):
                        with gr.Row():
                            index_rate = gr.Slider(0, 1, value=0.75, step=0.05, label="Index rate")
                            rms_mix_rate = gr.Slider(0, 1, value=0.25, step=0.05, label="Volume envelope")
                        with gr.Row():
                            filter_radius = gr.Slider(0, 10, value=3, step=1, label="Filter radius")
                            protect = gr.Slider(0, 0.5, value=0.33, step=0.01, label="Protect rate")
                    with gr.Accordion("Vocal enrichment", open=False):
                        with gr.Row():
                            split_audio = gr.Checkbox(label="Split audio (long files)", value=False)
                        with gr.Row():
                            autotune = gr.Checkbox(label="Autotune", value=False)
                            clean_audio = gr.Checkbox(label="Clean audio (noise reduction)", value=False)
                    with gr.Accordion("Speaker embeddings", open=False):
                        with gr.Row():
                            embedder_model = gr.Dropdown(
                                choices=["contentvec", "contentvec_base", "chinese-hubert-base",
                                         "japanese-hubert-base", "spin"],
                                value="contentvec", label="Embedder model"
                            )
                        export_format = gr.Radio(choices=["WAV", "MP3", "FLAC"],
                                                 value="WAV", label="Export format")

            with gr.Row():
                convert_btn = gr.Button("Convert vocals", variant="primary")
            converted_audio = gr.Audio(
                label="Converted vocals",
                type="filepath", interactive=False,
                waveform_options=gr.WaveformOptions(show_recording_waveform=False),
            )

        # ── Step 3: Song Mixing ──────────────────────────────────
        with gr.Accordion("Step 3: song mixing", open=False):
            with gr.Accordion("Options", open=False):
                with gr.Row():
                    vocal_vol = gr.Slider(0, 2, value=1.0, step=0.05, label="Main vocals gain")
                    inst_vol = gr.Slider(0, 2, value=1.0, step=0.05, label="Instrumentals gain")
            with gr.Row():
                mix_btn = gr.Button("Mix song cover", variant="primary")
            final_cover = gr.Audio(
                label="Song cover",
                type="filepath", interactive=False,
                waveform_options=gr.WaveformOptions(show_recording_waveform=False),
            )

        # ── EVENT WIRING ─────────────────────────────────────────
        def do_retrieve(source_type_val, url_val, local_val):
            if source_type_val == "YouTube URL":
                if not url_val: raise gr.Error("Please enter a YouTube URL.")
                return get_youtube_audio(url_val, os.path.join(root_dir, "workflow_output", "downloads"))
            else:
                if not local_val: raise gr.Error("Please upload an audio file.")
                return local_val

        def do_separate(audio_path, sep_choice):
            if not audio_path: raise gr.Error("No audio to separate!")
            v, i = separate_audio(
                input_audio_path=audio_path,
                output_dir=os.path.join(root_dir, "workflow_output", "separated"),
                use_demucs="demucs" in sep_choice.lower(),
            )
            return v, i

        def do_convert(vocal_path, m, idx, p, f0, ir, fr, rmr, prot, emb, fmt, split, at, clean):
            if not vocal_path: raise gr.Error("No vocals to convert!")
            if not m: raise gr.Error("No voice model selected!")
            out_path = os.path.join(root_dir, "workflow_output", "converted", "converted_vocals.wav")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            run_infer_script(
                pitch=p, filter_radius=fr, index_rate=ir,
                volume_envelope=rmr, protect=prot, f0_method=f0,
                input_path=vocal_path, output_path=out_path,
                pth_path=m, index_path=idx or "",
                split_audio=split, f0_autotune=at, f0_autotune_strength=1.0,
                clean_audio=clean, clean_strength=0.3, export_format=fmt,
                f0_file=None, embedder_model=emb, embedder_model_custom=None,
            )
            return out_path

        def do_mix(conv, inst, vv, iv):
            if not conv or not inst: raise gr.Error("Need both converted vocals and instrumental!")
            return mix_audio(conv, inst, vv, iv)

        retrieve_song_btn.click(do_retrieve, [source_type, yt_url, upload_audio], [input_audio])
        separate_vocals_btn.click(do_separate, [input_audio, sep_model], [vocals_preview, inst_preview])
        refresh_btn.click(refresh_models, [], [model_file, index_file])
        model_file.change(lambda m: match_index(m), [model_file], [index_file])
        convert_btn.click(
            do_convert,
            [vocals_preview, model_file, index_file, pitch, f0_method,
             index_rate, filter_radius, rms_mix_rate, protect,
             embedder_model, export_format, split_audio, autotune, clean_audio],
            [converted_audio],
        )
        mix_btn.click(do_mix, [converted_audio, inst_preview, vocal_vol, inst_vol], [final_cover])


# ═══════════════════════════════════════════════════════════════════
#  MODELS TAB
# ═══════════════════════════════════════════════════════════════════
def render_models_tab():
    with gr.Tab("Download"):
        with gr.Accordion("Voice models"):
            with gr.Row():
                voice_model_url = gr.Textbox(
                    label="Model URL",
                    info="URL to a zip file containing .pth and optionally .index.",
                )
                voice_model_name = gr.Textbox(
                    label="Model name",
                    info="Unique name for this voice model.",
                )
            with gr.Row(equal_height=True):
                download_voice_btn = gr.Button("Download 🌐", variant="primary", scale=19)
                download_voice_msg = gr.Textbox(label="Output message", interactive=False, scale=20)

            download_voice_btn.click(
                fn=download_custom_voice_model,
                inputs=[voice_model_url, voice_model_name],
                outputs=[download_voice_msg],
            )

        with gr.Accordion("Pretrained models", open=False):
            pretrained_list = get_pretrained_list()
            default_pt = "Titan" if "Titan" in pretrained_list else (pretrained_list[0] if pretrained_list else "")
            default_rates = get_pretrained_sample_rates(default_pt) if default_pt else []

            with gr.Row():
                pretrained_model = gr.Dropdown(
                    label="Pretrained model", choices=pretrained_list, value=default_pt,
                    info="Select the pretrained model to download.",
                )
                pretrained_sample_rate = gr.Dropdown(
                    label="Sample rate", choices=default_rates,
                    value=default_rates[0] if default_rates else None,
                    info="Select the target sample rate.",
                )

            pretrained_model.change(fn=update_pretrained_dropdown,
                                    inputs=[pretrained_model], outputs=[pretrained_sample_rate])

            with gr.Row(equal_height=True):
                download_pretrained_btn = gr.Button("Download 🌐", variant="primary", scale=19)
                download_pretrained_msg = gr.Textbox(label="Output message", interactive=False, scale=20)

            download_pretrained_btn.click(
                fn=download_custom_pretrained,
                inputs=[pretrained_model, pretrained_sample_rate],
                outputs=[download_pretrained_msg],
            )

    with gr.Tab("Upload"):
        with gr.Accordion("Voice models", open=True):
            gr.Markdown("Drag and drop your `.pth` and `.index` files here — "
                        "they will be automatically saved to `logs/<model_name>/`.")
            dropbox = gr.File(
                label="Drag your .pth file and .index file into this space.",
                type="filepath",
            )
            dropbox.upload(fn=save_drop_model, inputs=[dropbox], outputs=[dropbox])


# ═══════════════════════════════════════════════════════════════════
#  TRAIN TAB  — Clean Accordion Style with Optimal Defaults
# ═══════════════════════════════════════════════════════════════════
CPU_CORES = cpu_count()

def render_train_tab():
    # ── Step 1: Dataset Preprocessing ────────────────────────────
    with gr.Accordion("Step 1: dataset preprocessing", open=True):
        with gr.Row():
            train_model_name = gr.Textbox(
                label="Model name",
                info="A unique name for your voice model.",
                placeholder="e.g. MyVoice",
            )
            dataset_path = gr.Textbox(
                label="Dataset path",
                info="Path to folder containing audio files (.wav recommended).",
                placeholder="/content/dataset",
            )
        with gr.Accordion("Options", open=False):
            with gr.Row():
                sample_rate = gr.Dropdown(
                    choices=["32000", "40000", "48000"],
                    value="40000",
                    label="Sample rate",
                    info="40k is the best balance of quality and speed.",
                )
                cpu_threads = gr.Slider(
                    1, CPU_CORES, value=CPU_CORES, step=1,
                    label="CPU threads",
                    info="More threads = faster preprocessing.",
                )
            with gr.Row():
                cut_preprocess = gr.Dropdown(
                    choices=["Skip", "Simple", "Automatic"],
                    value="Automatic",
                    label="Audio split method",
                    info="Automatic splits based on silence. Simple splits by chunk length.",
                )
                normalization_mode = gr.Dropdown(
                    choices=["post_rms", "peak", "none"],
                    value="post_rms",
                    label="Normalization",
                    info="post_rms is recommended for consistent volume.",
                )
            with gr.Row():
                process_effects = gr.Checkbox(label="Process effects (filter audio)", value=True)
                noise_reduction = gr.Checkbox(label="Noise reduction (clean audio)", value=False)
                clean_strength = gr.Slider(0, 1, value=0.5, step=0.05, label="Clean strength")
            with gr.Row():
                chunk_len = gr.Slider(0.5, 5.0, value=3.0, step=0.5, label="Chunk length (sec)")
                overlap_len = gr.Slider(0.0, 0.5, value=0.3, step=0.05, label="Overlap length (sec)")

        with gr.Row(equal_height=True):
            preprocess_btn = gr.Button("Preprocess dataset", variant="primary", scale=2)
            preprocess_msg = gr.Textbox(label="Output message", interactive=False, scale=3)

        preprocess_btn.click(
            run_preprocess_script,
            inputs=[train_model_name, dataset_path, sample_rate, cpu_threads,
                    cut_preprocess, process_effects, noise_reduction,
                    clean_strength, chunk_len, overlap_len, normalization_mode],
            outputs=[preprocess_msg],
        )

    # ── Step 2: Feature Extraction ───────────────────────────────
    with gr.Accordion("Step 2: feature extraction", open=False):
        with gr.Row():
            extract_model_name = gr.Textbox(
                label="Model name",
                info="Must match the name from Step 1.",
                placeholder="e.g. MyVoice",
            )
        with gr.Accordion("Options", open=False):
            with gr.Row():
                extract_f0 = gr.Dropdown(
                    choices=["rmvpe", "crepe", "crepe-tiny", "fcpe"],
                    value="rmvpe",
                    label="F0 method",
                    info="rmvpe is fast & accurate. crepe for singing quality.",
                )
                extract_embedder = gr.Dropdown(
                    choices=["contentvec", "contentvec_base", "chinese-hubert-base",
                             "japanese-hubert-base", "spin"],
                    value="contentvec",
                    label="Embedder model",
                    info="contentvec is the standard choice.",
                )
            with gr.Row():
                extract_sr = gr.Dropdown(
                    choices=["32000", "40000", "48000"],
                    value="40000",
                    label="Sample rate",
                    info="Must match the sample rate from Step 1.",
                )
                extract_gpu = gr.Slider(0, 7, value=0, step=1, label="GPU index")
                extract_cpu = gr.Slider(1, CPU_CORES, value=CPU_CORES, step=1, label="CPU threads")
            with gr.Row():
                extract_vocoder = gr.Dropdown(
                    choices=["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"],
                    value="HiFi-GAN",
                    label="Vocoder",
                    info="HiFi-GAN is the standard. MRF or RefineGAN for experimental quality.",
                )
                include_mutes = gr.Slider(0, 10, value=2, step=1, label="Include mutes",
                                          info="Number of mute audio segments to include.")

        with gr.Row(equal_height=True):
            extract_btn = gr.Button("Extract features", variant="primary", scale=2)
            extract_msg = gr.Textbox(label="Output message", interactive=False, scale=3)

        # Map vocoder names to architecture IDs used by extract script
        vocoder_map = {"HiFi-GAN": 0, "MRF HiFi-GAN": 1, "RefineGAN": 2}

        def do_extract(name, f0, cpu, gpu, sr, vocoder, embedder, mutes):
            if not name: raise gr.Error("Please enter a model name.")
            voc_arch = vocoder_map.get(vocoder, 0)
            return run_extract_script(name, f0, int(cpu), int(gpu), int(sr),
                                      voc_arch, embedder, None, int(mutes))

        extract_btn.click(
            do_extract,
            inputs=[extract_model_name, extract_f0, extract_cpu, extract_gpu,
                    extract_sr, extract_vocoder, extract_embedder, include_mutes],
            outputs=[extract_msg],
        )

    # ── Step 3: Model Training ───────────────────────────────────
    with gr.Accordion("Step 3: model training", open=False):
        with gr.Row():
            training_model_name = gr.Textbox(
                label="Model name",
                info="Must match the name from Step 1 & 2.",
                placeholder="e.g. MyVoice",
            )
        with gr.Accordion("Options", open=False):
            with gr.Row():
                total_epochs = gr.Slider(1, 2000, value=500, step=10,
                                         label="Total epochs",
                                         info="300-800 is typical. More = longer training, risk of overtraining.")
                batch_size = gr.Slider(1, 64, value=8, step=1,
                                       label="Batch size",
                                       info="Higher = faster but needs more VRAM. 8 for T4, 12-16 for A100.")
            with gr.Row():
                train_sr = gr.Dropdown(
                    choices=["32000", "40000", "48000"],
                    value="40000",
                    label="Sample rate",
                    info="Must match previous steps.",
                )
                save_interval = gr.Slider(1, 100, value=25, step=5,
                                           label="Save interval (epochs)",
                                           info="How often to save checkpoints.")
            with gr.Row():
                train_gpu = gr.Slider(0, 7, value=0, step=1, label="GPU index")
                vocoder_train = gr.Dropdown(
                    choices=["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"],
                    value="HiFi-GAN",
                    label="Vocoder",
                )
            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    pretrained = gr.Checkbox(label="Use pretrained model", value=True,
                                             info="Highly recommended. Dramatically improves quality.")
                    save_only_latest = gr.Checkbox(label="Save only latest checkpoint", value=True,
                                                   info="Saves disk space by keeping only the latest G/D files.")
                    save_weights = gr.Checkbox(label="Save weight files (.pth)", value=True,
                                               info="Required to use the model for inference.")
                with gr.Row():
                    use_warmup = gr.Checkbox(label="Learning rate warmup", value=False)
                    warmup_duration = gr.Slider(1, 50, value=5, step=1, label="Warmup duration (epochs)")
                    cleanup = gr.Checkbox(label="Clean up old files on start", value=False)
                with gr.Row():
                    index_algorithm = gr.Dropdown(
                        choices=["Auto", "Faiss", "KMeans"],
                        value="Auto",
                        label="Index algorithm",
                    )

        with gr.Row(equal_height=True):
            train_btn = gr.Button("Train model", variant="primary", scale=2)
            stop_btn = gr.Button("Stop training", variant="stop", scale=1)
            train_msg = gr.Textbox(label="Output message", interactive=False, scale=3)

        def do_train(name, save_freq, save_latest, save_w, epochs, sr, bs, gpu,
                     warmup, warmup_dur, use_pt, do_cleanup, idx_algo, vocoder):
            if not name: raise gr.Error("Please enter a model name.")
            return run_train_script(
                model_name=name,
                epoch_save_frequency=int(save_freq),
                save_only_latest_net_models=save_latest,
                save_weight_models=save_w,
                total_epoch_count=int(epochs),
                sample_rate=int(sr),
                batch_size=int(bs),
                gpu=int(gpu),
                use_warmup=warmup,
                warmup_duration=int(warmup_dur),
                pretrained=use_pt,
                cleanup=do_cleanup,
                index_algorithm=idx_algo,
                vocoder=vocoder,
            )

        train_btn.click(
            do_train,
            inputs=[training_model_name, save_interval, save_only_latest, save_weights,
                    total_epochs, train_sr, batch_size, train_gpu,
                    use_warmup, warmup_duration, pretrained, cleanup,
                    index_algorithm, vocoder_train],
            outputs=[train_msg],
        )
        stop_btn.click(stop_train_script)

    # ── Step 4: Generate Index (optional) ────────────────────────
    with gr.Accordion("Step 4: generate index (optional)", open=False):
        with gr.Row():
            index_model_name = gr.Textbox(
                label="Model name",
                info="Must match the trained model name.",
                placeholder="e.g. MyVoice",
            )
            index_algo = gr.Dropdown(
                choices=["Auto", "Faiss", "KMeans"],
                value="Auto",
                label="Index algorithm",
            )
        with gr.Row(equal_height=True):
            index_btn = gr.Button("Generate index", variant="primary", scale=2)
            index_msg = gr.Textbox(label="Output message", interactive=False, scale=3)

        def do_index(name, algo):
            if not name: raise gr.Error("Please enter a model name.")
            return run_index_script(name, algo)

        index_btn.click(do_index, inputs=[index_model_name, index_algo], outputs=[index_msg])


# ═══════════════════════════════════════════════════════════════════
#  SETTINGS TAB  — Clean Accordion Style
# ═══════════════════════════════════════════════════════════════════
def render_settings_tab():
    with gr.Accordion("About NEWRVC", open=True):
        gr.Markdown(
            "### NEWRVC ❤️\n"
            "A high-performance RVC voice conversion app with a clean, modern interface.\n\n"
            "- **Core Engine**: [codename-rvc-fork-4](https://github.com/codename0og/codename-rvc-fork-4)\n"
            "- **UI Inspiration**: [Ultimate RVC](https://github.com/JackismyShephard/ultimate-rvc)\n"
            "- **Workflow**: yt-dlp, audio-separator (MDX / Demucs)\n"
        )
    with gr.Accordion("Device info", open=False):
        import torch
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        gr.Markdown(f"**Device**: {device}\n\n**GPU**: {gpu_name}\n\n**CPU threads**: {CPU_CORES}")


# ═══════════════════════════════════════════════════════════════════
#  BUILD THE APP
# ═══════════════════════════════════════════════════════════════════
with gr.Blocks(title="NEWRVC ❤️", theme=theme, css=css) as app:
    gr.HTML("<h1>NEWRVC ❤️</h1>")

    with gr.Tab("Generate", elem_id="generate-tab"):
        render_generate_tab()

    with gr.Tab("Models", elem_id="manage-tab"):
        render_models_tab()

    with gr.Tab("Train", elem_id="train-tab"):
        render_train_tab()

    with gr.Tab("Settings", elem_id="settings-tab"):
        render_settings_tab()


if __name__ == "__main__":
    should_share = "--share" in sys.argv
    app.launch(server_port=7897, inbrowser=True, share=should_share)
