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

# Workflow imports
from workflow.youtube_dl import get_youtube_audio
from workflow.audio_separator import separate_audio

# CSS & Theme
theme_path = os.path.join(root_dir, "assets", "themes", "ultimate_red.json")
theme = gr.Theme.load(theme_path) if os.path.exists(theme_path) else "Default"

css = """
h1 { text-align: center; margin-top: 20px; margin-bottom: 20px; }
#generate-tab-button { font-weight: bold !important; }
#manage-tab-button   { font-weight: bold !important; }
#audio-tab-button    { font-weight: bold !important; }
#settings-tab-button { font-weight: bold !important; }
"""

CPU_CORES = cpu_count()
model_root = os.path.join(core_dir, "logs")


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════
def get_voice_model_names():
    """Return voice model NAMES (folder names that contain a .pth)."""
    if not os.path.exists(model_root):
        return []
    names = []
    for name in sorted(os.listdir(model_root)):
        folder = os.path.join(model_root, name)
        if not os.path.isdir(folder):
            continue
        has_pth = any(
            f.endswith((".pth", ".uvmp"))
            and not (f.startswith("G_") or f.startswith("D_"))
            for f in os.listdir(folder)
        )
        if has_pth:
            names.append(name)
    return names


def resolve_model_paths(model_name):
    """Given a model name, return (pth_path, index_path) automatically."""
    if not model_name:
        return "", ""
    folder = os.path.join(model_root, model_name)
    if not os.path.isdir(folder):
        return "", ""
    pth_path, index_path = "", ""
    for f in os.listdir(folder):
        full = os.path.join(folder, f)
        if f.endswith((".pth", ".uvmp")) and not (f.startswith("G_") or f.startswith("D_")):
            pth_path = full
        elif f.endswith(".index") and "trained" not in f:
            index_path = full
    return pth_path, index_path


def mix_audio(vocals_path, instrumental_path, vocals_volume=1.0, instrumental_volume=1.0):
    voc_data, voc_sr = sf.read(vocals_path)
    inst_data, inst_sr = sf.read(instrumental_path)
    if inst_sr != voc_sr:
        import librosa
        inst_data = librosa.resample(
            inst_data.T if inst_data.ndim > 1 else inst_data,
            orig_sr=inst_sr, target_sr=voc_sr,
        )
        if inst_data.ndim > 1:
            inst_data = inst_data.T
    min_len = min(len(voc_data), len(inst_data))
    voc_data, inst_data = voc_data[:min_len], inst_data[:min_len]
    if voc_data.ndim == 1 and inst_data.ndim > 1:
        voc_data = np.stack([voc_data, voc_data], axis=-1)
    elif voc_data.ndim > 1 and inst_data.ndim == 1:
        inst_data = np.stack([inst_data, inst_data], axis=-1)
    mixed = (voc_data * vocals_volume) + (inst_data * instrumental_volume)
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    out_path = os.path.join(root_dir, "workflow_output", "covers", "cover_output.wav")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, mixed, voc_sr)
    return out_path


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
        raise gr.Error("Failed to download the model.")
    downloaded_zip = None
    for f in os.listdir(zips_folder):
        if f.endswith(".zip"):
            downloaded_zip = os.path.join(zips_folder, f)
            break
    if not downloaded_zip:
        raise gr.Error("No ZIP file found after download.")
    temp_extract = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(downloaded_zip, "r") as zf:
            zf.extractall(temp_extract)
    except Exception as e:
        shutil.rmtree(temp_extract)
        raise gr.Error(f"Failed to extract zip: {e}")
    pth_file, index_file = None, None
    for walk_root, _, files in os.walk(temp_extract):
        for f in files:
            if f.endswith(".pth") and "G_" not in f and "D_" not in f:
                pth_file = os.path.join(walk_root, f)
            elif f.endswith(".index") and "trained" not in f:
                index_file = os.path.join(walk_root, f)
    if not pth_file:
        shutil.rmtree(temp_extract)
        raise gr.Error("No valid .pth model found in archive.")
    final_folder = os.path.join(model_root, model_name)
    os.makedirs(final_folder, exist_ok=True)
    shutil.move(pth_file, os.path.join(final_folder, f"{model_name}.pth"))
    if index_file:
        shutil.move(index_file, os.path.join(final_folder, f"{model_name}.index"))
    shutil.rmtree(temp_extract)
    os.remove(downloaded_zip)
    return f"[+] Voice Model '{model_name}' ready!"


# ═══════════════════════════════════════════════════════════════════
#  BUILD THE APP  — Matching Ultimate RVC layout exactly
#
#  Top tabs:  Generate | Models | Audio | Settings
#
#  Generate > Song covers > One-click | Multi-step
#  Models   > Download | Upload | Train | Delete
# ═══════════════════════════════════════════════════════════════════
with gr.Blocks(title="NEWRVC ❤️", theme=theme, css=css) as app:
    gr.HTML("<h1>NEWRVC ❤️</h1>")

    # ──────────────────────────────────────────────────────────────
    #  TAB: Generate
    # ──────────────────────────────────────────────────────────────
    with gr.Tab("Generate", elem_id="generate-tab"):
        with gr.Tab("Song covers"):

            voice_names = get_voice_model_names()
            default_voice = voice_names[0] if voice_names else None

            # ─── One-click (flat layout, no steps) ───────────────
            with gr.Tab("One-click"):
                with gr.Row():
                    with gr.Column():
                        source_type_oc = gr.Dropdown(
                            choices=["YouTube link/local path", "Local file", "Cached song"],
                            value="YouTube link/local path",
                            label="Source type",
                            info="The type of source to retrieve a song from.",
                        )
                    with gr.Column():
                        source_oc = gr.Textbox(
                            label="Source",
                            info="Link to a song on YouTube or the full path of a local audio file.",
                        )

                voice_model_oc = gr.Dropdown(
                    label="Voice model",
                    choices=voice_names,
                    value=default_voice,
                    info="Select a model to use for voice conversion.",
                )

                with gr.Accordion("Options", open=False):
                    with gr.Row():
                        pitch_oc = gr.Slider(-24, 24, value=0, step=1, label="Pitch (semitones)")
                        f0_method_oc = gr.Dropdown(
                            choices=["rmvpe", "crepe", "crepe-tiny", "fcpe", "hybrid[rmvpe+fcpe]"],
                            value="rmvpe", label="F0 method",
                        )
                    with gr.Accordion("Advanced", open=False):
                        with gr.Row():
                            index_rate_oc = gr.Slider(0, 1, value=0.75, step=0.05, label="Index rate")
                            rms_mix_rate_oc = gr.Slider(0, 1, value=0.25, step=0.05, label="Volume envelope")
                        with gr.Row():
                            filter_radius_oc = gr.Slider(0, 10, value=3, step=1, label="Filter radius")
                            protect_oc = gr.Slider(0, 0.5, value=0.33, step=0.01, label="Protect rate")
                        with gr.Row():
                            split_audio_oc = gr.Checkbox(label="Split audio", value=False)
                            autotune_oc = gr.Checkbox(label="Autotune", value=False)
                            clean_audio_oc = gr.Checkbox(label="Clean audio", value=False)
                        with gr.Row():
                            embedder_oc = gr.Dropdown(
                                choices=["contentvec", "contentvec_base", "chinese-hubert-base",
                                         "japanese-hubert-base", "spin"],
                                value="contentvec", label="Embedder model",
                            )
                            export_format_oc = gr.Radio(
                                choices=["WAV", "MP3", "FLAC"], value="WAV", label="Export format",
                            )

                with gr.Row():
                    reset_oc_btn = gr.Button("Reset options")
                    generate_oc_btn = gr.Button("Generate", variant="primary")

                song_cover_oc = gr.Audio(
                    label="Song cover", type="filepath", interactive=False,
                    waveform_options=gr.WaveformOptions(show_recording_waveform=False),
                )

                # One-click: full pipeline in one go
                def one_click_generate(source, model_name, p, f0, ir, fr, rmr, prot,
                                       split, at, clean, emb, fmt):
                    if not source:
                        raise gr.Error("Please enter a source URL or path.")
                    if not model_name:
                        raise gr.Error("No voice model selected!")

                    pth_path, index_path = resolve_model_paths(model_name)
                    if not pth_path:
                        raise gr.Error(f"No .pth found for model '{model_name}'.")

                    # Step 1: Download / retrieve
                    gr.Info("Retrieving song...")
                    dl_dir = os.path.join(root_dir, "workflow_output", "downloads")
                    if source.startswith("http"):
                        audio_path = get_youtube_audio(source, dl_dir)
                    else:
                        audio_path = source

                    # Step 2: Separate vocals
                    gr.Info("Separating vocals...")
                    sep_dir = os.path.join(root_dir, "workflow_output", "separated")
                    vocals_path, inst_path = separate_audio(
                        input_audio_path=audio_path,
                        output_dir=sep_dir,
                        use_demucs=False,
                    )

                    # Step 3: Convert vocals
                    gr.Info("Converting vocals...")
                    out_path = os.path.join(root_dir, "workflow_output", "converted", "converted_vocals.wav")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    run_infer_script(
                        pitch=p, filter_radius=fr, index_rate=ir,
                        volume_envelope=rmr, protect=prot, f0_method=f0,
                        input_path=vocals_path, output_path=out_path,
                        pth_path=pth_path, index_path=index_path,
                        split_audio=split, f0_autotune=at, f0_autotune_strength=1.0,
                        clean_audio=clean, clean_strength=0.3, export_format=fmt,
                        f0_file=None, embedder_model=emb, embedder_model_custom=None,
                    )

                    # Step 4: Mix
                    gr.Info("Mixing song cover...")
                    cover = mix_audio(out_path, inst_path)
                    return cover

                generate_oc_btn.click(
                    one_click_generate,
                    inputs=[source_oc, voice_model_oc, pitch_oc, f0_method_oc,
                            index_rate_oc, filter_radius_oc, rms_mix_rate_oc, protect_oc,
                            split_audio_oc, autotune_oc, clean_audio_oc, embedder_oc, export_format_oc],
                    outputs=[song_cover_oc],
                )

                reset_oc_btn.click(
                    lambda: [0, "rmvpe", 0.75, 0.25, 3, 0.33, False, False, False, "contentvec", "WAV"],
                    outputs=[pitch_oc, f0_method_oc, index_rate_oc, rms_mix_rate_oc,
                             filter_radius_oc, protect_oc, split_audio_oc, autotune_oc,
                             clean_audio_oc, embedder_oc, export_format_oc],
                    show_progress="hidden",
                )

            # ─── Multi-step ──────────────────────────────────────
            with gr.Tab("Multi-step"):

                # Step 0: Song Retrieval
                with gr.Accordion("Step 0: song retrieval", open=True):
                    with gr.Row():
                        with gr.Column():
                            source_type_ms = gr.Dropdown(
                                choices=["YouTube link/local path", "Local file"],
                                value="YouTube link/local path",
                                label="Source type",
                            )
                        with gr.Column():
                            source_ms = gr.Textbox(label="Source", placeholder="https://youtube.com/watch?v=...")
                            local_file_ms = gr.Audio(
                                label="Source", type="filepath", visible=False,
                                waveform_options=gr.WaveformOptions(show_recording_waveform=False),
                            )
                    source_type_ms.input(
                        lambda t: (gr.update(visible=t == "YouTube link/local path"),
                                   gr.update(visible=t == "Local file")),
                        inputs=source_type_ms,
                        outputs=[source_ms, local_file_ms],
                        show_progress="hidden",
                    )
                    with gr.Row():
                        retrieve_btn = gr.Button("Retrieve song", variant="primary")
                    song_output = gr.Audio(
                        label="Song", type="filepath", interactive=False,
                        waveform_options=gr.WaveformOptions(show_recording_waveform=False),
                    )

                # Step 1: Vocal Separation
                with gr.Accordion("Step 1: vocal separation", open=False):
                    with gr.Accordion("Options", open=False):
                        with gr.Row():
                            sep_model_ms = gr.Dropdown(
                                choices=["UVR-MDX-NET-Voc_FT", "htdemucs_ft"],
                                value="UVR-MDX-NET-Voc_FT", label="Separation model",
                            )
                    with gr.Row():
                        separate_btn = gr.Button("Separate vocals", variant="primary")
                    with gr.Row():
                        primary_stem = gr.Audio(
                            label="Primary stem", type="filepath", interactive=False,
                            waveform_options=gr.WaveformOptions(show_recording_waveform=False),
                        )
                        secondary_stem = gr.Audio(
                            label="Secondary stem", type="filepath", interactive=False,
                            waveform_options=gr.WaveformOptions(show_recording_waveform=False),
                        )

                # Step 2: Vocal Conversion
                with gr.Accordion("Step 2: vocal conversion", open=False):
                    voice_model_ms = gr.Dropdown(
                        label="Voice model", choices=voice_names, value=default_voice,
                        info="Select a model to use for voice conversion.",
                    )
                    with gr.Accordion("Options", open=False):
                        with gr.Row():
                            pitch_ms = gr.Slider(-24, 24, value=0, step=1, label="Pitch (semitones)")
                            f0_method_ms = gr.Dropdown(
                                choices=["rmvpe", "crepe", "crepe-tiny", "fcpe", "hybrid[rmvpe+fcpe]"],
                                value="rmvpe", label="F0 method",
                            )
                        with gr.Accordion("Advanced", open=False):
                            with gr.Accordion("Voice synthesis", open=False):
                                with gr.Row():
                                    index_rate_ms = gr.Slider(0, 1, value=0.75, step=0.05, label="Index rate")
                                    rms_mix_rate_ms = gr.Slider(0, 1, value=0.25, step=0.05, label="Volume envelope")
                                with gr.Row():
                                    filter_radius_ms = gr.Slider(0, 10, value=3, step=1, label="Filter radius")
                                    protect_ms = gr.Slider(0, 0.5, value=0.33, step=0.01, label="Protect rate")
                            with gr.Accordion("Vocal enrichment", open=False):
                                with gr.Row():
                                    split_ms = gr.Checkbox(label="Split audio", value=False)
                                with gr.Row():
                                    autotune_ms = gr.Checkbox(label="Autotune", value=False)
                                    clean_ms = gr.Checkbox(label="Clean audio", value=False)
                            with gr.Accordion("Speaker embeddings", open=False):
                                with gr.Row():
                                    embedder_ms = gr.Dropdown(
                                        choices=["contentvec", "contentvec_base", "chinese-hubert-base",
                                                 "japanese-hubert-base", "spin"],
                                        value="contentvec", label="Embedder model",
                                    )
                                export_format_ms = gr.Radio(
                                    choices=["WAV", "MP3", "FLAC"], value="WAV", label="Export format",
                                )
                    with gr.Row():
                        convert_reset_btn = gr.Button("Reset options")
                        convert_btn = gr.Button("Convert vocals", variant="primary")
                    converted_vocals_out = gr.Audio(
                        label="Converted vocals", type="filepath", interactive=False,
                        waveform_options=gr.WaveformOptions(show_recording_waveform=False),
                    )

                # Step 3: Song Mixing
                with gr.Accordion("Step 3: song mixing", open=False):
                    with gr.Accordion("Options", open=False):
                        with gr.Row():
                            main_gain = gr.Slider(0, 2, value=1.0, step=0.05, label="Main vocals gain")
                            inst_gain = gr.Slider(0, 2, value=1.0, step=0.05, label="Instrumentals gain")
                    with gr.Row():
                        mix_reset_btn = gr.Button("Reset options")
                        mix_btn = gr.Button("Mix song cover", variant="primary")
                    song_cover_ms = gr.Audio(
                        label="Song cover", type="filepath", interactive=False,
                        waveform_options=gr.WaveformOptions(show_recording_waveform=False),
                    )

                # ── Multi-step event wiring ──────────────────────
                def ms_retrieve(src_type, src, local):
                    if src_type == "YouTube link/local path":
                        if not src:
                            raise gr.Error("Please enter a source.")
                        if src.startswith("http"):
                            return get_youtube_audio(src, os.path.join(root_dir, "workflow_output", "downloads"))
                        return src
                    else:
                        if not local:
                            raise gr.Error("Please upload an audio file.")
                        return local

                def ms_separate(audio, sep):
                    if not audio:
                        raise gr.Error("No audio to separate!")
                    v, i = separate_audio(
                        input_audio_path=audio,
                        output_dir=os.path.join(root_dir, "workflow_output", "separated"),
                        use_demucs="demucs" in sep.lower(),
                    )
                    return v, i

                def ms_convert(vocal, model_name, p, f0, ir, fr, rmr, prot, emb, fmt, split, at, clean):
                    if not vocal:
                        raise gr.Error("No vocals to convert!")
                    if not model_name:
                        raise gr.Error("No voice model selected!")
                    pth, idx = resolve_model_paths(model_name)
                    if not pth:
                        raise gr.Error(f"No .pth found for '{model_name}'.")
                    out = os.path.join(root_dir, "workflow_output", "converted", "converted_vocals.wav")
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    run_infer_script(
                        pitch=p, filter_radius=fr, index_rate=ir,
                        volume_envelope=rmr, protect=prot, f0_method=f0,
                        input_path=vocal, output_path=out,
                        pth_path=pth, index_path=idx,
                        split_audio=split, f0_autotune=at, f0_autotune_strength=1.0,
                        clean_audio=clean, clean_strength=0.3, export_format=fmt,
                        f0_file=None, embedder_model=emb, embedder_model_custom=None,
                    )
                    return out

                def ms_mix(conv, inst, vg, ig):
                    if not conv or not inst:
                        raise gr.Error("Need both converted vocals and instrumental!")
                    return mix_audio(conv, inst, vg, ig)

                retrieve_btn.click(ms_retrieve, [source_type_ms, source_ms, local_file_ms], [song_output])
                separate_btn.click(ms_separate, [song_output, sep_model_ms], [primary_stem, secondary_stem])
                convert_btn.click(
                    ms_convert,
                    [primary_stem, voice_model_ms, pitch_ms, f0_method_ms,
                     index_rate_ms, filter_radius_ms, rms_mix_rate_ms, protect_ms,
                     embedder_ms, export_format_ms, split_ms, autotune_ms, clean_ms],
                    [converted_vocals_out],
                )
                mix_btn.click(ms_mix, [converted_vocals_out, secondary_stem, main_gain, inst_gain], [song_cover_ms])

    # ──────────────────────────────────────────────────────────────
    #  TAB: Models  (sub-tabs: Download | Upload | Train | Delete)
    # ──────────────────────────────────────────────────────────────
    with gr.Tab("Models", elem_id="manage-tab"):

        # ─── Download ────────────────────────────────────────────
        with gr.Tab("Download"):
            with gr.Accordion("Voice models"):
                with gr.Row():
                    vm_url = gr.Textbox(
                        label="Model URL",
                        info="Should point to a zip file containing a .pth model file and optionally also an .index file.",
                    )
                    vm_name = gr.Textbox(
                        label="Model name",
                        info="Enter a unique name for the voice model.",
                    )
                with gr.Row(equal_height=True):
                    dl_voice_btn = gr.Button("Download 🌐", variant="primary", scale=19)
                    dl_voice_msg = gr.Textbox(label="Output message", interactive=False, scale=20)
                dl_voice_btn.click(download_custom_voice_model, [vm_url, vm_name], [dl_voice_msg])

            with gr.Accordion("Pretrained models", open=False):
                pt_list = get_pretrained_list()
                default_pt = "Titan" if "Titan" in pt_list else (pt_list[0] if pt_list else "")
                default_rates = get_pretrained_sample_rates(default_pt) if default_pt else []
                with gr.Row():
                    pt_model = gr.Dropdown(
                        label="Pretrained model", choices=pt_list, value=default_pt,
                        info="Select the pretrained model you want to download.",
                    )
                    pt_sr = gr.Dropdown(
                        label="Sample rate", choices=default_rates,
                        value=default_rates[0] if default_rates else None,
                        info="Select the sample rate for the pretrained model.",
                    )
                pt_model.change(
                    lambda m: gr.update(
                        choices=get_pretrained_sample_rates(m),
                        value=get_pretrained_sample_rates(m)[0] if get_pretrained_sample_rates(m) else None,
                    ),
                    inputs=[pt_model], outputs=[pt_sr],
                )
                with gr.Row(equal_height=True):
                    dl_pt_btn = gr.Button("Download 🌐", variant="primary", scale=19)
                    dl_pt_msg = gr.Textbox(label="Output message", interactive=False, scale=20)
                dl_pt_btn.click(
                    lambda m, s: core_download_pretrained_model(m, s) or f"[+] Downloaded {m} ({s})!",
                    [pt_model, pt_sr], [dl_pt_msg],
                )

        # ─── Upload ──────────────────────────────────────────────
        with gr.Tab("Upload"):
            with gr.Accordion("Voice models", open=True):
                gr.Markdown(
                    "Drag and drop your `.pth` and `.index` files here — "
                    "they will be automatically saved to `logs/`."
                )
                dropbox = gr.File(
                    label="Drag your .pth file and .index file into this space.",
                    type="filepath",
                )
                dropbox.upload(fn=save_drop_model, inputs=[dropbox], outputs=[dropbox])

        # ─── Train ───────────────────────────────────────────────
        with gr.Tab("Train"):
            # Step 1: Preprocessing
            with gr.Accordion("Step 1: dataset preprocessing", open=True):
                with gr.Row():
                    train_name = gr.Textbox(label="Model name", placeholder="e.g. MyVoice")
                    dataset_path = gr.Textbox(label="Dataset path", placeholder="/content/dataset")
                with gr.Accordion("Options", open=False):
                    with gr.Row():
                        train_sr = gr.Dropdown(["32000", "40000", "48000"], value="40000", label="Sample rate")
                        cpu_threads = gr.Slider(1, CPU_CORES, value=CPU_CORES, step=1, label="CPU threads")
                    with gr.Row():
                        cut_preprocess = gr.Dropdown(
                            ["Skip", "Simple", "Automatic"], value="Automatic", label="Split method",
                        )
                        normalization = gr.Dropdown(
                            ["post_rms", "peak", "none"], value="post_rms", label="Normalization",
                        )
                    with gr.Row():
                        process_fx = gr.Checkbox(label="Filter audio", value=True)
                        noise_red = gr.Checkbox(label="Noise reduction", value=False)
                        clean_str = gr.Slider(0, 1, value=0.5, step=0.05, label="Clean strength")
                    with gr.Row():
                        chunk_len = gr.Slider(0.5, 5.0, value=3.0, step=0.5, label="Chunk length (sec)")
                        overlap_len = gr.Slider(0.0, 0.5, value=0.3, step=0.05, label="Overlap (sec)")
                with gr.Row(equal_height=True):
                    preprocess_btn = gr.Button("Preprocess dataset", variant="primary", scale=2)
                    preprocess_msg = gr.Textbox(label="Output message", interactive=False, scale=3)
                preprocess_btn.click(
                    run_preprocess_script,
                    [train_name, dataset_path, train_sr, cpu_threads,
                     cut_preprocess, process_fx, noise_red, clean_str, chunk_len, overlap_len, normalization],
                    [preprocess_msg],
                )

            # Step 2: Extraction
            with gr.Accordion("Step 2: feature extraction", open=False):
                with gr.Accordion("Options", open=False):
                    with gr.Row():
                        ext_f0 = gr.Dropdown(
                            ["rmvpe", "crepe", "crepe-tiny", "fcpe"], value="rmvpe", label="F0 method",
                        )
                        ext_emb = gr.Dropdown(
                            ["contentvec", "contentvec_base", "chinese-hubert-base",
                             "japanese-hubert-base", "spin"],
                            value="contentvec", label="Embedder model",
                        )
                    with gr.Row():
                        ext_voc = gr.Dropdown(
                            ["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"],
                            value="HiFi-GAN", label="Vocoder",
                        )
                        ext_gpu = gr.Slider(0, 7, value=0, step=1, label="GPU index")
                        ext_cpu = gr.Slider(1, CPU_CORES, value=CPU_CORES, step=1, label="CPU threads")
                    with gr.Row():
                        ext_mutes = gr.Slider(0, 10, value=2, step=1, label="Include mutes")
                with gr.Row(equal_height=True):
                    extract_btn = gr.Button("Extract features", variant="primary", scale=2)
                    extract_msg = gr.Textbox(label="Output message", interactive=False, scale=3)

                voc_map = {"HiFi-GAN": 0, "MRF HiFi-GAN": 1, "RefineGAN": 2}

                def do_extract(f0, cpu, gpu, voc, emb, mutes):
                    name = train_name.value if hasattr(train_name, "value") else ""
                    sr_val = train_sr.value if hasattr(train_sr, "value") else "40000"
                    return run_extract_script(
                        name, f0, int(cpu), int(gpu), int(sr_val),
                        voc_map.get(voc, 0), emb, None, int(mutes),
                    )

                extract_btn.click(
                    lambda f0, cpu, gpu, sr, voc, emb, mutes, name: run_extract_script(
                        name, f0, int(cpu), int(gpu), int(sr),
                        voc_map.get(voc, 0), emb, None, int(mutes),
                    ),
                    [ext_f0, ext_cpu, ext_gpu, train_sr, ext_voc, ext_emb, ext_mutes, train_name],
                    [extract_msg],
                )

            # Step 3: Training
            with gr.Accordion("Step 3: model training", open=False):
                with gr.Accordion("Options", open=False):
                    with gr.Row():
                        total_epochs = gr.Slider(1, 2000, value=500, step=10, label="Total epochs")
                        batch_sz = gr.Slider(1, 64, value=8, step=1, label="Batch size")
                    with gr.Row():
                        save_interval = gr.Slider(1, 100, value=25, step=5, label="Save interval")
                        t_gpu = gr.Slider(0, 7, value=0, step=1, label="GPU index")
                    with gr.Row():
                        t_vocoder = gr.Dropdown(
                            ["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"],
                            value="HiFi-GAN", label="Vocoder",
                        )
                    with gr.Accordion("Advanced", open=False):
                        with gr.Row():
                            use_pretrained = gr.Checkbox(label="Use pretrained", value=True)
                            save_latest = gr.Checkbox(label="Save only latest", value=True)
                            save_weights = gr.Checkbox(label="Save weights (.pth)", value=True)
                        with gr.Row():
                            warmup = gr.Checkbox(label="LR warmup", value=False)
                            warmup_dur = gr.Slider(1, 50, value=5, step=1, label="Warmup epochs")
                            do_cleanup = gr.Checkbox(label="Cleanup old files", value=False)
                        with gr.Row():
                            idx_algo = gr.Dropdown(
                                ["Auto", "Faiss", "KMeans"], value="Auto", label="Index algorithm",
                            )
                with gr.Row(equal_height=True):
                    train_btn = gr.Button("Train model", variant="primary", scale=2)
                    stop_btn = gr.Button("Stop training", variant="stop", scale=1)
                    train_msg = gr.Textbox(label="Output message", interactive=False, scale=3)

                train_btn.click(
                    lambda name, sf_, sl, sw, ep, sr, bs, gpu, wu, wd, pt, cl, ia, vc: run_train_script(
                        model_name=name, epoch_save_frequency=int(sf_),
                        save_only_latest_net_models=sl, save_weight_models=sw,
                        total_epoch_count=int(ep), sample_rate=int(sr),
                        batch_size=int(bs), gpu=int(gpu),
                        use_warmup=wu, warmup_duration=int(wd),
                        pretrained=pt, cleanup=cl, index_algorithm=ia, vocoder=vc,
                    ),
                    [train_name, save_interval, save_latest, save_weights,
                     total_epochs, train_sr, batch_sz, t_gpu,
                     warmup, warmup_dur, use_pretrained, do_cleanup, idx_algo, t_vocoder],
                    [train_msg],
                )
                stop_btn.click(stop_train_script)

        # ─── Delete ──────────────────────────────────────────────
        with gr.Tab("Delete"):
            with gr.Accordion("Delete voice model", open=True):
                del_model = gr.Dropdown(
                    label="Voice model",
                    choices=get_voice_model_names(),
                    info="Select a voice model to delete.",
                )
                with gr.Row(equal_height=True):
                    del_btn = gr.Button("Delete", variant="stop", scale=19)
                    del_msg = gr.Textbox(label="Output message", interactive=False, scale=20)

                def delete_voice_model(name):
                    if not name:
                        raise gr.Error("No model selected.")
                    folder = os.path.join(model_root, name)
                    if os.path.isdir(folder):
                        shutil.rmtree(folder)
                        return f"[+] Deleted '{name}'."
                    raise gr.Error(f"Model folder '{name}' not found.")

                del_btn.click(delete_voice_model, [del_model], [del_msg])

    # ──────────────────────────────────────────────────────────────
    #  TAB: Audio
    # ──────────────────────────────────────────────────────────────
    with gr.Tab("Audio", elem_id="audio-tab"):
        gr.Markdown("### Manage audio files")
        gr.Markdown("Generated audio files are saved in `workflow_output/`.")

    # ──────────────────────────────────────────────────────────────
    #  TAB: Settings
    # ──────────────────────────────────────────────────────────────
    with gr.Tab("Settings", elem_id="settings-tab"):
        with gr.Accordion("About NEWRVC", open=True):
            gr.Markdown(
                "### NEWRVC ❤️\n"
                "A high-performance RVC voice conversion app with a clean, modern interface.\n\n"
                "- **Core Engine**: codename-rvc-fork-4\n"
                "- **UI Inspiration**: Ultimate RVC\n"
                "- **Workflow**: yt-dlp, audio-separator (MDX / Demucs)\n"
            )
        with gr.Accordion("Device info", open=False):
            import torch
            device = "CUDA" if torch.cuda.is_available() else "CPU"
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            gr.Markdown(f"**Device**: {device}\n\n**GPU**: {gpu_name}\n\n**CPU threads**: {CPU_CORES}")


if __name__ == "__main__":
    should_share = "--share" in sys.argv
    app.launch(server_port=7897, inbrowser=True, share=should_share)
