<div align="center">

# New RVC ❤️

**High-Performance AI Voice Conversion — Powered by Ultimate RVC Architecture**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DDME36/NEWRVC/blob/main/NewRVC_Colab.ipynb)
[![Train In Colab](https://img.shields.io/badge/🎯_Training_Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/DDME36/NEWRVC/blob/main/NewRVC_Training.ipynb)

</div>

---

An advanced fork of [Ultimate RVC](https://github.com/JackismyShephard/ultimate-rvc) with a custom **Red theme**, optimized pipeline, and enhanced core engine features. New RVC provides a professional, clean Gradio 5 interface for generating AI song covers, speech synthesis, and training custom voice models.

![New RVC Web Interface](images/webui_generate.png?raw=true)

## 🔥 What Makes New RVC Different?

### ⚡ The SOTA Core Engine
Underneath the beautiful UI is an absolute beast of an engine, fully modernized:

- **Next-Gen Vocoders**: Native support for **RingFormer (v1/v2)** and **APEX-GAN (PCPH-GAN)** for flawless audio phase reconstruction and human-like high notes.
- **Flawless Audio Isolation**: Powered by **BS-RoFormer** (Viperx 1297) and **MDX23C**, completely eliminating instrumental bleed and isolating pristine a cappellas.
- **PyTorch 2.x Acceleration**: Generator models are supercharged with `torch.compile(backend="inductor")`, boosting inference speeds by up to 2x-3x on supported GPUs.
- **Smart Auto-Tuning Trainer**: No more guessing epochs! The trainer actively monitors validation loss and automatically extracts the absolute `[model_name]_best_epoch.pth` at the peak of audio perfection.
- **Lightning Fast Setup**: Integrated with the `uv` package manager, reducing Colab installation times from minutes to mere seconds.

### 🎵 Recommended Settings (FAQ)
- **Embedder Model**: `contentvec` remains the undisputed gold standard. It perfectly captures phonetics and pronunciation without bleeding the original singer's vocal tone.
- **Pitch Extraction (F0 Method)**: `rmvpe` is the absolute best algorithm for most use cases, highly resistant to background noise and vocal artifacts. Use `fcpe` for extremely high pitches or to prioritize speed.

### 🎨 Custom Red Theme
A meticulously crafted `#ef4444` red accent theme with Google's Asap font — professional and premium-feeling.

## Features (Inherited from Ultimate RVC)

- One-click generation: Source → Separate → Convert → Mix in a single button
- Speech synthesis: TTS with any RVC voice model
- Custom configuration save/load system
- Gradio 5 with Python 3.12+ support

## ☁️ Google Colab

Don't have a strong GPU? Run New RVC directly in the cloud:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DDME36/NEWRVC/blob/main/NewRVC_Colab.ipynb)

## 🎯 Training Notebook (Recommended for Training)

For **dedicated voice model training**, use the standalone training notebook instead of the Gradio UI.
It calls the training functions directly — no UI overhead, no event loop — resulting in **significantly faster training**:

[![Open Training Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DDME36/NEWRVC/blob/main/NewRVC_Training.ipynb)

### Why use the Training Notebook?

| | Gradio UI | Training Notebook |
|---|-----------|-------------------|
| **Speed** | Slower (UI rendering, websocket) | ⚡ Direct function calls |
| **Hyperparameters** | Limited controls | Full control (LR, decay, seed) |
| **Monitoring** | Log output only | TensorBoard integration |
| **VRAM** | Higher (UI + model) | Lower (model only) |

### Recommended Settings

| Use Case | Dataset | Epochs | Batch | Vocoder | LR | Precision |
|----------|---------|--------|-------|---------|----|-----------|
| 🎙️ Speech | 10-30 min | 200-400 | 8-12 | HiFi-GAN | 1e-4 | FP16 |
| 🎵 Singing | 30-60 min | 400-800 | 6-8 | RingFormer_v2 | 5e-5 | BF16 |
| 🔄 Cross-Gender | 30-60 min | 500-800 | 6-8 | RefineGAN | 5e-5 | FP16 |

> **Note:** RingFormer and APEX-GAN don't have pretrained models yet — use `Pretrained=None` and train longer (800+ epochs).

## 💻 Local Setup

### Prerequisites
- Git
- Windows or Debian-based Linux

### Install

```console
git clone https://github.com/DDME36/NEWRVC.git
cd NEWRVC
./urvc install
```

### Run

```console
./urvc run
```

Once you see `Running on local URL: http://127.0.0.1:7860`, click the link to open the app.

### Update

```console
./urvc update
```

### Development Mode

```console
./urvc dev
```

## Usage

### Download Models

Navigate to `Models` > `Download`, paste the URL to a zip containing `.pth` and `.index` files, give it a unique name, and click **Download**.

### Generate Song Covers

**One-click**: Select source type, paste URL or upload file, choose voice model, click **Generate**.

**Multi-step**: Use the accordion steps to fine-tune each stage individually.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `URVC_MODELS_DIR` | Models storage directory | `./models` |
| `URVC_AUDIO_DIR` | Audio files directory | `./audio` |
| `URVC_TEMP_DIR` | Temporary files directory | `./temp` |
| `YT_COOKIEFILE` | YouTube cookies for download | None |
| `URVC_CONFIG` | Custom config name to load | Default |
| `URVC_ACCELERATOR` | `cuda` or `rocm` | `cuda` |

## 🏆 Credits

- **UI Architecture**: [Ultimate RVC](https://github.com/JackismyShephard/ultimate-rvc) by JackismyShephard
- **Core Engine**: codename-rvc-fork-4
- **Tools**: yt-dlp, audio-separator (MDX / Demucs), Gradio 5

## Terms of Use

The use of converted voice for the following purposes is prohibited:
- Criticizing or attacking individuals
- Political advocacy or opposing specific ideologies
- Selling voice models or generated voice clips
- Impersonation with malicious intent
- Fraudulent purposes leading to identity theft

## Disclaimer

I am not liable for any direct, indirect, consequential, incidental, or special damages arising out of or in any way connected with the use/misuse or inability to use this software.
