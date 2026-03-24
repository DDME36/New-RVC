<div align="center">

# 🚀 NEWRVC

**High-Performance Voice Conversion — Now with a Stunning Minimalist UI**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MIGHTYBIT/NEWRVC/blob/main/NewRVC_Colab.ipynb)

</div>

<br/>

**NEWRVC** is a completely overhauled, streamlined interface for the cutting-edge `codename-rvc-fork-4` core engine. We stripped away the convoluted, outdated Applio tabs and rebuilt a blazing-fast, **White/Red minimalist Gradio 5 UI** inspired by Ultimate RVC.

---

### 🔥 What Makes NEWRVC Different?

1. **🎵 The 4-Step Song Cover Pipeline**
   Create AI covers on a single elegant page. 
   **Input** (YouTube/File) ➔ **Separate** (MDX/Demucs) ➔ **Convert** (RVC) ➔ **Mix**. No more bouncing between 4 different tabs or messing with external tools.

2. **⬇️ Smart Custom Model Downloader**
   Sick of your downloaded models keeping their random `.zip` or unformatted names? Our custom downloader lets you input a **Model URL** and a **Custom Model Name**. NEWRVC automatically extracts, cleans, renames your `.pth` and `.index` files to your exact chosen name, and neatly places them in your `logs/` directory. 

3. **🎨 White & Red Aesthetic**
   A highly polished, meticulously designed UI theme (`ultimate_red`) featuring Google's Asap font, custom `#ef4444` accents, and a clean interface that feels premium.

4. **⚡ The Absolute Best Core Engine**
   Underneath the beautiful UI is the absolute beast of an engine:
   - Native **RingFormer** support
   - Experimental **PCPH-GAN** architecture
   - Next-gen **Spin Models** embedder
   - `uv` Python package manager integration for Colab builds that take seconds, not minutes.

---

## 💻 Local Installation (Windows)

1. **Clone the Repo**
   ```bash
   git clone https://github.com/DDME36/NEWRVC.git
   cd NEWRVC
   ```

2. **Install `uv` (Fastest Python Package Manager)**
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Create Environment and Install**
   ```bash
   uv venv .venv --python 3.11
   .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

4. **Launch the App**
   ```bash
   python app.py
   ```

---

## ☁️ Google Colab

Don't have a strong GPU? Run NEWRVC directly in the cloud in just one click:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MIGHTYBIT/NEWRVC/blob/main/NewRVC_Colab.ipynb)

---

### 🏆 Credits
- **Core Engine**: Built upon the fantastic work of [codename-rvc-fork-4](https://github.com/codename0og/codename-rvc-fork-4).
- **UI Inspiration**: Heavily inspired by the clean layout of [Ultimate RVC](https://github.com/JackismyShephard/ultimate-rvc).
- **Tools**: yt-dlp, audio-separator (MDX Kim / Demucs).
