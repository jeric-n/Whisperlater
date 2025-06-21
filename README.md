# Whisperlater

**Whisperlater** is a self-hosted, high-performance AI transcription service powered by OpenAI's Whisper `large-v3` model. It provides a sleek, dark-themed web interface to transcribe audio and video files with state-of-the-art accuracy, optimized for both clean audio and noisy environments like gaming streams and speeches in public.

The entire application is containerized with Docker and accelerated by NVIDIA GPUs, making deployment simple and performance incredibly fast.

---

## Key Features

- 🚀 **High Performance:** Utilizes `faster-whisper`, a CTranslate2 reimplementation of Whisper that is up to 4 times faster and uses 50% less memory.
- 💡 **State-of-the-Art Accuracy:** Powered by the `large-v3` model, offering the best available transcription quality and multilingual support.
- 🌐 **Sleek Web Interface:** A modern, "perfect black" dark theme that's easy on the eyes and simple to use.
- 🐳 **Dockerized for Easy Deployment:** Get the entire service running with a single `docker-compose` command. No need to manage Python dependencies or model downloads manually.
- ⚡ **GPU Accelerated:** Natively supports NVIDIA GPUs via the Docker container for maximum transcription speed.
- 🧠 **Intelligent & Robust:**
  - **Voice Activity Detection (VAD):** Intelligently filters out silence and non-speech noise before transcription, increasing accuracy and speed.
  - **Optimized Parameters:** Comes pre-tuned for high accuracy and resistance to background noise.
  - **Robust Memory Management:** Explicit garbage collection and GPU cache clearing after each job ensures stability for long-running, continuous use.
- 📄 **Multiple Output Formats:** Download your transcriptions as either plain text (`.txt`) or timestamped subtitles (`.srt`).
- scalable **Handles Large Files with Ease:** The streaming architecture can process files of any size (e.g., multi-hour podcasts or meetings) with low, constant memory usage.

## Tech Stack

- **Backend:** Python 3, Flask
- **AI Model:** OpenAI Whisper `large-v3`
- **Inference Engine:** `faster-whisper`
- **Containerization:** Docker, Docker Compose
- **Acceleration:** NVIDIA CUDA

## Getting Started

Follow these instructions to get your own instance of Whisperlater running.

### Prerequisites

- **Docker:** [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose:** Usually included with Docker Desktop. If not, [install Docker Compose](https://docs.docker.com/compose/install/).
- **NVIDIA GPU:** A CUDA-enabled NVIDIA GPU is required. *3070 and above recommended*
- **NVIDIA Container Toolkit:** This allows Docker to access your GPU. [Installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Installation & Running

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/jeric-n/whisperlater.git
    cd whisperlater
    ```

2. **Build the Docker Image:**
    This command builds the container, installs all dependencies, and downloads/converts the Whisper model. **This first build will take a significant amount of time** (15-30 minutes depending on your internet and CPU speed) as it downloads the ~3 GB model. Subsequent builds will be much faster.

    ```bash
    docker-compose build
    ```

3. **Start the Service:**
    Once the build is complete, start the application in detached mode.

    ```bash
    docker-compose up -d
    ```

4. **Access the Web UI:**
    Open your web browser and navigate to:
    **`http://localhost:5000`**

The service is now running! You can upload a file to begin transcribing.

To stop the service, run `docker-compose down`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- This project would not be possible without the incredible work of the teams behind [OpenAI's Whisper](https://github.com/openai/whisper) and [SYSTRAN's faster-whisper](https://github.com/SYSTRAN/faster-whisper).
- Gemini 2.5 Pro for enabling me to get me started with self-hosting AI models, the python libraries, optimizations and the idea itself within a week's time.
