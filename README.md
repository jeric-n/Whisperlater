# Whisperlater - High-Quality Self Host Whisper Transcription and Translations Server

This project provides a simple and powerful web server for generating high-quality transcriptions of audio and video files. It uses a combination of state-of-the-art tools to deliver fast, accurate, and well-formatted results. Supports translations.

The application is containerized with Docker for easy setup and deployment.

## Key Features

*   **High-Speed Transcription**: Powered by `faster-whisper`, a CTranslate2 reimplementation of OpenAI's Whisper model, providing significant speedups on GPU.
*   **High Accuracy**: Utilizes the `large-v3` Whisper model for state-of-the-art transcription quality.
*   **Voice Activity Detection (VAD)**: Uses the Silero VAD model to pre-process audio, removing long silences and improving transcription speed and accuracy by focusing only on speech segments.
*   **Polished Output**:
    *   Generates both plain text (`.txt`) and subtitle (`.srt`) files.
    *   Automatically splits long sentences into shorter, more readable subtitle chunks.
*   **Simple Web Interface**: An easy-to-use, single-page web UI for uploading files and selecting transcription options (task, language, format).
*   **Easy Deployment**: Fully containerized with Docker, with all models pre-loaded into the image for fast startup.

## Prerequisites

To run this application, you will need:

1.  **An NVIDIA GPU**: The Docker image is configured to use CUDA for GPU acceleration.
2.  **Docker**: [Install Docker](https://docs.docker.com/engine/install/) on your system.
3. *Possibly Optional: NVIDIA CUDA Toolkit 12 and CudNN 8 on your host system may help.*

## How to Run

Follow these steps to build and run the application using Docker.

### 1. Clone the Repository

```bash
git clone https://github.com/jeric-n/Whisperlater.git
cd your-repo-name
```

### 2. Build the Docker Image

Build the image from the `Dockerfile`. This will download all dependencies, the model and the model conversion, so it may take some time (10-20 minutes depending on your internet connection).

```bash
docker build -t Whisperlater .
```

### 3. Run the Docker Container

Once the image is built, run it as a container. The `--gpus all` flag gives the container access to your GPU.

```bash
docker run --gpus all -p 5000:5000 --name whisperlater-container -d Whisperlater
```

*   `--gpus all`: Exposes all available host GPUs to the container.
*   `-p 5000:5000`: Maps port 5000 on your host machine to port 5000 in the container.
*   `--name whisper-container`: Assigns a memorable name to the container for easy management.
*   `-d`: Runs the container in detached mode (in the background).

The server is now running!

## Usage

1.  Open your web browser and navigate to `http://localhost:5000`.
2.  Click "Choose File" to select an audio or video file.
3.  Select your desired options (Task, Language, Format).
4.  Click "Start Processing".
5.  Once processing is complete, a download link for the resulting `.txt` or `.srt` file will appear.

## Stopping the Container

To stop and remove the running container, use the name you assigned to it:

```bash
# Stop the container
docker stop whisperlater-container
```

