# ==============================================================================
# Dockerfile for the Faster-Whisper Transcription App
# ==============================================================================
# This file defines the container image for the application. It includes:
# - A CUDA-enabled base image for GPU acceleration.
# - System dependencies like Python and FFmpeg.
# - Python libraries including PyTorch, faster-whisper, and Flask.
# - Pre-downloading and converting the Whisper model for faster startup.
# - Pre-caching the Silero VAD model.
# - The application code itself.
# ==============================================================================


# --- Stage 1: Base Image and System Dependencies ---
# Use an official NVIDIA image with CUDA 12.1 and cuDNN 8 on Ubuntu 22.04.
# The 'devel' tag includes the CUDA toolkit, which is sometimes needed for compilation.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during package installation.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Update package lists and install essential system dependencies in a single layer.
# - python3-pip: For installing Python packages.
# - ffmpeg: For audio processing and conversion.
# - git: Required by some pip packages (e.g., for torch.hub).
# Cleaning up apt lists reduces the final image size.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*


# --- Stage 2: Python Environment and Dependencies ---
# Set the working directory for the application.
WORKDIR /app

# Copy the requirements file into the image. This is done before pip install
# to leverage Docker's layer caching.
COPY requirements.txt .

# Install Python packages using pip.
# --no-cache-dir is used to reduce image size by not storing the pip cache.
RUN \
    # Upgrade pip to the latest version.
    pip install --no-cache-dir --upgrade pip && \
    # 1. Install PyTorch first, using its specific index URL for CUDA 12.1 compatibility.
    #    This ensures the GPU-enabled version is installed correctly.
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    # 2. Install the rest of the packages from the requirements file.
    pip install --no-cache-dir -r requirements.txt


# --- Stage 3: AI Model Preparation ---
# This stage downloads the original Whisper model, converts it to the CTranslate2
# format for performance, and applies a necessary configuration fix.
# This is done during the build to avoid long download times on container startup.
RUN \
    echo "--- Preparing Whisper Model ---" && \
    # Install 'transformers' temporarily, as it's required by the converter script.
    pip install --no-cache-dir "transformers>=4.36.0" && \
    \
    # Run the CTranslate2 converter.
    # --model: The original Hugging Face model to convert.
    # --output_dir: Where to save the converted model.
    # --quantization float16: Use 16-bit floating point for a good balance of speed and accuracy on modern GPUs.
    # --force: Overwrite the output directory if it exists.
    echo "--- Downloading and Converting Whisper Model ---" && \
    ct2-transformers-converter \
        --model openai/whisper-large-v3 \
        --output_dir whisper-large-v3-ct2-float16 \
        --quantization float16 \
        --force && \
    \
    # FIX: The conversion process can generate an incomplete preprocessor_config.json.
    # This step explicitly creates a known-good version of the file, ensuring that
    # critical parameters like "num_mel_bins": 128 are correctly set.
    # Using a heredoc (<<EOF) is cleaner than a long, escaped echo string.
    echo "--- Applying preprocessor_config.json fix ---" && \
    cat <<EOF > whisper-large-v3-ct2-float16/preprocessor_config.json
{
  "chunk_length": 30,
  "feature_extractor_type": "WhisperFeatureExtractor",
  "feature_size": 128,
  "hop_length": 160,
  "n_fft": 400,
  "n_samples": 480000,
  "num_mel_bins": 128,
  "padding_side": "right",
  "padding_value": 0.0,
  "processor_class": "WhisperProcessor",
  "return_attention_mask": false,
  "sampling_rate": 16000
}
EOF


# --- Stage 4: Application Code and VAD Model Caching ---
# Copy the application source code into the image.
COPY app.py ./
COPY index.html ./

# Set the cache directory for PyTorch Hub inside the workdir.
ENV TORCH_HOME=/app/.cache/torch

# Pre-download (warm the cache) for the Silero VAD model.
# This prevents a download on the first run of the container, ensuring faster startup.
RUN python3 -c "import torch; torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)"


# --- Stage 5: Runtime Configuration ---
# Expose the port the Flask application will run on.
EXPOSE 5000

# Define the command to run the application when the container starts.
CMD ["python3", "app.py"]
