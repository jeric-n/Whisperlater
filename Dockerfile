# Step 1: Use a stable, official NVIDIA image with CUDA 12.1 and cuDNN 8
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file
COPY requirements.txt .

# --- Install Python Dependencies ---
# 1. Install PyTorch with its specific index URL for CUDA 12.1
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install the rest of the packages from requirements.txt, including the pinned ctranslate2
RUN pip install --no-cache-dir -r requirements.txt


# --- Download and Convert Whisper Model ---
RUN echo "--- Downloading and Converting Whisper Model ---" && \
    # We need to install transformers for the converter
    pip install --no-cache-dir "transformers>=4.36.0" && \
    # Run the converter
    ct2-transformers-converter --model openai/whisper-large-v3 --output_dir whisper-large-v3-ct2-float16 --quantization float16 --force && \
    #
    # --- THE DEFINITIVE FIX IS HERE ---
    # Overwrite/Create the preprocessor_config.json with a known-good version.
    # This guarantees the "num_mel_bins": 128 is present.
    echo '{ \
      "chunk_length": 30, \
      "feature_extractor_type": "WhisperFeatureExtractor", \
      "feature_size": 128, \
      "hop_length": 160, \
      "n_fft": 400, \
      "n_samples": 480000, \
      "num_mel_bins": 128, \
      "padding_side": "right", \
      "padding_value": 0.0, \
      "processor_class": "WhisperProcessor", \
      "return_attention_mask": false, \
      "sampling_rate": 16000 \
    }' > whisper-large-v3-ct2-float16/preprocessor_config.json && \
    #
    echo "--- Model Conversion and Config Fix Complete ---"


# Copy your application files
COPY app.py ./
COPY index.html ./

# Download Silero VAD model assets
ENV TORCH_HOME=/app/.cache/torch
RUN python3 -c "import torch; torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)"

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python3", "app.py"]
