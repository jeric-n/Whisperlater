# Dockerfile

# --- Stage 1: Base Image and System Dependencies ---
# Use an official NVIDIA image with CUDA 12.1 and cuDNN 8 on Ubuntu 22.04.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Update and install essential system dependencies.
# python3-pip is the only system dependency needed now.
# ffmpeg and git have been removed as they are no longer required by the refactored app.py.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- Stage 2: Python Environment and Dependencies ---
WORKDIR /app

COPY requirements.txt .

# Install Python packages.
# --no-cache-dir is used to reduce image size.
RUN \
    pip install --no-cache-dir --upgrade pip && \
    # 1. Install PyTorch first for CUDA 12.1 compatibility.
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    # 2. Install the rest of the packages from the simplified requirements file.
    pip install --no-cache-dir -r requirements.txt

# --- Stage 3: AI Model Preparation ---
# This stage downloads and converts the Whisper model to the optimized CTranslate2 format.
# This is done during the build to avoid long download times on container startup.
RUN \
    echo "--- Preparing Whisper Model ---" && \
    # Install 'transformers' temporarily for the converter script.
    pip install --no-cache-dir "transformers>=4.36.0" && \
    \
    # Run the CTranslate2 converter.
    echo "--- Downloading and Converting Whisper Model ---" && \
    ct2-transformers-converter \
        --model openai/whisper-large-v3 \
        --output_dir /models/whisper-large-v3-ct2-float16 \
        --quantization float16 \
        --force && \
    \
    # FIX: Ensure a known-good preprocessor_config.json is present.
    echo "--- Applying preprocessor_config.json fix ---" && \
    cat <<EOF > /models/whisper-large-v3-ct2-float16/preprocessor_config.json
{
  "chunk_length_s": 30,
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

# --- Stage 4: Application Code ---
# Copy the application source code into the image.
# These are commented out because docker-compose.yml uses a volume mount for development.
# For a production build, you would uncomment these lines.
# COPY app.py .
# COPY index.html .

# --- Stage 5: Runtime Configuration ---
# Expose the port the Flask application will run on.
EXPOSE 5000

# Define the default command to run the application.
# This will be used in production. The docker-compose.yml overrides this for development.
CMD ["python3", "-u", "app.py"]
