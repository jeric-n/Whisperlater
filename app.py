# app.py

import os
import torch
import torchaudio
import gc
import tempfile
import logging
from flask import Flask, request, render_template_string, jsonify
from faster_whisper import WhisperModel
from itertools import chain
from types import SimpleNamespace
import soundfile as sf
import numpy as np
import subprocess
import shutil

# 1. Flask Application Setup
# ==============================================================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Silero VAD Setup ---
torch.set_num_threads(1)
try:
    VAD_MODEL, VAD_UTILS = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
    )
except Exception as e:
    logging.error(f"Failed to load Silero VAD model: {e}", exc_info=True)
    VAD_MODEL, VAD_UTILS = None, None

# 2. Global Model Loading
# ==============================================================================
# Load the model once at startup for efficiency.
whisper_model = None
try:
    # Model is stored in a directory, not a single file.
    model_path = "/models/whisper-large-v3-ct2-float16"
    logging.info(f"Loading faster-whisper model from '{model_path}'...")
    whisper_model = WhisperModel(model_path, device="cuda", compute_type="float16")
    logging.info("Whisper model loaded successfully.")
except Exception as e:
    logging.error(f"FATAL: Failed to load Whisper model: {e}", exc_info=True)
    whisper_model = None

# 3. Helper Functions
# ==============================================================================


def convert_audio_to_wav(input_path, sampling_rate=16000):
    """
    Converts any audio file to a 16kHz mono WAV file using FFmpeg.
    This is necessary for compatibility with Silero VAD and soundfile.

    Args:
        input_path (str): Path to the input audio file.

    Returns:
        str: Path to the temporary converted WAV file.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg is not installed. Please install FFmpeg to use this feature."
        )

    temp_wav_file = tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False, dir=app.config["UPLOAD_FOLDER"]
    )
    output_path = temp_wav_file.name
    temp_wav_file.close()  # Close the file handle so ffmpeg can write to it

    logging.info(f"Converting '{os.path.basename(input_path)}' to 16kHz mono WAV...")

    command = [
        "ffmpeg",
        "-i",
        input_path,  # Input file
        "-ar",
        str(sampling_rate),  # Set audio sample rate to 16kHz
        "-ac",
        "1",  # Set audio channels to 1 (mono)
        "-f",
        "wav",  # Set output format to WAV
        "-y",  # Overwrite output file if it exists
        output_path,
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info("FFmpeg conversion successful.")
        return output_path
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg conversion failed for {input_path}.")
        logging.error(f"FFmpeg stderr: {e.stderr}")
        # Clean up the failed output file
        if os.path.exists(output_path):
            os.remove(output_path)
        raise  # Re-raise the exception to be caught by the main handler


def generate_silero_chunks(audio_path, vad_model, utils, chunk_duration_s=3600):
    """
    An efficient generator for processing large WAV files with Silero VAD.

    This function assumes the input is a 16kHz mono WAV file. It scans the
    file in small blocks to find speech timestamps, groups them into larger
    chunks (e.g., 1 hour), and yields these chunks for transcription.

    Args:
        audio_path (str): Path to the 16kHz mono WAV audio file.
        vad_model: The loaded Silero VAD model.
        utils (tuple): The VAD utility functions from torch.hub.
        chunk_duration_s (int): The maximum duration of each yielded audio chunk in seconds.

    Yields:
        dict: A dictionary containing:
              'start_time' (float): The start time of the chunk in seconds.
              'audio_chunk' (torch.Tensor): The audio data of the chunk.
    """
    (get_speech_timestamps, *_) = utils
    sampling_rate = 16000  # Silero VAD fixed sampling rate

    logging.info(f"Starting Silero VAD scan for '{os.path.basename(audio_path)}'...")
    all_speech_timestamps = []

    try:
        with sf.SoundFile(audio_path, "r") as audio_file:
            # Check if file is compatible (already converted, but good practice)
            if audio_file.samplerate != sampling_rate:
                raise ValueError(
                    f"Expected 16kHz sample rate, but file has {audio_file.samplerate}Hz. Conversion might have failed."
                )
            if audio_file.channels > 1:
                logging.warning(
                    f"File has {audio_file.channels} channels. Processing as mono."
                )

            block_size_samples = 30 * sampling_rate
            global_sample_offset = 0

            for block_num, block in enumerate(
                audio_file.blocks(
                    blocksize=block_size_samples, dtype="float32", fill_value=0
                )
            ):
                tensor_block = torch.from_numpy(block).float()
                if tensor_block.ndim > 1:
                    tensor_block = tensor_block.mean(dim=1)

                speech_ts_in_block = get_speech_timestamps(
                    tensor_block, vad_model, sampling_rate=sampling_rate
                )

                for ts in speech_ts_in_block:
                    all_speech_timestamps.append(
                        {
                            "start": ts["start"] + global_sample_offset,
                            "end": ts["end"] + global_sample_offset,
                        }
                    )

                global_sample_offset += len(tensor_block)
                if (block_num + 1) % 20 == 0:
                    processed_minutes = (block_num + 1) * 30 / 60
                    logging.info(
                        f"VAD Scan: Processed {processed_minutes:.1f} minutes of audio..."
                    )

    except Exception as e:
        logging.error(f"Error during VAD scanning of {audio_path}: {e}", exc_info=True)
        return

    logging.info(
        f"VAD scan complete. Found {len(all_speech_timestamps)} speech segments."
    )
    if not all_speech_timestamps:
        logging.warning("No speech detected in the entire file.")
        return

    logging.info(
        f"Grouping speech segments into chunks of max {chunk_duration_s} seconds."
    )
    max_duration_samples = chunk_duration_s * sampling_rate

    def _read_and_yield_chunk(start_sample, end_sample):
        start_time_sec = start_sample / sampling_rate
        logging.info(f"Preparing chunk from {start_time_sec:.2f}s onwards...")
        with sf.SoundFile(audio_path, "r") as audio_file:
            audio_file.seek(start_sample)
            num_frames = end_sample - start_sample
            audio_segment = audio_file.read(
                num_frames, dtype="float32", always_2d=False
            )

        audio_tensor = torch.from_numpy(audio_segment).float()
        duration_yielded = len(audio_tensor) / sampling_rate
        logging.info(
            f"Yielding chunk starting at {start_time_sec:.2f}s (duration: {duration_yielded:.2f}s)."
        )
        yield {"start_time": start_time_sec, "audio_chunk": audio_tensor}

    current_chunk_start_sample = all_speech_timestamps[0]["start"]
    last_speech_end_sample = 0

    for ts in all_speech_timestamps:
        if (ts["end"] - current_chunk_start_sample > max_duration_samples) and (
            ts["start"] > current_chunk_start_sample
        ):
            yield from _read_and_yield_chunk(
                current_chunk_start_sample, last_speech_end_sample
            )
            current_chunk_start_sample = ts["start"]
        last_speech_end_sample = ts["end"]

    yield from _read_and_yield_chunk(current_chunk_start_sample, last_speech_end_sample)
    logging.info("Finished processing all VAD chunks.")


def generate_and_split_chunks(segments_generator, max_words=13):
    """
    A generator that takes Whisper's segment generator and yields subtitle-ready chunks.
    """
    for segment in segments_generator:
        if hasattr(segment, "words") and segment.words:
            words = [w.word for w in segment.words]
        else:
            words = segment.text.strip().split()

        if not words:
            continue

        if len(words) > max_words:
            start_time, end_time = segment.start, segment.end
            duration = end_time - start_time
            duration_per_word = duration / len(words) if len(words) > 0 else 0
            current_word_index = 0
            while current_word_index < len(words):
                chunk_of_words = words[
                    current_word_index : current_word_index + max_words
                ]
                if not chunk_of_words:
                    break
                chunk_start = start_time + (current_word_index * duration_per_word)
                chunk_end = chunk_start + (len(chunk_of_words) * duration_per_word)
                chunk_text = " ".join(chunk_of_words).strip()
                yield {"timestamp": (chunk_start, chunk_end), "text": chunk_text}
                current_word_index += max_words
        else:
            yield {"timestamp": (segment.start, segment.end), "text": segment.text}


def format_as_srt(result_chunks_iterable):
    """
    Formats an iterable of subtitle chunks into the SRT file format.
    """
    srt_content_parts = []
    for i, chunk in enumerate(result_chunks_iterable, 1):
        start_time, end_time = chunk["timestamp"]
        text = chunk["text"].strip()
        start_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time * 1000) % 1000):03}"
        end_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time * 1000) % 1000):03}"
        srt_content_parts.append(f"{i}\n{start_srt} --> {end_srt}\n{text}\n")
    return "\n".join(srt_content_parts)


# 4. Flask Routes
# ==============================================================================


@app.route("/", methods=["GET"])
def index():
    """Serves the main HTML page."""
    try:
        return render_template_string(open("index.html").read())
    except FileNotFoundError:
        return ("Error: index.html not found.", 404)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Handles file upload and the main transcription process."""
    if not whisper_model:
        return jsonify({"error": "Whisper model is not available."}), 500
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    # --- Get transcription options ---
    task = request.form.get("task", "transcribe")
    language = request.form.get("language", "auto")
    output_format = request.form.get("format", "txt")
    language_code = language if language != "auto" else None
    enable_vad = request.form.get("enable_vad") == "on"

    # --- Advanced Whisper Settings ---
    try:
        temperature = tuple(
            float(t.strip())
            for t in request.form.get(
                "temperature", "0.0, 0.2, 0.4, 0.6, 0.8, 1.0"
            ).split(",")
        )
    except ValueError:
        temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    transcribe_options = {
        "language": language_code,
        "task": task,
        "beam_size": int(request.form.get("beam_size", 6)),
        "patience": float(request.form.get("patience", 2.0)),
        "length_penalty": float(request.form.get("length_penalty", 1.0)),
        "repetition_penalty": float(request.form.get("repetition_penalty", 1.1)),
        "no_repeat_ngram_size": int(request.form.get("no_repeat_ngram_size", 10)),
        "temperature": temperature,
        "compression_ratio_threshold": float(
            request.form.get("compression_ratio_threshold", 2.6)
        ),
        "log_prob_threshold": float(request.form.get("log_prob_threshold", -3.0)),
        "no_speech_threshold": float(request.form.get("no_speech_threshold", 1.0)),
        "condition_on_previous_text": request.form.get("condition_on_previous_text")
        == "on",
        "word_timestamps": request.form.get("word_timestamps") == "on",
        "suppress_tokens": [-1],
        "chunk_length": 30,
    }

    logging.info(
        f"New request: Task='{task}', Language='{language_code or 'auto-detect'}', Output='{output_format}', VAD={'Silero' if enable_vad else 'Whisper'}"
    )
    logging.info(f"Transcription options: {transcribe_options}")

    temp_path = None
    converted_wav_path = None
    try:
        # Save original uploaded file
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(file.filename)[1],
            delete=False,
            dir=app.config["UPLOAD_FOLDER"],
        ) as tmp_upload:
            temp_path = tmp_upload.name
            file.save(temp_path)

        # Convert the audio to a standard WAV format for processing
        converted_wav_path = convert_audio_to_wav(temp_path)
        processing_path = converted_wav_path

        if enable_vad:
            if not VAD_MODEL:
                return jsonify({"error": "Silero VAD model is not available."}), 500

            vad_chunk_generator = generate_silero_chunks(
                processing_path, VAD_MODEL, VAD_UTILS
            )

            def generate_segments_with_correct_time():
                for item in vad_chunk_generator:
                    chunk_start_time = item["start_time"]
                    audio_chunk = item["audio_chunk"]  # This is a torch.Tensor

                    logging.info(
                        f"Transcribing VAD chunk starting at {chunk_start_time:.2f}s..."
                    )

                    # *** THE FIX IS HERE: Convert tensor to NumPy array ***
                    segments, _ = whisper_model.transcribe(
                        audio_chunk.numpy(),  # <--- .numpy() is the fix
                        vad_filter=False,
                        **transcribe_options,
                    )

                    for segment in segments:
                        new_segment = SimpleNamespace()
                        new_segment.start = segment.start + chunk_start_time
                        new_segment.end = segment.end + chunk_start_time
                        new_segment.text = segment.text
                        if hasattr(segment, "words") and segment.words:
                            new_segment.words = [
                                SimpleNamespace(
                                    start=w.start + chunk_start_time,
                                    end=w.end + chunk_start_time,
                                    word=w.word,
                                    probability=w.probability,
                                )
                                for w in segment.words
                            ]
                        else:
                            new_segment.words = None
                        yield new_segment

            segments_generator = generate_segments_with_correct_time()
        else:
            # Transcribe directly from the converted WAV file (this path was already correct)
            segments_generator, _ = whisper_model.transcribe(
                processing_path, vad_filter=False, **transcribe_options
            )

        logging.info("Transcription complete. Formatting output...")
        output_filename_base = os.path.splitext(file.filename)[0]

        if output_format == "srt":
            output_filename = f"{output_filename_base}.srt"
            chunk_generator = generate_and_split_chunks(segments_generator)
            output_content = format_as_srt(chunk_generator)
        else:  # txt format
            output_filename = f"{output_filename_base}.txt"
            output_content = "\n".join(seg.text.strip() for seg in segments_generator)

        final_content = (
            output_content if output_content.strip() else "[No speech detected.]"
        )
        return jsonify({"filename": output_filename, "content": final_content})

    except Exception as e:
        logging.error(f"An error occurred during transcription: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during processing."}), 500

    finally:
        # Clean up both temporary files
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if converted_wav_path and os.path.exists(converted_wav_path):
            os.remove(converted_wav_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logging.info("Request cleanup complete.")


# 5. Application Start
# ==============================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
