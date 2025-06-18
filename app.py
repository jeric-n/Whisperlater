# app.py (Combined Silero VAD pre-processing with faster-whisper optimizations)

import os
import torch
import ffmpeg
import numpy as np
from flask import Flask, request, render_template_string, jsonify
from faster_whisper import WhisperModel  # Using faster-whisper
import tempfile
import logging

# --- Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Load Models (faster-whisper and VAD) ---
whisper_model = None
vad_model = None
get_speech_timestamps_fn = None  # To store the VAD function

try:
    # 1. Load faster-whisper Model
    model_path = "whisper-large-v3-ct2-float16"  # Your CTranslate2 converted model
    if not os.path.exists(model_path):
        # Attempt to guide the user if the model isn't converted
        logging.error(f"CTranslate2 model directory not found at '{model_path}'.")
        logging.error(
            "Please ensure you have converted the Whisper model to CTranslate2 format."
        )
        logging.error(
            "You can do this by running: pip install ctranslate2[transformers]"
        )
        logging.error(
            "Then: ct2-transformers-converter --model openai/whisper-large-v3 --output_dir whisper-large-v3-ct2-float16 --quantization float16 --force"
        )
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    logging.info(f"Loading faster-whisper model from '{model_path}'...")
    whisper_model = WhisperModel(model_path, device="cuda", compute_type="float16")

    # 2. Load Silero VAD model
    logging.info("Loading Silero VAD model...")
    vad_model_hub, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    vad_model = vad_model_hub  # Assign to the global variable
    get_speech_timestamps_fn = utils[0]  # Store the function

    logging.info("All models loaded successfully.")

except Exception as e:
    logging.error(f"FATAL: Failed to load models: {e}", exc_info=True)
    whisper_model = None
    vad_model = None


# --- Helper Functions (split_long_chunks, format_as_srt - no changes needed) ---
def split_long_chunks(chunks, max_words=13):
    new_chunks = []
    for chunk in chunks:
        words = chunk["text"].strip().split()
        if len(words) > max_words:
            start_time, end_time = chunk["timestamp"]
            duration = end_time - start_time
            duration_per_word = duration / len(words) if len(words) > 0 else 0
            current_word_index = 0
            while current_word_index < len(words):
                split_words = words[current_word_index : current_word_index + max_words]
                new_start = start_time + (current_word_index * duration_per_word)
                new_end = new_start + (len(split_words) * duration_per_word)
                new_chunks.append(
                    {"timestamp": (new_start, new_end), "text": " ".join(split_words)}
                )
                current_word_index += max_words
        else:
            new_chunks.append(chunk)
    return new_chunks


def format_as_srt(result_chunks):
    srt_content = ""
    for i, chunk in enumerate(result_chunks):
        start_time, end_time = chunk["timestamp"]
        text = chunk["text"].strip()
        if start_time is None or end_time is None:
            continue
        start_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time * 1000) % 1000):03}"
        end_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time * 1000) % 1000):03}"
        srt_content += f"{i + 1}\n{start_srt} --> {end_srt}\n{text}\n\n"
    return srt_content


# --- Flask Routes ---
@app.route("/", methods=["GET"])
def index():
    return render_template_string(open("index.html").read())


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if (
        not whisper_model or not vad_model or not get_speech_timestamps_fn
    ):  # Check all models
        return jsonify({"error": "A required model is not available."}), 500
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    task = request.form.get("task", "transcribe")
    language = request.form.get("language", "auto")
    output_format = request.form.get("format", "txt")
    language_code = language if language != "auto" else None

    logging.info(
        f"New request: Task='{task}', Language='{language_code or 'auto-detect'}', Output='{output_format}'"
    )

    temp_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(file.filename)[1],
            delete=False,
            dir=app.config["UPLOAD_FOLDER"],
        ) as tmp_upload:
            temp_path = tmp_upload.name
            file.save(temp_path)

        # 1. Load audio using FFmpeg and convert to NumPy array
        logging.info("Loading and converting audio file with FFmpeg...")
        SAMPLING_RATE = 16000  # Whisper and Silero VAD expect 16kHz
        try:
            out, _ = (
                ffmpeg.input(temp_path, threads=0)
                .output("-", format="s16le", ac=1, ar=SAMPLING_RATE)  # s16le for VAD
                .run(
                    cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                )
            )
        except ffmpeg.Error as e:
            logging.error(f"FFmpeg error: {e.stderr.decode()}")
            return jsonify({"error": "FFmpeg failed."}), 500

        audio_np_s16 = np.frombuffer(out, np.int16)  # For VAD

        # 2. Get speech timestamps using Silero VAD
        logging.info("Detecting speech segments with Silero VAD...")
        audio_tensor_for_vad = torch.from_numpy(audio_np_s16).float() / 32768.0

        # --- THIS IS THE FIX ---
        # Move the audio tensor to the GPU if CUDA is available
        if torch.cuda.is_available():
            audio_tensor_for_vad = audio_tensor_for_vad.to("cuda")
        # --- END OF FIX ---

        speech_timestamps = get_speech_timestamps_fn(  # Use the stored function
            audio_tensor_for_vad,  # Now on the correct device
            vad_model,
            sampling_rate=SAMPLING_RATE,
            threshold=float(0.4),
            min_silence_duration_ms=int(500),
        )

        if not speech_timestamps:
            return jsonify(
                {
                    "filename": os.path.splitext(file.filename)[0] + ".txt",
                    "content": "[No speech detected by VAD in the audio.]",
                }
            )

        # 3. Transcribe/Translate each speech chunk using faster-whisper
        all_processed_segments = []
        total_vad_chunks = len(speech_timestamps)

        # Define faster-whisper options based on our best-tuned settings
        fw_transcribe_options = {
            "beam_size": int(7),
            "language": language_code,
            "task": task,
            "temperature": tuple(float(t) for t in [0.0, 0.2, 0.4, 0.6, 0.8]),
            "log_prob_threshold": float(-1.0),
            "no_speech_threshold": float(
                0.4
            ),  # This applies to faster-whisper's internal check on the slice
            "compression_ratio_threshold": float(2.3),
            "condition_on_previous_text": True,
            "patience": float(1.5),
            "repetition_penalty": float(1.2),
            "no_repeat_ngram_size": int(5),
            "vad_filter": False,  # IMPORTANT: Disable faster-whisper's VAD for pre-segmented chunks
        }
        if task == "translate":  # Slightly different initial temp for translation
            fw_transcribe_options["temperature"] = tuple(
                float(t) for t in [0.2, 0.4, 0.6, 0.8, 1.0]
            )

        logging.info(
            f"Starting transcription with faster-whisper options: {fw_transcribe_options}"
        )

        for i, vad_segment in enumerate(speech_timestamps):
            vad_start_sample = vad_segment["start"]
            vad_end_sample = vad_segment["end"]

            # Offset for this VAD chunk's timestamps
            chunk_offset_seconds = vad_start_sample / SAMPLING_RATE

            # Slice audio for faster-whisper (expects float32, normalized)
            audio_slice_f32 = (
                audio_np_s16[vad_start_sample:vad_end_sample].astype(np.float32)
                / 32768.0
            )

            if audio_slice_f32.size == 0:
                logging.warning(f"Skipping empty audio slice from VAD chunk {i + 1}.")
                continue

            logging.info(
                f"Transcribing VAD chunk {i + 1}/{total_vad_chunks} (duration: {len(audio_slice_f32) / SAMPLING_RATE:.2f}s)..."
            )

            segments_generator, info = whisper_model.transcribe(
                audio_slice_f32, **fw_transcribe_options
            )

            for segment in segments_generator:
                abs_start = chunk_offset_seconds + segment.start
                abs_end = chunk_offset_seconds + segment.end
                all_processed_segments.append(
                    {"timestamp": (abs_start, abs_end), "text": segment.text}
                )

        if not all_processed_segments:
            return jsonify(
                {
                    "filename": os.path.splitext(file.filename)[0] + ".txt",
                    "content": "[Whisper detected no speech in VAD segments.]",
                }
            )

        logging.info("Transcription complete. Formatting output...")

        # 4. Post-process and format the output
        if output_format == "srt":
            final_chunks = split_long_chunks(all_processed_segments)
            output_content = format_as_srt(final_chunks)
            output_filename = os.path.splitext(file.filename)[0] + ".srt"
        else:  # .txt format
            output_content = "\n".join(
                chunk["text"].strip() for chunk in all_processed_segments
            )
            output_filename = os.path.splitext(file.filename)[0] + ".txt"

        return jsonify({"filename": output_filename, "content": output_content})

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred."}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Ensure Silero VAD model is on the same device as Whisper if GPU is used
    if vad_model and torch.cuda.is_available():
        vad_model.to("cuda")
    app.run(
        host="0.0.0.0", port=5000, debug=False, threaded=False
    )  # Debug=False for production
