# app.py (Final High-Performance Version with Full Anti-Hallucination)

import os
import torch
import ffmpeg
import numpy as np
from flask import Flask, request, render_template_string, jsonify
from faster_whisper import WhisperModel
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

# --- Load Optimized Whisper Model ---
model = None
try:
    model_path = "whisper-large-v3-ct2-float16"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model directory not found at '{model_path}'. Please run the conversion command first."
        )

    logging.info(f"Loading optimized CTranslate2 model from '{model_path}'...")
    model = WhisperModel(model_path, device="cuda", compute_type="float16")
    logging.info("Optimized model loaded successfully.")
except Exception as e:
    logging.error(f"FATAL: Failed to load the faster-whisper model: {e}", exc_info=True)
    model = None


# --- Helper Function: Splits long subtitle chunks ---
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
                new_start_time = start_time + (current_word_index * duration_per_word)
                new_end_time = new_start_time + (len(split_words) * duration_per_word)
                new_chunks.append(
                    {
                        "timestamp": (new_start_time, new_end_time),
                        "text": " ".join(split_words),
                    }
                )
                current_word_index += max_words
        else:
            new_chunks.append(chunk)
    return new_chunks


# --- Helper Function: Formats chunks to SRT ---
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
    if not model:
        return jsonify({"error": "Model not available"}), 500
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
        filename = file.filename
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(filename)[1],
            delete=False,
            dir=app.config["UPLOAD_FOLDER"],
        ) as tmp:
            temp_path = tmp.name
            file.save(temp_path)

        logging.info("Loading and converting audio file to numpy array...")
        try:
            out, _ = (
                ffmpeg.input(temp_path, threads=0)
                .output("-", format="s16le", ac=1, ar=16000)
                .run(
                    cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                )
            )
        except ffmpeg.Error as e:
            logging.error(f"FFmpeg error: {e.stderr.decode()}")
            return jsonify({"error": "FFmpeg failed."}), 500

        audio_np = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

        # --- Reworked Logic with faster-whisper AND Anti-Hallucination Settings ---
        segments_generator, info = model.transcribe(
            audio_np,
            beam_size=5,
            language=language_code,
            task=task,
            # --- Anti-Hallucination & Quality-Enhancing Parameters ---
            temperature=0.3,  # User-requested temperature
            no_repeat_ngram_size=10,  # Prevents looping phrases
            log_prob_threshold=-0.8,  # Filters out low-probability (garbage) tokens
            suppress_tokens=[-1],  # Suppresses tokens that are known to cause issues
        )

        logging.info("Transcription complete. Formatting output...")
        raw_chunks = [
            {"timestamp": (s.start, s.end), "text": s.text} for s in segments_generator
        ]

        if output_format == "srt":
            final_chunks = split_long_chunks(raw_chunks)
            output_content = format_as_srt(final_chunks)
            output_filename = os.path.splitext(filename)[0] + ".srt"
        else:
            output_content = " ".join(chunk["text"].strip() for chunk in raw_chunks)
            output_filename = os.path.splitext(filename)[0] + ".txt"

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
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=False)
