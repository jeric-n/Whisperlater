# app.py

import os
import torch
import gc  # Import the garbage collection module
from flask import Flask, request, render_template_string, jsonify
from faster_whisper import WhisperModel
import tempfile
import logging

# ==============================================================================
# 1. Application Setup
# ==============================================================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================================================================
# 2. Model Loading
# ==============================================================================
whisper_model = None
try:
    model_path = "/models/whisper-large-v3-ct2-float16"
    logging.info(f"Loading faster-whisper model from '{model_path}'...")
    whisper_model = WhisperModel(model_path, device="cuda", compute_type="float16")
    logging.info("Whisper model loaded successfully.")
except Exception as e:
    logging.error(f"FATAL: Failed to load Whisper model: {e}", exc_info=True)
    whisper_model = None


# ==============================================================================
# 3. Helper Functions
# ==============================================================================
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
        if start_time is None or end_time is None or not text:
            continue
        start_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time * 1000) % 1000):03}"
        end_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time * 1000) % 1000):03}"
        srt_content += f"{i + 1}\n{start_srt} --> {end_srt}\n{text}\n\n"
    return srt_content


# ==============================================================================
# 4. Flask API Routes
# ==============================================================================
@app.route("/", methods=["GET"])
def index():
    try:
        return render_template_string(open("index.html").read())
    except FileNotFoundError:
        return (
            "Error: index.html not found. Make sure it's in the same directory as app.py.",
            404,
        )


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if not whisper_model:
        return jsonify({"error": "Whisper model is not available."}), 500
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
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(file.filename)[1],
            delete=False,
            dir=app.config["UPLOAD_FOLDER"],
        ) as tmp_upload:
            temp_path = tmp_upload.name
            file.save(temp_path)

        # VAD Parameters
        vad_parameters = dict(
            threshold=0.45,
            min_speech_duration_ms=250,
            min_silence_duration_ms=350,
        )

        # Whisper Options
        fw_transcribe_options = dict(
            beam_size=5,
            language=language_code,
            task=task,
            temperature=[0.0, 0.2, 0.4, 0.6],
            log_prob_threshold=-0.8,
            no_speech_threshold=0.55,
            condition_on_previous_text=True,
            patience=1.0,
            repetition_penalty=1.2,
            no_repeat_ngram_size=10,
        )

        logging.info(f"Starting transcription with options: {fw_transcribe_options}")
        segments_generator, info = whisper_model.transcribe(
            temp_path,
            vad_filter=True,
            vad_parameters=vad_parameters,
            **fw_transcribe_options,
        )

        logging.info("Transcription complete. Formatting output...")

        output_filename = os.path.splitext(file.filename)[0]

        if output_format == "srt":
            # SRT requires the full list to process, so we materialize it.
            all_processed_segments = [
                {"timestamp": (seg.start, seg.end), "text": seg.text}
                for seg in segments_generator
            ]
            if not all_processed_segments:
                return jsonify(
                    {
                        "filename": f"{output_filename}.txt",
                        "content": "[No speech detected.]",
                    }
                )

            final_chunks = split_long_chunks(all_processed_segments)
            output_content = format_as_srt(final_chunks)
            output_filename += ".srt"

            # Explicitly delete large objects to free memory sooner
            del all_processed_segments
            del final_chunks

        else:  # txt format
            # This is memory-efficient: it avoids creating a full list in memory.
            # It processes one segment at a time.
            output_content = "\n".join(seg.text.strip() for seg in segments_generator)
            output_filename += ".txt"
            if not output_content:
                return jsonify(
                    {"filename": output_filename, "content": "[No speech detected.]"}
                )

        return jsonify({"filename": output_filename, "content": output_content})

    except Exception as e:
        logging.error(f"An error occurred during transcription: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during processing."}), 500

    finally:
        # --- CRITICAL ---
        # This block runs whether the try block succeeded or failed.
        # It's essential for cleaning up resources.

        # 1. Delete the temporary audio file from disk.
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        # 2. Free up GPU memory cache.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Trigger a full Python garbage collection cycle.
        gc.collect()
        logging.info("Request cleanup complete.")


# ==============================================================================
# 5. Application Start
# ==============================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
