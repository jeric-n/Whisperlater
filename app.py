# app.py (The Definitive, Polished Version for Maximum Quality)

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

# --- Load Models (faster-whisper and VAD) ---
whisper_model = None
vad_model = None
get_speech_timestamps_fn = None

try:
    # 1. Load faster-whisper Model
    model_path = "/app/whisper-large-v3-ct2-float16"
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
    vad_model = vad_model_hub
    get_speech_timestamps_fn = utils[0]

    logging.info("All models loaded successfully.")

except Exception as e:
    logging.error(f"FATAL: Failed to load models: {e}", exc_info=True)
    whisper_model = None
    vad_model = None


# --- Helper Functions ---
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


# --- Flask Routes ---
@app.route("/", methods=["GET"])
def index():
    return render_template_string(open("index.html").read())


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if not whisper_model or not vad_model:
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
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(file.filename)[1],
            delete=False,
            dir=app.config["UPLOAD_FOLDER"],
        ) as tmp_upload:
            temp_path = tmp_upload.name
            file.save(temp_path)

        SAMPLING_RATE = 16000
        logging.info("Loading and converting audio file with FFmpeg...")
        try:
            out, _ = (
                ffmpeg.input(temp_path, threads=0)
                .output("-", format="s16le", ac=1, ar=SAMPLING_RATE)
                .run(
                    cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                )
            )
        except ffmpeg.Error as e:
            logging.error(f"FFmpeg error: {e.stderr.decode()}")
            return jsonify({"error": "FFmpeg failed."}), 500
        audio_np_s16 = np.frombuffer(out, np.int16)

        logging.info("Detecting speech segments with Silero VAD...")
        audio_tensor_for_vad = torch.from_numpy(audio_np_s16.copy()).float() / 32768.0
        if torch.cuda.is_available():
            audio_tensor_for_vad = audio_tensor_for_vad.to("cuda")

        # --- User-Discovered Optimal VAD Settings ---
        # Using longer silence duration to group sentences for better context.
        speech_timestamps = get_speech_timestamps_fn(
            audio_tensor_for_vad,
            vad_model,
            sampling_rate=SAMPLING_RATE,
            threshold=float(0.35),  # A balanced threshold between 0.4 and your 0.5
            min_speech_duration_ms=int(150),
            min_silence_duration_ms=int(850),  # User-discovered setting for context
        )

        if not speech_timestamps:
            return jsonify(
                {
                    "filename": os.path.splitext(file.filename)[0] + ".txt",
                    "content": "[No speech detected by VAD.]",
                }
            )

        all_processed_segments = []
        total_vad_chunks = len(speech_timestamps)

        # --- FINAL, POLISHED PARAMETERS ---
        fw_transcribe_options = dict(
            beam_size=int(5),
            # best_of=int(5),
            language=language_code,
            task=task,
            temperature=tuple(float(t) for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            # --- Key changes for final polish ---
            log_prob_threshold=float(-1.0),
            no_speech_threshold=float(0.1),
            suppress_tokens=[-1],
            condition_on_previous_text=True,
            patience=float(1.7),
            # --- Surgical tools for the "Chris, Chris" repetition ---
            repetition_penalty=float(1.1),  # Increased soft penalty
            no_repeat_ngram_size=int(10),  # Added hard block for phrase loops
        )

        logging.info(
            f"Starting transcription with final polished options: {fw_transcribe_options}"
        )

        for i, vad_segment in enumerate(speech_timestamps):
            vad_start_sample = vad_segment["start"]
            vad_end_sample = vad_segment["end"]
            chunk_offset_seconds = vad_start_sample / SAMPLING_RATE
            audio_slice_f32 = (
                audio_np_s16[vad_start_sample:vad_end_sample].astype(np.float32)
                / 32768.0
            )
            if audio_slice_f32.size < 320:
                continue

            logging.info(
                f"Transcribing VAD chunk {i + 1}/{total_vad_chunks} (duration: {len(audio_slice_f32) / SAMPLING_RATE:.2f}s)..."
            )
            segments_generator, info = whisper_model.transcribe(
                audio_slice_f32, **fw_transcribe_options
            )

            for segment in segments_generator:
                all_processed_segments.append(
                    {
                        "timestamp": (
                            chunk_offset_seconds + segment.start,
                            chunk_offset_seconds + segment.end,
                        ),
                        "text": segment.text,
                    }
                )

        if not all_processed_segments:
            return jsonify(
                {
                    "filename": os.path.splitext(file.filename)[0] + ".txt",
                    "content": "[Whisper detected no speech in VAD segments.]",
                }
            )

        logging.info("Transcription complete. Formatting output...")

        if output_format == "srt":
            final_chunks = split_long_chunks(all_processed_segments)
            output_content = format_as_srt(final_chunks)
            output_filename = os.path.splitext(file.filename)[0] + ".srt"
        else:
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
    if vad_model and torch.cuda.is_available():
        vad_model.to("cuda")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
