# app.py

import os
import torch
import ffmpeg
import numpy as np
from flask import Flask, request, render_template_string, jsonify
from faster_whisper import WhisperModel
import tempfile
import logging

# ==============================================================================
# --- 1. Application Setup ---
# ==============================================================================

# Configure basic logging to monitor application status and errors.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize the Flask web application.
app = Flask(__name__)

# Create and configure a folder to temporarily store user uploads.
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================================================================
# --- 2. Model Loading ---
# ==============================================================================

# Initialize model variables to None. They will be loaded in the try-except block.
whisper_model = None
vad_model = None
get_speech_timestamps_fn = None

try:
    # --- Load faster-whisper Model ---
    # This is the core speech-to-text engine.
    model_path = "/app/whisper-large-v3-ct2-float16"
    logging.info(f"Loading faster-whisper model from '{model_path}'...")
    whisper_model = WhisperModel(model_path, device="cuda", compute_type="float16")

    # --- Load Silero VAD (Voice Activity Detection) Model ---
    # This model is used to detect speech segments in the audio, which allows
    # us to skip transcribing long periods of silence, improving efficiency.
    logging.info("Loading Silero VAD model...")
    vad_model_hub, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    vad_model = vad_model_hub
    get_speech_timestamps_fn = utils[0]  # Helper function from the VAD repo.

    logging.info("All models loaded successfully.")

except Exception as e:
    # If model loading fails, log the fatal error and prevent the app from starting.
    logging.error(f"FATAL: Failed to load models: {e}", exc_info=True)
    whisper_model = None
    vad_model = None


# ==============================================================================
# --- 3. Helper Functions ---
# ==============================================================================


def split_long_chunks(chunks, max_words=13):
    """
    Splits transcript segments that are too long for subtitles.
    This improves readability by breaking down long sentences into shorter,
    more manageable lines in the final SRT file.
    """
    new_chunks = []
    for chunk in chunks:
        words = chunk["text"].strip().split()
        if len(words) > max_words:
            # If a chunk is too long, split it into smaller sub-chunks.
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
            # If the chunk is short enough, keep it as is.
            new_chunks.append(chunk)
    return new_chunks


def format_as_srt(result_chunks):
    """
    Formats a list of timestamped text chunks into the standard SRT subtitle format.
    Example:
    1
    00:00:01,234 --> 00:00:03,456
    Hello, world.
    """
    srt_content = ""
    for i, chunk in enumerate(result_chunks):
        start_time, end_time = chunk["timestamp"]
        text = chunk["text"].strip()
        # Skip empty chunks.
        if start_time is None or end_time is None or not text:
            continue
        # Format timestamps into HH:MM:SS,ms
        start_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time * 1000) % 1000):03}"
        end_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time * 1000) % 1000):03}"
        # Append the formatted SRT block.
        srt_content += f"{i + 1}\n{start_srt} --> {end_srt}\n{text}\n\n"
    return srt_content


# ==============================================================================
# --- 4. Flask API Routes ---
# ==============================================================================


@app.route("/", methods=["GET"])
def index():
    """Serves the main HTML page for the application."""
    return render_template_string(open("index.html").read())


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    The main endpoint for handling file uploads and transcription.
    It processes the audio, runs VAD, transcribes with Whisper, and returns the result.
    """
    # --- Initial validation ---
    if not whisper_model or not vad_model:
        return jsonify(
            {
                "error": "A required model is not available. The server may be starting up or encountered an error."
            }
        ), 500
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # --- Get user-selected options from the form ---
    task = request.form.get("task", "transcribe")  # 'transcribe' or 'translate'
    language = request.form.get("language", "auto")  # e.g., 'en', 'es', or 'auto'
    output_format = request.form.get("format", "txt")  # 'txt' or 'srt'
    language_code = language if language != "auto" else None

    logging.info(
        f"New request: Task='{task}', Language='{language_code or 'auto-detect'}', Output='{output_format}'"
    )

    temp_path = None
    try:
        # --- Save uploaded file temporarily ---
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(file.filename)[1],
            delete=False,
            dir=app.config["UPLOAD_FOLDER"],
        ) as tmp_upload:
            temp_path = tmp_upload.name
            file.save(temp_path)

        # --- Audio Pre-processing with FFmpeg ---
        # Standardize the audio to 16kHz, 16-bit, mono PCM, which is required by Whisper.
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
            return jsonify({"error": "FFmpeg failed to process the audio file."}), 500
        audio_np_s16 = np.frombuffer(out, np.int16)

        # --- Voice Activity Detection (VAD) ---
        logging.info("Detecting speech segments with Silero VAD...")
        # Convert numpy array to a torch tensor for the VAD model.
        audio_tensor_for_vad = torch.from_numpy(audio_np_s16.copy()).float() / 32768.0
        if torch.cuda.is_available():
            audio_tensor_for_vad = audio_tensor_for_vad.to("cuda")

        # These VAD settings are tuned to group words into sentence-like chunks
        # for better contextual transcription.
        speech_timestamps = get_speech_timestamps_fn(
            audio_tensor_for_vad,
            vad_model,
            sampling_rate=SAMPLING_RATE,
            # VAD confidence threshold.
            # Higher: Less sensitive, less likely to detect non-speech as speech. Might miss quiet speech.
            # Lower: More sensitive, catches more speech but may misclassify noise as speech.
            threshold=float(0.35),
            # Minimum duration for a speech chunk in milliseconds.
            # Higher: Ignores very short sounds (e.g., coughs).
            # Lower: Catches very short words or sounds.
            min_speech_duration_ms=int(150),
            # Minimum duration of silence to treat as a split point (in ms).
            # Higher: Groups sentences together, providing more context to the model.
            # Lower: Creates more, smaller chunks, breaking up sentences at shorter pauses.
            min_silence_duration_ms=int(850),
        )

        if not speech_timestamps:
            return jsonify(
                {
                    "filename": os.path.splitext(file.filename)[0] + ".txt",
                    "content": "[No speech detected by VAD.]",
                }
            )

        # --- Transcription with faster-whisper ---
        all_processed_segments = []
        total_vad_chunks = len(speech_timestamps)

        # These transcription parameters are fine-tuned for high-quality results.
        fw_transcribe_options = dict(
            # Number of alternative sequences to explore.
            # Higher: Potentially more accurate but slower.
            # Lower: Faster but may be less accurate.
            beam_size=int(5),
            language=language_code,
            task=task,
            # Temperature for sampling. A tuple will fall back to lower temperatures
            # if the probability of the generated text is low.
            # Higher: More "creative" or random output.
            # Lower (closer to 0.0): More deterministic and conservative output.
            temperature=tuple(float(t) for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            # Log probability threshold for accepting generated tokens.
            # Higher (closer to 0): More conservative, only keeps high-confidence tokens.
            # Lower (more negative): More lenient, allows lower-confidence tokens.
            log_prob_threshold=float(-1.0),
            # Threshold for determining if a segment contains speech.
            # Higher: More tolerant of non-speech audio within a segment.
            # Lower: Stricter, more likely to classify a segment as silent.
            no_speech_threshold=float(0.1),
            # List of token IDs to suppress during generation (e.g., to prevent timestamps).
            suppress_tokens=[-1],
            # If True, uses the previous segment's text to prime the model for better context.
            condition_on_previous_text=True,
            # Controls temperature fallback.
            # Higher: Allows the model more attempts at a higher temperature before falling back.
            # Lower: Falls back to a lower temperature more quickly.
            patience=float(1.7),
            # Penalty for repeating tokens.
            # Higher (> 1.0): Reduces the likelihood of word/phrase repetition.
            # 1.0: No penalty.
            repetition_penalty=float(1.1),
            # Prevents n-grams (sequences of n words) from repeating.
            # Higher: Prevents longer phrases from repeating.
            no_repeat_ngram_size=int(10),
        )

        logging.info(
            f"Starting transcription with final polished options: {fw_transcribe_options}"
        )

        # Transcribe each speech chunk identified by VAD.
        for i, vad_segment in enumerate(speech_timestamps):
            vad_start_sample = vad_segment["start"]
            vad_end_sample = vad_segment["end"]
            chunk_offset_seconds = vad_start_sample / SAMPLING_RATE
            # Slice the audio for the current chunk.
            audio_slice_f32 = (
                audio_np_s16[vad_start_sample:vad_end_sample].astype(np.float32)
                / 32768.0
            )
            # Skip tiny, likely invalid chunks.
            if audio_slice_f32.size < 320:  # (0.02 seconds)
                continue

            logging.info(
                f"Transcribing VAD chunk {i + 1}/{total_vad_chunks} (duration: {len(audio_slice_f32) / SAMPLING_RATE:.2f}s)..."
            )
            # Run transcription on the audio slice.
            segments_generator, info = whisper_model.transcribe(
                audio_slice_f32, **fw_transcribe_options
            )

            # Process and store the results, adjusting timestamps to be absolute.
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

        # --- Format and Return Output ---
        logging.info("Transcription complete. Formatting output...")

        if output_format == "srt":
            final_chunks = split_long_chunks(all_processed_segments)
            output_content = format_as_srt(final_chunks)
            output_filename = os.path.splitext(file.filename)[0] + ".srt"
        else:  # "txt" format
            output_content = "\n".join(
                chunk["text"].strip() for chunk in all_processed_segments
            )
            output_filename = os.path.splitext(file.filename)[0] + ".txt"

        return jsonify({"filename": output_filename, "content": output_content})

    except Exception as e:
        # Catch-all for any unexpected errors during processing.
        logging.error(f"An error occurred during transcription: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during processing."}), 500
    finally:
        # --- Cleanup ---
        # Ensure the temporary file is deleted and GPU memory is cleared.
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==============================================================================
# --- 5. Application Start ---
# ==============================================================================

if __name__ == "__main__":
    # Move the VAD model to the GPU if available, after the Flask app forks.
    if vad_model and torch.cuda.is_available():
        vad_model.to("cuda")
    # Start the Flask development server.
    # threaded=False is often recommended for stability with GPU-based models in Flask.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
