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
VAD_MODEL, VAD_UTILS = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
)
(get_speech_timestamps, _, read_audio, *_) = VAD_UTILS

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
    # The app will run but /transcribe will return an error.
    whisper_model = None

# 3. Helper Functions
# ==============================================================================


def process_with_silero_vad(audio_path, vad_model, utils):
    """
    Processes an audio file with Silero VAD to extract speech chunks, merging them
    into larger chunks of approximately max_duration_s for better transcription quality.
    Returns a list of dictionaries, each with 'start_time' in seconds and 'audio_chunk' as a tensor.
    """
    (get_speech_timestamps, _, read_audio, *_) = utils
    wav = read_audio(audio_path, sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000)

    if not speech_timestamps:
        return []

    sampling_rate = 16000
    max_duration_s = 3600  # Updated to 1h

    # Merge timestamps into larger chunks
    merged_timestamps = []
    if speech_timestamps:
        current_start = speech_timestamps[0]["start"]

        for i in range(len(speech_timestamps) - 1):
            current_end = speech_timestamps[i]["end"]
            next_end = speech_timestamps[i + 1]["end"]

            # Check if the next chunk would exceed the max duration
            if (next_end - current_start) / sampling_rate > max_duration_s:
                merged_timestamps.append({"start": current_start, "end": current_end})
                current_start = speech_timestamps[i + 1]["start"]

        # Add the last chunk
        merged_timestamps.append(
            {"start": current_start, "end": speech_timestamps[-1]["end"]}
        )

    # Extract audio chunks based on merged timestamps
    speech_chunks_with_ts = []
    for ts in merged_timestamps:
        start_sec = ts["start"] / sampling_rate
        chunk = wav[ts["start"] : ts["end"]]
        speech_chunks_with_ts.append({"start_time": start_sec, "audio_chunk": chunk})

    return speech_chunks_with_ts


def generate_and_split_chunks(segments_generator, max_words=13):
    """
    A generator that takes Whisper's segment generator and yields subtitle-ready chunks.
    It splits long segments into smaller ones on-the-fly to improve readability.
    If a segment is longer than max_words, its duration is split evenly among the words.
    """
    for segment in segments_generator:
        # Use word-level timestamps if available to get a list of words, otherwise split text.
        if hasattr(segment, "words") and segment.words:
            words = [w.word for w in segment.words]
        else:
            words = segment.text.strip().split()

        if not words:
            continue  # Skip segments with no words.

        if len(words) > max_words:
            # Segment is too long, split it into smaller chunks with evenly distributed timing.
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
            # Yield the original segment as a chunk if it's already a good size.
            yield {"timestamp": (segment.start, segment.end), "text": segment.text}


def format_as_srt(result_chunks_iterable):
    """
    Formats an iterable of subtitle chunks (like from a generator) into the SRT file format.
    """
    srt_content_parts = []
    # Enumerate the iterable to get the subtitle index.
    for i, chunk in enumerate(result_chunks_iterable, 1):
        start_time, end_time = chunk["timestamp"]
        text = chunk["text"].strip()

        # Format start and end times to SRT's HH:MM:SS,ms format.
        start_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time * 1000) % 1000):03}"
        end_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time * 1000) % 1000):03}"

        srt_content_parts.append(f"{i}\n{start_srt} --> {end_srt}\n{text}\n")

    # Join all parts at the end for efficiency.
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

    # --- Get transcription options from the form ---
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
        temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # Fallback to default

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
    try:
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(file.filename)[1],
            delete=False,
            dir=app.config["UPLOAD_FOLDER"],
        ) as tmp_upload:
            temp_path = tmp_upload.name
            file.save(temp_path)

        if enable_vad:
            # Process with Silero VAD
            speech_chunks_with_ts = process_with_silero_vad(
                temp_path, VAD_MODEL, VAD_UTILS
            )
            if not speech_chunks_with_ts:
                return jsonify(
                    {
                        "filename": f"{os.path.splitext(file.filename)[0]}.txt",
                        "content": "[No speech detected.]",
                    }
                )

            def generate_segments_with_correct_time():
                """
                A generator that transcribes audio chunks one by one and yields segments
                with timestamps adjusted to the original audio file's timeline.
                """
                for item in speech_chunks_with_ts:
                    chunk_start_time = item["start_time"]
                    audio_chunk = item["audio_chunk"]

                    segments, _ = whisper_model.transcribe(
                        audio_chunk.numpy(),
                        vad_filter=False,  # VAD already applied
                        **transcribe_options,
                    )

                    for segment in segments:
                        new_segment = SimpleNamespace()
                        new_segment.start = segment.start + chunk_start_time
                        new_segment.end = segment.end + chunk_start_time
                        new_segment.text = segment.text

                        if hasattr(segment, "words") and segment.words:
                            new_segment.words = []
                            for w in segment.words:
                                new_word = SimpleNamespace()
                                new_word.start = w.start + chunk_start_time
                                new_word.end = w.end + chunk_start_time
                                new_word.word = w.word
                                new_word.probability = w.probability
                                new_segment.words.append(new_word)
                        else:
                            new_segment.words = None

                        yield new_segment

            segments_generator = generate_segments_with_correct_time()

        else:
            # Transcribe directly with faster-whisper's VAD
            segments_generator, _ = whisper_model.transcribe(
                temp_path,
                vad_filter=True,  # Use Whisper's VAD
                **transcribe_options,
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

        if not output_content.strip():
            final_content = "[No speech detected.]"
        else:
            final_content = output_content

        return jsonify({"filename": output_filename, "content": final_content})

    except Exception as e:
        logging.error(f"An error occurred during transcription: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during processing."}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logging.info("Request cleanup complete.")


# 5. Application Start
# ==============================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
