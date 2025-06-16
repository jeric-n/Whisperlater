# app.py (Final corrected version with StopIteration fix)

import os
import torch
import ffmpeg
import numpy as np
from flask import Flask, request, render_template_string, jsonify
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
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

# --- Load Whisper Model ---
pipe = None
try:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"

    logging.info(f"Loading model '{model_id}' on device '{device}'...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map="auto",
        load_in_8bit=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=8,
        return_timestamps=True,
        torch_dtype=torch_dtype,
    )
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"FATAL: Failed to load the Whisper model: {e}")
    pipe = None


# --- Helper Function to Format SRT ---
def format_as_srt(result_chunks):
    srt_content = ""
    for i, chunk in enumerate(result_chunks):
        start_time = chunk["timestamp"][0]
        end_time = chunk["timestamp"][1]
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
    if not pipe:
        return jsonify({"error": "Model not available"}), 500
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    task = request.form.get("task", "transcribe")
    language = request.form.get("language", "auto")
    output_format = request.form.get("format", "txt")

    generate_kwargs = {
        "temperature": 0.2,
        "no_repeat_ngram_size": 10,
        "logprob_threshold": -0.8,
    }
    if language != "auto":
        generate_kwargs["language"] = language

    logging.info(
        f"New request: Task='{task}', Language='{language}', Output='{output_format}'"
    )

    temp_path = None
    try:
        filename = file.filename
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(filename)[1],
            delete=False,
            dir=app.config["UPLOAD_FOLDER"],
        ) as tmp_upload:
            temp_path = tmp_upload.name
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
            return jsonify(
                {"error": "FFmpeg failed. Ensure it's installed and in PATH."}
            ), 500

        audio_np = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

        if task == "translate" and output_format == "srt":
            logging.info("Performing two-pass translation for SRT...")

            transcribe_kwargs = generate_kwargs.copy()
            transcribe_kwargs["task"] = "transcribe"
            pass1_result = pipe(audio_np.copy(), generate_kwargs=transcribe_kwargs)
            original_chunks = pass1_result["chunks"]

            translated_chunks = []
            total_chunks = len(original_chunks)
            for i, chunk in enumerate(original_chunks):
                logging.info(f"Translating chunk {i + 1}/{total_chunks}...")
                start_time, end_time = chunk["timestamp"]

                if start_time is None or end_time is None:
                    continue

                start_sample = int(start_time * 16000)
                end_sample = int(end_time * 16000)
                audio_slice = audio_np[start_sample:end_sample]

                # V-- THIS IS THE FIX: Check if the audio slice is empty --V
                if audio_slice.size == 0:
                    logging.warning(
                        f"Skipping empty audio chunk {i + 1} at {start_time:.2f}s."
                    )
                    continue
                # ^-- END OF FIX --^

                translate_kwargs = generate_kwargs.copy()
                translate_kwargs["task"] = "translate"
                pass2_result = pipe(audio_slice, generate_kwargs=translate_kwargs)

                translated_chunks.append(
                    {"timestamp": (start_time, end_time), "text": pass2_result["text"]}
                )

            output_content = format_as_srt(translated_chunks)
            output_filename = os.path.splitext(filename)[0] + ".srt"

        else:
            logging.info("Performing single-pass transcription/translation...")
            generate_kwargs["task"] = task
            result = pipe(audio_np.copy(), generate_kwargs=generate_kwargs)

            if output_format == "srt" and task == "transcribe":
                output_content = format_as_srt(result["chunks"])
                output_filename = os.path.splitext(filename)[0] + ".srt"
            else:
                output_content = result["text"]
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
