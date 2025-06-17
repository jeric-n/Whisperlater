# app.py (Final version with Silero VAD pre-processing and advanced anti-hallucination)

import os
import torch
import ffmpeg
import numpy as np
from flask import Flask, request, render_template_string, jsonify
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import tempfile
import logging

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load Models (Whisper and VAD) ---
pipe = None
vad_model = None
try:
    # 1. Load Whisper Model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"

    logging.info(f"Loading Whisper model '{model_id}' on device '{device}'...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
        device_map="auto", load_in_8bit=True
    )
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor, chunk_length_s=30, batch_size=8,
        return_timestamps=True, torch_dtype=torch_dtype,
    )
    
    # 2. Load Silero VAD model
    logging.info("Loading Silero VAD model...")
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    
    logging.info("All models loaded successfully.")

except Exception as e:
    logging.error(f"FATAL: Failed to load models: {e}", exc_info=True)
    pipe = None
    vad_model = None

# --- Helper Functions ---
def split_long_chunks(chunks, max_words=13):
    new_chunks = []
    for chunk in chunks:
        words = chunk['text'].strip().split()
        if len(words) > max_words:
            start_time, end_time = chunk['timestamp']
            duration = end_time - start_time
            duration_per_word = duration / len(words) if len(words) > 0 else 0
            current_word_index = 0
            while current_word_index < len(words):
                split_words = words[current_word_index : current_word_index + max_words]
                new_start = start_time + (current_word_index * duration_per_word)
                new_end = new_start + (len(split_words) * duration_per_word)
                new_chunks.append({'timestamp': (new_start, new_end), 'text': ' '.join(split_words)})
                current_word_index += max_words
        else:
            new_chunks.append(chunk)
    return new_chunks

def format_as_srt(result_chunks):
    srt_content = ""
    for i, chunk in enumerate(result_chunks):
        start_time, end_time = chunk['timestamp']
        text = chunk['text'].strip()
        if start_time is None or end_time is None: continue
        start_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time * 1000) % 1000):03}"
        end_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time * 1000) % 1000):03}"
        srt_content += f"{i + 1}\n{start_srt} --> {end_srt}\n{text}\n\n"
    return srt_content

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template_string(open("index.html").read())

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if not pipe or not vad_model: return jsonify({"error": "A required model is not available."}), 500
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400

    task = request.form.get('task', 'transcribe')
    language = request.form.get('language', 'auto')
    output_format = request.form.get('format', 'txt')

    # Advanced anti-hallucination parameters
    generate_kwargs = {
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8), # Fallback temperatures
        "suppress_tokens": [-1], # Required to enable temperature fallback
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -0.8,
        "no_repeat_ngram_size": 10
    }
    if language != "auto": generate_kwargs["language"] = language
        
    logging.info(f"New request: Task='{task}', Language='{language}', Output='{output_format}'")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[1], delete=False) as tmp_upload:
            temp_path = tmp_upload.name
            file.save(temp_path)
        
        # 1. Load audio using FFmpeg
        logging.info("Loading and converting audio file with FFmpeg...")
        SAMPLING_RATE = 16000
        try:
            out, _ = (ffmpeg.input(temp_path, threads=0).output("-", format="s16le", ac=1, ar=SAMPLING_RATE).run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True))
        except ffmpeg.Error as e:
            logging.error(f"FFmpeg error: {e.stderr.decode()}"); return jsonify({"error": "FFmpeg failed."}), 500
        
        audio_np_s16 = np.frombuffer(out, np.int16)

        # 2. Get speech timestamps using Silero VAD
        logging.info("Detecting speech segments with Silero VAD...")
        # Silero VAD requires a torch tensor
        audio_tensor = torch.from_numpy(audio_np_s16).float() / 32768.0
        speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=SAMPLING_RATE, min_silence_duration_ms=500)

        if not speech_timestamps:
            return jsonify({"filename": "result.txt", "content": "[No speech detected in the audio.]"})

        # 3. Transcribe/Translate each speech chunk
        processed_chunks = []
        total_chunks = len(speech_timestamps)
        for i, segment in enumerate(speech_timestamps):
            logging.info(f"Processing speech chunk {i+1}/{total_chunks}...")
            start_sample, end_sample = segment['start'], segment['end']
            audio_slice = audio_np_s16[start_sample:end_sample].astype(np.float32) / 32768.0
            
            if audio_slice.size == 0: continue

            # Set the task for Whisper
            generate_kwargs['task'] = task
            result = pipe(audio_slice, generate_kwargs=generate_kwargs)

            # Preserve the original, accurate VAD timestamp
            processed_chunks.append({
                'timestamp': (start_sample / SAMPLING_RATE, end_sample / SAMPLING_RATE),
                'text': result['text']
            })

        # 4. Post-process and format the output
        if output_format == 'srt':
            final_chunks = split_long_chunks(processed_chunks) # Apply word-limit splitting
            output_content = format_as_srt(final_chunks)
            output_filename = os.path.splitext(file.filename)[0] + ".srt"
        else: # .txt format
            output_content = "\n".join(chunk['text'].strip() for chunk in processed_chunks)
            output_filename = os.path.splitext(file.filename)[0] + ".txt"

        return jsonify({ "filename": output_filename, "content": output_content })

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred."}), 500
    finally:
        if temp_path and os.path.exists(temp_path): os.remove(temp_path)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
