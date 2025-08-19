# youtube_processing_pipeline.py

# ========================================================================================
# REQUIREMENTS
# ========================================================================================
# Please install the necessary libraries before running this script:
#
# pip install yt-dlp pydub huggingface_hub
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install pyannote.audio==3.1.1
# pip install pandas requests soundfile librosa
#
# You will also need FFmpeg. If it's not installed system-wide, you can install it via:
# # For Debian/Ubuntu:
# sudo apt update && sudo apt install ffmpeg
# # For macOS (using Homebrew):
# brew install ffmpeg
# # For Windows (using Chocolatey):
# choco install ffmpeg
# ========================================================================================

import os
import gc
import glob
import time
import shutil
import random
import zipfile
import datetime
import threading
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Data and Audio Processing
import pandas as pd
import torch
import librosa
import soundfile as sf
import yt_dlp
from pydub import AudioSegment

# APIs and Services
import requests
from huggingface_hub import login
from pyannote.audio import Pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# ========================================================================================
# SECTION 1: CONFIGURATION
# ========================================================================================
# --- List of YouTube URLs to process ---
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=iWsVWWWQhco",
    "https://www.youtube.com/watch?v=V8xNCcAMXaE",
    # Add more YouTube URLs here
]

# --- Directories ---
BASE_DIR = "youtube_pipeline"
INPUT_DIR = os.path.join(BASE_DIR, "1_downloaded_wav")
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "2_processed_output")

# --- Hardware & Parallelism ---
NUM_GPUS = 2  # Number of GPUs to use for processing
GPU_FILE_WORKERS = 1  # Number of files to process in parallel on a single GPU
MAX_SEGMENT_THREADS = 64  # Max concurrent threads for processing audio segments (API calls)

# --- External Services & Authentication ---
# Required for pyannote/speaker-diarization-3.1 model
HUGGINGFACE_TOKEN = "     " # Replace with your token
# ASR Service Endpoint
WHISPER_API_URL = "   "
LLM_API_KEY = " " # Your API key  for the ASR service

# --- API Call Settings ---
MAX_WHISPER_CONCURRENCY = 100 # Global cap for concurrent ASR HTTP requests
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 2.0


# ========================================================================================
# SECTION 2: UTILITY AND HELPER FUNCTIONS
# ========================================================================================

# To enforce system-wide concurrency limits across all GPUs and files
whisper_semaphore = threading.Semaphore(MAX_WHISPER_CONCURRENCY)

def now_str():
    """Returns the current timestamp as a formatted string."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    """Prints a log message with a timestamp."""
    print(f"[{now_str()}] {msg}", flush=True)

def ensure_dir(path):
    """Creates a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)

def save_and_zip_dir(dir_path):
    """Zips a directory and removes the original directory."""
    zip_path = dir_path + ".zip"
    if not os.path.isdir(dir_path):
        log(f"Warning: Directory not found, cannot zip: {dir_path}")
        return None
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dir_path):
            for f in files:
                fp = os.path.join(root, f)
                arcname = os.path.relpath(fp, start=dir_path)
                zipf.write(fp, arcname)
    shutil.rmtree(dir_path, ignore_errors=True)
    return zip_path

def robust_post(url, headers=None, files=None, data=None, timeout=120, max_retries=MAX_RETRIES):
    """Sends a POST request with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                jitter = random.uniform(0.5, 1.5)
                sleep_time = (RETRY_DELAY_SECONDS * (2 ** (attempt - 1))) * jitter
                log(f"Request failed. Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)

            resp = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            log(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise
    return None

# ========================================================================================
# SECTION 3: YOUTUBE AUDIO DOWNLOAD AND PREPARATION
# ========================================================================================

def download_audio(url, out_dir):
    """
    Downloads audio from a YouTube URL using yt-dlp and saves it as a WAV file.
    The audio is converted to 16kHz mono, which is ideal for most ASR models.
    """
    log(f"Starting download for URL: {url}")
    sanitized_title = ""
    try:
        # Get video metadata to create a safe filename
        with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            title = info_dict.get('title', 'untitled_video')
            # Sanitize title for use as a filename
            sanitized_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c in (' ', '-')]).rstrip()
            sanitized_title = sanitized_title.replace(" ", "_")

        temp_mp3_path = os.path.join(out_dir, f"{sanitized_title}.mp3")
        final_wav_path = os.path.join(out_dir, f"{sanitized_title}.wav")

        if os.path.exists(final_wav_path):
            log(f"WAV file already exists. Skipping download: {final_wav_path}")
            return final_wav_path

        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "outtmpl": os.path.join(out_dir, f"{sanitized_title}"),
            "quiet": False,
        }

        # Download and extract MP3
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Convert MP3 to 16kHz mono WAV
        log(f"Converting {temp_mp3_path} to WAV format...")
        audio = AudioSegment.from_mp3(temp_mp3_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(final_wav_path, format="wav")
        os.remove(temp_mp3_path) # Clean up the intermediate mp3 file

        log(f"Successfully downloaded and converted audio to: {final_wav_path}")
        return final_wav_path

    except Exception as e:
        log(f"ERROR: Failed to download or convert audio for URL {url}. Reason: {e}")
        return None

# ========================================================================================
# SECTION 4: CORE TRANSCRIPTION & DIARIZATION PIPELINE
# ========================================================================================

def call_whisper_api(audio_path, language="te"):
    """Sends an audio file to the Whisper API and returns the transcription."""
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    try:
        with open(audio_path, "rb") as audio_file:
            files = {"file": (os.path.basename(audio_path), audio_file, "audio/wav")}
            data = {"language": language, "model": "/workspace/models/whisper-large"}
            
            with whisper_semaphore: # Limit concurrent API calls
                resp = robust_post(WHISPER_API_URL, headers=headers, files=files, data=data)
            
            result = resp.json()
            return result.get("text", "").strip()
    except Exception as e:
        log(f"Whisper API failed for {audio_path}: {e}")
        return ""

def process_segment(segment_info, audio_path, output_dir, segment_idx):
    """
    Extracts an audio segment, saves it, and sends it for transcription.
    """
    try:
        turn, _, speaker = segment_info
        start, end = float(turn.start), float(turn.end)
        speaker_id = f"SPEAKER_{speaker}"
        duration = end - start

        if duration <= 0.5:  # Skip very short segments
            return None

        segment_filename = f"segment_{segment_idx:05d}.wav"
        segment_path = os.path.join(output_dir, segment_filename)

        # Extract segment using librosa
        y, sr = librosa.load(audio_path, sr=16000, offset=start, duration=duration)
        sf.write(segment_path, y, sr)

        if not os.path.exists(segment_path):
            return None

        # Get transcription for the segment
        text = call_whisper_api(segment_path, language="te")
        if not text: # Don't include segments with no transcription
            return None

        return {
            "start_time": start,
            "end_time": end,
            "duration": duration,
            "speaker_tag": speaker_id,
            "transcription": text,
            "file_name": segment_filename
        }
    except Exception as e:
        log(f"Segment processing error: {e}")
        return None

def process_one_file_on_gpu(audio_path, out_dir, gpu_local_id, hf_token):
    """
    Main processing function for a single audio file on a dedicated GPU.
    Performs diarization and then orchestrates segment transcription.
    """
    try:
        device = torch.device(f"cuda:{gpu_local_id}" if torch.cuda.is_available() else "cpu")
        log(f"Processing {os.path.basename(audio_path)} on device: {device}")

        # Load diarization pipeline on the assigned GPU
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        ).to(device)

        ensure_dir(out_dir)

        # Perform diarization
        diarization = diarization_pipeline(audio_path, num_speakers=2) # Assuming 2 speakers, can be dynamic
        segments = list(diarization.itertracks(yield_label=True))
        results = []

        # Process segments in parallel using a thread pool
        with ThreadPoolExecutor(max_workers=MAX_SEGMENT_THREADS) as pool:
            futures = [
                pool.submit(process_segment, seg_info, audio_path, out_dir, idx)
                for idx, seg_info in enumerate(segments)
            ]
            for fut in as_completed(futures):
                record = fut.result()
                if record:
                    results.append(record)

        if not results:
            log(f"No valid segments found for {audio_path}. Skipping.")
            shutil.rmtree(out_dir, ignore_errors=True)
            return False
            
        # Sort results chronologically and save to CSV
        results_sorted = sorted(results, key=lambda x: x['start_time'])
        df = pd.DataFrame(results_sorted)
        csv_path = os.path.join(out_dir, "metadata.csv")
        df.to_csv(csv_path, index=False)

        # Zip the output directory for easy download and cleanup
        zip_path = save_and_zip_dir(out_dir)
        log(f"Completed and zipped output for {os.path.basename(audio_path)} to: {zip_path}")
        return True

    except Exception as e:
        log(f"File processing failed for {audio_path}: {e}")
        import traceback; traceback.print_exc()
        shutil.rmtree(out_dir, ignore_errors=True) # Cleanup on failure
        return False
    finally:
        # Clean up GPU memory
        try:
            del diarization_pipeline
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def gpu_worker_process(gpu_id, files_for_gpu, base_output_dir, hf_token):
    """
    A dedicated process that manages all tasks for a single GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log(f"GPU worker {gpu_id} starting with {len(files_for_gpu)} files.")

    success_count = 0
    with ThreadPoolExecutor(max_workers=GPU_FILE_WORKERS) as pool:
        futures = []
        for idx, audio_path in enumerate(files_for_gpu):
            file_name_base = os.path.splitext(os.path.basename(audio_path))[0]
            out_dir = os.path.join(base_output_dir, f"{file_name_base}_output")
            futures.append(pool.submit(process_one_file_on_gpu, audio_path, out_dir, 0, hf_token))
        
        for fut in as_completed(futures):
            if fut.result():
                success_count += 1

    log(f"GPU worker {gpu_id} finished. Success: {success_count}/{len(files_for_gpu)}")


# ========================================================================================
# SECTION 5: MAIN ORCHESTRATOR
# ========================================================================================

def main():
    """
    Main function to orchestrate the entire pipeline.
    """
    start_time = time.time()
    log("Starting YouTube Transcription and Diarization Pipeline")

    # --- Step 1: Setup Directories ---
    ensure_dir(BASE_DIR)
    ensure_dir(INPUT_DIR)
    ensure_dir(OUTPUT_BASE_DIR)
    
    # --- Step 2: Login to Hugging Face ---
    if HUGGINGFACE_TOKEN:
        login(token=HUGGINGFACE_TOKEN)
        log("Successfully logged into Hugging Face Hub.")
    else:
        log("WARNING: Hugging Face token not found. Diarization may fail.")
        
    # --- Step 3: Download and Prepare Audio Files ---
    log("--- STAGE 1: Downloading and Preparing Audio ---")
    wav_files_to_process = []
    for url in YOUTUBE_URLS:
        wav_path = download_audio(url, INPUT_DIR)
        if wav_path:
            wav_files_to_process.append(wav_path)

    if not wav_files_to_process:
        log("No audio files were successfully downloaded. Exiting.")
        return

    # --- Step 4: Process Audio Files in Parallel on GPUs ---
    log(f"--- STAGE 2: Starting Diarization and Transcription for {len(wav_files_to_process)} files ---")
    
    # Distribute files evenly across the available GPUs
    shards = [[] for _ in range(NUM_GPUS)]
    for i, f in enumerate(wav_files_to_process):
        shards[i % NUM_GPUS].append(f)

    # Launch one process per GPU for parallel execution
    with ProcessPoolExecutor(max_workers=NUM_GPUS) as proc_pool:
        futures = []
        for gpu_id in range(NUM_GPUS):
            batch = shards[gpu_id]
            if not batch:
                continue
            futures.append(
                proc_pool.submit(gpu_worker_process, gpu_id, batch, OUTPUT_BASE_DIR, HUGGINGFACE_TOKEN)
            )
        
        # Wait for all GPU workers to complete
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                log(f"A GPU worker process encountered an unhandled error: {e}")

    # --- Step 5: Final Summary ---
    elapsed = time.time() - start_time
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    log(f"--- PIPELINE COMPLETE ---")
    log(f"Total files processed: {len(wav_files_to_process)}")
    log(f"Total elapsed time: {h}h {m}m {s}s")
    log(f"Find your zipped output folders in: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()
