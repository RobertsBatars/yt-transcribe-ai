#!/usr/bin/env python3

import os
import re
import shutil # For removing chunk folder
import yt_dlp
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment # For audio manipulation
from pydub.utils import make_chunks # For splitting audio

# --- Configuration ---
DOWNLOAD_FOLDER = "downloaded_audio_files" # Main folder for downloaded YouTube audio
TRANSCRIPTION_FOLDER = "transcribed_texts"

# Whisper API limits and safety margins for chunking
WHISPER_API_FILE_SIZE_LIMIT = 25 * 1024 * 1024  # 25 MB
SAFE_CHUNK_SIZE_BYTES = 24 * 1024 * 1024       # 24 MB target for chunks
# Assumed bitrate for MP3 files extracted by yt-dlp (must match yt-dlp settings)
MP3_BITRATE_KBPS = 192 
# --- End Configuration ---

def sanitize_filename(name):
    """
    Sanitizes a string to be used as a valid filename.
    """
    if name is None:
        name = "untitled"
    # For transcription file names, we might get full titles.
    # For audio file names during download, yt-dlp handles sanitization or we use a sanitized title.
    # This function is mainly for the final .txt transcript filename.
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_ ')
    if not name:
        name = "untitled_video" # Fallback for empty names after sanitization
    return name

def download_audio(youtube_url):
    """
    Downloads audio from a given YouTube URL using yt-dlp.
    Saves the audio as an MP3 file in the DOWNLOAD_FOLDER.
    """
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

    video_original_title = "untitled_video"
    # Sanitize title for use in the temporary audio filename to avoid issues
    # yt-dlp also has its own sanitization with --restrict-filenames, but good to have a base.
    sanitized_title_for_audio_filename = "untitled_video_file"

    try:
        temp_ydl_opts = {'quiet': True, 'skip_download': True, 'noplaylist': True}
        with yt_dlp.YoutubeDL(temp_ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_original_title = info.get('title', 'untitled_video')
            # Use a sanitized version of the title for the audio file name
            sanitized_title_for_audio_filename = sanitize_filename(video_original_title)
    except Exception as e:
        print(f"Could not fetch video info for {youtube_url} before download: {e}")
        # Fallback if title fetch fails
        video_original_title = f"video_{youtube_url.split('=')[-1]}"
        sanitized_title_for_audio_filename = sanitize_filename(video_original_title) # Sanitize fallback
        print(f"Using fallback title: {video_original_title}")


    # Output template for yt-dlp using the sanitized title
    output_template = os.path.join(DOWNLOAD_FOLDER, f"{sanitized_title_for_audio_filename}.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': str(MP3_BITRATE_KBPS), # Use configured bitrate
        }],
        'noplaylist': True,
        'quiet': False,
        # 'restrictfilenames': True, # yt-dlp option for safer filenames
    }

    downloaded_audio_path = None
    # Expected path after yt-dlp download and conversion to MP3
    expected_mp3_path = os.path.join(DOWNLOAD_FOLDER, f"{sanitized_title_for_audio_filename}.mp3")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Attempting to download audio for: {video_original_title} ({youtube_url})")
            error_code = ydl.download([youtube_url])
            if error_code != 0:
                print(f"Error during audio download for '{video_original_title}'. yt-dlp exit code: {error_code}")
                return None, video_original_title # Return original title for context

        if os.path.exists(expected_mp3_path):
            downloaded_audio_path = expected_mp3_path
            print(f"Audio successfully downloaded and converted to: {downloaded_audio_path}")
        else:
            # Fallback if specific mp3 not found (e.g. yt-dlp sanitization changed name slightly, or other format)
            print(f"Expected MP3 file not found at {expected_mp3_path}. This might indicate an issue with naming or conversion.")
            # This part may need more robust file finding if yt-dlp naming is unpredictable
            # For now, we rely on the constructed `expected_mp3_path`.
            # If it's not there, consider it a failure for this simplified script.
            print(f"Could not confirm audio file creation for {youtube_url} with title {video_original_title}.")
            return None, video_original_title
        
        return downloaded_audio_path, video_original_title

    except yt_dlp.utils.DownloadError as e:
        print(f"yt-dlp DownloadError for '{video_original_title}' ({youtube_url}): {e}")
        return None, video_original_title
    except Exception as e:
        print(f"An unexpected error occurred during download of '{video_original_title}' ({youtube_url}): {e}")
        return None, video_original_title

# --- Audio Transcription with Chunking Logic (adapted from local video transcriber) ---
def _transcribe_single_audio_file(audio_chunk_path, openai_api_key):
    """Transcribes a single audio file (or chunk) using OpenAI Whisper."""
    client = OpenAI(api_key=openai_api_key)
    try:
        with open(audio_chunk_path, "rb") as audio_file:
            chunk_size_mb = os.path.getsize(audio_chunk_path) / (1024 * 1024)
            print(f"Sending chunk {os.path.basename(audio_chunk_path)} ({chunk_size_mb:.2f} MB) to Whisper API...")
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="text"
            )
        return transcript
    except Exception as e:
        print(f"Error during OpenAI Whisper transcription for chunk {audio_chunk_path}: {e}")
        return None

def _split_and_transcribe_audio(full_audio_path, openai_api_key):
    """Splits a large audio file into manageable chunks and transcribes each."""
    base_full_audio_name = sanitize_filename(os.path.splitext(os.path.basename(full_audio_path))[0])
    # Chunks will be stored in a subfolder of DOWNLOAD_FOLDER
    chunks_subfolder = os.path.join(DOWNLOAD_FOLDER, f"{base_full_audio_name}_chunks")
    os.makedirs(chunks_subfolder, exist_ok=True)

    print(f"Splitting large audio file: {full_audio_path}")
    try:
        audio = AudioSegment.from_mp3(full_audio_path)
    except Exception as e:
        print(f"Could not load audio file {full_audio_path} with pydub: {e}")
        if os.path.exists(chunks_subfolder): shutil.rmtree(chunks_subfolder)
        return None

    bytes_per_second_audio = (MP3_BITRATE_KBPS * 1000) / 8.0
    if bytes_per_second_audio <= 0:
        print("Error: Audio bytes per second is zero or negative. Check MP3_BITRATE_KBPS.")
        if os.path.exists(chunks_subfolder): shutil.rmtree(chunks_subfolder)
        return None
        
    max_duration_seconds_per_chunk = (SAFE_CHUNK_SIZE_BYTES / bytes_per_second_audio) * 0.95 # 5% safety margin
    chunk_length_ms = int(max_duration_seconds_per_chunk * 1000)

    if chunk_length_ms <= 1000: # Chunks must be > 0s, realistically > 1s.
        print(f"Error: Calculated chunk length ({chunk_length_ms}ms) is too small. Cannot split effectively.")
        if os.path.exists(chunks_subfolder): shutil.rmtree(chunks_subfolder)
        return None
        
    print(f"Targeting audio chunks of up to {chunk_length_ms / 1000.0:.2f} seconds.")
    
    audio_chunks = make_chunks(audio, chunk_length_ms)
    full_transcription_parts = []
    all_chunks_processed_successfully = True

    for i, audio_segment_chunk in enumerate(audio_chunks):
        chunk_filename = os.path.join(chunks_subfolder, f"chunk_{base_full_audio_name}_{i}.mp3")
        print(f"Exporting audio chunk {i+1}/{len(audio_chunks)}: {chunk_filename}")
        try:
            audio_segment_chunk.export(chunk_filename, format="mp3", bitrate=f"{MP3_BITRATE_KBPS}k")
        except Exception as e:
            print(f"Error exporting audio chunk {chunk_filename}: {e}")
            all_chunks_processed_successfully = False; break

        if os.path.getsize(chunk_filename) >= WHISPER_API_FILE_SIZE_LIMIT:
            print(f"Critical Error: Exported chunk {chunk_filename} is too large. Aborting for this file.")
            try: os.remove(chunk_filename)
            except OSError: pass
            all_chunks_processed_successfully = False; break
        
        transcription_part = _transcribe_single_audio_file(chunk_filename, openai_api_key)
        try: os.remove(chunk_filename)
        except OSError as e: print(f"Warning: Could not delete audio chunk file {chunk_filename}: {e}")

        if transcription_part is None:
            print(f"Failed to transcribe audio chunk {i+1}/{len(audio_chunks)}. Aborting for this video.");
            all_chunks_processed_successfully = False; break
        full_transcription_parts.append(transcription_part)

    if os.path.exists(chunks_subfolder):
        try: shutil.rmtree(chunks_subfolder); print(f"Cleaned up chunk folder: {chunks_subfolder}")
        except OSError as e: print(f"Warning: Could not remove chunk folder {chunks_subfolder}: {e}")

    if not all_chunks_processed_successfully:
        print(f"Transcription for {full_audio_path} was incomplete due to errors."); return None
    return " ".join(full_transcription_parts) if full_transcription_parts else None

def transcribe_audio_manager(audio_file_path, openai_api_key):
    """Manages transcription, deciding whether to split the audio file or transcribe directly."""
    if not openai_api_key:
        print("OpenAI API key not provided. Skipping transcription."); return None
    if not audio_file_path or not os.path.exists(audio_file_path):
        print(f"Audio file not found at {audio_file_path}. Skipping transcription."); return None

    file_size = os.path.getsize(audio_file_path)
    if file_size < (WHISPER_API_FILE_SIZE_LIMIT * 0.98): # Using 98% of limit as threshold
        print(f"Audio file {os.path.basename(audio_file_path)} ({file_size/(1024*1024):.2f} MB) is within size limit. Transcribing directly.")
        return _transcribe_single_audio_file(audio_file_path, openai_api_key)
    else:
        print(f"Audio file {os.path.basename(audio_file_path)} ({file_size/(1024*1024):.2f} MB) exceeds threshold. Attempting to split.")
        return _split_and_transcribe_audio(audio_file_path, openai_api_key)
# --- End of Transcription Logic ---


def save_transcription(text_content, original_video_title):
    """Saves the transcribed text to a .txt file, named after the original video title."""
    if not os.path.exists(TRANSCRIPTION_FOLDER):
        os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)
    
    # Sanitize the original video title to create a safe filename for the transcript
    sanitized_transcript_filename = sanitize_filename(original_video_title)
    filepath = os.path.join(TRANSCRIPTION_FOLDER, f"{sanitized_transcript_filename}.txt")
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text_content)
        print(f"Transcription saved to: {filepath}")
    except Exception as e:
        print(f"Error saving transcription to {filepath}: {e}")

def process_youtube_links_file(links_filepath, current_openai_api_key):
    """Main processing loop for YouTube links."""
    try:
        with open(links_filepath, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and (line.startswith("http://") or line.startswith("https://"))]
    except FileNotFoundError:
        print(f"Error: Input file '{links_filepath}' not found."); return
    except Exception as e:
        print(f"Error reading file '{links_filepath}': {e}"); return

    if not urls: print(f"No valid YouTube URLs found in '{links_filepath}'."); return
    print(f"Found {len(urls)} URL(s) to process.")

    for i, url in enumerate(urls):
        print(f"\n--- Processing URL {i+1}/{len(urls)}: {url} ---")
        
        # download_audio returns (path_to_audio_file, original_video_title)
        audio_file_path, video_original_title = download_audio(url)
        
        if audio_file_path and os.path.exists(audio_file_path):
            print(f"Full audio downloaded to: {audio_file_path} (Original Title: '{video_original_title}')")
            
            # Use the new transcription manager that handles splitting
            transcription_text = transcribe_audio_manager(audio_file_path, current_openai_api_key)
            
            if transcription_text is not None:
                save_transcription(transcription_text, video_original_title)
            else:
                print(f"Failed to get transcription for '{video_original_title}' ({url}).")
            
            try: # Clean up the main downloaded audio file
                os.remove(audio_file_path)
                print(f"Cleaned up main audio file: {audio_file_path}")
            except OSError as e:
                print(f"Error deleting main audio file {audio_file_path}: {e}")
        else:
            print(f"Failed to download or locate audio for URL ({url}). Title: '{video_original_title}'. Skipping.")

def main():
    load_dotenv()
    print("YouTube Audio Downloader and Transcriber (with .env and chunking support)")
    print("=" * 70)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("\nOpenAI API key not found. Please set OPENAI_API_KEY in your .env file or environment.")
        while not openai_api_key:
            openai_api_key = input("Enter your OpenAI API key (or press Enter to skip all transcriptions): ").strip()
            if not openai_api_key: print("No API key entered. Transcription will be skipped."); break
    
    if openai_api_key: print("OpenAI API key loaded.")

    while True:
        input_file_path = input("\nEnter path to the text file containing YouTube URLs: ").strip()
        if os.path.isfile(input_file_path): break
        else: print(f"File not found: {input_file_path}. Please enter a valid file path.")
            
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)
        
    process_youtube_links_file(input_file_path, openai_api_key)
    
    print("\n--- All processing complete. ---")
    try: # Optional: Clean up the main download folder if it's empty
        if os.path.exists(DOWNLOAD_FOLDER) and not os.listdir(DOWNLOAD_FOLDER):
            os.rmdir(DOWNLOAD_FOLDER)
            print(f"Cleaned up empty download folder: {DOWNLOAD_FOLDER}")
    except OSError as e:
        print(f"Could not remove download folder {DOWNLOAD_FOLDER} (it might not be empty): {e}")

if __name__ == "__main__":
    main()
