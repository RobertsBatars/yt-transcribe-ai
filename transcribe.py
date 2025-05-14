#!/usr/bin/env python3

import os
import re
import yt_dlp
from openai import OpenAI
from dotenv import load_dotenv # Import load_dotenv

# --- Configuration ---
DOWNLOAD_FOLDER = "downloaded_audio_files"
TRANSCRIPTION_FOLDER = "transcribed_texts"
# --- End Configuration ---

def sanitize_filename(name):
    """
    Sanitizes a string to be used as a valid filename.
    Removes or replaces characters that are not allowed in filenames
    and ensures the name is not excessively long if necessary (not implemented here).
    """
    if name is None:
        name = "untitled"
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_ ')
    if not name:
        name = "untitled_video"
    return name

def download_audio(youtube_url):
    """
    Downloads audio from a given YouTube URL using yt-dlp.
    Saves the audio as an MP3 file in the DOWNLOAD_FOLDER.

    Args:
        youtube_url (str): The URL of the YouTube video.

    Returns:
        tuple: (path_to_audio_file, original_video_title)
               Returns (None, original_video_title) if download fails.
               original_video_title might be a fallback if fetching fails.
    """
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

    video_original_title = "untitled_video"
    sanitized_title_for_filename = "untitled_video_file"

    try:
        temp_ydl_opts = {'quiet': True, 'skip_download': True, 'noplaylist': True}
        with yt_dlp.YoutubeDL(temp_ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_original_title = info.get('title', 'untitled_video')
            sanitized_title_for_filename = sanitize_filename(video_original_title)
    except Exception as e:
        print(f"Could not fetch video info for {youtube_url} before download: {e}")
        video_original_title = f"video_{youtube_url.split('=')[-1]}"
        sanitized_title_for_filename = sanitize_filename(video_original_title)
        print(f"Using fallback title: {video_original_title}")

    output_template = os.path.join(DOWNLOAD_FOLDER, f"{sanitized_title_for_filename}.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'noplaylist': True,
        'quiet': False,
    }

    downloaded_audio_path = None
    expected_mp3_path = os.path.join(DOWNLOAD_FOLDER, f"{sanitized_title_for_filename}.mp3")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Attempting to download audio for: {video_original_title} ({youtube_url})")
            error_code = ydl.download([youtube_url])
            if error_code != 0:
                print(f"Error during audio download for '{video_original_title}'. yt-dlp exit code: {error_code}")
                return None, video_original_title

        if os.path.exists(expected_mp3_path):
            downloaded_audio_path = expected_mp3_path
            print(f"Audio successfully downloaded and converted to: {downloaded_audio_path}")
        else:
            print(f"Expected MP3 file not found at {expected_mp3_path}. Checking for alternatives...")
            for ext in ['m4a', 'wav', 'ogg', 'webm']:
                alt_path = os.path.join(DOWNLOAD_FOLDER, f"{sanitized_title_for_filename}.{ext}")
                if os.path.exists(alt_path):
                    downloaded_audio_path = alt_path
                    print(f"Found alternative audio file: {downloaded_audio_path}")
                    break
            if not downloaded_audio_path:
                print(f"Could not find downloaded audio file for '{video_original_title}' with expected name structure.")
                return None, video_original_title
        
        return downloaded_audio_path, video_original_title

    except yt_dlp.utils.DownloadError as e:
        print(f"yt-dlp DownloadError for '{video_original_title}' ({youtube_url}): {e}")
        return None, video_original_title
    except Exception as e:
        print(f"An unexpected error occurred during download of '{video_original_title}' ({youtube_url}): {e}")
        return None, video_original_title

def transcribe_audio_openai(audio_file_path, openai_api_key):
    """
    Transcribes the given audio file using OpenAI's Whisper API.
    """
    if not openai_api_key:
        print("OpenAI API key not provided. Skipping transcription.")
        return None
    if not audio_file_path or not os.path.exists(audio_file_path):
        print(f"Audio file not found at {audio_file_path}. Skipping transcription.")
        return None

    client = OpenAI(api_key=openai_api_key)
    print(f"Transcribing audio file: {audio_file_path}...")

    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        print("Transcription successful.")
        return transcript
    except Exception as e:
        print(f"Error during OpenAI Whisper transcription for {audio_file_path}: {e}")
        return None

def save_transcription(text_content, original_video_title_for_filename):
    """
    Saves the transcribed text to a .txt file.
    """
    if not os.path.exists(TRANSCRIPTION_FOLDER):
        os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)
    
    sanitized_transcript_filename_base = sanitize_filename(original_video_title_for_filename)
    filepath = os.path.join(TRANSCRIPTION_FOLDER, f"{sanitized_transcript_filename_base}.txt")
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text_content)
        print(f"Transcription saved to: {filepath}")
    except Exception as e:
        print(f"Error saving transcription to {filepath}: {e}")

def process_youtube_links_file(links_filepath, current_openai_api_key):
    """
    Main processing loop: reads URLs from file, downloads audio, transcribes, and saves.
    """
    try:
        with open(links_filepath, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and (line.startswith("http://") or line.startswith("https://"))]
    except FileNotFoundError:
        print(f"Error: Input file '{links_filepath}' not found.")
        return
    except Exception as e:
        print(f"Error reading file '{links_filepath}': {e}")
        return

    if not urls:
        print(f"No valid YouTube URLs found in '{links_filepath}'. Ensure URLs start with http:// or https://.")
        return

    print(f"Found {len(urls)} URL(s) to process.")

    for i, url in enumerate(urls):
        print(f"\n--- Processing URL {i+1}/{len(urls)}: {url} ---")
        
        audio_file_path, video_title = download_audio(url)
        
        if audio_file_path and os.path.exists(audio_file_path):
            print(f"Audio downloaded to: {audio_file_path} (Original Video Title: '{video_title}')")
            
            transcription_text = transcribe_audio_openai(audio_file_path, current_openai_api_key)
            
            if transcription_text is not None:
                save_transcription(transcription_text, video_title)
            else:
                print(f"Failed to transcribe audio for '{video_title}' ({url}).")
            
            try:
                os.remove(audio_file_path)
                print(f"Cleaned up audio file: {audio_file_path}")
            except OSError as e:
                print(f"Error deleting audio file {audio_file_path}: {e}")
        else:
            print(f"Failed to download or locate audio for '{video_title}' ({url}). Skipping transcription.")

def main():
    """
    Main function to run the script.
    """
    # Load environment variables from .env file
    load_dotenv() # THIS IS THE NEW LINE

    print("YouTube Audio Downloader and Transcriber (with .env support)")
    print("=" * 60)

    # Get OpenAI API Key from environment (now potentially loaded from .env)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("\nOpenAI API key not found in environment variables or .env file.")
        print("You can set it in a .env file (e.g., OPENAI_API_KEY='your_key') or enter it manually.")
        while not openai_api_key:
            openai_api_key = input("Enter your OpenAI API key (or press Enter to skip transcription): ").strip()
            if not openai_api_key:
                print("Transcription will be skipped as no API key was provided.")
                break
    
    if openai_api_key:
         print("OpenAI API key loaded.")
    else:
        print("Proceeding without OpenAI API key. Transcription step will be skipped for all videos.")

    while True:
        input_file_path = input("\nEnter the path to the text file containing YouTube URLs: ").strip()
        if os.path.isfile(input_file_path):
            break
        else:
            print(f"File not found: {input_file_path}. Please enter a valid file path.")
            
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)
        
    process_youtube_links_file(input_file_path, openai_api_key)
    
    print("\n--- All processing complete. ---")

if __name__ == "__main__":
    main()
