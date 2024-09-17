import os
import yt_dlp as youtube_dl
import whisper
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import isodate
from googleapiclient.discovery import build
import tempfile
import shutil
import html
from fuzzywuzzy import fuzz
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load Whisper large-v2 model for transcription (provides higher accuracy)
whisper_model = whisper.load_model("large-v2")

# Set up your YouTube API key
YOUTUBE_API_KEY = 'AIzaSyCQOiMG3INHF_xzlU2YZGG4gJOTqYp1Uc0'
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Load the T5 model for summarization
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# List of common words/phrases to recognize better in transcription
PHRASE_HINTS = {
    "Ruben Gallego": ["Ruben Gallego"],
    "Kari Lake": ["Kari Lake"],
    "Josh Stein": ["Josh Stein"],
    # Add other phrases...
}

# Hard-coded common transcription errors and corrections
COMMON_CORRECTIONS = {
    "Kerry Lake": "Kari Lake",
    "Donald Drumpf": "Donald Trump",
}

# Context-based corrections based on the surrounding content
CONTEXT_CORRECTIONS = {
    "Kerry Lake": ("Kari Lake", ["Republican", "Arizona"]),
}

# Function to replace common words/phrases using fuzzy matching
def replace_common_phrases(transcription, threshold=85):
    for key_phrase, alternatives in PHRASE_HINTS.items():
        for phrase in alternatives:
            words = transcription.split()
            for i, word in enumerate(words):
                if fuzz.ratio(word.lower(), phrase.lower()) >= threshold:
                    words[i] = key_phrase
            transcription = " ".join(words)
    return transcription

# Function to apply hard-coded corrections
def apply_common_corrections(transcription):
    for incorrect, correct in COMMON_CORRECTIONS.items():
        transcription = transcription.replace(incorrect, correct)
    return transcription

# Function to apply context-based corrections
def apply_context_corrections(transcription):
    for incorrect, (correct, context_terms) in CONTEXT_CORRECTIONS.items():
        if incorrect in transcription:
            for term in context_terms:
                if term in transcription:
                    transcription = transcription.replace(incorrect, correct)
                    break
    return transcription

# Function to summarize the transcription using T5
def summarize_transcription(text):
    # Preprocess the text for summarization
    inputs = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate the summary
    summary_ids = t5_model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the summary
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Transcribe audio using Whisper with all corrections and summarize if needed
def transcribe_audio_whisper(audio_file, channel_id):
    try:
        result = whisper_model.transcribe(audio_file)
        transcription = result['text']
        
        # Apply corrections
        transcription = replace_common_phrases(transcription)
        transcription = apply_common_corrections(transcription)
        transcription = apply_context_corrections(transcription)
        
        # If the transcription is from "2WAY with Mark Halperin", summarize it
        if channel_id == "UCq7OKQb6_1tbA73oSloIiZQ":  # ID for "2WAY with Mark Halperin"
            transcription = summarize_transcription(transcription)
        
        print(f"Transcription: {transcription}")
        return transcription
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

# Fetch recent videos from YouTube, filtering for videos under 2 minutes (except for 2WAY with Mark Halperin)
def get_recent_videos(channel_id):
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=10,
        order="date",
        type="video"  # Ensure we only fetch videos
    )
    response = request.execute()
    videos = response['items']

    # Check if it's the "2WAY with Mark Halperin" channel (no time limit)
    if channel_id == "UCq7OKQb6_1tbA73oSloIiZQ":
        return videos

    # Filter videos under 2 minutes
    video_ids = [video['id']['videoId'] for video in videos]
    details_request = youtube.videos().list(part="contentDetails", id=",".join(video_ids))
    details_response = details_request.execute()

    filtered_videos = []
    for video, details in zip(videos, details_response['items']):
        duration = isodate.parse_duration(details['contentDetails']['duration']).total_seconds()
        if duration < 120:  # Only keep videos under 2 minutes
            filtered_videos.append(video)

    return filtered_videos

# Download audio using youtube-dl, automatically handle file extensions and overwrite existing files
def download_audio(video_id):
    url = f'https://www.youtube.com/watch?v={video_id}'
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': f'{video_id}.%(ext)s',  # Use dynamic extension handling
        'noplaylist': True,
        'verbose': True,
        'nooverwrites': False  # Overwrite existing files automatically
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            downloaded_file = f"{video_id}.mp3"  # Return the .mp3 file since .webm was deleted
            return os.path.abspath(downloaded_file).replace("\\", "/")  # Ensure forward slashes in the file path
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

# Preprocess audio using FFmpeg (convert to WAV for Whisper), using a temporary directory for processing
def preprocess_audio(audio_file):
    # Create a temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    temp_audio_file = os.path.join(temp_dir, os.path.basename(audio_file))

    # Copy the downloaded file to the temporary directory
    try:
        shutil.copyfile(audio_file, temp_audio_file)
    except Exception as e:
        print(f"Error copying file to temporary directory: {e}")
        return None

    # Ensure paths use forward slashes
    temp_audio_file = temp_audio_file.replace("\\", "/")
    wav_file = f'{temp_audio_file}.wav'.replace("\\", "/")

    # FFmpeg command to convert the file to WAV format
    command = ['ffmpeg', '-v', 'verbose', '-i', temp_audio_file, '-ar', '16000', '-ac', '1', wav_file]
    try:
        subprocess.run(command, check=True)
        return wav_file
    except subprocess.CalledProcessError as e:
        print(f"Error during preprocessing: {e}")
        return None

# GUI components for video selection and transcription
root = tk.Tk()
root.title("YouTube Video Transcriber")
root.geometry("500x600")

# Function to populate the video list
def populate_videos():
    selected_channel = channel_combobox.get()  # Get selected channel
    if selected_channel:
        video_listbox.delete(0, tk.END)  # Clear previous entries
        channel_id = channel_mapping[selected_channel]  # Get channel ID
        recent_videos = get_recent_videos(channel_id)
        for video in recent_videos:
            video_title = video['snippet']['title']
            video_title_clean = html.unescape(video_title)  # Clean HTML-encoded characters
            video_id = video['id']['videoId']
            video_listbox.insert(tk.END, f"{video_title_clean} (ID: {video_id})")

# Download and transcribe selected video
def download_and_transcribe():
    selected_video = video_listbox.get(tk.ANCHOR)
    if not selected_video:
        messagebox.showerror("Error", "Please select a video.")
        return

    video_id = selected_video.split("(ID: ")[1].split(")")[0]
    channel_name = channel_combobox.get()
    channel_id = channel_mapping[channel_name]

    # Download and preprocess audio
    audio_file = download_audio(video_id)
    if audio_file:
        print(f"Downloaded audio file: {audio_file}")
        audio_file = preprocess_audio(audio_file)

    if audio_file:
        # Transcribe audio using Whisper
        transcription = transcribe_audio_whisper(audio_file, channel_id)
        if transcription:
            transcription_text.delete(1.0, tk.END)
            transcription_text.insert(tk.END, transcription)
        else:
            messagebox.showerror("Error", "Failed to transcribe the audio.")
    else:
        messagebox.showerror("Error", "Failed to download or preprocess the audio.")

# GUI elements for channel selection and video list
channel_mapping = {
    "Ruben Gallego": "UCxggVFesZy65a0WBT3_roXQ",
    "Josh Stein": "UCz1XsZYTzudZtHAIQvuEQAQ",
    "Joyce Craig": "UCBt2qbHd5ns7ryv3n0Y_bHw",
    "Jacky Rosen": "UCq2JO4WbdKPvTfcfmHPmWMw",
    "Sherrod Brown": "UCt_l7Nge_872rTm5Jvbo6Mw",
    "Bob Casey": "UCOak7SAWIvog_DN6dRMO3CA",
    "Kamala Harris": "UC0XBsJpPhOLg0k4x9ZwrWzw",
    "Tammy Baldwin": "UC_XjYCRbbI2_TDDjJidwk0Q",
    "2WAY with Mark Halperin": "UCq7OKQb6_1tbA73oSloIiZQ"  # No time limit for this channel
}

channel_combobox = ttk.Combobox(root, values=list(channel_mapping.keys()), state="readonly")
channel_combobox.set("Select a Channel")
channel_combobox.pack(pady=10)

load_videos_button = tk.Button(root, text="Load Videos", command=populate_videos)
load_videos_button.pack(pady=5)

video_listbox = tk.Listbox(root, height=10, width=50)
video_listbox.pack(pady=5)

transcribe_button = tk.Button(root, text="Transcribe Selected Video", command=download_and_transcribe)
transcribe_button.pack(pady=10)

transcription_text = tk.Text(root, height=10, width=50)
transcription_text.pack(pady=10)

# Start the GUI loop
root.mainloop()
cd