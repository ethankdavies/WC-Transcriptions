import os
import yt_dlp as youtube_dl
import whisper
import streamlit as st
import shutil
import tempfile
import isodate
from googleapiclient.discovery import build
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Set up YouTube API key
YOUTUBE_API_KEY = 'AIzaSyCQOiMG3INHF_xzlU2YZGG4gJOTqYp1Uc0'
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Load Whisper large-v2 model for transcription
whisper_model = whisper.load_model("large-v2")

# Load the T5 model for summarization
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Channel mapping for dropdown
channel_mapping = {
    "Ruben Gallego": "UCxggVFesZy65a0WBT3_roXQ",
    "Josh Stein": "UCz1XsZYTzudZtHAIQvuEQAQ",
    "Joyce Craig": "UCBt2qbHd5ns7ryv3n0Y_bHw",
    "Jacky Rosen": "UCq2JO4WbdKPvTfcfmHPmWMw",
    "Sherrod Brown": "UCt_l7Nge_872rTm5Jvbo6Mw",
    "Bob Casey": "UCOak7SAWIvog_DN6dRMO3CA",
    "Kamala Harris": "UC0XBsJpPhOLg0k4x9ZwrWzw",
    "Tammy Baldwin": "UC_XjYCRbbI2_TDDjJidwk0Q",
    "2WAY with Mark Halperin": "UCq7OKQb6_1tbA73oSloIiZQ"
}

# Function to fetch recent videos from a selected YouTube channel
def get_recent_videos(channel_id):
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=10,
        order="date",
        type="video"
    )
    response = request.execute()
    videos = response['items']

    # Filter videos based on the duration
    video_ids = [video['id']['videoId'] for video in videos]
    details_request = youtube.videos().list(part="contentDetails", id=",".join(video_ids))
    details_response = details_request.execute()

    filtered_videos = []
    for video, details in zip(videos, details_response['items']):
        duration = isodate.parse_duration(details['contentDetails']['duration']).total_seconds()
        if duration < 120 or channel_id == "UCq7OKQb6_1tbA73oSloIiZQ":  # No time limit for 2WAY with Mark Halperin
            filtered_videos.append(video)
    
    return filtered_videos

# Function to download and transcribe YouTube videos
def download_and_transcribe(video_id):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': '%(id)s.%(ext)s',
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=True)
            audio_file = f"{info_dict['id']}.mp3"
            
            # Transcribe the downloaded audio
            result = whisper_model.transcribe(audio_file)
            transcription = result['text']
            os.remove(audio_file)  # Clean up the audio file after transcribing
            return transcription

    except Exception as e:
        return f"Error: {str(e)}"

# Function to summarize the transcription using T5
def summarize_transcription(text):
    # Preprocess the text for summarization
    inputs = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate the summary
    summary_ids = t5_model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the summary
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Streamlit Web Interface
st.title("YouTube Transcriber")

# Channel selection dropdown
selected_channel = st.selectbox("Select a YouTube Channel", list(channel_mapping.keys()))
channel_id = channel_mapping[selected_channel]

# Fetch and display recent videos
if selected_channel:
    st.write(f"Fetching recent videos from {selected_channel}...")
    recent_videos = get_recent_videos(channel_id)
    video_options = {video['snippet']['title']: video['id']['videoId'] for video in recent_videos}
    
    # Video selection dropdown
    selected_video_title = st.selectbox("Select a Video", list(video_options.keys()))
    selected_video_id = video_options[selected_video_title]
    
    # Checkbox to choose whether to summarize
    summarize_option = st.checkbox("Summarize the transcription (for 2WAY with Mark Halperin)")
    
    # Transcribe button
    if st.button("Transcribe"):
        st.write(f"Transcribing video: {selected_video_title}...")
        transcription = download_and_transcribe(selected_video_id)
        
        if summarize_option and selected_channel == "2WAY with Mark Halperin":
            transcription = summarize_transcription(transcription)
        
        st.write("Transcription Result:")
        st.write(transcription)
