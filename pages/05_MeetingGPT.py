import os
from pathlib import Path
from pydub import AudioSegment
import streamlit as st
import subprocess
import math
import openai
import glob


CACHE_DIR = Path("./.cache")
UPLOAD_DIR = CACHE_DIR / "files"
CHUNK_DIR = CACHE_DIR / "chunks"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
  page_title="MeetingGPT", 
  page_icon="💼"
)

st.title("Meeting GPT")

st.markdown(
"""
# MeetingGPT

Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)


def extract_audio_from_video(video_path):
  # video와 같은 위치, 같은 이름의 .mp3 파일로 저장
  audio_path = video_path.replace("mp4", "mp3")
  
  command = [
    "ffmpeg", 
    "-y" # overwrite
    "-i", 
    video_path, 
    "-vn", # no video
    audio_path
  ]
  
  subprocess.run(command)


def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
  track = AudioSegment.from_mp3(audio_path)
  
  chunk_len = chunk_size * 60 * 1000

  chunks = math.ceil(len(track) / chunk_len)

  for i in range(chunks):
    start_time = i * chunk_len
    end_time = (i + 1) * chunk_len

    chunk = track[start_time:end_time]

    chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
  files = glob.glob(f"{chunk_folder}/*.mp3")
  files.sort()

  for file in files:
    
    # rb: reading as binary
    # a: append mode
    with open(file, "rb") as audio_file, open(destination, "a") as text_file:
      
      transript = openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file, 
        language="ko"
      )

      text_file.write(transript.text)


with st.sidebar:
  video = st.file_uploader("Video", type=["mp4", "avi", "mkv", "mov"])


if video:
  video_path = str(UPLOAD_DIR / video.name)
  audio_path = video_path.replace("mp4", "mp3")
  audio_chunks_path = str(CHUNK_DIR)
  trascript_path = video_path.replace("mp4", "txt")
  
  has_transcript = os.path.exists(trascript_path)

  if has_transcript:
    st.info("Transcript already exists.")
    st.stop()

  with st.status("Loading video..."):
    # save video file
    with open(video_path, "wb") as f:
      video_content = video.read()
      f.write(video_content)
    

  with st.status("Extracting audio..."):
    extract_audio_from_video(video_path)
  
  with st.status("Cutting audio segments..."):
    # 10분 단위로 자름
    cut_audio_in_chunks(audio_path, 10, audio_chunks_path) 

  with st.status("Transcribing audio..."):
    transcribe_chunks(audio_chunks_path, trascript_path)

