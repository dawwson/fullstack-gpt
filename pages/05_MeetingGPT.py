import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    "-y", # overwrite
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
  transcript_path = video_path.replace("mp4", "txt")
  
  has_transcript = os.path.exists(transcript_path)

  if has_transcript:
    transcript_tab, summary_tab, qa_tab = st.tabs(["Transcript", "Summary", "Q&A"])

    with transcript_tab:
      with open(transcript_path, "r") as file:
        st.write(file.read())

    with summary_tab:
      start = st.button("Generate summary")

      if start:
        llm = ChatOpenAI(
          temperature=0.1,
        )

        loader = TextLoader(transcript_path)

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
          chunk_size=800,
          chunk_overlap=50,
        )

        docs = loader.load_and_split(text_splitter=splitter)

        first_summary_prompt = ChatPromptTemplate.from_template(
          """
          Write a concise summary of the following:
          "{text}"
          CONCISE SUMMARY:
          """
        )

        first_summary_chain = first_summary_prompt | llm | StrOutputParser()

        summary = first_summary_chain.invoke({"text": docs[0].page_content})

        refine_prompt = ChatPromptTemplate.from_template(
          """
          Your job is to produce a final summary.
          We have provided an existing summary up to a certain point: {existing_summary}
          We have the opportunity to refine the existing summary (only if needed) with some more context below.
          ---------
          {context}
          ---------
          Given the new context, refine the original summary.
          If the context isn't useful, RETURN the original summary.
          """
        )

        refine_chain = refine_prompt | llm | StrOutputParser()

        with st.status("Summarizing...") as status:
          
          for i, doc in enumerate(docs[1:]):
            status.update(label=f"Processing document {i+1} / {len(docs)-1}")
            summary = refine_chain.invoke({"existing_summary": summary, "context": doc.page_content})
        
        st.write(summary)


  else:
    with st.status("Loading video...") as status:
      # save video file
      with open(video_path, "wb") as f:
        video_content = video.read()
        f.write(video_content)

      status.update(label="Extracting audio...")
      extract_audio_from_video(video_path)
    
      status.update(label="Cutting audio segments...")
      cut_audio_in_chunks(audio_path, 10, audio_chunks_path) # 10분 단위로 자름

      status.update(label="Transcribing audio...")
      transcribe_chunks(audio_chunks_path, transcript_path)


  
