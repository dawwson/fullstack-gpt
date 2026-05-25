
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
import streamlit as st


st.set_page_config(
  page_title="SiteGPT", 
  page_icon="🖥️"
)

st.title("Site GPT")

st.markdown(
"""
Ask questions about the content of a website.

Start by writing the URL of the website on the sidebar.
"""
)

html2text_transformer = Html2TextTransformer()

with st.sidebar:
  url = st.text_input(
    "Write down a URL",
    placeholder="https://www.example.com"
  )


if url:
  # async chromium loader
  loader = AsyncChromiumLoader([url])
  
  docs = loader.load()
  
  # HTML 태그 제거
  transformed = html2text_transformer.transform_documents(docs)

  st.write(transformed)