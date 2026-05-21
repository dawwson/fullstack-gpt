from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
import streamlit as st

st.set_page_config(page_title="Document GPT", page_icon="📄")

st.markdown(
  """
  Welcome!
              
  Use this chatbot to ask questions to an AI about your files!
  """
)

def embed_file(file):
  file_content = file.read()
  file_path = f"./.cache/files/{file.name}"

  with open(file_path, "wb") as f:
    f.write(file_content)
  
  splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
  )

  loader = UnstructuredLoader(file_path)

  docs = filter_complex_metadata(loader.load_and_split(text_splitter=splitter))

  embeddings = OpenAIEmbeddings()

  cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

  cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

  vectorstore = FAISS.from_documents(docs, cached_embeddings)

  retriever = vectorstore.as_retriever()

  return retriever


file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx"])

if file:
  retriever = embed_file(file)
  retriever.invoke("winston")

