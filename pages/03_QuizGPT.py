from langchain_community.retrievers import WikipediaRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
import streamlit as st

st.set_page_config(
  page_title="Quiz GPT", 
  page_icon="❓"
)

st.title("Quiz GPT")


@st.cache_resource(show_spinner="Loading file...")  
def split_file(file):
  file_content = file.read()
  file_path = f"./.cache/quiz_files/{file.name}"

  with open(file_path, "wb") as f:
    f.write(file_content)
  
  splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
  )

  loader = UnstructuredLoader(file_path)

  docs = loader.load_and_split(text_splitter=splitter)

  return docs


with st.sidebar:
  choice = st.selectbox(
    "Choose what you want to use", 
    (
      "File",
      "Wikipedia Article"
    )
  )

  if choice == "File":
    file = st.file_uploader("Upload a .docx, .txt, or .pdf file", type=["docx", "pdf", "txt"])
  
    if file:
      docs = split_file(file)
  
  else:
    topic = st.text_input("Name of the article")
    
    if topic:
      retriever = WikipediaRetriever(top_k_results=5)
      
      with st.status("Searching Wikipedia..."):
        docs = retriever._get_relevant_documents(topic)


