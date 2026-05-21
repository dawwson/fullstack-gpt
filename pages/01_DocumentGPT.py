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

  Upload your files on the sidebar
  """
)

# 동일한 파일이 업로드되면 함수 실행을 건너 뛰고 캐싱했던 결과 반환
@st.cache_resource(show_spinner="Embedding file...")  
def embed_file(file):
  file_content = file.read()
  file_path = f".cache/files/{file.name}"

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

  cache_dir = LocalFileStore(f".cache/embeddings/{file.name}")

  cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

  vectorstore = FAISS.from_documents(docs, cached_embeddings)

  retriever = vectorstore.as_retriever()

  return retriever


# 메시지를 화면에 표시하고, 메시지 기록에 저장하는 함수
def send_message(role, message, save=True):
  with st.chat_message(role):
    st.markdown(message)
  if save:
    st.session_state.messages.append({"role": role, "message": message})


# 메시지 기록을 화면에 표시하는 함수
def display_history():
  for message in st.session_state.messages:
    send_message(message["role"], message["message"], save=False)

with st.sidebar:
  file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx"])


if file:
  retriever = embed_file(file)

  send_message("ai", "I'm ready! Ask away!", save=False)

  display_history()

  message = st.chat_input("Ask a question about your file...")

  if message:
    send_message("human", message)
    send_message("ai", "dkjdkjskksdj")
else:
  st.session_state.messages = []
    

