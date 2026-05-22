from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
import streamlit as st

st.set_page_config(
  page_title="Private GPT", 
  page_icon="📄"
)

st.markdown(
  """
  Welcome!
              
  Use this chatbot to ask questions to an AI about your files!

  Upload your files on the sidebar
  """
)

with st.sidebar:
  file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx"])


class ChatCallbackHandler(BaseCallbackHandler):
  message = ""
  
  def on_llm_start(self, *args, **kwargs):
    # LLM이 시작될 때 빈 메시지 박스를 생성하여 실시간으로 토큰을 표시할 준비를 한다.
    self.message_box = st.empty()
  
  def on_llm_end(self, *args, **kwargs):
    # LLM이 종료될 때 메시지 박스를 업데이트하여 완료된 메시지를 표시한다.
    save_message("ai", self.message)

  def on_llm_new_token(self, token, *args, **kwargs):
    # LLM이 새로운 토큰을 생성할 때마다 생성된 토큰을 메시지에 추가하고, 메시지 박스를 업데이트하여 실시간으로 토큰이 표시되도록 한다.
    self.message += token
    self.message_box.markdown(self.message)


# LLM 초기화
llm = ChatOllama(
  model="mistral:latest",
  temperature=0.1,
  streaming=True,
  callbacks=[
    ChatCallbackHandler(),
  ]
)


# 동일한 파일이 업로드되면 함수 실행을 건너 뛰고 캐싱했던 결과 반환
@st.cache_resource(show_spinner="Embedding file...")  
def embed_file(file):
  file_content = file.read()
  file_path = f"./.cache/private_files/{file.name}"

  with open(file_path, "wb") as f:
    f.write(file_content)
  
  splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
  )

  loader = UnstructuredLoader(file_path)

  docs = filter_complex_metadata(loader.load_and_split(text_splitter=splitter))

  embeddings = OllamaEmbeddings(model="mistral:latest")

  cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")

  cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

  vectorstore = FAISS.from_documents(docs, cached_embeddings)

  retriever = vectorstore.as_retriever()

  return retriever


# 메시지를 메시지 기록에 저장하는 함수
def save_message(role, message):
  st.session_state.messages.append({"role": role, "message": message})


# 메시지를 화면에 표시하고, 메시지 기록에 저장하는 함수
def send_message(role, message, save=True):
  with st.chat_message(role):
    st.markdown(message)
  if save:
    save_message(role, message)


# 메시지 기록을 화면에 표시하는 함수
def display_history():
  for message in st.session_state.messages:
    send_message(message["role"], message["message"], save=False)


# retriever로 검색한 결과를 포맷팅하는 함수
def format_docs(docs):
  return "\n\n".join(document.page_content for document in docs)


# LLM에 전달할 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages(
  [
    (
      "system", 
      """
      Answer the question using ONLY the following context.
      If you don't know the answer, just say you don't know.
      DON'T make anything up.

      Context: {context}
      """
    ),
    ("human", "{question}"),
  ]
)


if file:
  retriever = embed_file(file)

  send_message("ai", "I'm ready! Ask away!", save=False)

  display_history()

  message = st.chat_input("Ask a question about your file...")

  if message:
    send_message("human", message)
    
    chain = (
      {
        # retriever로 검색한 결과를 format_docs 함수에 전달하여 포맷팅
        "context": retriever | RunnableLambda(format_docs), 
        # 질문은 그대로 LLM에 전달
        "question": RunnablePassthrough(), 
      }
      | prompt 
      | llm
    )
    
    # 체인 실행 결과(LLM의 답변)를 response에 저장
    with st.chat_message("ai"):
      response = chain.invoke(message)

else:
  # 파일이 없으면 메시지 기록 초기화
  st.session_state.messages = []
    

