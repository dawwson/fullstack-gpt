from pathlib import Path

from langchain_classic.memory import ConversationBufferMemory
import streamlit as st
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader


REPOSITORY_URL = "https://github.com/dawwson/fullstack-gpt"
APP_CODE_URL = f"{REPOSITORY_URL}/blob/main/pages/01_DocumentGPT_Challenge.py"
CACHE_DIR = Path(".cache")

MEMORY_KEY = "chat_history"
INPUT_KEY = "question"
OUTPUT_KEY = "answer"


st.set_page_config(page_title="Document GPT Challenge", page_icon="📄")

st.title("Document GPT Challenge")
st.markdown(
    """
    Upload a document and ask questions about it.
    """
)

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
    )
    file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx"])

    st.divider()
    st.link_button("GitHub Repository", REPOSITORY_URL)
    st.link_button("Streamlit App Code", APP_CODE_URL)

class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        self.message = ""
        self.message_box = st.empty()

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

    def on_llm_end(self, *args, **kwargs):
        save_message("ai", self.message)


# LLM 초기화
llm = ChatOpenAI(
  temperature=0.1,
  streaming=True,
  callbacks=[
    ChatCallbackHandler(),
  ]
)

# 메모리 초기화
memory = ConversationBufferMemory(
    memory_key=MEMORY_KEY,
    input_key=INPUT_KEY,
    output_key=OUTPUT_KEY,
    return_messages=True,
)


def save_message(role, message):
    st.session_state.messages.append({"role": role, "message": message})


def send_message(role, message, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(role, message)


def display_history():
    for message in st.session_state.messages:
        send_message(message["role"], message["message"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file, api_key):
  file_content = file.read()
  file_name = file.name
  
  file_path = str(CACHE_DIR / "files" / file_name)
  embeddings_cache_dir = str(CACHE_DIR / "embeddings" / file_name)

  with open(file_path, "wb") as f:
    f.write(file_content)

    # 1. 문서 로드 & 분할
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1200,
        chunk_overlap=300,
    )

    loader = UnstructuredLoader(file_path)
    
    docs = filter_complex_metadata(loader.load_and_split(text_splitter=splitter))

    # 2. 임베딩 생성 & 캐싱
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    local_cache_dir = LocalFileStore(embeddings_cache_dir)
    
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, local_cache_dir)

    # 3. 벡터 스토어 생성 및 retriever 변환
    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    
    return retriever


# 프롬프트 생성
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an AI assistant performing document-based question answering.
            Answer using ONLY the provided context.
            If the context does not contain the answer, say "I don't know."
            Keep answers short and clear.
            Use the chat history only when it helps understand the user's latest question.

            Context:
            {context}
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


# 파
if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if file and st.session_state.file_name != file.name:
    st.session_state.messages = []
    st.session_state.file_name = file.name

if not openai_api_key:
    st.info("Enter your OpenAI API key in the sidebar to start.")
elif not file:
    st.session_state.messages = []
    st.session_state.file_name = None
    st.info("Upload a file in the sidebar to start chatting.")
else:
    retriever = embed_file(file, openai_api_key)

    send_message("ai", "I'm ready! Ask away!", save=False)
    
    display_history()

    question = st.chat_input("Ask a question about your file...")

    if question:
        send_message("human", question)

        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "chat_history": RunnableLambda(load_memory),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

        with st.chat_message("ai"):
            chain.invoke(question)
