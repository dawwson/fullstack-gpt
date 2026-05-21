from pathlib import Path

import streamlit as st
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from openai import AuthenticationError, OpenAIError


REPOSITORY_URL = "https://github.com/dawwson/fullstack-gpt"
APP_CODE_URL = f"{REPOSITORY_URL}/blob/main/app.py"
CACHE_DIR = Path(".cache")

# Streamlit 페이지의 기본 메타데이터와 첫 화면 안내 문구를 설정한다.
st.set_page_config(page_title="Document GPT Challenge", page_icon="📄")

st.title("Document GPT Challenge")
st.markdown(
    """
    Upload a document and ask questions about it.
    """
)

# 사이드바에는 사용자 API 키 입력, 문서 업로드, 과제 코드 링크를 배치한다.
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
    # OpenAI 응답을 토큰 단위로 받아 Streamlit 화면에 실시간으로 표시한다.
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


def init_session_state():
    # Streamlit은 위젯 입력마다 스크립트를 다시 실행한다.
    # 같은 세션에서 유지해야 할 값은 session_state에 명시적으로 둔다.
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("file_name", None)


def save_message(role, message):
    # 화면 재렌더링과 다음 질문의 chat_history 구성을 위해 메시지를 세션에 저장한다.
    init_session_state()
    messages = st.session_state.get("messages", [])
    messages.append({"role": role, "message": message})
    st.session_state["messages"] = messages


def send_message(role, message, save=True):
    # 채팅 말풍선을 그리고, 필요한 경우 같은 내용을 세션 기록에도 남긴다.
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(role, message)


def display_history():
    # rerun 이후에도 같은 세션의 이전 대화를 화면에 다시 그린다.
    init_session_state()
    for message in st.session_state.get("messages", []):
        send_message(message["role"], message["message"], save=False)


def format_docs(docs):
    # 검색된 문서 조각들을 LLM 프롬프트에 넣기 쉬운 하나의 문자열로 합친다.
    return "\n\n".join(document.page_content for document in docs)


def get_chat_history(messages):
    # 저장된 채팅 기록을 LangChain 프롬프트가 이해할 수 있는 메시지 객체 목록(dict)으로 변환한다.
    chat_history = []
    for message in messages:
        if message["role"] == "human":
            chat_history.append(HumanMessage(content=message["message"]))
        elif message["role"] == "ai":
            chat_history.append(AIMessage(content=message["message"]))
    return chat_history


def show_invalid_api_key_error():
    # 인증 실패는 traceback 대신 사용자가 바로 이해할 수 있는 안내 UI로 표시한다.
    st.error(
        "OpenAI API key가 유효하지 않습니다. "
        "사이드바에 올바른 API key를 다시 입력해 주세요."
    )


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file_name, file_content, api_key):
    # 업로드 파일과 임베딩 결과를 로컬에 캐싱해 같은 파일의 반복 처리 비용을 줄인다.
    files_dir = CACHE_DIR / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    file_path = files_dir / file_name
    embeddings_cache_dir = CACHE_DIR / "embeddings" / file_name

    file_path.write_bytes(file_content)

    # 1. 문서를 로드하고 검색에 적합한 크기의 chunk로 분할한다.
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1200,
        chunk_overlap=300,
    )

    loader = UnstructuredLoader(str(file_path))

    docs = filter_complex_metadata(loader.load_and_split(text_splitter=splitter))

    # 2. 사용자 API 키로 임베딩을 만들고, 같은 chunk는 파일 캐시에 재사용한다.
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    local_cache_dir = LocalFileStore(str(embeddings_cache_dir))

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        local_cache_dir,
    )

    # 3. FAISS 벡터 스토어를 만들고 질문과 유사한 chunk를 찾는 retriever로 변환한다.
    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    return retriever


# 검색 결과와 대화 기록을 함께 받아 답변하도록 프롬프트를 구성한다.
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


init_session_state()

# 파일이 바뀌면 이전 문서에 대한 대화 기록은 더 이상 유효하지 않으므로 초기화한다.
if file and st.session_state.get("file_name") != file.name:
    st.session_state["messages"] = []
    st.session_state["file_name"] = file.name

# API 키와 파일이 모두 준비된 뒤에만 RAG 파이프라인과 LLM 호출을 실행한다.
if not openai_api_key:
    st.info("Enter your OpenAI API key in the sidebar to start.")
elif not file:
    st.session_state["messages"] = []
    st.session_state["file_name"] = None
    st.info("Upload a file in the sidebar to start chatting.")
else:
    # 파일을 벡터 검색기로 변환하고, 사용자 API 키로 스트리밍 LLM을 초기화한다.
    try:
        retriever = embed_file(file.name, file.getvalue(), openai_api_key)
    except AuthenticationError:
        show_invalid_api_key_error()
        st.stop()
    except OpenAIError as error:
        st.error(f"OpenAI API 요청 중 오류가 발생했습니다: {error}")
        st.stop()

    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
        openai_api_key=openai_api_key,
    )

    send_message("ai", "I'm ready! Ask away!", save=False)
    
    display_history()

    question = st.chat_input("Ask a question about your file...")

    if question:
        send_message("human", question)
        # 현재 질문은 별도 입력으로 들어가므로, history에는 그 이전 대화만 포함한다.
        chat_history = get_chat_history(st.session_state.get("messages", [])[:-1])

        # 질문 -> 관련 문서 검색 -> 프롬프트 구성 -> LLM 호출 순서로 RAG 체인을 실행한다.
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "chat_history": RunnableLambda(lambda _: chat_history),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

        with st.chat_message("ai"):
            try:
                chain.invoke(question)
            except AuthenticationError:
                show_invalid_api_key_error()
            except OpenAIError as error:
                st.error(f"OpenAI API 요청 중 오류가 발생했습니다: {error}")
