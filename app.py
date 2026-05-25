from bs4 import BeautifulSoup
import os

os.environ.setdefault("USER_AGENT", "CloudflareSiteGPT/1.0")

import streamlit as st
from langchain_community.document_loaders import SitemapLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


REPOSITORY_URL = "https://github.com/dawwson/fullstack-gpt"
APP_CODE_URL = f"{REPOSITORY_URL}/blob/main/app.py"
SITEMAP_URL = "https://developers.cloudflare.com/sitemap-0.xml"
PRODUCT_FILTERS = [
  r"^https://developers\.cloudflare\.com/ai-gateway/.*",
  r"^https://developers\.cloudflare\.com/vectorize/.*",
  r"^https://developers\.cloudflare\.com/workers-ai/.*",
]
PRODUCT_NAMES = {
  "/ai-gateway/": "AI Gateway",
  "/vectorize/": "Vectorize",
  "/workers-ai/": "Workers AI",
}


st.set_page_config(
  page_title="Cloudflare SiteGPT",
  page_icon="☁️",
)

st.title("Cloudflare SiteGPT")
st.markdown(
"""
Ask questions about Cloudflare's official AI Gateway, Vectorize, and Workers AI documentation.
"""
)


def init_session_state():
  st.session_state.setdefault("messages", [])


def parse_page(soup: BeautifulSoup):
  # 반복되는 내비게이션과 푸터를 제거해서 검색 문맥에 본문이 더 많이 들어가게 한다.
  for tag in soup.find_all(["header", "footer", "nav", "script", "style"]):
    tag.decompose()

  main = soup.find("main")
  text_source = main if main else soup

  return (
    text_source.get_text(" ", strip=True)
    .replace("Copy as Markdown Copied!", "")
    .replace("↗", "")
  )


@st.cache_resource(show_spinner="Loading Cloudflare documentation...")
def load_cloudflare_docs(openai_api_key):
  splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
  )

  loader = SitemapLoader(
    SITEMAP_URL,
    filter_urls=PRODUCT_FILTERS,
    parsing_function=parse_page,
    continue_on_failure=True,
  )
  loader.requests_per_second = 2

  docs = loader.load_and_split(text_splitter=splitter)
  embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
  vector_store = FAISS.from_documents(docs, embeddings)

  return vector_store.as_retriever(search_kwargs={"k": 8})


def format_docs(docs):
  return "\n\n".join(
    f"Source: {doc.metadata.get('source', 'Unknown')}\n"
    f"Last modified: {doc.metadata.get('lastmod', 'Unknown')}\n"
    f"Content: {doc.page_content}"
    for doc in docs
  )


def get_product_name(source):
  for path, name in PRODUCT_NAMES.items():
    if path in source:
      return name

  return "공식"


def format_source(docs):
  for doc in docs:
    source = doc.metadata.get("source")

    if not source:
      continue

    product_name = get_product_name(source)
    return f"자세한 내용은 [Cloudflare {product_name} 문서]({source})를 참조하세요."

  return ""


def make_chain(openai_api_key):
  prompt = ChatPromptTemplate.from_messages([
    (
      "system",
      """
You answer questions about Cloudflare's official documentation for AI Gateway,
Cloudflare Vectorize, and Workers AI.

Use ONLY the context below. If the context does not contain the answer, say you
do not know. Do not make anything up.

Answer in the same language as the user's question when possible.
Do not include source URLs. The application will add source links separately.

Context:
{context}
""",
    ),
    ("human", "{question}"),
  ])

  llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini-2024-07-18",
    openai_api_key=openai_api_key,
  )

  return prompt | llm


init_session_state()

with st.sidebar:
  openai_api_key = st.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="sk-...",
  )

  st.divider()
  st.link_button("GitHub Repository", REPOSITORY_URL)
  st.link_button("Streamlit App Code", APP_CODE_URL)
  st.caption(f"Docs sitemap: {SITEMAP_URL}")

if not openai_api_key:
  st.info("Enter your OpenAI API key in the sidebar to start.")
  st.stop()

for message in st.session_state["messages"]:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

question = st.chat_input("Ask about AI Gateway, Vectorize, or Workers AI")

if question:
  st.session_state["messages"].append({
    "role": "user",
    "content": question,
  })

  with st.chat_message("user"):
    st.markdown(question)

  with st.chat_message("assistant"):
    try:
      retriever = load_cloudflare_docs(openai_api_key)
      docs = retriever.invoke(question)
      chain = make_chain(openai_api_key)
      result = chain.invoke({
        "context": format_docs(docs),
        "question": question,
      })
      answer = result.content
      source = format_source(docs)

      if source:
        answer = f"{answer}\n\n{source}"

      st.markdown(answer)
      st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
      })
    except Exception as e:
      error_message = f"문서를 불러오거나 답변을 생성하는 중 오류가 발생했습니다: {e}"
      st.error(error_message)
      st.session_state["messages"].append({
        "role": "assistant",
        "content": error_message,
      })
