from bs4 import BeautifulSoup
import os
import re

# SitemapLoader가 Cloudflare 문서를 요청할 때 사용할 식별자다.
# 이미 외부에서 USER_AGENT를 설정했다면 그 값을 우선한다.
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

# 1단계 프롬프트: 검색된 문서 조각마다 독립적인 후보 답변을 만들고 관련도 점수를 매긴다.
# 이후 choose_prompt가 여러 후보 중 가장 좋은 답변을 고르는 데 이 점수를 사용한다.
answers_prompt = ChatPromptTemplate.from_template(
  """
Using ONLY the following context, answer the user's question.
If the context does not contain the answer, say you don't know.
Do not make anything up.
Do not include source URLs.

Then, give a score to the answer between 0 and 5.
The score should be high if the answer is related to the user's question, and low otherwise.
If there is no relevant content, the score is 0.
Always provide scores with your answers.

Context: {context}

Examples:

Question: How far away is the moon?
Answer: The moon is 384,400 km away.
Score: 5

Question: How far away is the sun?
Answer: I don't know.
Score: 0

Your turn!

Question: {question}
"""
)

# 2단계 프롬프트: 문서별 후보 답변 중 최종 답변 하나를 선택한다.
# URL은 화면에 직접 노출하지 않기 위해 Source ID만 반환하게 하고, 링크 문장은 코드에서 붙인다.
choose_prompt = ChatPromptTemplate.from_messages([
  (
    "system",
    """
Use ONLY the following pre-existing answers to answer the user's question.

Use the answer with the highest score. If multiple answers have the same score,
prefer the most specific and recent one.

Do not include source URLs in the answer.
Return exactly this format:

Answer: <final answer>
Source ID: <the numeric Source ID for the answer you used>

Pre-existing answers:
{answers}
""",
  ),
  ("human", "{question}"),
])


def init_session_state():
  # Streamlit은 입력마다 스크립트를 다시 실행하므로 대화 기록을 session_state에 보관한다.
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
  # 긴 문서를 겹치는 조각으로 나누면 문맥 경계에서 중요한 내용이 잘리는 일을 줄일 수 있다.
  splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
  )

  # Cloudflare 전체 사이트맵 중 과제 대상 제품 문서만 로드한다.
  loader = SitemapLoader(
    SITEMAP_URL,
    filter_urls=PRODUCT_FILTERS,
    parsing_function=parse_page,
    continue_on_failure=True,
  )
  loader.requests_per_second = 2

  docs = loader.load_and_split(text_splitter=splitter)

  # 사용자 API 키로 임베딩을 만들고, FAISS retriever를 캐시해 같은 세션에서 재사용한다.
  embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
  vector_store = FAISS.from_documents(docs, embeddings)

  return vector_store.as_retriever(search_kwargs={"k": 8})


def get_product_name(source):
  # 출처 URL 경로를 사람이 읽기 좋은 제품명으로 바꿔 링크 문장에 사용한다.
  for path, name in PRODUCT_NAMES.items():
    if path in source:
      return name

  return "공식"


def format_source(source):
  # 최종 답변에는 선택된 출처 하나만 Markdown 링크로 표시한다.
  if source:
    product_name = get_product_name(source)
    return f"자세한 내용은 [Cloudflare {product_name} 문서]({source})를 참조하세요."

  return ""


def make_llm(openai_api_key):
  # temperature를 낮춰 문서 기반 답변과 Source ID 형식이 흔들리지 않게 한다.
  return ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini-2024-07-18",
    openai_api_key=openai_api_key,
  )


def get_answers(docs, question, llm):
  answers_chain = answers_prompt | llm
  answers = []

  # 검색된 각 문서 조각을 별도로 평가한다. 이렇게 하면 관련 없는 조각은 낮은 점수를 받는다.
  for index, doc in enumerate(docs, start=1):
    answer = answers_chain.invoke({
      "question": question,
      "context": doc.page_content,
    }).content

    answers.append({
      "id": index,
      "answer": answer,
      "source": doc.metadata.get("source"),
      "date": doc.metadata.get("lastmod", "Unknown"),
    })

  return answers


def parse_chosen_answer(content, answers):
  # choose_chain의 구조화된 텍스트 응답에서 답변 본문과 Source ID를 분리한다.
  answer_match = re.search(
    r"Answer:\s*(.*?)(?:\n\s*Source ID:|$)",
    content,
    flags=re.DOTALL | re.IGNORECASE,
  )
  source_match = re.search(r"Source ID:\s*(\d+)", content, flags=re.IGNORECASE)

  answer = answer_match.group(1).strip() if answer_match else content.strip()
  source_id = int(source_match.group(1)) if source_match else None
  source = None

  if source_id:
    # Source ID를 실제 문서 URL로 매핑해서 화면에 표시할 링크를 결정한다.
    source = next(
      (item["source"] for item in answers if item["id"] == source_id),
      None,
    )

  if not source and answers:
    # 모델이 Source ID를 빠뜨린 경우에도 답변 끝에 출처 하나는 붙일 수 있게 한다.
    source = answers[0]["source"]

  return answer, source


def choose_answer(answers, question, llm):
  choose_chain = choose_prompt | llm

  # 후보 답변에는 Source URL을 포함하지만, 최종 출력에서는 Source ID만 사용하게 한다.
  result = choose_chain.invoke({
    "question": question,
    "answers": "\n\n".join(
      f"Source ID: {answer['id']}\n"
      f"Answer candidate:\n{answer['answer']}\n"
      f"Source URL: {answer['source']}\n"
      f"Last modified: {answer['date']}"
      for answer in answers
    ),
  })

  return parse_chosen_answer(result.content, answers)


init_session_state()

with st.sidebar:
  # OpenAI 호출 비용은 사용자가 입력한 API 키로 처리한다.
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
  # 사용자 메시지를 먼저 저장하고 화면에 표시한 뒤 답변 생성을 시작한다.
  st.session_state["messages"].append({
    "role": "user",
    "content": question,
  })

  with st.chat_message("user"):
    st.markdown(question)

  with st.chat_message("assistant"):
    try:
      status_placeholder = st.empty()

      with status_placeholder.status("답변 생성 중...", expanded=True) as status:
        # RAG 흐름: 검색 -> 후보 답변 생성/점수화 -> 최종 답변 선택 -> 출처 링크 1개 추가.
        status.update(label="Cloudflare 문서를 불러오는 중...")
        retriever = load_cloudflare_docs(openai_api_key)

        status.update(label="관련 문서를 검색하는 중...")
        docs = retriever.invoke(question)

        status.update(label="답변 후보를 생성하는 중...")
        llm = make_llm(openai_api_key)
        answers = get_answers(docs, question, llm)

        status.update(label="최종 답변을 정리하는 중...")
        answer, source = choose_answer(answers, question, llm)

        status.update(label="답변 생성 완료", state="complete")

      status_placeholder.empty()
      source_link = format_source(source)

      if source_link:
        answer = f"{answer}\n\n{source_link}"

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
