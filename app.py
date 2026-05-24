import json
from pathlib import Path

import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader


REPOSITORY_URL = "https://github.com/dawwson/fullstack-gpt"
APP_CODE_URL = f"{REPOSITORY_URL}/blob/main/pages/03_QuizGPT.py"
CACHE_DIR = Path(".cache/quiz_files")

# OpenAI 함수 호출에 사용할 스키마다.
# 모델은 일반 텍스트가 아니라 이 구조에 맞춘 arguments로 퀴즈를 반환한다.
QUIZ_FUNCTION = {
  "name": "create_quiz",
  "description": "Create a multiple-choice quiz from the provided context.",
  "parameters": {
    "type": "object",
    "properties": {
      "questions": {
        "type": "array",
        "description": "The quiz questions.",
        "items": {
          "type": "object",
          "properties": {
            "question": {
              "type": "string",
              "description": "The question text.",
            },
            "answers": {
              "type": "array",
              "description": "Exactly four answer choices. One answer must be correct.",
              "items": {
                "type": "object",
                "properties": {
                  "answer": {
                    "type": "string",
                    "description": "The answer text.",
                  },
                  "correct": {
                    "type": "boolean",
                    "description": "Whether this answer is the correct one.",
                  },
                },
                "required": ["answer", "correct"],
              },
            },
          },
          "required": ["question", "answers"],
        },
      },
    },
    "required": ["questions"],
  },
}


st.set_page_config(
  page_title="Quiz GPT",
  page_icon="❓",
)

st.title("Quiz GPT")


def init_session_state():
  # Streamlit은 위젯 이벤트마다 스크립트를 다시 실행하므로,
  # 생성된 퀴즈와 채점 상태는 session_state에 보관한다.
  st.session_state.setdefault("quiz", None)
  st.session_state.setdefault("quiz_source", None)
  st.session_state.setdefault("score", None)
  st.session_state.setdefault("attempt", 0)


def reset_quiz():
  # 파일, 위키피디아 주제, 난이도가 바뀌면 이전 퀴즈는 더 이상 유효하지 않다.
  st.session_state["quiz"] = None
  st.session_state["quiz_source"] = None
  st.session_state["score"] = None
  st.session_state["attempt"] = 0


def format_docs(docs):
  # LangChain Document 목록을 프롬프트에 넣기 쉬운 하나의 문자열로 합친다.
  return "\n\n".join(document.page_content for document in docs)


@st.cache_resource(show_spinner="Loading file...")
def split_file(file):
  # 업로드된 파일을 캐시 폴더에 저장한 뒤, LLM 입력에 적당한 크기로 분할한다.
  CACHE_DIR.mkdir(parents=True, exist_ok=True)

  file_content = file.read()
  file_path = CACHE_DIR / file.name

  with open(file_path, "wb") as f:
    f.write(file_content)

  splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
  )

  loader = UnstructuredLoader(str(file_path))
  docs = loader.load_and_split(text_splitter=splitter)

  return docs


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
  # 선택한 주제와 관련된 한국어 위키피디아 문서를 가져온다.
  retriever = WikipediaRetriever(top_k_results=5, lang="ko")
  docs = retriever.invoke(term)
  return docs


def make_llm(openai_api_key):
  # create_quiz 함수 호출을 강제해서 퀴즈 결과를 안정적인 JSON 인자로 받는다.
  return ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini-2024-07-18",
    openai_api_key=openai_api_key,
  ).bind(
    functions=[QUIZ_FUNCTION],
    function_call={"name": "create_quiz"},
  )


def generate_quiz(docs, difficulty, openai_api_key):
  # 문서 내용과 난이도를 프롬프트에 넣고, 함수 호출 결과에서 퀴즈 데이터를 꺼낸다.
  prompt = ChatPromptTemplate.from_messages([
    (
      "system",
      """
You are a helpful teacher.

Based ONLY on the following context, create 10 multiple-choice questions.
The selected difficulty is: {difficulty}.

Rules:
- Each question must have exactly 4 answers.
- Exactly one answer per question must be correct.
- Make the questions easy when the difficulty is Easy.
- Make the questions balanced when the difficulty is Medium.
- Make the questions require careful reading and inference when the difficulty is Hard.
- Return the quiz by calling the create_quiz function.

Context: {context}
""",
    )
  ])

  chain = prompt | make_llm(openai_api_key)
  response = chain.invoke({
    "context": format_docs(docs),
    "difficulty": difficulty,
  })

  function_call = response.additional_kwargs.get("function_call", {})
  arguments = function_call.get("arguments")

  if not arguments:
    raise ValueError("The model did not return quiz data through function calling.")

  # function_call.arguments는 문자열이므로 Python dict로 변환한다.
  return json.loads(arguments)


def display_quiz(quiz):
  # 생성된 문제를 radio 버튼으로 표시하고, 제출 시 사용자의 선택을 채점한다.
  with st.form(f"questions_form_{st.session_state['attempt']}"):
    selections = []

    for index, question in enumerate(quiz["questions"]):
      st.write(f"Q{index + 1}. {question['question']}")

      selected = st.radio(
        "Select an option",
        [answer["answer"] for answer in question["answers"]],
        index=None,
        key=f"question_{st.session_state['attempt']}_{index}",
      )
      selections.append(selected)

    submitted = st.form_submit_button("Submit Exam")

  if submitted:
    score = 0

    for selected, question in zip(selections, quiz["questions"]):
      correct_answer = next(
        answer["answer"] for answer in question["answers"] if answer["correct"]
      )

      if selected == correct_answer:
        score += 1

    st.session_state["score"] = score


def display_score(quiz):
  # 만점이면 축하 효과를 보여주고, 만점이 아니면 같은 퀴즈를 다시 풀 수 있게 한다.
  score = st.session_state["score"]

  if score is None:
    return

  total = len(quiz["questions"])
  st.subheader(f"Score: {score} / {total}")

  if score == total:
    st.success("Perfect score!")
    st.balloons()
  else:
    st.error("Not a perfect score. Try again.")

    if st.button("Retake Exam"):
      st.session_state["score"] = None
      st.session_state["attempt"] += 1
      st.rerun()


init_session_state()

# 사이드바는 API 키, 난이도, 퀴즈 생성에 사용할 자료 선택을 담당한다.
with st.sidebar:
  docs = None
  # source_id는 현재 선택된 자료와 난이도를 식별하는 값이다.
  # 캐시/session_state에 남아 있는 이전 퀴즈가 현재 입력과 맞는지 비교할 때 사용한다.
  source_id = None

  openai_api_key = st.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="sk-...",
  )

  difficulty = st.selectbox(
    "Exam difficulty",
    (
      "Easy",
      "Medium",
      "Hard",
    ),
  )

  choice = st.selectbox(
    "Choose what you want to use",
    (
      "File",
      "Wikipedia Article",
    ),
  )

  if choice == "File":
    file = st.file_uploader(
      "Upload a .docx, .txt, or .pdf file",
      type=["docx", "pdf", "txt"],
    )

    if file:
      docs = split_file(file)
      source_id = f"file:{file.name}:{difficulty}"

  else:
    topic = st.text_input("Name of the article")

    if topic:
      docs = wiki_search(topic)
      source_id = f"wiki:{topic}:{difficulty}"

  st.divider()
  st.link_button("GitHub Repository", REPOSITORY_URL)
  st.link_button("Streamlit App Code", APP_CODE_URL)

if not openai_api_key:
  st.info("Enter your OpenAI API key in the sidebar to start.")

elif not docs:
  st.markdown(
    """
    Welcome to QuizGPT.

    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
  )

else:
  # source_id가 달라졌다면 사용자가 다른 자료나 난이도를 선택한 것이므로,
  # session_state에 남아 있던 이전 퀴즈와 점수를 초기화한다.
  if source_id != st.session_state["quiz_source"]:
    reset_quiz()

  st.markdown(
    "QuizGPT creates a 10-question multiple-choice exam from your selected file "
    "or Wikipedia article."
  )

  if st.button("Generate Quiz"):
    try:
      # LLM 호출은 시간이 걸릴 수 있으므로 사용자에게 진행 상태를 표시한다.
      with st.status("Generating quiz...", expanded=True) as status:
        st.write("Reading the selected content and asking the model to create questions.")
        quiz = generate_quiz(docs, difficulty, openai_api_key)
        status.update(label="Quiz generated.", state="complete", expanded=False)

      st.session_state["quiz"] = quiz
      st.session_state["quiz_source"] = source_id
      st.session_state["score"] = None
      st.session_state["attempt"] = 0
    except Exception as e:
      st.exception(e)

  if st.session_state["quiz"]:
    display_quiz(st.session_state["quiz"])
    display_score(st.session_state["quiz"])
