
from bs4 import BeautifulSoup
from langchain_community.document_loaders import SitemapLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

with st.sidebar:
  url = st.text_input(
    "Write down a URL",
    placeholder="https://www.example.com"
  )


llm = ChatOpenAI(
  temperature=0.1
)


answers_prompt = ChatPromptTemplate.from_template(
"""
Using ONLY the following context answer the user's question.
If you can't just say you don't know, don't make anything up.

Then, give a score to the answer between 0 and 5.
The score should be high if the answer is related to the user's question, and low otherwise.
If there is no relevant content, the score is 0.
Always provide scores with your answers

Make sure to include the answer's score even if it's 0.

Context: {context}

Examples:

Question: How far away is the moon?
Answer: The moon is 384,400 km away.
Score: 5

Question: How far away is the sun?
Answer: I don't know
Score: 0

Your turn!

Question: {question}
"""
)


choose_prompt = ChatPromptTemplate.from_messages([
  (
    "system",
    """
    Use ONLY the following pre-existing answers to answer the user's question.

    Use the answers that have the highest score (more helpful) and favor the most recent ones.

    Site sources and return the sources of the answers as they are, do not change them.

    Answers: {answers}
    """,
  ),
  ("human", "{question}"),
])


# header, footer 제거한 나머지 text 반환
def parse_page(soup: BeautifulSoup):
  header = soup.find("header")
  footer = soup.find("footer")
  
  if header:
    header.decompose()
  if footer:
    footer.decompose()
  
  return (
    str(soup.get_text())
    .replace("\n", " ")
    .replace("↗", "")
  )


@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
  splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
  )

  loader = SitemapLoader(
      url,
      # TODO: url 수정 필요
      filter_urls=[
        r"^(.*\/ai-gateway\/).*", # /ai-gateway/를 포함하는 URL만 필터링
      ],
      parsing_function=parse_page,
      # FIXME: rate limit 문제로 추가하였음. 모든 페이지를 불러오지 못하므로 다른 해결방안 모색이 필요함
      blocksize=10, # 10개씩 나눠서 긁어옴
      blocknum=0,
      continue_on_failure=True,
    )
    
  loader.requests_per_second = 1
    
  docs = loader.load_and_split(text_splitter=splitter)

  embeddings = OpenAIEmbeddings()

  vector_store = FAISS.from_documents(docs, embeddings)

  retriever = vector_store.as_retriever()

  return retriever

def get_answers(inputs):
  docs = inputs["docs"]
  question = inputs["question"]

  answers_chain = answers_prompt | llm

  return {
    "question": question,
    "answers": [
      {
        "answer": answers_chain.invoke(
            {
              "question": question,
              "context": doc.page_content
            }
          ).content,
        "source": doc.metadata["source"],
        "date": doc.metadata["lastmod"],
      } for doc in docs
    ]
  }


def choose_answer(inputs):
  answers = inputs["answers"]
  question = inputs["question"]

  choose_chain = choose_prompt | llm
  
  return choose_chain.invoke({
    "question": question,
    "answers": "\n\n".join(
      f"{answer['answer']}\n" + 
      f"Source:{answer['source']}\n" +
      f"Date:{answer['date']}\n" 
      for answer in answers)
  })


if url:
  
  if ".xml" not in url:
    with st.sidebar:
      st.error("Please write down a Sitemap URL.")
  
  else:
    retriever = load_website(url)

    query = st.text_input("Ask a question to the website.")

    if query:
      chain = (
        {
          "docs": retriever, 
          "question": RunnablePassthrough()
        }
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
      )
      
      result = chain.invoke(query)
      st.write(result.content)
    
