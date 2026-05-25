
from bs4 import BeautifulSoup
from langchain_community.document_loaders import SitemapLoader
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



@st.cache_data(show_spinner="Loading website...")
def load_website(url):
  splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
  )

  loader = SitemapLoader(
      url,
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

  return docs


if url:
  
  if ".xml" not in url:
    with st.sidebar:
      st.error("Please write down a Sitemap URL.")
  
  else:
    docs = load_website(url)
    st.write(docs)
