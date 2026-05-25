
from langchain_community.document_loaders import SitemapLoader
from langchain_community.document_transformers import Html2TextTransformer
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

html2text_transformer = Html2TextTransformer()

with st.sidebar:
  url = st.text_input(
    "Write down a URL",
    placeholder="https://www.example.com"
  )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
  loader = SitemapLoader(
      url,
      # FIXME: rate limit 문제로 추가하였음. 모든 페이지를 불러오지 못하므로 다른 해결방안 모색이 필요함
      blocksize=10, # 10개씩 나눠서 긁어옴
      blocknum=0,
      continue_on_failure=True,
    )
    
  loader.requests_per_second = 1
    
  return loader.load()


if url:
  
  if ".xml" not in url:
    with st.sidebar:
      st.error("Please write down a Sitemap URL.")
  
  else:
    docs = load_website(url)
    st.write(docs)
