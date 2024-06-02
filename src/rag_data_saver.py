from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
from langchain_community.document_loaders import WebBaseLoader
from constants import MY_FAISS_INDEX

# BeautifulSoup : HTML 및 XML 문서를 파싱하고 구문 분석하는 데 사용되는 파이썬 라이브러리. 주로 웹 스크레이핑(웹 페이지에서 데이터 추출) 작업에서 사용되며, 웹 페이지의 구조를 이해하고 필요한 정보를 추출하는 데 유용
loader = WebBaseLoader(
    web_paths=("https://www.aitimes.com/news/articleView.html?idxno=159102"
               , "https://www.aitimes.com/news/articleView.html?idxno=159072"
               , "https://www.aitimes.com/news/articleView.html?idxno=158943"
               ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "article", # 태그
            attrs={"id": ["article-view-content-div"]}, # 태그의 ID 값들
        )
    ),
)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(data)
print(f'len(splits[0].page_content) : {len(splits[1].page_content)}')
print(f'splits : {splits}')

vectorstore = FAISS.from_documents(splits,
                                   embedding = OllamaEmbeddings(model="EEVE-Korean-Instruct-10.8B-v1.0:latest"),
                                   distance_strategy = DistanceStrategy.COSINE # 코사인 유사도 측정. 값이 클수록 더 유사함을 의미
                                  )

# 로컬에 DB 저장
vectorstore.save_local(MY_FAISS_INDEX)