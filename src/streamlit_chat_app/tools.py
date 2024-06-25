from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from constants import MY_NEWS_INDEX, MY_PDF_INDEX
from embeddings import embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.tools.tavily_search import TavilySearchResults

TOOL_AUTO = "auto"
SAVED_NEWS_SEARCH_TOOL_NAME = "saved_news_search"
PDF_SEARCH_TOOL_NAME = "pdf_search"
WEB_SEARCH_TOOL_NAME = "web_search"

# 로컬 DB 불러오기
vectorstore1 = FAISS.load_local(MY_NEWS_INDEX, 
                               embeddings, 
                               allow_dangerous_deserialization=True)
retriever1 = vectorstore1.as_retriever(search_type="similarity", search_kwargs={"k": 3}) # 유사도 높은 3문장 추출
vectorstore2 = FAISS.load_local(MY_PDF_INDEX, 
                               embeddings, 
                               allow_dangerous_deserialization=True)
retriever2 = vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k": 3}) # 유사도 높은 3문장 추출

retriever_tool1 = create_retriever_tool(
    retriever1,
    name=SAVED_NEWS_SEARCH_TOOL_NAME,
    description="""
아래와 같은 정보를 검색할 때에는 이 도구를 사용해야 한다
- 엔비디아의 스타트업 인수 관련 내용
- 퍼플렉시티 관련 내용 (회사가치, 투자 등)
- 라마3 관련 내용
""",
)

retriever_tool2 = create_retriever_tool(
    retriever2,
    name=PDF_SEARCH_TOOL_NAME,
    description="""
다음과 같은 정보를 검색할 때에는 이 도구를 사용해야 한다
- 생성형 AI 신기술 도입에 따른 선거 규제 연구 관련 내용
- 생성 AI 규제 연구 관련 내용
- 생성 AI 연구 관련 내용
"""
)

tavily_search = TavilySearchResults(k=3)

retriever_tool3 = create_retriever_tool(
    tavily_search,
    name=WEB_SEARCH_TOOL_NAME,
    description="""
웹에서 검색하고자 할 때, 이 도구를 사용해야 한다
"""
)

tools = [retriever_tool1, retriever_tool2, retriever_tool3]