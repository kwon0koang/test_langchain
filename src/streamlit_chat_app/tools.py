from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from constants import MY_NEWS_INDEX, MY_PDF_INDEX
from embeddings import embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
import json
from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from llm import gemma2
from typing import List, Union
from langchain_community.tools import Tool
from langchain_core.documents.base import Document

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
다음과 같은 정보를 검색할 때에는 이 도구를 사용해야 한다:
- 엔비디아의 스타트업 인수 관련 내용
- 퍼플렉시티 관련 내용 (회사가치, 투자 등)
- 라마3 관련 내용
""",
)

retriever_tool2 = create_retriever_tool(
    retriever2,
    name=PDF_SEARCH_TOOL_NAME,
    description="""
다음과 같은 정보를 검색할 때에는 이 도구를 사용해야 한다:
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

# tools = [retriever_tool1, retriever_tool2, retriever_tool3]
tools = [retriever_tool1, retriever_tool2]

# 사이드바에 표시할 tool 목록 셋팅
options_in_sidebar = [
    (None, '선택 안함'),
    (SAVED_NEWS_SEARCH_TOOL_NAME, '저장된 뉴스 검색'),
    (PDF_SEARCH_TOOL_NAME, '저장된 PDF 검색'),
    # (WEB_SEARCH_TOOL_NAME, 'WEB 검색'),
    (TOOL_AUTO, '도구 자동 선택 (BETA)'),
]

# ==========================================================================================================================================================================================

# 적합한 tool 추출 위한 프롬프트
prompt_for_extract_actions = hub.pull("kwonempty/extract-actions-for-ollama")

def get_tools(query) -> str:
    """
    사용 가능한 도구들의 이름과 설명을 JSON 문자열 형식으로 변환하여 반환
    """
    # tools 리스트에서 각 도구의 이름, 설명을 딕셔너리 형태로 추출
    tool_info = [{"tool_name": tool.name, "tool_description": tool.description} for tool in tools]
    
    print(f"get_tools / tool_info: {tool_info}")
    
    # tool_info 리스트를 JSON 문자열 형식으로 변환하여 반환
    return json.dumps(tool_info, ensure_ascii=False)

chain_for_extract_actions = (
    {"tools": get_tools, "question": RunnablePassthrough()}
    | prompt_for_extract_actions 
    | gemma2
    | StrOutputParser()
    )

def get_retriever_by_tool_name(name: str) -> VectorStoreRetriever:
    """
    도구 이름을 통해 검색기 반환
    """
    for tool in tools:
        if tool.name == name:
            return tool.func.keywords['retriever']
    return None

# ==========================================================================================================================================================================================
    
def get_documents_from_actions(actions_json: str, tools: List[Tool]) -> List[Document]:
    """
    주어진 JSON 문자열을 파싱하여 해당 액션에 대응하는 검색기를 찾아서 
    액션을 실행 후 검색된 문서를 반환
    
    :param actions_json: 액션과 그 입력이 포함된 JSON 문자열
    :param tools: 사용 가능한 도구들의 리스트
    :return: 액션을 통해 검색된 문서들의 리스트
    """
    print(f"get_documents_from_actions / actions_json: {actions_json}")
    
    # JSON 문자열을 파싱
    try:
        actions = json.loads(actions_json)
    except json.JSONDecodeError:
        raise ValueError("유효하지 않은 JSON 문자열")

    # 파싱된 객체가 리스트인지 확인
    if not isinstance(actions, list):
        raise ValueError("제공된 JSON은 액션 리스트를 나타내야 함")

    documents = []

    # 각 액션을 처리
    for action in actions:
        if not isinstance(action, dict) or 'action' not in action or 'action_input' not in action:
            continue  # 유효하지 않은 액션은 건너뜀

        tool_name = action['action']
        action_input = action['action_input']
        print(f"get_documents_from_actions / tool_name: {tool_name} / action_input: {action_input}")
        
        # if tool_name == "None": # 사용할 도구 없음. 바로 빈 document 리턴
        #     print(f"get_documents_from_actions / 사용할 도구 없음. 바로 빈 document 리턴")
        #     return []
        # 사용할 도구 없으면 오류 발생시켜서 streaming chain 사용하게끔 할 것
        if tool_name == "None": # 사용할 도구 없음. 바로 빈 document 리턴
            raise ValueError("사용할 도구 없음")
        
        retriever = get_retriever_by_tool_name(tool_name)
        
        if retriever:
            # 액션 입력으로 검색기 실행
            retrieved_docs = retriever.invoke(action_input)
            documents.extend(retrieved_docs)
        
    print(f"get_documents_from_actions / len(documents): {len(documents)}")
    return documents
