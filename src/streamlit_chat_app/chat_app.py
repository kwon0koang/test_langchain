import os
import sys

# 현재 파일 경로
current_dir = os.path.dirname(os.path.abspath(__file__))
# src 디렉토리 경로
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# src 디렉토리를 sys.path에 추가
sys.path.append(parent_dir)

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
import json
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from streamlit.runtime.state import SessionStateProxy
from langchain_core.vectorstores import VectorStoreRetriever
import time
from langchain_core.runnables import RunnablePassthrough
from langchain.tools.retriever import create_retriever_tool
from langchain_core.pydantic_v1 import BaseModel, Field
from constants import MY_NEWS_INDEX, MY_PDF_INDEX
from embeddings import embeddings

TOOL_AUTO = "auto"
SAVED_NEWS_SEARCH_TOOL_NAME = "saved_news_search"
PDF_SEARCH_TOOL_NAME = "pdf_search"

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        # print(f"kyk / token: [{token}] / text: {self.text}")
        self.container.markdown(self.text)
        # self.container.markdown(self.text, unsafe_allow_html=True)
        
st.title("권봇 🤖")

llm = ChatOllama(model="EEVE-Korean-Instruct-10.8B-v1.0:latest", temperature=0)
llama_llm = ChatOllama(model="llama3:8b", temperature=0)
# qwen2_llm = ChatOllama(model="qwen2:latest", temperature=0)

query = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, professional assistant named 권봇. answer me in Korean no matter what"),
    MessagesPlaceholder(variable_name="messages"),
])

chain = query | llm

# ==========================================================================================================================================================================================

# 사이드바에 tool 목록 셋팅

options = [
    (None, '선택 안함'),
    (SAVED_NEWS_SEARCH_TOOL_NAME, '저장된 뉴스 검색'),
    (PDF_SEARCH_TOOL_NAME, '저장된 PDF 검색'),
    (TOOL_AUTO, '도구 자동 선택 (BETA)'),
]

option_names, option_display_names = zip(*options)  # 옵션 코드를 추출
selected_option_display_name = st.sidebar.selectbox('도구 🛠️', option_display_names)
selected_option_name = next(name for name, display_name in options if display_name == selected_option_display_name)

# ==========================================================================================================================================================================================

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
    name="saved_news_search",
    description="""
아래와 같은 정보를 검색할 때에는 이 도구를 사용해야 한다
- 엔비디아의 스타트업 인수
- 퍼플렉시티 관련 내용 (회사가치, 투자 등)
- 라마3 관련 내용
""",
)

retriever_tool2 = create_retriever_tool(
    retriever2,
    name="pdf_search",
    description="""
다음과 같은 정보를 검색할 때에는 이 도구를 사용해야 한다
- 생성형 AI 신기술 도입에 따른 선거 규제 연구
- 생성 AI 규제 연구
- 생성 AI 연구
"""
)

tools = [retriever_tool1, retriever_tool2]

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """
너는 유능한 업무 보조자야.
context를 사용해서 question에 대한 답을 말해줘.
정답을 모르면 모른다고만 해.
""")
    , MessagesPlaceholder(variable_name="messages"),
    ("human", "{user_input}")
])

retrieved_docs = []
def get_page_contents_with_metadata(docs) -> str: 
    global retrieved_docs
    retrieved_docs = docs
    
    result = ""
    for i, doc in enumerate(docs):
        if i > 0:
            result += "\n\n"
        result += f"## 본문: {doc.page_content}\n### 출처: {doc.metadata['source']}"
    return result

# tool명으로 retriever 찾기
def get_retriever_by_tool_name(name) -> VectorStoreRetriever:
    print(f"get_retriever_by_tool_name / name: {name}")
    for tool in tools:
        if tool.name == name:
            # print(tool.func) # functools.partial(<function _get_relevant_documents at 0x1487dd6c0>, retriever=VectorStoreRetriever(tags=['FAISS', 'HuggingFaceEmbeddings'], vectorstore=<langchain_community.vectorstores.faiss.FAISS object at 0x317e52ea0>, search_kwargs={'k': 5}), document_prompt=PromptTemplate(input_variables=['page_content'], template='{page_content}'), document_separator='\n\n')
            return tool.func.keywords['retriever']
    return None

# 문서 검색 후 새 메시지 리스트 가져오기
def get_new_messages_after_doc_retrieval(messages_dict) -> dict:
    print("========================")
    print(f"messages_dict: {messages_dict}") # {'messages': [HumanMessage(content='라마3 성능은?')]}
    messages = messages_dict["messages"]
    print(f"messages: {messages}")
    last_human_message = messages[-1].content
    print(f"last_human_message: {last_human_message}")
    
    selected_tool = ""
    if TOOL_AUTO == selected_option_name:
        selected_tool = chain_for_select_tool.invoke(last_human_message) # LLM 한테 tool 선택하게 하기
        print(f"chain_for_select_tool.invoke 결과 / selected_tool: {selected_tool}")
        # 후보정 Start ============================
        # 가끔 "웹 검색(saved_news_search)" 요런 형식으로 말함 ㅜ
        if PDF_SEARCH_TOOL_NAME in selected_tool:
            selected_tool = PDF_SEARCH_TOOL_NAME
        elif SAVED_NEWS_SEARCH_TOOL_NAME in selected_tool:
            selected_tool = SAVED_NEWS_SEARCH_TOOL_NAME
        else:
            selected_tool = ""
        # 후보정 End ============================
    else:
        selected_tool = selected_option_name
    retriever = get_retriever_by_tool_name(selected_tool)
    
    if retriever is None:
        raise ValueError(f"{selected_tool} retriever가 없어요")
    
    global retrieved_docs
    retrieved_docs = retriever.invoke(last_human_message)
    
    new_human_message = HumanMessage(content=f"""
<question>
{last_human_message}
</question>

<context>
{get_page_contents_with_metadata(retrieved_docs)}
</context>

# answer :
""")
    
    messages_without_last = messages[:-1]
    return {"messages": messages_without_last, "user_input": new_human_message}

# 출처 가져오기
def get_metadata_sources(docs) -> str: 
    sources = set()
    for doc in docs:
        source = doc.metadata['source']
        is_pdf = source.endswith('.pdf')
        if (is_pdf):
            file_path = doc.metadata['source']
            file_name = os.path.basename(file_path)
            source = f"[{file_name} ({int(doc.metadata['page']) + 1}페이지)](file://{file_path})"
        sources.add(source)
    return "\n\n".join(sources)

# AI 메시지 뒤에 출처 붙이기
def parse(ai_message: AIMessage) -> str:
    """Parse the AI message and add source."""
    return f"{ai_message.content}\n\n[출처]\n\n{get_metadata_sources(retrieved_docs)}"

agent_chain = (
    get_new_messages_after_doc_retrieval
    | agent_prompt
    | llm
    | parse
)

# ==========================================================================================================================================================================================

# 적합한 tool 추출 위한 프롬프트
prompt_for_select_tool = ChatPromptTemplate.from_messages([
    ("system", """
You have "tools" that can answer "question".
Using "tools" as a guide, choose a "tool" that can answer "question".
Without saying anything else, say the "tool_name" of the selected "tool" in English.
If there is no appropriate "tool", say "None".

<tools>
{tools}
</tools>

<question>
{question}
</question>

# answer :
"""
    )
])

# tool 목록 가져오기
def get_tools(query):
    tool_info = [{"tool_name": tool.name, "tool_description": tool.description} for tool in tools]
    print(f"get_tools / {tool_info}")
    return json.dumps(tool_info, ensure_ascii=False)

# 적합한 tool 추출 위한 체인
chain_for_select_tool = (
    {"tools": get_tools, "question": RunnablePassthrough()}
    | prompt_for_select_tool 
    # | llm
    | llama_llm
    # | qwen2_llm
    | StrOutputParser()
    )

# ==========================================================================================================================================================================================

if "messages" not in st.session_state:
    # st.session_state.messages = [AIMessage(type="ai", content="무엇을 도와드릴까요?")]
    st.session_state.messages = []

# 10개까지만 표시되도록 메시지 수 제한
MAX_MESSAGES_COUNT = 10
if len(st.session_state.messages) >= MAX_MESSAGES_COUNT:
    st.session_state.messages = st.session_state.messages[2:]

for msg in st.session_state.messages:
    print(f"for msg in st.session_state.messages / msg.content: {msg.content}")
    st.chat_message(msg.type).write(msg.content)

query = st.chat_input()
if query:
    st.write("") # agent_chain 검색 중 메시지 띄울 때, 이전 메시지가 잠깐 보이는 오류가 있어서, 빈 글 하나 썼더니 오류 해결됨
    
    st.session_state.messages.append(HumanMessage(type="human", content=query))
    st.chat_message("human").write(query)

    with st.chat_message("ai"):
        print(f"messages: {st.session_state.messages}")
        
        stream_handler = StreamHandler(st.empty())
        
        response = ""
        print(f"selected_option_name: {selected_option_name}")
        if selected_option_name:
            try:
                with st.spinner("검색 중이에요 🔍"):
                    response = agent_chain.invoke({"messages": st.session_state.messages})
                    print(f"agent_chain.invoke / response: {response}")
                    st.markdown(response)
                    time.sleep(0.1)
                    st.session_state.messages.append(AIMessage(type="ai", content=response))
            except Exception as e:
                st.write("적절한 검색 도구를 찾지 못했어요, 아는 만큼 답변할게요 🫠")
                # 도구 찾기에 실패했기 때문에 LLM한테 그냥 물어보기
                with st.spinner(""):
                    response = chain.invoke({"messages": st.session_state.messages}, {"callbacks": [stream_handler]})
                    print(f"chain.invoke / response: {response}")
                    time.sleep(0.1)
                    st.session_state.messages.append(AIMessage(type="ai", content=response.content))
            
            
            
            
            # with st.spinner("검색 중이에요 🔍"):
            #     try:
            #         response = agent_chain.invoke({"messages": st.session_state.messages})
            #         print(f"agent_chain.invoke / response: {response}")
            #         st.markdown(response)
            #         time.sleep(0.1)
            #         st.session_state.messages.append(AIMessage(type="ai", content=response))
            #     except Exception as e:
            #         st.write("적절한 검색 도구를 찾지 못했어요 😔")
            #         # 도구 찾기에 실패했기 때문에 LLM한테 그냥 물어보기
            #         with st.spinner(""):
            #             response = chain.invoke({"messages": st.session_state.messages}, {"callbacks": [stream_handler]})
            #             print(f"chain.invoke / response: {response}")
            #             time.sleep(0.1)
            #             st.session_state.messages.append(AIMessage(type="ai", content=response.content))
        else:
            with st.spinner(""):
                response = chain.invoke({"messages": st.session_state.messages}, {"callbacks": [stream_handler]})
                print(f"chain.invoke / response: {response}")
                time.sleep(0.1)
                st.session_state.messages.append(AIMessage(type="ai", content=response.content))

# # 다운로드할 파일 경로 지정
# file_path = f"{os.getcwd()}/assets/생성형_AI_신기술_도입에_따른_선거_규제_연구_결과보고서.pdf"

# # 파일이 존재하는지 확인
# if os.path.exists(file_path):
#     # 파일 내용 로드
#     with open(file_path, "rb") as file:
#         file_contents = file.read()

#     # 파일 다운로드 버튼 추가
#     st.download_button(
#         label="파일 다운로드",
#         data=file_contents,
#         file_name=os.path.basename(file_path),
#         mime="text/plain"  # 파일 형식에 맞게 변경하세요. 예: 'application/pdf', 'image/png'
#     )
# else:
#     st.error("파일을 찾을 수 없습니다.")