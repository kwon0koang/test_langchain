import os
import sys

# 현재 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 상위 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# 상위의 상위 디렉토리를 sys.path에 추가
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))))

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
import json
from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from streamlit.runtime.state import SessionStateProxy
from langchain_core.vectorstores import VectorStoreRetriever
import time
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain.tools.retriever import create_retriever_tool
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Union
from langchain_community.tools import Tool
from langchain_core.documents.base import Document
from datetime import datetime
from utils import current_date, perform_groundedness_check, grounded_result_mapping
from callbacks import StreamCallback
from tools import tools, options_in_sidebar, chain_for_extract_actions, get_documents_from_actions, get_retriever_by_tool_name, TOOL_AUTO, SAVED_NEWS_SEARCH_TOOL_NAME, PDF_SEARCH_TOOL_NAME, WEB_SEARCH_TOOL_NAME
from llm import gemma2

st.set_page_config(
    page_title="권봇", # 페이지 제목
    page_icon=":robot_face:", # 페이지 아이콘
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("권봇 🤖")

# eeve = ChatOllama(model="EEVE-Korean-Instruct-10.8B-v1.0:latest", temperature=0)
# llama = ChatOllama(model="llama3:8b", temperature=0)
# qwen2 = ChatOllama(model="qwen2:latest", temperature=0)

# ==========================================================================================================================================================================================

# 옵션명 추출
option_names, option_display_names = zip(*options_in_sidebar)

if 'selected_option_name' not in st.session_state:
    st.session_state.selected_option_name = None

# 선택된 옵션 업데이트 함수
def update_selected_option():
    selected_option_display_name = st.session_state.selected_option_display_name
    for name, display_name in options_in_sidebar:
        if display_name == selected_option_display_name:
            st.session_state['selected_option_name'] = name
            break
    print(f"update_selected_option / selected_option_display_name: {selected_option_display_name} / selected_option_name: {st.session_state['selected_option_name']}")

# 사이드바에 selectbox 생성
selected_option_display_name = st.sidebar.selectbox(
    '도구 🛠️',
    option_display_names,
    on_change=update_selected_option,
    key='selected_option_display_name'
)

# ==========================================================================================================================================================================================

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """
너는 정확하고 신뢰할 수 있는 답변을 제공하는 유능한 업무 보조자야.
아래의 context를 사용해서 question에 대한 답변을 작성해줘.

다음 지침을 따라주세요:
1. 답변은 반드시 한국어로 작성해야 해.
2. context에 있는 정보만을 사용해서 답변해야 해.
3. 정답을 확실히 알 수 없다면 "주어진 정보로는 답변하기 어렵습니다."라고만 말해.
4. 답변 시 추측하거나 개인적인 의견을 추가하지 마.
5. 가능한 간결하고 명확하게 답변해.
""")
    , MessagesPlaceholder(variable_name="messages"),
    ("human", """
     
# question: 
{question}

# context: 
{context}

# answer: 
""")
])

default_prompt = ChatPromptTemplate.from_messages([
    ("system", """
너는 정확하고 신뢰할 수 있는 답변을 제공하는 유능한 업무 보조자야.
다음 질문에 최선을 다해서 대답해줘.
"""
    )
    , MessagesPlaceholder(variable_name="messages")
    , ("human", "{question}")
])

if 'retrieved_docs' not in st.session_state:
    st.session_state.retrieved_docs = []
    
def get_page_contents_string(docs) -> str: 
    """
    문서 리스트를 받아 각 문서의 본문 내용을 이어붙인 문자열을 생성
    """
    st.session_state.retrieved_docs = docs
    
    result = ""
    
    for i, doc in enumerate(docs):
        if i > 0:
            result += "\n"
            
        # if 'url' in doc:
        #     # Web 검색
        #     result += f"## 본문: {doc['content']}\n### 출처: {doc['url']}"
        # else:
        #     # Vector DB 검색
        #     result += f"## 본문: {doc.page_content}\n### 출처: {doc.metadata['source']}"
        
        # LLM 답변 이후에 parse 함수로 출처 붙여줄거니까 본문만 이어붙인 문자열 생성하자
        if 'url' in doc:
            # Web 검색
            result += f"{doc['content']}"
        else:
            # Vector DB 검색
            result += f"{doc.page_content}"
    
    return result

def check_context(inputs: dict) -> bool:
    """
    context 존재 여부 확인
    
    :return: 문자열이 비어있지 않으면 True, 비어있으면 False
    """
    result = bool(inputs['context'].strip())
    print(f"check_context / result: {result}")
    return result

def retrieved_docs_and_get_messages(messages: List[BaseMessage], selected_option_name: str) -> dict:
    """
    쿼리에 따라 문서를 검색하고, 해당 문서들의 본문 내용과 출처를 포함한 문자열을 반환
    """
    print("========================")
    print(f"retrieved_docs_and_get_messages / messages: {messages}")
    query = messages[-1].content # last human message
    print(f"retrieved_docs_and_get_messages / query: {query}")
    
    if TOOL_AUTO == selected_option_name:
        actions_json = chain_for_extract_actions.invoke(query)
        
        # 예외 답변 후처리
        actions_json = actions_json.replace("```json", "")
        actions_json = actions_json.replace("```", "")
        
        st.session_state.retrieved_docs = get_documents_from_actions(actions_json, tools)
    else:
        retriever = get_retriever_by_tool_name(selected_option_name)
        st.session_state.retrieved_docs = retriever.invoke(query)
    
    messages_without_last = messages[:-1]
    
    if len(st.session_state.retrieved_docs) <= 0:
        return {"messages": messages_without_last
            , "context": ""
            , "question": query}
    
    return {"messages": messages_without_last
            , "context": get_page_contents_string(st.session_state.retrieved_docs)
            , "question": query}

def get_metadata_sources(docs) -> str: 
    """
    문서 리스트에서 각 문서의 출처 추출해서 문자열로 반환
    """
    sources = set()
    
    for doc in docs:
        if 'url' in doc:
            # Web 검색
            sources.add(doc['url'])
        else:
            # Vector DB 검색
            source = doc.metadata['source']
            is_pdf = source.endswith('.pdf')
            if (is_pdf):
                file_path = doc.metadata['source']
                file_name = os.path.basename(file_path)
                source = f"[{file_name} ({int(doc.metadata['page']) + 1}페이지)](file://{file_path})"
            sources.add(source)
        
    return "\n\n".join(sources)

def parse(ai_message: AIMessage) -> str:
    """
    AI 메시지 파싱해서 내용에 출처 추가
    """
    return f"{ai_message.content}\n\n<span style='color:gray;'>[출처]</span>\n\n{get_metadata_sources(st.session_state.retrieved_docs)}"

with_context_chain = (
    RunnablePassthrough()
    | RunnableLambda(lambda x: {
        "messages": x["messages"]
        , "context": x["context"]
        , "question": x["question"]
        })
    | agent_prompt
    # | gemma2
    # | parse
)

without_context_chain = (
    RunnablePassthrough()
    | RunnableLambda(lambda x: {
        "messages": x["messages"]
        ,"question": x["question"]
        })
    | default_prompt
    # | gemma2
    # | StrOutputParser()
)

agent_chain = (
    RunnablePassthrough()
    | RunnableLambda(lambda x: retrieved_docs_and_get_messages(x["messages"], st.session_state.selected_option_name))
    | RunnableBranch(
        (lambda x: check_context(x), with_context_chain),
        without_context_chain  # default
    )
)

# ==========================================================================================================================================================================================

if "messages" not in st.session_state:
    # st.session_state.messages = [AIMessage(type="ai", content="무엇을 도와드릴까요?")]
    st.session_state.messages = []

# 8개까지만 표시되도록 메시지 수 제한
MAX_MESSAGES_COUNT = 8
if len(st.session_state.messages) >= MAX_MESSAGES_COUNT:
    st.session_state.messages = st.session_state.messages[2:]

for msg in st.session_state.messages:
    print(f"for msg in st.session_state.messages / msg.content: {msg.content}")
    st.chat_message(msg.type).markdown(msg.content
                                       , unsafe_allow_html=True
                                       )

query = st.chat_input()
if query:
    st.write("") # agent_chain 검색 중 메시지 띄울 때, 이전 메시지가 잠깐 보이는 오류가 있어서, 빈 글 하나 썼더니 오류 해결됨
    
    st.session_state.messages.append(HumanMessage(type="human", content=query))
    st.chat_message("human").markdown(query
                                      , unsafe_allow_html=True
                                      )

    with st.chat_message("ai"):
        print(f"messages: {st.session_state.messages}")
        
        streaming_llm = ChatOllama(model="ko-gemma-2-9b-it.Q5_K_M:latest"
                            , temperature=0
                            , callbacks=[StreamCallback(st.empty(), initial_text="")]
                            )
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
너는 정확하고 신뢰할 수 있는 답변을 제공하는 유능한 업무 보조자야.
오늘 날짜는 {current_date}야. 무엇을 요청받았을 때 항상 정확한 현재 날짜를 제공해줘.
항상 한국어로 대답해줘.
다음 질문에 최선을 다해서 대답해줘.
""")
            , MessagesPlaceholder(variable_name="messages"),
        ])
        streaming_chain = prompt | streaming_llm
        streaming_agent_chain = agent_chain | streaming_llm
        
        response = ""
        print(f"selected_option_name: {st.session_state.selected_option_name}")
        
        if st.session_state.selected_option_name is None:
            with st.spinner(""):
                # response = chain.invoke({"messages": st.session_state.messages}, {"callbacks": [stream_handler]})
                response = streaming_chain.invoke({"messages": st.session_state.messages})
                print(f"chain.invoke / response: {response}")
                time.sleep(0.1)
                st.session_state.messages.append(AIMessage(type="ai", content=response.content))
        else:
            try:
                with st.spinner("검색 중이에요 🔍"):
                    # response = agent_chain.invoke({"messages": st.session_state.messages})
                    response = streaming_agent_chain.invoke({"messages": st.session_state.messages})
                    
                    print(f"agent_chain.invoke / response: {response}")
                    
                    # st.markdown(response
                    #             , unsafe_allow_html=True
                    #             )
                    
                    time.sleep(0.1)
                    
                    if (len(st.session_state.retrieved_docs) <= 0):
                        # 검색 문서 없으면
                        st.session_state.messages.append(AIMessage(type="ai", content=response))
                    else:
                        # 검색 문서 있으면 출처 표시 및 LLM 답변과 문서 관련성 검증
                        
                        # 출처 표시
                        source_msg = f"<span style='color:gray;'>[출처]</span>\n\n{get_metadata_sources(st.session_state.retrieved_docs)}"
                        st.markdown(source_msg
                                    , unsafe_allow_html=True
                                    )
                        
                        # LLM 답변과 문서 관련성 검증
                        grounded_result = perform_groundedness_check(
                            answer=response.content
                            , context=get_page_contents_string(st.session_state.retrieved_docs)
                            )
                        grounded_label, grounded_color = grounded_result_mapping.get(grounded_result, ("알 수 없음", "gray"))
                        grounded_msg = f'<span style="color:gray;">문서 검증 결과:</span> <span style="color:{grounded_color}; font-weight:bold;">{grounded_label}</span>'
                        st.markdown(grounded_msg
                                    , unsafe_allow_html=True
                                    )
                        
                        # 검색 문서 초기화
                        st.session_state.retrieved_docs = []
                        
                        llm_resp_and_grounded_msg = f"{response.content}\n\n{source_msg}\n\n{grounded_msg}"
                        st.session_state.messages.append(AIMessage(type="ai", content=llm_resp_and_grounded_msg))
            except Exception as e:
                print(f"error: {e}")
                
                if e == "사용할 도구 없음":
                    st.markdown("### ❗️ 검색에 실패했어요, 아는 만큼 답변할게요")
                else:
                    st.markdown("### ❗️ 답변 중 오류가 발생했어요, 아는 만큼 답변할게요")
                
                with st.spinner(""):
                    # response = chain.invoke({"messages": st.session_state.messages}, {"callbacks": [stream_handler]})
                    response = streaming_chain.invoke({"messages": st.session_state.messages})
                    print(f"chain.invoke / response: {response}")
                    time.sleep(0.1)
                    st.session_state.messages.append(AIMessage(type="ai", content=response.content))