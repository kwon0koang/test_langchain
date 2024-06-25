import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
from callbacks import StreamCallback
from tools import tools, TOOL_AUTO, SAVED_NEWS_SEARCH_TOOL_NAME, PDF_SEARCH_TOOL_NAME, WEB_SEARCH_TOOL_NAME

st.title("권봇 🤖")

eeve_llm = ChatOllama(model="EEVE-Korean-Instruct-10.8B-v1.0:latest", temperature=0)
llama_llm = ChatOllama(model="llama3:8b", temperature=0)
# qwen2_llm = ChatOllama(model="qwen2:latest", temperature=0)

# ==========================================================================================================================================================================================

# 사이드바에 tool 목록 셋팅

options = [
    (None, '선택 안함'),
    (SAVED_NEWS_SEARCH_TOOL_NAME, '저장된 뉴스 검색'),
    (PDF_SEARCH_TOOL_NAME, '저장된 PDF 검색'),
    (WEB_SEARCH_TOOL_NAME, 'WEB 검색'),
    (TOOL_AUTO, '도구 자동 선택 (BETA)'),
]

# 옵션명 추출
option_names, option_display_names = zip(*options)

if 'selected_option_name' not in st.session_state:
    st.session_state.selected_option_name = None

# 선택된 옵션 업데이트 함수
def update_selected_option():
    selected_option_display_name = st.session_state.selected_option_display_name
    st.session_state.selected_option_name = next(name for name, display_name in options if display_name == selected_option_display_name)

# 사이드바에 selectbox 생성
selected_option_display_name = st.sidebar.selectbox(
    '도구 🛠️',
    option_display_names,
    on_change=update_selected_option,
    key='selected_option_display_name'
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

# tool명으로 retriever 찾기
def get_retriever_by_tool_name(name) -> VectorStoreRetriever:
    print(f"get_retriever_by_tool_name / name: {name}")
    for tool in tools:
        if tool.name == name:
            # print(tool.func) # functools.partial(<function _get_relevant_documents at 0x1487dd6c0>, retriever=VectorStoreRetriever(tags=['FAISS', 'HuggingFaceEmbeddings'], vectorstore=<langchain_community.vectorstores.faiss.FAISS object at 0x317e52ea0>, search_kwargs={'k': 5}), document_prompt=PromptTemplate(input_variables=['page_content'], template='{page_content}'), document_separator='\n\n')
            return tool.func.keywords['retriever']
    return None

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
            
        if 'url' in doc:
            # Web 검색
            result += f"## 본문: {doc['content']}\n### 출처: {doc['url']}"
        else:
            # Vector DB 검색
            result += f"## 본문: {doc.page_content}\n### 출처: {doc.metadata['source']}"
    
    return result


# 문서 검색 후 새 메시지 리스트 가져오기
def get_new_messages_after_doc_retrieval(messages_dict) -> dict:
    print("========================")
    print(f"messages_dict: {messages_dict}") # {'messages': [HumanMessage(content='라마3 성능은?')]}
    messages = messages_dict["messages"]
    print(f"messages: {messages}")
    last_human_message = messages[-1].content
    print(f"last_human_message: {last_human_message}")
    
    selected_tool = ""
    if TOOL_AUTO == st.session_state.selected_option_name:
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
        selected_tool = st.session_state.selected_option_name
    retriever = get_retriever_by_tool_name(selected_tool)
    
    if retriever is None:
        raise ValueError(f"{selected_tool} retriever가 없어요")
    
    global retrieved_docs
    retrieved_docs = retriever.invoke(last_human_message)
    
    print(f"retrieved_docs: {retrieved_docs}")
    
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

# AI 메시지 뒤에 출처 붙이기
def parse(ai_message: AIMessage) -> str:
    """Parse the AI message and add source."""
    return f"{ai_message.content}\n\n[출처]\n\n{get_metadata_sources(retrieved_docs)}"

agent_chain = (
    get_new_messages_after_doc_retrieval
    | agent_prompt
    | eeve_llm
    | parse
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
        
        streaming_eeve_llm = ChatOllama(model="EEVE-Korean-Instruct-10.8B-v1.0:latest"
                            , temperature=0
                            , callbacks=[StreamCallback(st.empty(), initial_text="")]
                            )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful, professional assistant named 권봇. answer me in Korean no matter what"),
            MessagesPlaceholder(variable_name="messages"),
        ])
        streaming_chain = prompt | streaming_eeve_llm
        
        response = ""
        print(f"selected_option_name: {st.session_state.selected_option_name}")
        
        if st.session_state.selected_option_name:
            try:
                with st.spinner("검색 중이에요 🔍"):
                    response = agent_chain.invoke({"messages": st.session_state.messages})
                    print(f"agent_chain.invoke / response: {response}")
                    st.markdown(response)
                    time.sleep(0.1)
                    st.session_state.messages.append(AIMessage(type="ai", content=response))
            except Exception as e:
                print(f"error: {e}")
                st.write("적절한 검색 도구를 찾지 못했어요, 아는 만큼 답변할게요 🫠")
                # 도구 찾기에 실패했기 때문에 LLM한테 그냥 물어보기
                with st.spinner(""):
                    # response = chain.invoke({"messages": st.session_state.messages}, {"callbacks": [stream_handler]})
                    response = streaming_chain.invoke({"messages": st.session_state.messages})
                    print(f"chain.invoke / response: {response}")
                    time.sleep(0.1)
                    st.session_state.messages.append(AIMessage(type="ai", content=response.content))
        else:
            with st.spinner(""):
                # response = chain.invoke({"messages": st.session_state.messages}, {"callbacks": [stream_handler]})
                response = streaming_chain.invoke({"messages": st.session_state.messages})
                print(f"chain.invoke / response: {response}")
                time.sleep(0.1)
                st.session_state.messages.append(AIMessage(type="ai", content=response.content))