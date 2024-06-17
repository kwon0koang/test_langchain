from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
from streamlit.runtime.state import SessionStateProxy
from langchain_core.vectorstores import VectorStoreRetriever
import time
import os
from langchain_core.runnables import RunnablePassthrough
from langchain.tools.retriever import create_retriever_tool
from langchain_core.pydantic_v1 import BaseModel, Field
from constants import MY_FAISS_INDEX, MY_PDF_INDEX
from embeddings import embeddings
# from streamlit_chat_app.agent_chain import agent_chain

WEB_TOOL_NAME = "web_search"
PDF_TOOL_NAME = "pdf_search"

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

llm = ChatOllama(model="EEVE-Korean-Instruct-10.8B-v1.0:latest", temperature=0.1)

query = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, professional assistant named 권봇. answer me in Korean no matter what"),
    MessagesPlaceholder(variable_name="messages"),
])

chain = query | llm

# ==========================================================================================================================================================================================

options = [
    (None, '-'),
    (WEB_TOOL_NAME, 'WEB'),
    (PDF_TOOL_NAME, 'PDF'),
]

option_names, option_display_names = zip(*options)  # 옵션 코드를 추출
selected_option_display_name = st.sidebar.selectbox('도구 🛠️', option_display_names)
selected_option_name = next(name for name, display_name in options if display_name == selected_option_display_name)
def get_selected_option_name() -> str:
    return str(selected_option_name)

# ==========================================================================================================================================================================================

# 로컬 DB 불러오기
vectorstore1 = FAISS.load_local(MY_FAISS_INDEX, 
                               embeddings, 
                               allow_dangerous_deserialization=True)
retriever1 = vectorstore1.as_retriever(search_type="similarity", search_kwargs={"k": 3}) # 유사도 높은 3문장 추출
vectorstore2 = FAISS.load_local(MY_PDF_INDEX, 
                               embeddings, 
                               allow_dangerous_deserialization=True)
retriever2 = vectorstore2.as_retriever(search_type="similarity", search_kwargs={"k": 3}) # 유사도 높은 3문장 추출

retriever_tool1 = create_retriever_tool(
    retriever1,
    name="web_search",
    description="엔비디아, 퍼플렉시티, 라마3 관련 정보를 검색한다. 엔비디아, 퍼플렉시티, 라마3 관련 정보는 이 도구를 사용해야 한다",
)

retriever_tool2 = create_retriever_tool(
    retriever2,
    name="pdf_search",
    description="생성형 AI 신기술 도입에 따른 선거 규제 연구 관련 정보를 검색한다. 생성형 AI 신기술 도입에 따른 선거 규제 연구 관련 정보는 이 도구를 사용해야 한다",
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

def get_retriever_by_tool_name(name) -> VectorStoreRetriever:
    print(f"get_retriever_by_tool_name / name: {name}")
    for tool in tools:
        if tool.name == name:
            # print(tool.func) # functools.partial(<function _get_relevant_documents at 0x1487dd6c0>, retriever=VectorStoreRetriever(tags=['FAISS', 'HuggingFaceEmbeddings'], vectorstore=<langchain_community.vectorstores.faiss.FAISS object at 0x317e52ea0>, search_kwargs={'k': 5}), document_prompt=PromptTemplate(input_variables=['page_content'], template='{page_content}'), document_separator='\n\n')
            return tool.func.keywords['retriever']
    return None

def get_new_messages_after_doc_retrieval(messages_dict) -> dict:
    print("========================")
    print(f"messages_dict: {messages_dict}") # {'messages': [HumanMessage(content='라마3 성능은?')]}
    messages = messages_dict["messages"]
    print(f"messages: {messages}")
    last_human_message = messages[-1].content
    print(f"last_human_message: {last_human_message}")
    
    selected_tool = selected_option_name
    retriever = get_retriever_by_tool_name(selected_tool)
    
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

def get_metadata_sources(docs) -> str: 
    sources = set()
    for doc in docs:
        source = doc.metadata['source']
        is_pdf = source.endswith('.pdf')
        if (is_pdf):
            file_path = doc.metadata['source']
            file_name = os.path.basename(file_path)
            source = f"{file_name} ({int(doc.metadata['page']) + 1}페이지)"
        sources.add(source)
    return "\n\n".join(sources)

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

if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(type="ai", content="무엇을 도와드릴까요?")]

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
            # response = agent_chain.invoke({"messages": st.session_state.messages}, {"callbacks": [stream_handler]})
            # print(f"agent_chain.invoke / response: {response}")
            # time.sleep(0.1)
            # st.write(response)
            # st.session_state.messages.append(AIMessage(type="ai", content=response))
            with st.spinner("검색 중이에요 🔍"):
                response = agent_chain.invoke({"messages": st.session_state.messages})
                print(f"agent_chain.invoke / response: {response}")
                st.markdown(response)
                time.sleep(0.1)
                st.session_state.messages.append(AIMessage(type="ai", content=response))
        else:
            with st.spinner(""):
                response = chain.invoke({"messages": st.session_state.messages}, {"callbacks": [stream_handler]})
                print(f"chain.invoke / response: {response}")
                time.sleep(0.1)
                st.session_state.messages.append(AIMessage(type="ai", content=response.content))
        