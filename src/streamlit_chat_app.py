from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

llm = ChatOllama(model="EEVE-Korean-Instruct-10.8B-v1.0:latest")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, professional assistant named 권봇. answer me in Korean no matter what"),
    MessagesPlaceholder(variable_name="messages"),
])

chain = prompt | llm
        
# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(role="ai", content="무엇을 도와드릴까요?")]

for msg in st.session_state.messages:
    print(f"for msg in st.session_state.messages / msg.content: {msg.content}")
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(role="human", content=prompt))
    st.chat_message("human").write(prompt)

    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    # with st.chat_message("assistant"):
    #     stream_handler = StreamHandler(st.empty())
    #     llm = ChatOpenAI(openai_api_key=openai_api_key, streaming=True, callbacks=[stream_handler])
    #     response = llm.invoke(st.session_state.messages)
    #     st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))
    with st.chat_message("ai"):
        print(f"messages: {st.session_state.messages}")
        stream_handler = StreamHandler(st.empty())
        response = chain.invoke({"messages": st.session_state.messages}, {"callbacks": [stream_handler]})
        st.session_state.messages.append(AIMessage(role="ai", content=response.content))