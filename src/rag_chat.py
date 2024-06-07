from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import OllamaEmbeddings
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from llm import llm
from constants import MY_FAISS_INDEX
from embeddings import embeddings

# 로컬 DB 불러오기
vectorstore = FAISS.load_local(MY_FAISS_INDEX
                               , embeddings
                               , allow_dangerous_deserialization=True
                               )

retriever = vectorstore.as_retriever(search_type="similarity"
                                     , search_kwargs={"k": 5})

# prompt = hub.pull("rlm/rag-prompt") # https://smith.langchain.com/hub/rlm/rag-prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """
너는 유능한 업무 보조자야.
context를 사용해서 question에 대한 답을 말해줘.
정답을 모르면 모른다고만 해.
""")
    , MessagesPlaceholder(variable_name="messages"),
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

def replace_last_message(messages, new_item):
    if not messages:
        raise ValueError("List is empty")
    new_messages = messages[:]
    new_messages[-1] = new_item
    return new_messages

def get_new_messages_after_doc_retrieval(messages_dict):
    print("========================")
    print(f"messages_dict: {messages_dict}") # {'messages': [HumanMessage(content='라마3 성능은?')]}
    messages = messages_dict["messages"]
    print(f"messages: {messages}")
    last_human_message = messages[-1].content
    print(f"last_human_message: {last_human_message}")
    
    global retrieved_docs
    retrieved_docs = retriever.invoke(last_human_message)
    
    new_human_message = HumanMessage(content=f"""
# question : {last_human_message}

# context ========================
{get_page_contents_with_metadata(retrieved_docs)}
==================================

# answer :
""")
    
    new_messages = replace_last_message(messages, new_human_message)
    return {"messages": new_messages}

def get_metadata_sources(docs) -> str: 
    sources = set()
    for doc in docs:
        sources.add(doc.metadata['source'])
    return "\n".join(sources)

def parse(ai_message: AIMessage) -> str:
    """Parse the AI message and add source."""
    return f"{ai_message.content}\n\n[출처]\n{get_metadata_sources(retrieved_docs)}"

chain = (
    get_new_messages_after_doc_retrieval
    | prompt
    | llm
    | parse
)