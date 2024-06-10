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
다음 context를 사용해서 question에 대한 답을 말해줘.
정답을 모르면 모른다고만 해.

<question>
{question}
</question>

<context>
{context}
</context>

# answer :
"""
    ),
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

def get_metadata_sources(docs) -> str: 
    sources = set()
    for doc in docs:
        sources.add(doc.metadata['source'])
    return "\n".join(sources)

def parse(ai_message: AIMessage) -> str:
    """Parse the AI message and add source."""
    return f"{ai_message.content}\n\n[출처]\n{get_metadata_sources(retrieved_docs)}"

chain = (
    {"context": retriever | get_page_contents_with_metadata, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parse
)