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
from llm import llm
from constants import MY_FAISS_INDEX

# 로컬 DB 불러오기
embeddings = OllamaEmbeddings(model="EEVE-Korean-Instruct-10.8B-v1.0:latest")
vectorstore = FAISS.load_local(MY_FAISS_INDEX, 
                               embeddings, 
                               allow_dangerous_deserialization=True
                               )

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# prompt = hub.pull("rlm/rag-prompt") # https://smith.langchain.com/hub/rlm/rag-prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """
너는 유능한 업무 보조자야.
다음 context를 사용해서 question에 대한 답과 출처를 심플하게 말해줘.
정답을 모르면 모른다고만 해.

# question : {question}

# context : {context}

# answer :
"""
    ),
])

# extract page_content
def get_page_contents(docs):
    return "\n\n".join(f'{doc.page_content}' for doc in docs)

# extract page_content with metadata
def get_page_contents_with_metadata(docs):
    return "\n\n".join(f'{doc.page_content} [출처 : {doc.metadata["source"]}]' for doc in docs)

chain = (
    {"context": retriever | get_page_contents_with_metadata, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
