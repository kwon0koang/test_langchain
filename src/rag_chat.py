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
from langchain.memory import ConversationSummaryBufferMemory
from llm import llm

# # BeautifulSoup : HTML 및 XML 문서를 파싱하고 구문 분석하는 데 사용되는 파이썬 라이브러리. 주로 웹 스크레이핑(웹 페이지에서 데이터 추출) 작업에서 사용되며, 웹 페이지의 구조를 이해하고 필요한 정보를 추출하는 데 유용
# loader = WebBaseLoader(
#     web_paths=("https://www.aitimes.com/news/articleView.html?idxno=159102"
#                , "https://www.aitimes.com/news/articleView.html?idxno=159072"
#                , "https://www.aitimes.com/news/articleView.html?idxno=158943"
#                ),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             "article", # 태그
#             attrs={"id": ["article-view-content-div"]}, # 태그의 ID 값들
#         )
#     ),
# )
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# splits = text_splitter.split_documents(data)
# print(f'len(splits[0].page_content) : {len(splits[1].page_content)}')
# print(f'splits : {splits}')

# vectorstore = FAISS.from_documents(splits,
#                                    embedding = OllamaEmbeddings(model="EEVE-Korean-Instruct-10.8B-v1.0:latest"),
#                                    distance_strategy = DistanceStrategy.COSINE # 코사인 유사도 측정. 값이 클수록 더 유사함을 의미
#                                   )

# # 로컬에 DB 저장
# MY_FAISS_INDEX = "MY_FAISS_INDEX"
# vectorstore.save_local(MY_FAISS_INDEX)

# 로컬 DB 불러오기
MY_FAISS_INDEX = "MY_FAISS_INDEX"
embeddings = OllamaEmbeddings(model="EEVE-Korean-Instruct-10.8B-v1.0:latest")
vectorstore = FAISS.load_local(MY_FAISS_INDEX, 
                               embeddings, 
                               allow_dangerous_deserialization=True
                               )

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=80,
    memory_key="chat_history",
    return_messages=True,
)

def load_memory(input):
    print(input)
    return memory.load_memory_variables({})["chat_history"]

prompt = ChatPromptTemplate.from_messages([
    ("system", """
너는 유능한 업무 보조자야.
다음 context를 사용해서 질문에 대한 답과 출처를 심플하게 말해줘.
정답을 모르면 모른다고만 해.

# context : {context}
"""
    ),
    # MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# extract page_content with metadata
def get_page_contents_with_metadata(docs):
    return "\n\n".join(f'{doc.page_content} [출처 : {doc.metadata["source"]}]' for doc in docs)

chain = (
    {"context": retriever | get_page_contents_with_metadata, "question": RunnablePassthrough}
    | prompt
    | llm
    | StrOutputParser()
)
