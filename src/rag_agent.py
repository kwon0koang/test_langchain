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


llm = ChatOllama(model="EEVE-Korean-Instruct-10.8B-v1.0:latest")

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

# prompt = hub.pull("rlm/rag-prompt") # https://smith.langchain.com/hub/rlm/rag-prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

question : {question}

context : {context}
"""
    ),
])

# extract page_content
def get_page_contents(docs):
    return "\n\n\n".join(f'{doc.page_content}' for doc in docs)

chain = (
    {"context": retriever | get_page_contents, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
