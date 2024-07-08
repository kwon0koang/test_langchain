from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import MarkdownTextSplitter
import pymupdf4llm
from constants import MY_PDF_INDEX
from embeddings import embeddings
from langchain_core.documents import Document
from typing import List, Union
import re

loader = PyMuPDFLoader(f"{os.getcwd()}/assets/생성형_AI_신기술_도입에_따른_선거_규제_연구_결과보고서.pdf")
data = loader.load()

def remove_newlines_except_after_period(text):
    """마침표 다음의 줄바꿈을 제외한 모든 줄바꿈을 제거"""
    return re.sub(r'(?<!\.)(\n|\r\n)', ' ', text)

def process_pages(pages: List[Document]) -> List[Document]:
    return [Document(page_content=remove_newlines_except_after_period(page.page_content), metadata=page.metadata) for page in pages]

processed_data = process_pages(data)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(processed_data)

# data = pymupdf4llm.to_markdown(f"{os.getcwd()}/assets/생성형_AI_신기술_도입에_따른_선거_규제_연구_결과보고서.pdf")
# text_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=100)
# splits = text_splitter.split_documents([data])

print(f'len(splits[0].page_content) : {len(splits[0].page_content)}')
print(f'splits : {splits}')

vectorstore = FAISS.from_documents(splits,
                                   embedding = embeddings,
                                  )

# 로컬에 DB 저장
vectorstore.save_local(MY_PDF_INDEX)