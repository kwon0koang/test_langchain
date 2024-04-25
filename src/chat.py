from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, professional assistant named 권봇. answer me in Korean no matter what"),
    MessagesPlaceholder(variable_name="messages"),
])

llm = ChatOllama(model="EEVE-Korean-Instruct-10.8B-v1.0:latest")

output_parser = StrOutputParser()

chain = prompt | llm | output_parser
