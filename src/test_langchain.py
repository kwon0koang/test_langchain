from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3:latest")
baseMessage = llm.invoke("What is Yeouido?")
print(f'baseMessage : {baseMessage}')