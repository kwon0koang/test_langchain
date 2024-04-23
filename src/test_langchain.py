from langchain_community.chat_models import ChatOllama

if __name__ == "__main__":
    llm = ChatOllama(model="llama3:latest")
    baseMessage = llm.invoke("What is Yeouido?")
    print(f'baseMessage : {baseMessage}')