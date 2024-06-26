from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Union
from langserve.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages.base import BaseMessage
from langserve.schema import CustomUserType
from typing import Tuple
from langserve import add_routes
from llm import llm
from chain import chain as chain
from chat import chain as chat_chain
from rag_chain import chain as rag_chain
from rag_chat import chain as rag_chat_chain
from rag_chat2 import chain as rag_chat_chain2
from dotenv import load_dotenv

# 환경변수 로드 (.env)
load_dotenv()

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("chain/playground")

add_routes(
    app, 
    llm, 
    path="/llm"
)

add_routes(
    app, 
    chain, 
    path="/chain"
)

class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ..., # Pydantic에서 필수 필드임을 의미
        description="The chat messages representing the current conversation.",
    )
    
add_routes(
    app,
    chat_chain.with_types(input_type=InputChat),
    path="/chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)

add_routes(
    app, 
    rag_chain, 
    path="/rag_chain"
)
    
add_routes(
    app,
    rag_chat_chain.with_types(input_type=InputChat),
    path="/rag_chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)
    
add_routes(
    app,
    rag_chat_chain2.with_types(input_type=InputChat),
    path="/rag_chat2",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)

# http://localhost:8000/chain?query=What is stock?
@app.get("/chain")
def query_chain(query: str):
    result = chain.invoke(query)
    return result

@app.get("/rag_chain")
def query_chain(query: str):
    result = rag_chain.invoke(query)
    return result

if __name__ == "__main__":
    import uvicorn
    
    # uvicorn: ASGI(Asynchronous Server Gateway Interface) 서버를 구현한 비동기 경량 웹 서버
    uvicorn.run(app, host="localhost", port=8000)