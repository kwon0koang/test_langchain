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
from chain import chain as chain
from chat import chain as chat_chain
from rag_chain import chain as rag_chain
from rag_chat import chain as rag_chat
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
    chain, 
    path="/chain"
)

class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
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
    
# class ChatHistory(CustomUserType):
#     chat_history: List[Tuple[str, str]] = Field(
#         ...,
#         examples=[[("human input", "ai response")]],
#         extra={"widget": {"type": "chat", "input": "question", "output": "answer"}},
#     )
#     question: str


# def _format_to_messages(input: ChatHistory) -> List[BaseMessage]:
#     """Format the input to a list of messages."""
#     history = input.chat_history
#     user_input = input.question

#     messages = []

#     for human, ai in history:
#         messages.append(HumanMessage(content=human))
#         messages.append(AIMessage(content=ai))
#     messages.append(HumanMessage(content=user_input))
#     return messages
    
add_routes(
    app,
    rag_chat.with_types(input_type=InputChat),
    path="/rag_chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="localhost", port=8000)