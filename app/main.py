from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from smolagents import CodeAgent, OpenAIServerModel, FinalAnswerTool, DuckDuckGoSearchTool
from fastapi.middleware.cors import CORSMiddleware
import os



# FastAPI setup
app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load system prompt
with open('./system-prompt.md', 'r', encoding='utf-8') as file:
    system_prompt = file.read()

# Model setup
model = OpenAIServerModel(
    temperature=0.5,
    model_id=os.environ["OPENROUTER_MODEL_NAME"],
    api_base=os.environ["OPENROUTER_API_BASE"],
    api_key=os.environ["OPENROUTER_API_KEY_1"],
)

# Agent setup
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), FinalAnswerTool()],
    model=model,
    stream_outputs=False,
)

# In-memory session chat
conversation_memory = f"system: {system_prompt}\n"

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def read_root():
    return {"message": "Hello people!"}

@app.head("/")
async def head_root():
    return Response(status_code=200)

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    global conversation_memory

    # Add user input to memory
    conversation_memory += f"user: {request.message}\n"

    # Run agent
    response = agent.run(task=conversation_memory, max_steps=3)

    # Extract clean reply
    reply = response if isinstance(response, str) else getattr(response, "final_answer", str(response))

    # Add assistant reply to memory
    conversation_memory += f"assistant: {reply}\n"

    return {"reply": reply}

