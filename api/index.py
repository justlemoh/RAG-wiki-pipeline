import os
import sys

# Add project root to sys.path so imports work correctly on Render
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Import run_wiki_agent from agents
from agents.wiki_agent import run_wiki_agent

app = FastAPI(title="RAG Wiki Query Pipeline")

# Mount static folder (paths are relative to the project root on Render)
app.mount("/static", StaticFiles(directory="static"), name="static")

class Message(BaseModel):
    role: str  # 'user' or 'model'
    text: str

class QueryRequest(BaseModel):
    query: str
    history: list[Message] = []
    k: int = 3

@app.get("/")
async def read_index():
    return FileResponse("templates/index.html")

@app.post("/api/query")
async def query_rag(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        answer, sources = run_wiki_agent(req.query, history=req.history)
        return {
            "answer": answer,
            "sources": sources,
            "chunks": sources  # backward-compat alias
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api.index:app", host="127.0.0.1", port=8000, reload=True)
