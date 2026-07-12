import os
import sys

# Add project root to sys.path so imports work correctly on Render
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Import run_rag_pipeline from rag_query
from rag_query import run_rag_pipeline

app = FastAPI(title="RAG Wiki Query Pipeline")

# Mount static folder (paths are relative to the project root on Render)
app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryRequest(BaseModel):
    query: str
    k: int = 3

@app.get("/")
async def read_index():
    return FileResponse("templates/index.html")

@app.post("/api/query")
async def query_rag(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        answer, chunks = run_rag_pipeline(req.query, k=req.k)
        return {
            "answer": answer,
            "chunks": chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api.index:app", host="127.0.0.1", port=8000, reload=True)
