from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from pipeline import run_pipeline

app = FastAPI(title="Knowledge Gap Finder")

app.mount("/static", StaticFiles(directory="frontend"), name="static")


class SearchRequest(BaseModel):
    query: str
    limit: int = 100
    top_k: int = 10
    force_refresh: bool = False


@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")


@app.post("/search")
def search(request: SearchRequest):
    results = run_pipeline(
        query=request.query,
        limit=request.limit,
        top_k=request.top_k,
        force_refresh=request.force_refresh
    )
    return results


@app.get("/health")
def health():
    return {"status": "ok"}