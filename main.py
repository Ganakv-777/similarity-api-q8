from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import httpx
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="InfoCore Similarity Service - AI Pipe")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (modify if needed)
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Request model
class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom != 0 else 0.0

# Get embeddings from AI Pipe
async def get_embeddings(text_list: List[str]):
    token = os.getenv("AI_PIPE_TOKEN")
    base_url = os.getenv("AI_PIPE_BASE_URL", "https://aipipe.org/openai/v1")

    if not token:
        raise HTTPException(status_code=500, detail="Missing AI_PIPE_TOKEN in environment")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "text-embedding-3-small",
        "input": text_list
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{base_url}/embeddings", headers=headers, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"AI Pipe embedding failed: {response.text}")

    return [item["embedding"] for item in response.json()["data"]]

# Main similarity endpoint
@app.post("/similarity")
async def compute_similarity(request: SimilarityRequest):
    embeddings = await get_embeddings(request.docs + [request.query])
    doc_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    scores = [cosine_similarity(query_embedding, e) for e in doc_embeddings]
    top_indices = np.argsort(scores)[::-1][:3]

    return {
        "query": request.query,
        "matches": [request.docs[i] for i in top_indices],
        "scores": [scores[i] for i in top_indices]
    }

# Health check endpoint
@app.get("/")
def root():
    return {"message": "✅ AI Pipe Similarity API is running!"}
