import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "cache" / "embeddings.json"

model = None

def load_model():
    global model
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    return model


def embed_abstracts(papers):
    m = load_model()
    abstracts = [p.get("abstract", "") for p in papers]
    embeddings = m.encode(abstracts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def save_embeddings(embeddings, paper_ids):
    data = {
        "paper_ids": paper_ids,
        "embeddings": embeddings.tolist()
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(data, f)


def load_embeddings():
    if not CACHE_PATH.exists():
        return None, None
    with open(CACHE_PATH, "r") as f:
        data = json.load(f)
    embeddings = np.array(data["embeddings"], dtype=np.float32)
    paper_ids = data["paper_ids"]
    return embeddings, paper_ids


def get_embeddings(papers, force_refresh=False):
    paper_ids = [p["paper_id"] for p in papers]

    if not force_refresh:
        embeddings, cached_ids = load_embeddings()
        if cached_ids is not None and cached_ids == paper_ids:
            print("Loaded embeddings from cache")
            return embeddings

    print("Generating embeddings...")
    embeddings = embed_abstracts(papers)
    save_embeddings(embeddings, paper_ids)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings