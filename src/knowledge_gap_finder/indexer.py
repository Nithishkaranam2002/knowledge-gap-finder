import numpy as np
import faiss
import pickle
from rank_bm25 import BM25Okapi
from pathlib import Path

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
BM25_PATH = INDEX_DIR / "bm25_index.pkl"
FAISS_PATH = INDEX_DIR / "faiss_index.bin"
PAPERS_PATH = INDEX_DIR / "indexed_papers.pkl"


def tokenize(text):
    return text.lower().split()


def build_bm25(papers):
    corpus = [tokenize(p.get("title", "") + " " + p.get("abstract", "")) for p in papers]
    bm25 = BM25Okapi(corpus)
    return bm25


def build_faiss(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    vectors = np.array(embeddings, dtype=np.float32)
    index.add(vectors)
    return index


def build_index(papers, embeddings, force_refresh=False):
    if not force_refresh and BM25_PATH.exists() and FAISS_PATH.exists():
        print("Loading indexes from cache")
        with open(BM25_PATH, "rb") as f:
            bm25 = pickle.load(f)
        faiss_index = faiss.read_index(str(FAISS_PATH))
        with open(PAPERS_PATH, "rb") as f:
            indexed_papers = pickle.load(f)
        return bm25, faiss_index, indexed_papers

    print("Building BM25 and FAISS indexes...")
    bm25 = build_bm25(papers)
    faiss_index = build_faiss(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    faiss.write_index(faiss_index, str(FAISS_PATH))
    with open(PAPERS_PATH, "wb") as f:
        pickle.dump(papers, f)

    print(f"Index built: {len(papers)} papers")
    return bm25, faiss_index, papers