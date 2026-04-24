import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
model = None

def load_model():
    global model
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    return model


def bm25_search(bm25, papers, query, top_k=10):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(ranked):
        results.append((idx, scores[idx], rank + 1))
    return results


def faiss_search(faiss_index, papers, query, top_k=10):
    m = load_model()
    query_vec = m.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = faiss_index.search(query_vec, top_k)
    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        results.append((idx, dist, rank + 1))
    return results


def reciprocal_rank_fusion(bm25_results, faiss_results, k=60):
    scores = {}
    for idx, _, rank in bm25_results:
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
    for idx, _, rank in faiss_results:
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def retrieve(query, bm25, faiss_index, papers, top_k=10):
    bm25_results = bm25_search(bm25, papers, query, top_k)
    faiss_results = faiss_search(faiss_index, papers, query, top_k)
    fused = reciprocal_rank_fusion(bm25_results, faiss_results)

    retrieved = []
    for idx, score in fused[:top_k]:
        paper = papers[idx].copy()
        paper["rrf_score"] = round(score, 6)
        retrieved.append(paper)
    return retrieved