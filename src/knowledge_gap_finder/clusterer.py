import numpy as np
from bertopic import BERTopic
from sklearn.cluster import KMeans
import pickle
from pathlib import Path

CLUSTER_CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "clusters.pkl"


def cluster_topics(papers, embeddings, num_topics=5, force_refresh=False):
    if not force_refresh and CLUSTER_CACHE.exists():
        print("Loading clusters from cache")
        with open(CLUSTER_CACHE, "rb") as f:
            return pickle.load(f)

    print("Clustering papers into topics...")
    
    n_papers = len(papers)
    n_clusters = min(num_topics, n_papers // 2) if n_papers >= 4 else 2

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    clusters = {}
    for i, label in enumerate(labels):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(papers[i])

    labeled_clusters = []
    for label, cluster_papers in clusters.items():
        words = extract_keywords(cluster_papers)
        labeled_clusters.append({
            "cluster_id":   label,
            "label":        words,
            "papers":       cluster_papers,
            "paper_count":  len(cluster_papers)
        })

    labeled_clusters.sort(key=lambda x: x["paper_count"], reverse=True)

    CLUSTER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CLUSTER_CACHE, "wb") as f:
        pickle.dump(labeled_clusters, f)

    print(f"Created {len(labeled_clusters)} clusters")
    return labeled_clusters


def extract_keywords(papers, top_n=4):
    from collections import Counter
    stopwords = {
        "the","a","an","of","in","and","to","for","is","are","with",
        "on","this","that","we","by","as","from","be","it","our","at",
        "which","was","were","has","have","can","using","used","based",
        "paper","propose","proposed","show","results","also","two","new"
    }
    word_freq = Counter()
    for p in papers:
        words = p.get("abstract", "").lower().split()
        for w in words:
            w = w.strip(".,;:()")
            if len(w) > 3 and w not in stopwords:
                word_freq[w] += 1
    top_words = [w for w, _ in word_freq.most_common(top_n)]
    return " | ".join(top_words)