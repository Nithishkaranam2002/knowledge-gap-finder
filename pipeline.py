from src.knowledge_gap_finder.fetcher import fetch_papers
from src.knowledge_gap_finder.embedder import get_embeddings
from src.knowledge_gap_finder.indexer import build_index
from src.knowledge_gap_finder.retriever import retrieve
from src.knowledge_gap_finder.clusterer import cluster_topics
from src.knowledge_gap_finder.scorer import rank_gaps, evaluate_rankings
from src.knowledge_gap_finder.explainer import explain_all_gaps


def run_pipeline(query, limit=100, top_k=10, force_refresh=False):
    print(f"\n=== Knowledge Gap Finder ===")
    print(f"Query: '{query}'")

    papers = fetch_papers(query, limit=limit, force_refresh=force_refresh)
    if not papers:
        return {"error": "No papers found for this query"}

    embeddings = get_embeddings(papers, force_refresh=force_refresh)
    bm25, faiss_index, indexed_papers = build_index(papers, embeddings, force_refresh=force_refresh)
    retrieved = retrieve(query, bm25, faiss_index, indexed_papers, top_k=top_k)
    clusters = cluster_topics(indexed_papers, embeddings, force_refresh=force_refresh)
    ranked = rank_gaps(clusters)
    explanations = explain_all_gaps(ranked)
    metrics = evaluate_rankings(ranked)

    print(f"\nDone! Found {len(explanations)} knowledge gaps")
    return {
        "query":        query,
        "total_papers": len(papers),
        "retrieved":    retrieved,
        "gaps":         explanations,
        "metrics":      metrics
    }


def print_results(results):
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print(f"\n{'='*60}")
    print(f"Query: {results['query']}")
    print(f"Total papers indexed: {results['total_papers']}")
    print(f"{'='*60}")

    print(f"\nTop Retrieved Papers:")
    for i, p in enumerate(results["retrieved"][:5], 1):
        print(f"  {i}. [{p['rrf_score']}] {p['title']}")

    print(f"\nKnowledge Gaps Discovered:")
    for i, gap in enumerate(results["gaps"], 1):
        print(f"\n  Gap #{i} | Score: {gap['gap_score']}")
        print(f"  Topic: {gap['label']}")
        print(f"  {gap['summary']}")
        print(f"  Why a gap: {gap['why_gap']}")
        if gap["top_papers"]:
            print(f"  Key papers: {gap['top_papers'][0]}")

    print(f"\nEvaluation Metrics:")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v}")