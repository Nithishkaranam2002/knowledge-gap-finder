import numpy as np
from collections import Counter
from datetime import datetime


def compute_trend(cluster):
    current_year = datetime.now().year
    recent_cutoff = current_year - 2

    years = [p.get("year") for p in cluster["papers"] if p.get("year")]
    if not years:
        return 0.0

    recent = sum(1 for y in years if y >= recent_cutoff)
    older = sum(1 for y in years if y < recent_cutoff)

    if older == 0:
        return 1.0 if recent > 0 else 0.0

    trend = (recent - older) / (older + 1)
    return round(min(max(trend, -1.0), 1.0), 4)


def compute_citation_demand(cluster):
    citations = [p.get("citations", 0) for p in cluster["papers"]]
    if not citations:
        return 0.0
    avg = np.mean(citations)
    normalized = min(avg / 500.0, 1.0)
    return round(float(normalized), 4)


def compute_venue_prestige(cluster):
    top_venues = {
        "nature", "science", "neurips", "icml", "iclr", "acl", "emnlp",
        "cvpr", "iccv", "sigkdd", "www", "sigir", "aaai", "ijcai", "arxiv"
    }
    venues = [p.get("venue", "").lower() for p in cluster["papers"]]
    if not venues:
        return 0.0
    matches = sum(1 for v in venues if any(t in v for t in top_venues))
    prestige = matches / len(venues)
    return round(prestige, 4)


def compute_gap_score(cluster):
    density = min(cluster["paper_count"] / 20.0, 1.0)
    trend = compute_trend(cluster)
    citation_demand = compute_citation_demand(cluster)
    prestige = compute_venue_prestige(cluster)

    trend_normalized = (trend + 1) / 2

    gap_score = (
        (1 - density)          * 0.35 +
        trend_normalized       * 0.25 +
        citation_demand        * 0.25 +
        prestige               * 0.15
    )
    return round(gap_score, 4)


def rank_gaps(clusters):
    scored = []
    for c in clusters:
        score = compute_gap_score(c)
        scored.append({
            "cluster_id":       c["cluster_id"],
            "label":            c["label"],
            "paper_count":      c["paper_count"],
            "gap_score":        score,
            "trend":            compute_trend(c),
            "citation_demand":  compute_citation_demand(c),
            "venue_prestige":   compute_venue_prestige(c),
            "papers":           c["papers"]
        })
    scored.sort(key=lambda x: x["gap_score"], reverse=True)
    return scored


def evaluate_rankings(ranked_clusters, expert_labels=None):
    total = len(ranked_clusters)
    if total == 0:
        return {}

    top_half = ranked_clusters[:total // 2]
    bottom_half = ranked_clusters[total // 2:]

    top_avg_score = np.mean([c["gap_score"] for c in top_half])
    bottom_avg_score = np.mean([c["gap_score"] for c in bottom_half])

    scores = [c["gap_score"] for c in ranked_clusters]
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    if expert_labels:
        predicted = [1 if c["gap_score"] >= mean_score else 0 for c in ranked_clusters]
        actual = expert_labels[:total]
        tp = sum(1 for p, a in zip(predicted, actual) if p == 1 and a == 1)
        fp = sum(1 for p, a in zip(predicted, actual) if p == 1 and a == 0)
        fn = sum(1 for p, a in zip(predicted, actual) if p == 0 and a == 1)
        tn = sum(1 for p, a in zip(predicted, actual) if p == 0 and a == 0)

        precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy   = (tp + tn) / total if total > 0 else 0.0

        return {
            "precision":        round(precision, 4),
            "recall":           round(recall, 4),
            "f1_score":         round(f1, 4),
            "accuracy":         round(accuracy, 4),
            "top_avg_score":    round(float(top_avg_score), 4),
            "bottom_avg_score": round(float(bottom_avg_score), 4),
            "mean_gap_score":   round(float(mean_score), 4),
            "std_gap_score":    round(float(std_score), 4),
        }

    return {
        "top_avg_score":    round(float(top_avg_score), 4),
        "bottom_avg_score": round(float(bottom_avg_score), 4),
        "mean_gap_score":   round(float(mean_score), 4),
        "std_gap_score":    round(float(std_score), 4),
    }