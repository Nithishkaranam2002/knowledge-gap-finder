from collections import Counter
from datetime import datetime


def explain_gap(cluster):
    papers = cluster["papers"]
    label = cluster["label"]
    gap_score = cluster["gap_score"]
    paper_count = cluster["paper_count"]
    trend = cluster["trend"]
    citation_demand = cluster["citation_demand"]
    venue_prestige = cluster["venue_prestige"]

    current_year = datetime.now().year
    years = [p.get("year") for p in papers if p.get("year")]
    avg_year = round(sum(years) / len(years)) if years else current_year
    latest_year = max(years) if years else current_year

    citations = [p.get("citations", 0) for p in papers]
    avg_citations = round(sum(citations) / len(citations)) if citations else 0

    top_papers = sorted(papers, key=lambda x: x.get("citations", 0), reverse=True)[:3]
    top_titles = [p["title"] for p in top_papers]

    density_note = get_density_note(paper_count)
    trend_note = get_trend_note(trend)
    citation_note = get_citation_note(avg_citations)

    explanation = {
        "cluster_id":       cluster["cluster_id"],
        "label":            label,
        "gap_score":        gap_score,
        "summary":          build_summary(label, paper_count, density_note, trend_note, citation_note, latest_year),
        "why_gap":          build_why_gap(paper_count, avg_citations, trend, venue_prestige),
        "top_papers":       top_titles,
        "stats": {
            "paper_count":      paper_count,
            "avg_year":         avg_year,
            "latest_year":      latest_year,
            "avg_citations":    avg_citations,
            "trend":            trend,
            "citation_demand":  citation_demand,
            "venue_prestige":   venue_prestige,
        }
    }
    return explanation


def get_density_note(paper_count):
    if paper_count <= 2:
        return "very few papers exist"
    elif paper_count <= 5:
        return "limited research exists"
    elif paper_count <= 10:
        return "moderate research exists"
    else:
        return "substantial research exists"


def get_trend_note(trend):
    if trend > 0.5:
        return "rapidly growing research area"
    elif trend > 0:
        return "slowly growing research area"
    elif trend == 0:
        return "stagnant research area"
    else:
        return "declining research area"


def get_citation_note(avg_citations):
    if avg_citations >= 500:
        return "very high citation demand"
    elif avg_citations >= 100:
        return "high citation demand"
    elif avg_citations >= 20:
        return "moderate citation demand"
    else:
        return "low citation demand"


def build_summary(label, paper_count, density_note, trend_note, citation_note, latest_year):
    return (
        f"This cluster focuses on '{label}'. "
        f"Currently {density_note} in this area ({paper_count} papers), "
        f"with the most recent publication in {latest_year}. "
        f"It is a {trend_note} with {citation_note}."
    )


def build_why_gap(paper_count, avg_citations, trend, venue_prestige):
    reasons = []
    if paper_count <= 5:
        reasons.append("very few papers have explored this topic")
    if avg_citations < 20:
        reasons.append("existing work has low citation impact suggesting unrecognized potential")
    if trend < 0:
        reasons.append("research activity has declined recently leaving open questions")
    if venue_prestige < 0.3:
        reasons.append("limited presence in top venues indicates underexplored territory")
    if not reasons:
        reasons.append("the topic shows potential for deeper investigation despite existing work")
    return " and ".join(reasons).capitalize() + "."


def explain_all_gaps(ranked_clusters):
    explanations = []
    for cluster in ranked_clusters:
        exp = explain_gap(cluster)
        explanations.append(exp)
    return explanations