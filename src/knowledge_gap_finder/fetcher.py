import requests
import sqlite3
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "cache" / "papers.db"


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id    TEXT PRIMARY KEY,
            title       TEXT,
            abstract    TEXT,
            authors     TEXT,
            year        INTEGER,
            citations   INTEGER,
            venue       TEXT,
            field       TEXT,
            source      TEXT,
            query       TEXT
        )
    """)
    conn.commit()
    conn.close()


def cache_papers(papers, query):
    conn = sqlite3.connect(DB_PATH)
    for p in papers:
        conn.execute("""
            INSERT OR IGNORE INTO papers
            (paper_id, title, abstract, authors, year, citations, venue, field, source, query)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            p.get("paper_id", ""),
            p.get("title", ""),
            p.get("abstract", ""),
            json.dumps(p.get("authors", [])),
            p.get("year"),
            p.get("citations", 0),
            p.get("venue", ""),
            p.get("field", ""),
            p.get("source", ""),
            query,
        ))
    conn.commit()
    conn.close()


def load_cached_papers(query):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT * FROM papers WHERE query = ?", (query,)
    ).fetchall()
    conn.close()
    cols = ["paper_id","title","abstract","authors","year","citations","venue","field","source","query"]
    papers = []
    for row in rows:
        p = dict(zip(cols, row))
        p["authors"] = json.loads(p["authors"])
        papers.append(p)
    return papers


def is_cached(query):
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE query = ?", (query,)
    ).fetchone()[0]
    conn.close()
    return count > 0


def fetch_semantic_scholar(query, limit=100):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "paperId,title,abstract,authors,year,citationCount,venue,fieldsOfStudy"
    }
    papers = []
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("data", []):
            abstract = item.get("abstract") or ""
            if not abstract.strip():
                continue
            papers.append({
                "paper_id":  item.get("paperId", ""),
                "title":     item.get("title", ""),
                "abstract":  abstract,
                "authors":   [a["name"] for a in item.get("authors", [])],
                "year":      item.get("year"),
                "citations": item.get("citationCount", 0),
                "venue":     item.get("venue", ""),
                "field":     (item.get("fieldsOfStudy") or [""])[0],
                "source":    "semantic_scholar",
            })
    except Exception as e:
        print(f"Semantic Scholar error: {e}")
    return papers


def fetch_arxiv(query, limit=100):
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit,
    }
    papers = []
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        ns = "{http://www.w3.org/2005/Atom}"
        root = ET.fromstring(resp.text)
        for entry in root.findall(f"{ns}entry"):
            abstract = (entry.findtext(f"{ns}summary") or "").strip()
            if not abstract:
                continue
            arxiv_id = (entry.findtext(f"{ns}id") or "").split("/")[-1]
            authors = [a.findtext(f"{ns}name") or "" for a in entry.findall(f"{ns}author")]
            published = entry.findtext(f"{ns}published") or ""
            year = int(published[:4]) if published else None
            papers.append({
                "paper_id":  f"arxiv_{arxiv_id}",
                "title":     (entry.findtext(f"{ns}title") or "").strip(),
                "abstract":  abstract,
                "authors":   authors,
                "year":      year,
                "citations": 0,
                "venue":     "arXiv",
                "field":     "",
                "source":    "arxiv",
            })
        time.sleep(3)
    except Exception as e:
        print(f"ArXiv error: {e}")
    return papers


def fetch_papers(query, limit=100, force_refresh=False):
    init_db()
    if not force_refresh and is_cached(query):
        print(f"Loaded from cache: '{query}'")
        return load_cached_papers(query)

    print(f"Fetching papers for: '{query}'")
    papers = []
    papers += fetch_semantic_scholar(query, limit // 2)
    papers += fetch_arxiv(query, limit // 2)

    seen, unique = set(), []
    for p in papers:
        key = p["title"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(p)

    print(f"Total unique papers: {len(unique)}")
    cache_papers(unique, query)
    return unique