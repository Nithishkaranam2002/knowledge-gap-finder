# Knowledge Gap Finder

A research discovery system that helps students and researchers find under-explored 
areas in any academic field using Information Retrieval and Natural Language Processing.

---

## What This Project Does

When you want to start a new research project, the hardest part is figuring out 
what has not been studied yet. You normally have to manually read through hundreds 
of papers over days or weeks just to find a gap.

This system automates that entire process. You type any research topic, and within 
60 seconds it tells you exactly which subtopics are under-explored and why — ranked 
by a Gap Score we designed using four signals: research density, publication trend, 
citation demand, and venue prestige.

---

## How It Works

ou type a topic
↓
Fetch papers from Semantic Scholar + ArXiv APIs
↓
Convert abstracts to 384-dimensional vectors (SentenceTransformers)
↓
Build BM25 keyword index + FAISS vector index
↓
Hybrid search using Reciprocal Rank Fusion
↓
Group papers into 5 subtopic clusters (KMeans)
↓
Calculate Gap Score for each cluster
↓
Generate plain English explanations
↓
Display results on web dashboard

---

## Gap Score Formula

The core of our system. Each cluster gets a score between 0 and 1:
Gap Score = (1 - density)      × 0.35
+ trend_normalized   × 0.25
+ citation_demand    × 0.25
+ venue_prestige     × 0.15

- **Research Density** — fewer papers = bigger gap (35% weight)
- **Publication Trend** — growing or recently abandoned areas signal opportunity (25% weight)
- **Citation Demand** — highly cited but small cluster = underexplored important area (25% weight)
- **Venue Prestige** — low presence in top venues = underrecognized area (15% weight)

---

## Project Structure
knowledge-gap-finder/
│
├── src/knowledge_gap_finder/
│   ├── fetcher.py       # Fetch papers from Semantic Scholar + ArXiv, cache to SQLite
│   ├── embedder.py      # Generate semantic embeddings using SentenceTransformers
│   ├── indexer.py       # Build BM25 keyword index and FAISS vector index
│   ├── retriever.py     # Hybrid retrieval using Reciprocal Rank Fusion
│   ├── clusterer.py     # KMeans topic clustering on paper embeddings
│   ├── scorer.py        # Gap Score calculation and evaluation metrics
│   └── explainer.py     # Plain language gap explanation generation
│
├── frontend/
│   └── index.html       # Full web UI with sidebar navigation
│
├── data/
│   └── cache/           # SQLite database and index files (auto-created)
│
├── tests/               # Unit tests
├── api.py               # FastAPI backend
├── pipeline.py          # End-to-end pipeline orchestration
├── main.py              # CLI entry point
├── pyproject.toml       # Project dependencies
└── README.md

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Data Sources | Semantic Scholar API + ArXiv API |
| Local Storage | SQLite |
| Keyword Search | BM25 (rank-bm25) |
| Vector Search | FAISS (faiss-cpu) |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Clustering | KMeans (scikit-learn) |
| Result Merging | Reciprocal Rank Fusion |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML + CSS + JavaScript |

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Nithishkaranam2002/knowledge-gap-finder.git
cd knowledge-gap-finder
```

### 2. Create virtual environment

```bash
uv venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
uv pip install requests sentence-transformers faiss-cpu rank-bm25 bertopic pandas numpy matplotlib plotly fastapi uvicorn python-multipart
```

### 4. Run the web application

```bash
.venv/bin/uvicorn api:app --reload --port 8000
```

Open your browser and go to:
http://127.0.0.1:8000

### 5. Or run from command line

```bash
.venv/bin/python main.py "your research topic" --limit 100 --top-k 10
```

---

## Example Results

Query: **"federated learning privacy"**

| Gap | Score | Topic | Papers | Trend |
|---|---|---|---|---|
| Gap 1 | 0.6975 | models, quantum, machine, learning | 3 | +1.0 growing |
| Gap 2 | 0.4788 | distances, minimax, different, pairwise | 3 | -0.75 declining |
| Gap 3 | 0.455 | data, machine, learning, statistics | 4 | -0.8 declining |
| Gap 4 | 0.4367 | fairness, learning, machine, fair | 6 | -0.67 declining |
| Gap 5 | 0.3703 | learning, machine, models, data | 9 | -0.78 declining |

Evaluation Metrics:
- Mean Gap Score: 0.4877
- Top Cluster Avg: 0.5881
- Bottom Cluster Avg: 0.4207
- Score Separation: 0.1674

---

## Web UI Features

- **Dashboard** — Search interface with stats bar and gap cards ranked by score
- **Clusters** — Table view comparing all 5 clusters side by side
- **Analytics** — Evaluation metrics and bar charts for gap scores and paper counts
- **Retrieved Papers** — Top papers from hybrid search with RRF scores

---

## CLI Options

```bash
.venv/bin/python main.py "topic" --limit 100 --top-k 10 --refresh
```

| Option | Default | Description |
|---|---|---|
| topic | required | Research topic to search |
| --limit | 100 | Number of papers to fetch |
| --top-k | 10 | Number of retrieved papers to show |
| --refresh | False | Force refresh cache |

---

## Evaluation

We evaluated the system across three test queries:
- machine learning fairness
- knowledge graph embedding  
- federated learning privacy

Results showed consistent score separation between top and bottom ranked clusters 
(average gap of 0.1674), confirming the Gap Score formula successfully differentiates 
high-gap from low-gap research areas.

Estimated metrics based on manual review:
- Precision: 0.75
- Recall: 0.70
- F1 Score: 0.72

---

## Limitations

- Corpus limited to 50-100 papers per query due to free API rate limits
- ArXiv API does not provide citation counts so those papers show zero citations
- Cluster count is fixed at 5 for all queries
- Evaluation based on manual team review, not formal expert annotation

---

## Team

| Name | Student ID |
|---|---|
| Nithish Karanam | 11823599 |
| Aravind Reddy Janke | 11857421 |
| Rahul Kalapala | 11791251 |

**Course:** CSCE 5200 - Information Retrieval
**University:** University of North Texas
**Semester:** Spring 2026

---

## References

- Blei, D. M., Ng, A. Y., Jordan, M. I. (2003). Latent Dirichlet Allocation. JMLR.
- Ostendorff, T. et al. (2020). Document-level Definition Detection in Scholarly Documents.
- Swanson, D. R. (1986). Fish oil, Raynaud's Syndrome, and Undiscovered Public Knowledge.
- Reimers, N. and Gurevych, I. (2019). Sentence-BERT. EMNLP 2019.
- Johnson, J., Douze, M., Jegou, H. (2019). Billion-scale similarity search with GPUs.
- Robertson, S. and Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
