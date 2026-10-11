# 🎬 Movie Recommender System

A content-based movie recommendation system that combines **TF-IDF lexical similarity, BERT-based semantic embeddings (MiniLM) and metadata-driven reranking**. It runs on a dataset of 1.2M+ movies that I scraped and built from the TMDB API, and is served through a **FastAPI** backend with a **Streamlit** frontend that offers live search-as-you-type.

## Overview

Keyword-based recommenders often return noisy results: titles that share words but differ in plot, tone or context. This project uses a hybrid pipeline instead. It compares plot descriptions and metadata, filters out unsuitable candidates, and reranks the results using franchise relationships, cast and director overlap, genres, keywords and voting statistics.

It covers the full workflow: dataset collection and preprocessing, feature and embedding generation, recommendation logic, a REST API and a user-facing app.

## Key Features

* **Self-built dataset:** 1.2M+ movie records collected and enriched through the TMDB API.
* **Hybrid similarity:** TF-IDF combined with BERT-based `all-MiniLM-L6-v2` sentence embeddings of plot overviews.
* **Semantic matching:** finds related plots even when they share no vocabulary.
* **Metadata-aware reranking:** franchise/sequel detection, cast and director overlap, genre and keyword overlap, and vote-count weighting.
* **Candidate filtering:** released status, minimum vote count, minimum runtime and genre overlap.
* **FastAPI backend:** recommendation and autocomplete endpoints. Data and indexes are loaded once at startup.
* **Streamlit frontend:** live search-as-you-type and ranked result cards showing rating, year, vote count and match score.

## Tech Stack

| Technology | Purpose |
| --- | --- |
| Python | Core implementation |
| Pandas, NumPy, SciPy | Data processing and sparse/dense numerics |
| scikit-learn | TF-IDF vectorization |
| Sentence-Transformers (`all-MiniLM-L6-v2`) | Plot-overview embeddings |
| FastAPI + Uvicorn | Backend API |
| Streamlit | Frontend |
| Joblib, Parquet, NumPy `.npy` | Storing precomputed artifacts |
| TMDB API | Source of movie and TV metadata |

## How It Works

```text
 TMDB API ─► collection & enrichment ─► cleaned movie dataset
                                              │
                      ┌───────────────────────┴──────────────────────┐
                      ▼                                              ▼
        TF-IDF (title, keywords,                     MiniLM embeddings of plot
        cast, director)                              overviews (L2-normalised)
                      └───────────────────────┬──────────────────────┘
                                              ▼
                        hybrid similarity = 0.15 · TF-IDF + 0.85 · semantic
                                              ▼
                          candidate filtering  →  metadata reranking
                                              ▼
                                     top-N recommendations
                                              ▼
                                 FastAPI  ──►  Streamlit UI
```

### 1. Dataset

Built by scraping the TMDB API with multithreaded requests, retries with backoff and checkpointed resumption, then cleaning, normalising and deduplicating the records into one enriched corpus (overview, genres, keywords, cast, director, ratings and more). The scraping code is not part of this repository.

### 2. TF-IDF similarity

Unigram TF-IDF (capped at 50,000 features) over titles, keywords, cast and directors. Cosine similarity finds candidates that share meaningful terms or people.

### 3. Semantic similarity

Plot overviews are encoded with `all-MiniLM-L6-v2`, a distilled BERT-based sentence-transformer. Embeddings are generated on a GPU and aligned to the dataset on `(id, content_type)`. This matters because movie and TV IDs overlap in TMDB, so joining on `id` alone silently attaches the wrong embedding.

### 4. Hybrid score

```
similarity = 0.15 × TF-IDF + 0.85 × semantic
```

Short synopses share too little vocabulary for TF-IDF to carry the ranking, so the semantic component is weighted more heavily. Both vectors are L2-normalised once at startup, so similarity is a plain dot product.

### 5. Candidate filtering

* Released status
* Minimum vote count and minimum runtime (removes shorts and obscure stubs)
* At least two shared genres, so one generic tag cannot qualify a candidate

### 6. Metadata reranking

| Signal | Effect |
| --- | --- |
| Franchise detection | Large boost for sequels and franchise entries, injected as guaranteed candidates even if their text similarity is low |
| Director / cast overlap | Bonus scaled to the similarity score, so a weak match cannot be lifted by a shared actor |
| Genre / keyword overlap | Small proportional bonuses |
| Vote-count weighting | Scales metadata bonuses by how much voting evidence exists |
| Rating | Small quality contribution |

## Repository Structure

```text
├── api.py                          # FastAPI backend
├── frontend.py                     # Streamlit frontend
├── recommender.ipynb               # Feature building and recommendation prototype
├── bert_embedding_generation.ipynb # MiniLM embedding generation (GPU)
├── requirements.txt
└── README.md
```

## Getting Started

### Requirements

* Python 3.10+
* About 16 GB of RAM (the embeddings and TF-IDF matrix are held in memory)
* The four data artifacts listed below

### 1. Install

```bash
git clone https://github.com/taarushgaba/movie-recommendation-system.git
cd movie-recommendation-system
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 2. Data artifacts

The data files are several GB, so they are **not** included in this repository. Generate them with the notebooks and place them next to `api.py`:

| File | Produced by |
| --- | --- |
| `recommendation_df_slim.parquet` | `recommender.ipynb` |
| `tfidf_matrix.joblib` | `recommender.ipynb` |
| `title_indices.joblib` | `recommender.ipynb` |
| `overview_embeddings_aligned.npy` | `bert_embedding_generation.ipynb` output, aligned to the dataset in `recommender.ipynb` on `(id, content_type)` |

### 3. Run

Start the backend. The first start takes a while because it loads several GB of data:

```bash
uvicorn api:app
```

In a second terminal:

```bash
streamlit run frontend.py
```

The API runs at `http://127.0.0.1:8000` (interactive docs at `/docs`) and the app at `http://localhost:8501`.

## API

| Endpoint | Description |
| --- | --- |
| `GET /titles?q=mission&limit=8` | Title autocomplete, prefix matches first |
| `POST /recommend` | Top-N similar movies |

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"title": "The Godfather", "n": 10}'
```

## Limitations

* Similarity comes from plot overviews, cast and keywords, so tone, pacing and visual style are not captured.
* Films with vague or generic synopses get looser matches than films with distinctive plots.
* Franchise detection is title-based, so connected films with unrelated titles (such as a prequel with a different name) can be missed.
* Reranking weights are hand-tuned heuristics. No quantitative evaluation (Precision@K, Recall@K) has been run yet.
* Memory-heavy: similarity is computed against all 1.2M rows per query.

## Future Improvements

* Evaluate with Precision@K / Recall@K and benchmark against a TF-IDF-only baseline
* Embed overview, tagline and keywords together
* Try a larger embedding model such as `all-mpnet-base-v2`
* Approximate nearest-neighbour search (e.g. FAISS) instead of scoring every row
* Franchise detection from TMDB collection metadata
* Automated tests for the recommender and API

## Acknowledgements

Movie metadata comes from [The Movie Database (TMDB)](https://www.themoviedb.org/). This project is not affiliated with or endorsed by TMDB.
