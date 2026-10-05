from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import math, re, joblib, time

app = FastAPI()

print("Loading data...")
t0 = time.time()
df = pd.read_parquet("recommendation_df_slim.parquet")
print(f"parquet loaded in {time.time()-t0:.1f}s, shape={df.shape}")

t1 = time.time()
tfidf_matrix = joblib.load("tfidf_matrix.joblib")
print(f"tfidf loaded in {time.time()-t1:.1f}s, type={type(tfidf_matrix)}")

t2 = time.time()
indices = joblib.load("title_indices.joblib")
print(f"indices loaded in {time.time()-t2:.1f}s")

t3 = time.time()
overview_embeddings_aligned = np.load("overview_embeddings_aligned.npy").astype(np.float32)
# Pre-normalize once so similarity is a plain dot product (no sklearn upcast/copy per request)
_norms = np.linalg.norm(overview_embeddings_aligned, axis=1, keepdims=True)
_norms[_norms == 0] = 1
overview_embeddings_aligned /= _norms
print(f"embeddings loaded in {time.time()-t3:.1f}s, shape={overview_embeddings_aligned.shape}")
print("Loaded.")

# tfidf_matrix already L2-normalized by TfidfVectorizer (default norm='l2'),
# so its dot product with a query row is already cosine similarity.

valid = (
    (df["vote_count"].fillna(0) >= 100) &
    (df["content_type"] == "movie") &
    (df["status"].fillna("").str.lower() == "released")
)

# Precomputed lookup for fast autocomplete: one row per unique title,
# keeping the highest vote_count version, sorted by popularity.
movie_df = df[df["content_type"] == "movie"]
_title_lookup = (
    movie_df.sort_values("vote_count", ascending=False)
    .drop_duplicates(subset="title")
    [["title", "vote_count"]]
    .reset_index(drop=True)
)
_title_lookup["title_lower"] = _title_lookup["title"].str.lower()

def get_franchise_base(title):
    """Return the index of the base movie in the franchise group."""
    key = franchise_key(title)
    group = _franchise_groups.get(key, [])
    if not group:
        return None
    # Pick the one with highest vote_count (or earliest release_date if you prefer)
    return max(group, key=lambda i: df.iloc[i]["vote_count"])


def clean_tokens(value):
    if pd.isna(value) or not str(value).strip():
        return set()
    value = str(value).lower().replace("|", " ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return set(value.split())

def clean_keyword_tokens(value):
    if pd.isna(value) or not str(value).strip():
        return set()
    parts = str(value).lower().split("|")
    return {re.sub(r"[^a-z0-9\s]", "", p.strip()) for p in parts if p.strip()}

def top_cast(value, k=5):
    if pd.isna(value) or not str(value).strip():
        return set()
    names = str(value).split("|")[:k]
    cleaned = set()
    for n in names:
        n = re.sub(r"[^a-z0-9\s]", "", n.lower().strip())
        if n:
            cleaned.add(n)
    return cleaned

def is_same_franchise(title_a, title_b, n_words=3):
    def clean(t):
        words = str(t).lower().split()
        cleaned = []
        for w in words:
            w = re.sub(r"[^a-z0-9]", "", w)
            if w and w not in {"the", "a", "an"}:
                cleaned.append(w)
        return cleaned
    a, b = clean(title_a), clean(title_b)
    if not a or not b:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    # The shorter title's words must be (almost) entirely the shared prefix —
    # not just any N words — so a coincidental partial phrase doesn't match.
    prefix_len = min(n_words, len(shorter))
    if prefix_len < len(shorter) - 1:  # shorter title has extra words beyond the prefix
        return False
    if len(shorter) == 1:
        # A single generic word ("captain", "american") isn't a reliable
        # franchise signal unless it's genuinely rare as a title-starter.
        group = _franchise_groups.get(shorter[0], [])
        if len(group) > 20:
            return False
    shorter_prefix = " ".join(shorter[:prefix_len])
    longer_prefix = " ".join(longer[:prefix_len])
    return shorter_prefix == longer_prefix

def franchise_key(title):
    words = [w for w in str(title).lower().split() if w not in {"the", "a", "an"}]
    if not words:
        return ""
    return re.sub(r"[^a-z0-9]", "", words[0])

# Precompute a cheap grouping key so same-franchise titles can be pulled in
# as guaranteed candidates, even if their raw similarity rank is low.
df["_franchise_key"] = df["title"].apply(franchise_key)
_franchise_groups = df.groupby("_franchise_key").indices

def hybrid_recommend(title, n=10, candidate_pool=300,
                      tfidf_weight=0.15, semantic_weight=0.85):
    title_clean = title.lower().strip()
    if title_clean not in indices:
        return None

    matching_indices = [i for i in indices[title_clean] if df.iloc[i]["content_type"] == "movie"]
    if not matching_indices:
        return None

    idx = max(matching_indices, key=lambda i: df.iloc[i]["vote_count"])

    tfidf_sims = tfidf_matrix.dot(tfidf_matrix[idx].T).toarray().flatten()
    semantic_sims = overview_embeddings_aligned @ overview_embeddings_aligned[idx]
    sims = tfidf_weight * tfidf_sims + semantic_weight * semantic_sims

    top_k = min(candidate_pool * 3, len(sims) - 1)
    top_unsorted = np.argpartition(sims, -top_k)[-top_k:]
    candidate_indices = top_unsorted[np.argsort(sims[top_unsorted])[::-1]]

    q_genre = clean_tokens(df.iloc[idx]["genres"])
    query_title = df.iloc[idx]["title"]

    min_genre_overlap = min(2, len(q_genre)) if q_genre else 0
    candidates = [i for i in candidate_indices
                  if i not in matching_indices and valid.iloc[i]
                  and df.iloc[i]["vote_count"] >= 50
                  and (pd.isna(df.iloc[i].get("runtime")) or df.iloc[i]["runtime"] >= 40)
                  and len(clean_tokens(df.iloc[i]["genres"]) & q_genre) >= min_genre_overlap][:candidate_pool]

    # Guarantee real franchise entries are considered even if their raw
    # similarity rank falls outside the top candidate window.
    query_key = franchise_key(query_title)
    if query_key:
        # key match is just a cheap pre-filter; is_same_franchise does the real check
        franchise_rows = [i for i in _franchise_groups.get(query_key, [])
                           if i not in matching_indices and valid.iloc[i]
                           and df.iloc[i]["vote_count"] >= 50
                           and (pd.isna(df.iloc[i].get("runtime")) or df.iloc[i]["runtime"] >= 40)
                           and is_same_franchise(query_title, df.iloc[i]["title"])]
        candidates = list(dict.fromkeys(franchise_rows + candidates))

    q_dir = clean_tokens(df.iloc[idx]["director"])
    q_cast = top_cast(df.iloc[idx]["cast"])
    q_kw = clean_keyword_tokens(df.iloc[idx]["keywords"])

    rows = []
    for i in candidates:
        m = df.iloc[i]
        sim = sims[i]
        director_bonus = 0.15 * sim if (q_dir & clean_tokens(m["director"])) else 0
        cast_bonus = 0.08 * sim * min(len(q_cast & top_cast(m["cast"])), 2)
        genre_bonus = 0.10 * sim * (len(q_genre & clean_tokens(m["genres"])) / max(len(q_genre), 1))
        kw_bonus = 0.05 * sim * min(len(q_kw & clean_keyword_tokens(m["keywords"])), 3)
        franchise_bonus = 0.35 if is_same_franchise(query_title, m["title"]) else 0
        quality_factor = min(math.log10(m["vote_count"] + 1) / math.log10(1000), 1)
        quality = min(m["vote_average"] / 10, 1) * 0.05 if m["vote_average"] else 0
        score = sim + (director_bonus + cast_bonus + genre_bonus + kw_bonus) * quality_factor + quality + franchise_bonus
        rows.append((i, score))

    rows.sort(key=lambda x: x[1], reverse=True)
    sel_idx = [x[0] for x in rows[:n]]
    out = df.iloc[sel_idx][["title", "vote_average", "vote_count", "release_date"]].copy()
    out["similarity"] = [sims[i] for i in sel_idx]
    out["hybrid_score"] = [x[1] for x in rows[:n]]
    return out.reset_index(drop=True)


class RecommendRequest(BaseModel):
    title: str
    n: int = 10


@app.get('/')
def home():
    return {'message': 'Movie recommender API'}


@app.get('/debug/franchise-size')
def debug_franchise_size(word: str):
    word = word.lower().strip()
    group = _franchise_groups.get(word, [])
    return {"word": word, "group_size": len(group)}


@app.get('/titles')
def titles(q: str = "", limit: int = 10):
    q = q.lower().strip()
    if not q:
        return []
    matches = _title_lookup[_title_lookup["title_lower"].str.startswith(q)]
    if len(matches) < limit:
        contains = _title_lookup[
            _title_lookup["title_lower"].str.contains(q, regex=False)
            & ~_title_lookup["title_lower"].str.startswith(q)
        ]
        matches = pd.concat([matches, contains])
    return matches["title"].head(limit).tolist()


@app.post('/recommend')
def recommend(data: RecommendRequest):
    result = hybrid_recommend(data.title, n=data.n)
    if result is None:
        return {"error": f"'{data.title}' not found."}
    return result.to_dict(orient="records")