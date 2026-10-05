import streamlit as st
import requests
import pandas as pd
from streamlit_searchbox import st_searchbox

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Movie Recommender", layout="wide", page_icon="🎬")

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background: #0e0e14;
    }

    h1 {
        color: #f0f0f5;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #83839a;
        margin-bottom: 1.8rem;
    }

    div[data-baseweb="input"] {
        background: #1a1a24 !important;
        border: 1px solid #2c2c3a !important;
        border-radius: 8px !important;
    }
    input { color: #fff !important; }

    div.stButton > button[kind="primary"] {
        background: #e63946;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #d32f3e;
    }

    div.stButton > button:not([kind="primary"]) {
        background: #1a1a24;
        color: #c8c8d4;
        border: 1px solid #2c2c3a;
        border-radius: 20px;
        font-size: 0.82rem;
        padding: 0.25rem 0.8rem;
    }
    div.stButton > button:not([kind="primary"]):hover {
        border-color: #e63946;
        color: #e63946;
    }

    .movie-card {
        background: #15151e;
        border: 1px solid #22222e;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.75rem;
    }
    .movie-rank {
        color: #e63946;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .movie-name {
        color: #fff;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0.1rem 0 0.45rem 0;
    }
    .badge {
        display: inline-block;
        background: #22222e;
        color: #a8a8b8;
        border-radius: 5px;
        padding: 0.1rem 0.5rem;
        font-size: 0.76rem;
        margin-right: 0.35rem;
    }
    .badge-match {
        background: #e6394622;
        color: #e63946;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommender")
st.markdown('<div class="subtitle">Search a movie you like to get similar recommendations.</div>', unsafe_allow_html=True)

if "search_title" not in st.session_state:
    st.session_state.search_title = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

def search_titles(searchterm: str):
    st.session_state.last_query = searchterm
    if not searchterm or len(searchterm) < 2:
        return []
    try:
        resp = requests.get(f"{API_URL}/titles", params={"q": searchterm, "limit": 8}, timeout=5)
        return resp.json()
    except requests.exceptions.RequestException:
        return []

col1, col2 = st.columns([5, 1])
with col1:
    picked = st_searchbox(
        search_titles,
        placeholder="Type a movie title... (e.g. Mission)",
        key="movie_searchbox",
        clear_on_submit=False,
    )
with col2:
    search_clicked = st.button("Recommend", type="primary", use_container_width=True)

if "last_picked" not in st.session_state:
    st.session_state.last_picked = None

if search_clicked and st.session_state.last_query.strip():
    st.session_state.search_title = st.session_state.last_query.strip()
    st.session_state.last_picked = picked
elif picked and picked != st.session_state.last_picked:
    st.session_state.last_picked = picked
    st.session_state.search_title = picked

n_results = st.slider("Number of recommendations", 5, 20, 10)

if st.session_state.search_title:
    title = st.session_state.search_title
    st.markdown(f"### Because you liked **{title}**")

    payload = {"title": title, "n": n_results}
    with st.spinner("Finding great matches..."):
        response = requests.post(f"{API_URL}/recommend", json=payload)
    data = response.json()

    if isinstance(data, dict) and "error" in data:
        st.error(data["error"])
    else:
        left, right = st.columns(2)
        for i, movie in enumerate(data, start=1):
            year = str(movie.get("release_date", ""))[:4] or "—"
            rating = movie.get("vote_average", 0)
            votes = int(movie.get("vote_count", 0))
            match_pct = round(movie.get("similarity", 0) * 100)

            target = left if i % 2 else right
            target.markdown(f"""
            <div class="movie-card">
                <div class="movie-rank">#{i}</div>
                <div class="movie-name">{movie['title']}</div>
                <span class="badge">⭐ {rating:.1f}</span>
                <span class="badge">{year}</span>
                <span class="badge">{votes:,} votes</span>
                <span class="badge badge-match">{match_pct}% match</span>
            </div>
            """, unsafe_allow_html=True)