import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Config:
    # Root directory
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    CORPUS_FILE = DATA_DIR / "corpus" / "corpus.jsonl"
    PUBLIC_EVAL_FILE = DATA_DIR / "questions" / "eval_public.jsonl"
    HIDDEN_EVAL_FILE = DATA_DIR / "questions" / "eval_hidden.jsonl"
    PROCESSED_DIR = DATA_DIR / "processed"
    RESULTS_DIR = PROJECT_ROOT / "benchmark_results"

    # LLM Settings
    _st_secrets = {}
    try:
        import streamlit as _st
        _st_secrets = _st.secrets
    except Exception:
        pass

    GEMINI_API_KEY = (
        os.getenv("GEMINI_API_KEY") 
        or os.getenv("GOOGLE_API_KEY") 
        or os.getenv("GOOGLE_GENAI_API_KEY") 
        or _st_secrets.get("GEMINI_API_KEY", "") 
        or _st_secrets.get("GOOGLE_API_KEY", "")
        or ""
    )
    GEMINI_MODEL = os.getenv("GEMINI_MODEL") or _st_secrets.get("GEMINI_MODEL", "gemini-2.5-flash")

    # TigerGraph Settings
    TIGERGRAPH_HOST = os.getenv("TIGERGRAPH_HOST", "")
    TIGERGRAPH_USERNAME = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")
    TIGERGRAPH_PASSWORD = os.getenv("TIGERGRAPH_PASSWORD", "tigergraph")
    TIGERGRAPH_GRAPHNAME = os.getenv("TIGERGRAPH_GRAPHNAME", "OlympicGraph")
    TIGERGRAPH_SECRET = os.getenv("TIGERGRAPH_SECRET", "")
    TIGERGRAPH_API_TOKEN = os.getenv("TIGERGRAPH_API_TOKEN", "")

    # Vector / Embedding Settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # AVI (Agentic Value Index) Hyperparameters
    AVI_EPSILON = 0.15          # Stopping threshold for marginal information gain
    AVI_LATENCY_WEIGHT = 0.05   # Lambda penalty for latency
    MAX_AGENT_HOPS = 6          # Maximum investigation depth

    @classmethod
    def ensure_dirs(cls):
        cls.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
