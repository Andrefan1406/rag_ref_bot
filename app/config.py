from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "index"

PDF_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"

FAISS_PATH = INDEX_DIR / "docs.faiss"
META_PATH = INDEX_DIR / "docs_meta.jsonl"
VEC_PATH = INDEX_DIR / "tfidf_vectorizer.json"
