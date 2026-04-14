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

EMB_PATH = Path("data/embeddings.npy")  # файл с эмбеддингами (N x D)

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
# OLLAMA_LLM_MODEL = "gemma3:1b"
# OLLAMA_LLM_MODEL = "llama3"
# OLLAMA_LLM_MODEL = "llama3.1:8b"
OLLAMA_LLM_MODEL = "llama3.2:3b"
