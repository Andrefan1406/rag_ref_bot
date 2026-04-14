import json
import numpy as np
import faiss
import httpx
from pathlib import Path
import httpx
import asyncio

from ..config import FAISS_PATH, META_PATH, EMB_PATH, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL

EMBED_URL = f"{OLLAMA_BASE_URL.rstrip('/')}/api/embeddings"
EMBED_MODEL = OLLAMA_EMBED_MODEL

EMBED_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=120.0,
    write=120.0,
    pool=10.0,
)

_client = httpx.AsyncClient(timeout=EMBED_TIMEOUT)


# -------------------------
# META 
# -------------------------
def load_meta() -> list[dict]:
    if not META_PATH.exists():
        return []
    metas = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

def save_meta(metas: list[dict]) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


# -------------------------
# EMBEDDINGS storage
# -------------------------
def load_embeddings() -> np.ndarray | None:
    if not EMB_PATH.exists():
        return None
    return np.load(EMB_PATH).astype(np.float32)

def save_embeddings(emb: np.ndarray) -> None:
    EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, emb.astype(np.float32))


# -------------------------
# OLLAMA embeddings
# -------------------------
async def embed_text(text: str) -> np.ndarray:
    payload = {
        "model": EMBED_MODEL,
        "prompt": text,
    }

    r = await _client.post(EMBED_URL, json=payload)
    r.raise_for_status()

    vec = r.json()["embedding"]
    return np.array(vec, dtype=np.float32)

# async def embed_texts(texts: list[str]) -> np.ndarray:
#     # Последовательно (надёжнее для старта). Потом можно распараллелить.
#     vectors = []
#     for t in texts:
#         vectors.append(await embed_text(t))
#     return np.vstack(vectors).astype(np.float32)
async def embed_texts(texts: list[str], concurrency: int = 6) -> np.ndarray:
    sem = asyncio.Semaphore(concurrency)

    async def _one(t: str):
        async with sem:
            return await embed_text(t)

    vectors = await asyncio.gather(*[_one(t) for t in texts])
    return np.vstack(vectors).astype(np.float32)

# -------------------------
# FAISS index
# -------------------------
def rebuild_faiss_from_embeddings(emb: np.ndarray) -> dict:
    if emb.size == 0:
        return {"ok": True, "message": "No chunks yet"}

    # cosine similarity = inner product по L2-нормированным векторам
    emb = emb.astype(np.float32).copy()
    faiss.normalize_L2(emb)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_PATH))
    return {"ok": True, "chunks": int(emb.shape[0]), "dim": int(dim)}


async def rebuild_index_from_meta() -> dict:
    """
    Полная пересборка: читает META, заново считает эмбеддинги и строит FAISS.
    Использовать если нужно полностью восстановиться.
    """
    metas = load_meta()
    if not metas:
        return {"ok": True, "message": "No chunks yet"}

    texts = [m["text"] for m in metas]
    emb = await embed_texts(texts)
    save_embeddings(emb)
    return rebuild_faiss_from_embeddings(emb)


# -------------------------
# Search (теперь async)
# -------------------------
async def search(query: str, top_k: int = 5) -> list[dict]:
    if not (FAISS_PATH.exists() and META_PATH.exists() and EMB_PATH.exists()):
        return []

    index = faiss.read_index(str(FAISS_PATH))
    metas = load_meta()
    if not metas:
        return []

    q = await embed_text(query)
    q = q.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(q)

    D, I = index.search(q, top_k)

    out = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metas):
            continue
        m = metas[idx]
        out.append({
            "score": float(score),
            "source_file": m.get("source_file"),
            "chunk_id": m.get("chunk_id"),
            "title": m.get("title"),
            "year": m.get("year"),
            "text": m.get("text", "")[:1400],
        })
    return out


# -------------------------
# Add chunks (теперь async)
# -------------------------
async def add_chunks(chunks: list[str], meta_common: dict) -> dict:
    """
    Добавляет чанки + эмбеддинги в файлы и пересобирает индекс.
    """
    metas = load_meta()
    emb_old = load_embeddings()
    if emb_old is None:
        emb_old = np.zeros((0, 1), dtype=np.float32)  # временно, заменим после первых эмбеддингов

    # 1) metas
    start = len(metas)
    for i, ch in enumerate(chunks):
        metas.append({
            **meta_common,
            "chunk_id": i,
            "text": ch
        })

    # 2) embeddings для новых чанков
    new_emb = await embed_texts(chunks)

    # 3) склейка embeddings
    if start == 0:
        emb_all = new_emb
    else:
        # emb_old может быть "пустышкой" (0,1) если файла не было
        emb_old_real = load_embeddings()
        if emb_old_real is None:
            emb_all = new_emb
        else:
            emb_all = np.vstack([emb_old_real, new_emb]).astype(np.float32)

    save_meta(metas)
    save_embeddings(emb_all)

    # 4) rebuild FAISS
    return rebuild_faiss_from_embeddings(emb_all)

def clear_vector_store() -> dict:
    """
    Полностью очищает векторную базу:
    - faiss index
    - meta.jsonl
    - vectorizer.json
    """
    for path in [FAISS_PATH, META_PATH]:
        if path.exists():
            path.unlink()

    return {"ok": True, "message": "База очищена"}
