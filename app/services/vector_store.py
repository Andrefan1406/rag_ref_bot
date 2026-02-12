import json
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

from ..config import FAISS_PATH, META_PATH, VEC_PATH

MAX_FEATURES = 20000  # чтобы не раздувало память

def load_meta() -> list[dict]:
    if not META_PATH.exists():
        return []
    metas = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

def save_meta(metas: list[dict]) -> None:
    with open(META_PATH, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

def fit_vectorizer(texts: list[str]) -> TfidfVectorizer:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=MAX_FEATURES)
    vec.fit(texts)
    return vec

def vectorizer_to_json(vec: TfidfVectorizer) -> dict:
    return {
        "vocabulary_": {k: int(v) for k, v in vec.vocabulary_.items()},
        "idf_": vec.idf_.tolist(),
        "ngram_range": list(vec.ngram_range),
        "min_df": vec.min_df,
        "max_features": vec.max_features,
        "token_pattern": vec.token_pattern,
        "lowercase": vec.lowercase,
    }

def vectorizer_from_json(d: dict) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        ngram_range=tuple(d["ngram_range"]),
        min_df=d["min_df"],
        max_features=d["max_features"],
        token_pattern=d["token_pattern"],
        lowercase=d["lowercase"],
    )
    vec.vocabulary_ = d["vocabulary_"]
    vec.idf_ = np.array(d["idf_"], dtype=np.float64)
    vec._tfidf._idf_diag = None
    return vec

def save_vectorizer(vec: TfidfVectorizer) -> None:
    with open(VEC_PATH, "w", encoding="utf-8") as f:
        json.dump(vectorizer_to_json(vec), f, ensure_ascii=False)

def load_vectorizer():
    if not VEC_PATH.exists():
        return None
    with open(VEC_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return vectorizer_from_json(d)

def rebuild_index_from_meta() -> dict:
    metas = load_meta()
    if not metas:
        return {"ok": True, "message": "No chunks yet"}

    texts = [m["text"] for m in metas]
    vec = fit_vectorizer(texts)
    X = vec.transform(texts).astype(np.float32)

    # ВАЖНО: делаем плотное представление, но ограничили max_features
    X_dense = X.toarray()
    faiss.normalize_L2(X_dense)

    dim = X_dense.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X_dense)

    faiss.write_index(index, str(FAISS_PATH))
    save_vectorizer(vec)

    return {"ok": True, "chunks": len(metas), "dim": dim}

def search(query: str, top_k: int = 5) -> list[dict]:
    if not (FAISS_PATH.exists() and META_PATH.exists() and VEC_PATH.exists()):
        return []
    index = faiss.read_index(str(FAISS_PATH))
    vec = load_vectorizer()
    metas = load_meta()
    if vec is None or not metas:
        return []

    # TF-IDF -> dense
    q = vec.transform([query]).astype(np.float32).toarray()
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


def add_chunks(chunks: list[str], meta_common: dict) -> dict:
    """
    Добавляет чанки в META и пересобирает индекс.
    meta_common: например {"source_file": "...", "title": "...", "year": 2022}
    """
    metas = load_meta()

    start_id = 0
    # chunk_id внутри одного файла обычно начинается с 0, но это не принципиально.
    # Мы сохраним chunk_id так, как пришёл список.
    for i, ch in enumerate(chunks):
        metas.append({
            **meta_common,
            "chunk_id": i,
            "text": ch
        })

    save_meta(metas)
    return rebuild_index_from_meta()
