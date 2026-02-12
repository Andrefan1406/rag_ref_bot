import re
import requests
from pathlib import Path
import fitz  # pymupdf

def safe_filename(name: str, max_len: int = 80) -> str:
    name = name or "paper"
    name = re.sub(r"[^a-zA-Z0-9а-яА-Я._-]+", "_", name).strip("_")
    return (name[:max_len] or "paper") + ".pdf"

def looks_like_pdf(content: bytes, content_type: str) -> bool:
    ct = (content_type or "").lower()
    head = content[:10]
    return ("pdf" in ct) or head.startswith(b"%PDF-")

def download_pdf(url: str, out_path: Path) -> tuple[bool, str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, timeout=60, allow_redirects=True, headers=headers)
    if r.status_code >= 400:
        return False, f"download status={r.status_code}"
    if not looks_like_pdf(r.content, r.headers.get("Content-Type", "")):
        return False, f"not pdf (content-type={r.headers.get('Content-Type','')})"
    out_path.write_bytes(r.content)
    if out_path.stat().st_size < 5000:
        return False, "file too small, likely not a real pdf"
    return True, "ok"

def extract_text_from_pdf(path: Path) -> str:
    doc = fitz.open(path)
    pages = []
    for page in doc:
        t = page.get_text("text").strip()
        if t:
            pages.append(t)
    doc.close()
    text = "\n".join(pages)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        c = text[start:end].strip()
        if c:
            chunks.append(c)
        start = max(end - overlap, 0)
        if end == n:
            break
    return chunks
