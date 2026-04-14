import re
import requests
from pathlib import Path
import fitz  # pymupdf

# ---- limits (чтобы не утянуть гигантский PDF в память) ----
MAX_PDF_MB = 50
MAX_PDF_BYTES = MAX_PDF_MB * 1024 * 1024

def safe_filename(name: str, max_len: int = 80) -> str:
    name = name or "paper"
    name = re.sub(r"[^a-zA-Z0-9а-яА-Я._-]+", "_", name).strip("_")
    return (name[:max_len] or "paper") + ".pdf"

def looks_like_pdf(content: bytes, content_type: str | None = None) -> bool:
    ct = (content_type or "").lower()
    head = content[:10]
    return ("pdf" in ct) or head.startswith(b"%PDF-")

def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# -------------------- OLD (disk) --------------------
def download_pdf(url: str, out_path: Path) -> tuple[bool, str]:
    """
    Старый режим: скачать PDF и сохранить на диск.
    Оставляем для совместимости (/upload_pdf и т.п.)
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, timeout=60, allow_redirects=True, headers=headers)
    if r.status_code >= 400:
        return False, f"download status={r.status_code}"
    if not looks_like_pdf(r.content, r.headers.get("Content-Type", "")):
        return False, f"not pdf (content-type={r.headers.get('Content-Type','')})"
    out_path.write_bytes(r.content)
    if out_path.stat().st_size < 5000:
        return False, "файл слишком маленький для pdf"
    return True, "ok"

def extract_text_from_pdf(path: Path) -> str:
    """
    Старый режим: извлечь текст из PDF файла на диске.
    Оставляем для совместимости.
    """
    doc = fitz.open(path)
    pages = []
    for page in doc:
        t = page.get_text("text").strip()
        if t:
            pages.append(t)
    doc.close()
    return _clean_text("\n".join(pages))

# -------------------- NEW (bytes, no-disk) --------------------
def download_pdf_bytes(url: str, timeout: int = 60) -> tuple[bool, str, bytes]:
    """
    Новый режим (Вариант 3): скачать PDF в память (bytes), НЕ сохраняя на диск.
    Есть лимит MAX_PDF_MB.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        with requests.get(url, timeout=timeout, allow_redirects=True, headers=headers, stream=True) as r:
            if r.status_code >= 400:
                return False, f"http {r.status_code}", b""

            # Если сервер отдал Content-Length — отсекаем гигантов заранее
            cl = r.headers.get("Content-Length")
            if cl and cl.isdigit() and int(cl) > MAX_PDF_BYTES:
                return False, f"too large: {int(cl)//1024//1024}MB", b""

            # Быстрая проверка "похоже на PDF" по первым байтам
            first = next(r.iter_content(chunk_size=2048), b"")
            if not looks_like_pdf(first, r.headers.get("Content-Type")):
                return False, "not a pdf", b""

            data = bytearray()
            data.extend(first)

            for chunk in r.iter_content(chunk_size=1024 * 256):  # 256KB
                if not chunk:
                    continue
                data.extend(chunk)
                if len(data) > MAX_PDF_BYTES:
                    return False, f"too large > {MAX_PDF_MB}MB", b""

            if len(data) < 5000:
                return False, "файл слишком маленький для pdf", b""

            return True, "ok", bytes(data)

    except Exception as e:
        return False, type(e).__name__, b""

def extract_text_from_pdf_bytes(content: bytes) -> str:
    """
    Извлечь текст из PDF, который находится в памяти (bytes).
    """
    doc = fitz.open(stream=content, filetype="pdf")
    pages = []
    for page in doc:
        t = page.get_text("text").strip()
        if t:
            pages.append(t)
    doc.close()
    return _clean_text("\n".join(pages))

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
