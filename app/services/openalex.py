import requests
from ..config import OPENALEX_WORKS_URL


def looks_like_pdf(content: bytes, content_type: str | None) -> bool:
    ct = (content_type or "").lower()
    head = content[:10]
    return ("pdf" in ct) or head.startswith(b"%PDF-")


def check_pdf_url(pdf_url: str, timeout: int = 20) -> tuple[bool, str]:
    try:
        # 1) пробуем HEAD
        h = requests.head(pdf_url, timeout=timeout, allow_redirects=True)
        if h.status_code >= 400:
            # fallback: небольшой GET (первые байты)
            r = requests.get(
                pdf_url,
                timeout=timeout,
                allow_redirects=True,
                stream=True,
                headers={"Range": "bytes=0-2048"},
            )
            if r.status_code >= 400:
                return False, f"http {r.status_code}"
            chunk = next(r.iter_content(chunk_size=2048), b"")
            return (True, "ok") if looks_like_pdf(chunk, r.headers.get("Content-Type")) else (False, "not a pdf")

        # 2) HEAD успешен, но проверим первые байты (Content-Type бывает врёт)
        r2 = requests.get(
            pdf_url,
            timeout=timeout,
            allow_redirects=True,
            stream=True,
            headers={"Range": "bytes=0-2048"},
        )
        if r2.status_code >= 400:
            return False, f"http {r2.status_code}"
        chunk = next(r2.iter_content(chunk_size=2048), b"")
        return (True, "ok") if looks_like_pdf(chunk, r2.headers.get("Content-Type")) else (False, "not a pdf")

    except Exception as e:
        return False, type(e).__name__


def search_openalex(query: str, n: int = 20, lang: str = "ru", only_oa: bool = True) -> list[dict]:
    filt = []
    if lang:
        filt.append(f"language:{lang}")
    if only_oa:
        filt.append("is_oa:true")

    params = {
        "search": query,
        "filter": ",".join(filt) if filt else None,
        "sort": "cited_by_count:desc",
        "per_page": min(max(n, 1), 200),
    }

    r = requests.get(OPENALEX_WORKS_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    results = []
    for w in data.get("results", []) or []:
        primary = w.get("primary_location") or {}
        ids = w.get("ids") or {}
        results.append(
            {
                "openalex_id": w.get("id"),
                "title": w.get("title"),
                "year": w.get("publication_year"),
                "doi": ids.get("doi"),
                "landing_page_url": primary.get("landing_page_url") or w.get("url"),
                "pdf_url": primary.get("pdf_url"),
                "cited_by_count": w.get("cited_by_count"),
            }
        )
    return results
