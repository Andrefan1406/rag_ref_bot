import asyncio
import httpx
from ..config import OPENALEX_WORKS_URL


def looks_like_pdf(content: bytes, content_type: str | None) -> bool:
    ct = (content_type or "").lower()
    head = content[:10]
    return ("pdf" in ct) or head.startswith(b"%PDF-")


async def check_pdf_url(
    client: httpx.AsyncClient,
    pdf_url: str,
    timeout: float = 20.0,
) -> tuple[bool, str]:
    try:
        # 1) пробуем HEAD
        h = await client.head(pdf_url, timeout=timeout, follow_redirects=True)
        if h.status_code >= 400:
            # fallback: небольшой GET (первые байты)
            r = await client.get(
                pdf_url,
                timeout=timeout,
                follow_redirects=True,
                headers={"Range": "bytes=0-2048"},
            )
            if r.status_code >= 400:
                return False, f"http {r.status_code}"
            chunk = r.content[:2048]
            return (True, "ok") if looks_like_pdf(chunk, r.headers.get("Content-Type")) else (False, "not a pdf")

        # 2) HEAD успешен, но проверим первые байты
        r2 = await client.get(
            pdf_url,
            timeout=timeout,
            follow_redirects=True,
            headers={"Range": "bytes=0-2048"},
        )
        if r2.status_code >= 400:
            return False, f"http {r2.status_code}"
        chunk = r2.content[:2048]
        return (True, "ok") if looks_like_pdf(chunk, r2.headers.get("Content-Type")) else (False, "not a pdf")

    except Exception as e:
        return False, type(e).__name__


async def search_openalex(
    query: str,
    n: int = 20,
    lang: str = "ru",
    only_oa: bool = True,
    timeout: float = 30.0,
) -> list[dict]:
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

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await client.get(OPENALEX_WORKS_URL, params=params)
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


async def check_pdfs(results: list[dict], concurrency: int = 10) -> None:
    """
    Мутирует results: добавляет pdf_ok / pdf_reason.
    concurrency — сколько проверок одновременно.
    """
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(follow_redirects=True) as client:

        async def one(it: dict):
            pdf_url = it.get("pdf_url")
            if not pdf_url:
                it["pdf_ok"] = False
                it["pdf_reason"] = "no_pdf_url"
                return

            async with sem:
                ok, reason = await check_pdf_url(client, pdf_url)
                it["pdf_ok"] = bool(ok)
                it["pdf_reason"] = reason

        await asyncio.gather(*(one(it) for it in results))