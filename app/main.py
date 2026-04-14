from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import fitz
import time
import httpx

from .schemas import (
    SearchRequest,
    IngestRequest,
    ChatRequest,
    OpenAlexSearchRequest,
    OpenAlexImportRequest,
    DebateRequest
)
from .config import PDF_DIR
from .services.openalex import search_openalex, check_pdf_url, check_pdfs
from .services.pdf_ingest import (
    safe_filename,
    download_pdf,
    extract_text_from_pdf,
    download_pdf_bytes,
    extract_text_from_pdf_bytes,
    chunk_text,
)
from .services.vector_store import clear_vector_store, add_chunks, search as vs_search
from .services.llm import generate_answer, generate_debate
from .graphs.rag_graph import rag_graph

app = FastAPI(title="RAG Ref Bot")

# Минимальный UI (static/index.html) доступен по /ui
app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")

# чтобы потом легко подключить фронтенд
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"ok": True, "service": "rag_ref_bot"}


def interleave(a: list[dict], b: list[dict]) -> list[dict]:
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out

@app.post("/openalex/search")
async def api_openalex_search(req: OpenAlexSearchRequest):
    t_start = time.perf_counter()

    lang = (req.lang or "").strip()

    # -------------------------
    # 1) SEARCH (OpenAlex)
    # -------------------------
    t_search_start = time.perf_counter()

    if lang in ("ru|en", "en|ru"):
        per_lang = min(max(req.n * 2, 20), 200)

        ru_results = await search_openalex(req.query, n=per_lang, lang="ru", only_oa=req.only_oa)
        en_results = await search_openalex(req.query, n=per_lang, lang="en", only_oa=req.only_oa)

        for it in ru_results:
            it["_lang"] = "ru"
        for it in en_results:
            it["_lang"] = "en"

        results = ru_results + en_results
    else:
        results = await search_openalex(req.query, n=req.n, lang=lang, only_oa=req.only_oa)
        for it in results:
            it["_lang"] = lang or ""

    t_search_end = time.perf_counter()

    # -------------------------
    # 2) CHECK PDF URL
    # -------------------------
    t_pdf_start = time.perf_counter()

    await check_pdfs(results, concurrency=10)

    t_pdf_end = time.perf_counter()

    # -------------------------
    # 3) SORT / MIX
    # -------------------------
    t_rank_start = time.perf_counter()

    if lang in ("ru|en", "en|ru"):
        ru = [x for x in results if x.get("_lang") == "ru"]
        en = [x for x in results if x.get("_lang") == "en"]

        ru_ok = [x for x in ru if x["pdf_ok"]]
        en_ok = [x for x in en if x["pdf_ok"]]
        ru_no = [x for x in ru if not x["pdf_ok"]]
        en_no = [x for x in en if not x["pdf_ok"]]

        def by_cites(x: dict):
            return (-(x.get("cited_by_count") or 0), -(x.get("year") or 0))

        ru_ok.sort(key=by_cites)
        en_ok.sort(key=by_cites)
        ru_no.sort(key=by_cites)
        en_no.sort(key=by_cites)

        mixed_ok = interleave(ru_ok, en_ok)
        mixed_no = interleave(ru_no, en_no)

        results = (mixed_ok + mixed_no)[:req.n]
    else:
        results.sort(
            key=lambda x: (
                not x.get("pdf_ok", False),
                -(x.get("cited_by_count") or 0),
                -(x.get("year") or 0),
            )
        )
        results = results[:req.n]

    t_rank_end = time.perf_counter()

    t_end = time.perf_counter()

    # -------------------------
    # RESPONSE
    # -------------------------
    return {
        "results": results,
        "timing": {
            "total_sec": round((t_end - t_start) , 0),
            "search_ms": round((t_search_end - t_search_start) * 1000, 1),
            "pdf_check_ms": round((t_pdf_end - t_pdf_start) * 1000, 1),
            "ranking_ms": round((t_rank_end - t_rank_start) * 1000, 1),
            "items_total": len(results),
        },
    }

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    path = Path("static") / "favicon.ico"
    if path.exists():
        return FileResponse(path)
    return {}


@app.post("/openalex/import")
async def api_openalex_import(req: OpenAlexImportRequest):
    """
    Вариант 3: НЕ храним PDF на диске.
    Скачиваем PDF в память (bytes) -> извлекаем текст -> чанк -> добавляем в индекс.
    """
    added: list[dict] = []
    skipped: list[dict] = []

    t0 = time.perf_counter()

    # Один клиент на весь импорт (быстрее + меньше накладных расходов)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for item in req.items:
            if not item.pdf_url:
                skipped.append({"title": item.title, "reason": "no pdf_url"})
                continue

            # перепроверяем PDF ещё раз, даже если pdf_ok пришёл с фронта
            try:
                r = await client.get(item.pdf_url, timeout=60)
                r.raise_for_status()
                pdf_bytes = r.content
            except Exception as e:
                skipped.append({
                    "title": item.title,
                    "reason": f"download failed: {type(e).__name__}: {e}",
                    "url": item.pdf_url,
                })
                continue

            # скачиваем PDF в память
            # ok_dl, reason_dl, pdf_bytes = download_pdf_bytes(item.pdf_url, timeout=60)
            # if not ok_dl:
            #     skipped.append({"title": item.title, "reason": reason_dl, "url": item.pdf_url})
            #     continue

            # читаем число страниц
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                pages_count = doc.page_count
                doc.close()
            except Exception as e:
                skipped.append(
                    {"title": item.title, "reason": f"cannot read pdf pages: {type(e).__name__}: {e}"}
                )
                continue

            # извлекаем текст из bytes
            try:
                text = extract_text_from_pdf_bytes(pdf_bytes)
            except Exception as e:
                skipped.append({"title": item.title, "reason": f"extract failed: {type(e).__name__}: {e}"})
                continue
            finally:
                # освобождаем память пораньше
                pdf_bytes = b""

            if len(text) < 200:
                skipped.append({"title": item.title, "reason": "pdf has no text (scan?) -> нужен OCR"})
                continue

            chunks = chunk_text(text)

            # виртуальное имя, чтобы в /chat было понятно, откуда кусок
            virt_name = f"openalex_{(item.openalex_id or '').split('/')[-1]}".strip("_") or "openalex_doc"

            await add_chunks(
                chunks,
                meta_common={
                    "source": "openalex",
                    "source_file": virt_name,
                    "title": item.title,
                    "year": item.year,
                    "doi": item.doi,
                    "landing_page_url": item.landing_page_url,
                    "pdf_url": item.pdf_url,
                    "openalex_id": item.openalex_id,
                },
            )

            added.append({"title": item.title, "pages": pages_count, "chunks": len(chunks)})

    t1 = time.perf_counter()

    return {
        "added": added,
        "skipped": skipped,
        "timing": {
            "total_min": round((t1 - t0) /60, 1),
            "added_count": len(added),
            "skipped_count": len(skipped),
        },
    }

# -------------------- Search in DB --------------------
@app.post("/search")
async def api_search_db(req: SearchRequest):
    results = await vs_search(req.query, top_k=req.n)
    return {"results": results}

# -------------------- Сохранение PDF на диск --------------------
@app.post("/ingest")
async def api_ingest(req: IngestRequest):
    
    added = []
    skipped = []

    for item in req.items:
        if not item.pdf_url:
            skipped.append({"title": item.title, "reason": "no pdf_url (upload PDF вручную через /upload_pdf)"})
            continue

        pdf_name = safe_filename(item.title or "paper")
        pdf_path = PDF_DIR / pdf_name

        ok, reason = download_pdf(item.pdf_url, pdf_path)
        if not ok:
            skipped.append({"title": item.title, "reason": reason, "url": item.pdf_url})
            continue

        try:
            text = extract_text_from_pdf(pdf_path)
        except Exception as e:
            skipped.append({"title": item.title, "reason": f"extract failed: {type(e).__name__}: {e}", "pdf": pdf_name})
            continue

        if len(text) < 200:
            skipped.append({"title": item.title, "reason": "pdf has no text (scan?) -> нужен OCR", "pdf": pdf_name})
            continue

        chunks = chunk_text(text)

        await add_chunks(
            chunks,
            meta_common={
                "source_file": pdf_name,
                "title": item.title,
                "year": item.year,
            },
        )

        added.append({"title": item.title, "pdf": pdf_name, "chunks": len(chunks)})

    return {"added": added, "skipped": skipped}


# -------------------- Manual upload --------------------
@app.post("/upload_pdf")
async def api_upload_pdf(file: UploadFile = File(...), title: str = "uploaded_pdf", year: int | None = None):
    """
    Когда ссылка не работает — пользователь загружает PDF вручную.
    Этот режим сохраняет PDF на диск, потому что пользователь его передал.
    """
    pdf_name = safe_filename(title)
    pdf_path = PDF_DIR / pdf_name

    content = await file.read()
    pdf_path.write_bytes(content)

    try:
        text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        return {"ok": False, "reason": f"extract failed: {type(e).__name__}: {e}", "pdf": pdf_name}

    if len(text) < 200:
        return {"ok": False, "reason": "pdf has no text (scan?) -> нужен OCR", "pdf": pdf_name}

    chunks = chunk_text(text)

    await add_chunks(
        chunks,
        meta_common={
            "source_file": pdf_name,
            "title": title,
            "year": year,
        },
    )

    return {"ok": True, "pdf": pdf_name, "chunks": len(chunks)}


# -------------------- Chat --------------------
@app.post("/chat")
async def api_chat(req: ChatRequest):
    result = await rag_graph.ainvoke({
        "question": req.question,
        "top_k": req.top_k,
    })
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    }

@app.post("/chat_debate")
async def api_chat_debate(req: DebateRequest):
    t0 = time.perf_counter() 
    # 1) Берём источники из твоей базы (как в обычном /chat)
    t_search = time.perf_counter()
    sources = await vs_search(req.question, top_k=req.top_k)
    t_search_end = time.perf_counter() 

    # 2) Собираем context строкой (как у тебя обычно делается)    
    context = "\n\n".join([s.get("text", "") for s in sources])

    # 3) Запускаем дебаты
    t_deb_start = time.perf_counter()
    data = await generate_debate(
        question=req.question,
        context=context,
        rounds=req.rounds,
        critic_style=req.critic_style,
    )
    t_deb_end = time.perf_counter()
    total_duration = time.perf_counter()  - t0

    return {
        "final_answer": data["final_answer"],
        "turns": data["turns"],
        "sources": sources,
        "timing":{
            "total": round(total_duration, 3),
            "search": round(t_search_end - t_search, 3),
            "debate": round(t_deb_end - t_deb_start, 3)
        }
    }


@app.post("/admin/clear_db")
def api_clear_db():
    return clear_vector_store()


