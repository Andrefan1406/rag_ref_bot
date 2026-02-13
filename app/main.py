from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .schemas import (
    SearchRequest,
    IngestRequest,
    ChatRequest,
    OpenAlexSearchRequest,
    OpenAlexImportRequest,
)
from .config import PDF_DIR
from .services.openalex import search_openalex, check_pdf_url
from .services.pdf_ingest import (
    safe_filename,
    download_pdf,
    extract_text_from_pdf,
    download_pdf_bytes,
    extract_text_from_pdf_bytes,
    chunk_text,
)
from .services.vector_store import add_chunks, search as vs_search

app = FastAPI(title="RAG Ref Bot (MVP, no-ollama)")

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


# -------------------- OpenAlex --------------------
@app.post("/openalex/search")
def api_openalex_search(req: OpenAlexSearchRequest):
    results = search_openalex(req.query, n=req.n, lang=req.lang, only_oa=req.only_oa)

    # проверяем pdf_url на "живой PDF"
    for it in results:
        pdf_url = it.get("pdf_url")
        if pdf_url:
            ok, reason = check_pdf_url(pdf_url)
            it["pdf_ok"] = ok
            it["pdf_reason"] = reason
        else:
            it["pdf_ok"] = False
            it["pdf_reason"] = "no_pdf_url"

    # сначала те, у кого pdf_ok=True, затем по cited_by_count desc
    results.sort(key=lambda x: (not x.get("pdf_ok", False), -(x.get("cited_by_count") or 0)))

    return {"results": results}


@app.post("/openalex/import")
def api_openalex_import(req: OpenAlexImportRequest):
    """
    Вариант 3: НЕ храним PDF на диске.
    Скачиваем PDF в память (bytes) -> извлекаем текст -> чанк -> добавляем в индекс.
    """
    added = []
    skipped = []

    for item in req.items:
        if not item.pdf_url:
            skipped.append({"title": item.title, "reason": "no pdf_url"})
            continue

        # перепроверяем PDF ещё раз, даже если pdf_ok пришёл с фронта
        ok_pdf, reason_pdf = check_pdf_url(item.pdf_url)
        if not ok_pdf:
            skipped.append({"title": item.title, "reason": f"pdf not ok: {reason_pdf}", "url": item.pdf_url})
            continue

        # скачиваем PDF в память
        ok_dl, reason_dl, pdf_bytes = download_pdf_bytes(item.pdf_url, timeout=60)
        if not ok_dl:
            skipped.append({"title": item.title, "reason": reason_dl, "url": item.pdf_url})
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

        add_chunks(
            chunks,
            meta_common={
                "source": "openalex",
                "source_file": virt_name,  # файла нет, но идентификатор удобен
                "title": item.title,
                "year": item.year,
                "doi": item.doi,
                "landing_page_url": item.landing_page_url,
                "pdf_url": item.pdf_url,
                "openalex_id": item.openalex_id,
            },
        )

        added.append({"title": item.title, "source_file": virt_name, "chunks": len(chunks)})

    return {"added": added, "skipped": skipped}


# -------------------- Search in DB --------------------
@app.post("/search")
def api_search_db(req: SearchRequest):
    results = vs_search(req.query, top_k=req.n)
    return {"results": results}


# -------------------- Legacy ingest (keeps PDFs on disk) --------------------
@app.post("/ingest")
def api_ingest(req: IngestRequest):
    """
    Старый режим (как было): скачиваем PDF на диск в PDF_DIR.
    Можно потом тоже перевести на bytes, если захочешь.
    """
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

        add_chunks(
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

    add_chunks(
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
def api_chat(req: ChatRequest):
    hits = vs_search(req.question, top_k=req.top_k)

    if not hits:
        return {"answer": "В базе нет документов или ничего не найдено.", "sources": []}

    # MVP ответ без LLM: выдаём лучшие фрагменты
    answer_lines = ["Нашёл релевантные фрагменты:"]
    for h in hits:
        src = h.get("source_file") or "source"
        answer_lines.append(f"- ({src}#chunk{h['chunk_id']}) {h['text'][:320]}...")

    return {"answer": "\n".join(answer_lines), "sources": hits}
