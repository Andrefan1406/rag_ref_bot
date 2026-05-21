# server.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import shutil
from pathlib import Path
import asyncio
import json
import time
import uuid
from fastapi.responses import StreamingResponse
import re

from rag_core import init_debate_apps, ask_debate_rag_direct, connect_kb, add_document_to_kb

app = FastAPI(title="RAG Bot Backend")

UPLOAD_JOBS = {}

KB_TITLES_FILE = Path("data/kb/_kb_titles.json")

DEFAULT_KB_TITLES = {
    "default": "Оптика и светотехника",
    "ovk": "ОВ и К",
    "bess": "BESS",
}


def load_kb_titles():
    KB_TITLES_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not KB_TITLES_FILE.exists():
        save_kb_titles(DEFAULT_KB_TITLES)
        return DEFAULT_KB_TITLES.copy()

    with open(KB_TITLES_FILE, "r", encoding="utf-8") as f:
        titles = json.load(f)

    for kb_name, title in DEFAULT_KB_TITLES.items():
        titles.setdefault(kb_name, title)

    save_kb_titles(titles)
    return titles


def save_kb_titles(titles: dict):
    KB_TITLES_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(KB_TITLES_FILE, "w", encoding="utf-8") as f:
        json.dump(titles, f, ensure_ascii=False, indent=2)

def slugify_kb_name(title: str) -> str:
    translit_map = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
        "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
        "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
        "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
        "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch",
        "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
        "э": "e", "ю": "yu", "я": "ya",
    }

    text = title.lower().strip()
    text = "".join(translit_map.get(ch, ch) for ch in text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")

    return text

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],    
)

class KBRequest(BaseModel):
    kb_name: str

class CreateKBRequest(BaseModel):
    title: str

class AskRequest(BaseModel):
    question: str

@app.post("/api/switch-kb")
async def switch_kb(req: KBRequest):
    titles = load_kb_titles()

    if req.kb_name not in titles:
        raise HTTPException(status_code=400, detail="Неизвестная база знаний")

    connect_kb(req.kb_name)

    return {
        "success": True,
        "active_kb": req.kb_name,
        "title": titles[req.kb_name],
    }

@app.post("/api/create-kb")
async def create_kb_endpoint(req: CreateKBRequest):
    title = (req.title or "").strip()

    if not title:
        raise HTTPException(status_code=400, detail="Название базы не может быть пустым")

    titles = load_kb_titles()
    kb_name = slugify_kb_name(title)

    if not kb_name:
        raise HTTPException(status_code=400, detail="Не удалось сформировать системное имя базы")

    original_kb_name = kb_name
    counter = 2

    while kb_name in titles:
        kb_name = f"{original_kb_name}_{counter}"
        counter += 1

    connect_kb(kb_name)

    titles[kb_name] = title
    save_kb_titles(titles)

    return {
        "success": True,
        "kb_name": kb_name,
        "title": title,
    }

@app.get("/api/kbs")
async def get_kbs():
    titles = load_kb_titles()

    return {
        "success": True,
        "items": [
            {"kb_name": kb_name, "title": title}
            for kb_name, title in titles.items()
        ]
    }

@app.get("/api/kbs/{kb_name}/books")
async def get_kb_books(kb_name: str):
    books = set()

    # 1. Файлы, загруженные через интерфейс
    uploads_dir = Path("data/uploads") / kb_name

    if uploads_dir.exists() and uploads_dir.is_dir():
        allowed_suffixes = {".pdf", ".djvu"}

        for file in uploads_dir.iterdir():
            if file.is_file() and file.suffix.lower() in allowed_suffixes:
                books.add(file.name)

    # 2. Старые книги из чанков базы знаний
    chunks_path = Path("data/kb") / kb_name / "my_chunks.pkl"

    if chunks_path.exists():
        import pickle

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue

            metadata = chunk.get("metadata", {})
            source = metadata.get("source")

            if source:
                books.add(source)

    return {
        "books": sorted(books)
    }
    
@app.on_event("startup")
async def startup():
    print("🔌 Загрузка моделей и инициализация графа без уточнения аспектов...")
    init_debate_apps()
    print("✅ Бэкенд готов к работе")

@app.post("/api/ask")
async def ask(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Вопрос пустой")

    try:
        print(f"🟢 Вопрос: {question}")
        answer = ask_debate_rag_direct(question)
        return {"answer": answer or "Ответ не сформирован."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/upload-doc")
async def upload_doc(
    kb_name: str = Form(...),
    file: UploadFile = File(...)
):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()

    if suffix not in [".pdf", ".djvu"]:
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только PDF и DJVU"
        )

    upload_id = str(uuid.uuid4())

    upload_dir = Path("data/uploads") / kb_name
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(filename).name
    saved_path = upload_dir / safe_name

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    UPLOAD_JOBS[upload_id] = {
        "queue": queue,
        "done": False,
        "error": None,
        "result": None,
    }

    def push_progress(data: dict):
        loop.call_soon_threadsafe(queue.put_nowait, data)

    async def background_job():
        try:
            result = await asyncio.to_thread(
                add_document_to_kb,
                str(saved_path),
                kb_name,
                Path(filename).stem,
                push_progress
            )

            UPLOAD_JOBS[upload_id]["done"] = True
            UPLOAD_JOBS[upload_id]["result"] = result

            push_progress({
                "percentage": 100,
                "status_text": "Документ успешно добавлен",
                "book_name": Path(filename).stem,
                "eta": "готово",
                "done": True,
                "result": result
            })

        except Exception as e:
            UPLOAD_JOBS[upload_id]["done"] = True
            UPLOAD_JOBS[upload_id]["error"] = str(e)

            push_progress({
                "percentage": 100,
                "status_text": "Ошибка",
                "book_name": Path(filename).stem,
                "eta": "",
                "done": True,
                "error": str(e)
            })

    asyncio.create_task(background_job())

    return {
        "success": True,
        "upload_id": upload_id,
        "filename": filename
    }

@app.get("/api/kbs/{kb_name}/books")
async def get_kb_books(kb_name: str):
    upload_dir = Path("data/uploads") / kb_name
    if not upload_dir.exists():
        return {"books": []}
    
    # Собираем файлы с расширениями .pdf и .djvu
    books = [f.name for f in upload_dir.iterdir() if f.is_file() and f.suffix.lower() in [".pdf", ".djvu"]]
    # Сортируем по алфавиту
    books.sort()
    return {"books": books}

@app.get("/api/upload-status/{upload_id}")
async def upload_status(upload_id: str):
    job = UPLOAD_JOBS.get(upload_id)

    if not job:
        raise HTTPException(status_code=404, detail="Загрузка не найдена")

    async def event_stream():
        queue = job["queue"]

        try:
            while True:
                data = await queue.get()

                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                if data.get("done"):
                    break

        except (asyncio.CancelledError, ConnectionResetError):
            print(f"⚠️ Клиент отключился от SSE: {upload_id}")

        finally:
            if job.get("done"):
                UPLOAD_JOBS.pop(upload_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    ) 

# Старый endpoint оставлен как алиас, чтобы старые версии index.html не падали.
@app.post("/api/start")
async def start_alias(req: AskRequest):
    return await ask(req)

@app.post("/api/upload-doc")
async def upload_doc(
    kb_name: str = Form(...),
    file: UploadFile = File(...)
):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()

    if suffix not in [".pdf", ".djvu"]:
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только PDF и DJVU"
        )

    upload_dir = Path("data/uploads") / kb_name
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(filename).name
    saved_path = upload_dir / safe_name

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = add_document_to_kb(
        file_path=str(saved_path),
        kb_name=kb_name,
        source_name=Path(safe_name).stem
    )

    return {
        "success": True,
        "message": "Документ добавлен в базу знаний",
        "result": result
    }    

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
