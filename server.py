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

from rag_core import init_debate_apps, ask_debate_rag_direct, connect_kb, add_document_to_kb

app = FastAPI(title="RAG Bot Backend")

UPLOAD_JOBS = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],    
)

class KBRequest(BaseModel):
    kb_name: str

class AskRequest(BaseModel):
    question: str

@app.post("/api/switch-kb")
async def switch_kb(req: KBRequest):
    titles = {
        "default": "Оптика и светотехника",
        "ovk": "ОВ и К",
        "bess": "BESS",
    }

    if req.kb_name not in titles:
        raise HTTPException(status_code=400, detail="Неизвестная база знаний")

    connect_kb(req.kb_name)

    return {
        "success": True,
        "active_kb": req.kb_name,
        "title": titles[req.kb_name],
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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
