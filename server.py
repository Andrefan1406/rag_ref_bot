# server.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import shutil
from pathlib import Path

from rag_core import init_debate_apps, ask_debate_rag_direct, connect_kb, add_document_to_kb

app = FastAPI(title="RAG Bot Backend")

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
