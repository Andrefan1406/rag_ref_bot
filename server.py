# server.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
import uvicorn
import shutil
from pathlib import Path
import asyncio
import json
import time
import uuid
from fastapi.responses import StreamingResponse
import re

from rag_core import (
    init_debate_apps,
    ask_debate_rag_direct,
    start_debate_rag,
    continue_debate_rag,
    build_hypothesis_node,
    connect_kb,
    add_document_to_kb,
    delete_book_from_kb,
    append_source_chunks_to_answer,
    ollama_chat
)
app = FastAPI(title="RAG Bot Backend")

UPLOAD_JOBS = {}

CHAT_SESSIONS = {}

KB_TITLES_FILE = Path("data/kb/_kb_titles.json")

KB_BOOKS_FILE = Path("data/kb/_kb_books.json")

RESEARCHES_FILE = Path("data/researches.json")
RESEARCH_STATES_DIR = Path("data/research_states")



def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def make_research_title(question: str) -> str:
    title = re.sub(r"\s+", " ", (question or "").strip())
    if not title:
        return "Новое исследование"
    return title[:70] + ("..." if len(title) > 70 else "")


def load_researches():
    RESEARCHES_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not RESEARCHES_FILE.exists():
        return []

    with open(RESEARCHES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_researches(items: list):
    RESEARCHES_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RESEARCHES_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def get_research_state_path(research_id: str) -> Path:
    RESEARCH_STATES_DIR.mkdir(parents=True, exist_ok=True)
    return RESEARCH_STATES_DIR / f"{research_id}.json"


def save_research_state(research_id: str, state: dict):
    state_path = get_research_state_path(research_id)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)


def load_research_state(research_id: str):
    state_path = get_research_state_path(research_id)
    if not state_path.exists():
        return None

    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def upsert_research_meta(research_id: str, **kwargs):
    items = load_researches()
    found = None

    for item in items:
        if item.get("id") == research_id:
            found = item
            break

    if found is None:
        found = {"id": research_id, "created_at": now_ts()}
        items.insert(0, found)

    found.update(kwargs)
    found["updated_at"] = now_ts()

    save_researches(items)
    return found


def get_session_state_or_404(session_id: str):
    state = CHAT_SESSIONS.get(session_id)

    if state is None:
        state = load_research_state(session_id)
        if state is not None:
            CHAT_SESSIONS[session_id] = state

    if state is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")

    return state

DEFAULT_KB_TITLES = {
    "default": "Оптика и светотехника",
    "ovk": "ОВ и К",
    "bess": "BESS",
}

def load_kb_books():
    KB_BOOKS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not KB_BOOKS_FILE.exists():
        return {}

    with open(KB_BOOKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_kb_book(kb_name: str, book_name: str):
    books = load_kb_books()

    books.setdefault(kb_name, [])

    if book_name not in books[kb_name]:
        books[kb_name].append(book_name)
        books[kb_name].sort()

    with open(KB_BOOKS_FILE, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=2)

def delete_kb_book(kb_name: str, book_name: str):
    books = load_kb_books()

    if kb_name in books:
        books[kb_name] = [
            book for book in books[kb_name]
            if book != book_name
        ]

    with open(KB_BOOKS_FILE, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=2)

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

class HypothesisRequest(BaseModel):
    question: str
    kb_name: Optional[str] = None

class RegenerateHypothesisRequest(BaseModel):
    session_id: str

class ContinueRequest(BaseModel):
    session_id: str
    hypothesis: str = ""

class PlanRequest(BaseModel):
    session_id: str
    hypothesis: Optional[str] = None

class AnalysisRequest(BaseModel):
    session_id: str

class ResearchUpdateRequest(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None
    current_stage: Optional[str] = None
    state: Optional[dict[str, Any]] = None

class ApplyClarificationRequest(BaseModel):
    session_id: str
    comment: str

class ExcludeSourceRequest(BaseModel):
    session_id: str
    fragment_index: int

def format_research_fragments(state: dict, limit: int = 12, max_chars_per_fragment: int = 1200) -> str:
    fragments = state.get("used_fragments") or state.get("retrieved_items") or []
    parts = []

    for idx, frag in enumerate(fragments[:limit], start=1):
        if not isinstance(frag, dict):
            continue

        metadata = frag.get("metadata") or {}
        source = frag.get("source") or metadata.get("source") or "Источник не указан"
        page = frag.get("page") or metadata.get("page") or frag.get("pages") or ""
        text = (frag.get("text") or frag.get("content") or "").strip()

        if not text:
            continue

        if len(text) > max_chars_per_fragment:
            text = text[:max_chars_per_fragment] + "..."

        page_text = f", стр. {page}" if page else ""
        parts.append(f"[Фрагмент {idx}] {source}{page_text}\n{text}")

    if parts:
        return "\n\n---\n\n".join(parts)

    context = (state.get("retrieved_context") or state.get("additional_context") or "").strip()
    return context[:12000] if context else "Фрагменты источников не найдены."

def append_stage_message(state: dict, role: str, stage: str, content: str):
    messages = state.get("messages") or []
    messages.append({
        "role": role,
        "stage": stage,
        "content": content,
        "created_at": now_ts()
    })
    state["messages"] = messages

@app.get("/api/researches")
async def get_researches():
    return {
        "success": True,
        "items": load_researches()
    }

@app.get("/api/researches/{research_id}")
async def get_research(research_id: str):
    items = load_researches()
    meta = next((item for item in items if item.get("id") == research_id), None)

    if not meta:
        raise HTTPException(status_code=404, detail="Исследование не найдено")

    state = load_research_state(research_id) or {}
    CHAT_SESSIONS[research_id] = state

    return {
        "success": True,
        "research": meta,
        "state": state
    }

@app.post("/api/researches")
async def create_research(req: HypothesisRequest):
    question = (req.question or "").strip()

    if not question:
        raise HTTPException(status_code=400, detail="Вопрос пустой")

    research_id = str(uuid.uuid4())
    kb_name = req.kb_name or "default"
    title = make_research_title(question)

    state = {
        "id": research_id,

        # Исходный вопрос
        "question": question,

        # База знаний
        "kb_name": kb_name,

        # Этапы исследования
        "hypothesis": "",
        "refined_hypothesis": "",
        "plan": "",
        "analysis": "",
        "final_answer": "",

        # Служебное
        "current_stage": "question",
        "completed_stages": ["question"],
        "status": "draft",

        # История сообщений
        "messages": [
            {
                "role": "user",
                "stage": "question",
                "content": question,
                "created_at": now_ts()
            }
        ]
    }

    save_research_state(research_id, state)
    upsert_research_meta(
        research_id,
        title=title,
        kb_name=kb_name,
        question=question,
        current_stage="question",
        status="draft"
    )

    return {
        "success": True,
        "research_id": research_id,
        "research": upsert_research_meta(research_id),
        "state": state
    }

@app.post("/api/researches/{research_id}/save")
async def save_research(research_id: str, req: ResearchUpdateRequest):
    state = load_research_state(research_id) or {}

    if req.state:
        state.update(req.state)

    if req.current_stage:
        state["current_stage"] = req.current_stage

    if req.status:
        state["status"] = req.status

    save_research_state(research_id, state)

    meta_updates = {}
    if req.title is not None:
        meta_updates["title"] = req.title
    if req.current_stage is not None:
        meta_updates["current_stage"] = req.current_stage
    if req.status is not None:
        meta_updates["status"] = req.status

    meta = upsert_research_meta(research_id, **meta_updates)

    return {"success": True, "research": meta, "state": state}


@app.delete("/api/researches/{research_id}")
async def delete_research(research_id: str):
    items = [item for item in load_researches() if item.get("id") != research_id]
    save_researches(items)

    state_path = get_research_state_path(research_id)
    state_path.unlink(missing_ok=True)
    CHAT_SESSIONS.pop(research_id, None)

    return {"success": True}


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
    books_data = load_kb_books()
    books = set(books_data.get(kb_name, []))

    if kb_name == "default":
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

@app.delete("/api/kbs/{kb_name}/books/{book_name}")
async def delete_book(kb_name: str, book_name: str):
    try:
        result = delete_book_from_kb(kb_name, book_name)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("message"))

        delete_kb_book(kb_name, book_name)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/kbs/{kb_name}")
async def delete_kb_endpoint(kb_name: str):
    titles = load_kb_titles()

    if kb_name not in titles:
        raise HTTPException(status_code=404, detail="База знаний не найдена")

    if len(titles) <= 1:
        raise HTTPException(status_code=400, detail="Нельзя удалить последнюю базу знаний")

    deleted_title = titles.pop(kb_name)
    save_kb_titles(titles)

    books = load_kb_books()
    books.pop(kb_name, None)

    with open(KB_BOOKS_FILE, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=2)

    kb_dir = Path("data/kb") / kb_name
    temp_dir = Path("data/temp") / kb_name
    upload_dir = Path("data/uploads") / kb_name

    shutil.rmtree(kb_dir, ignore_errors=True)
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(upload_dir, ignore_errors=True)

    next_kb = next(iter(titles.keys()))
    connect_kb(next_kb)

    return {
        "success": True,
        "deleted_kb": kb_name,
        "deleted_title": deleted_title,
        "next_kb": next_kb,
        "next_title": titles[next_kb]
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
    
@app.post("/api/hypothesis")
async def create_hypothesis(req: HypothesisRequest):
    question = (req.question or "").strip()

    if not question:
        raise HTTPException(status_code=400, detail="Вопрос пустой")

    try:
        kb_name = req.kb_name or "default"
        session_id = str(uuid.uuid4())

        state = start_debate_rag(question)
        state["id"] = session_id
        state["kb_name"] = kb_name
        state["current_stage"] = "hypothesis"
        state["completed_stages"] = ["question", "sources", "hypothesis"]
        state["status"] = "in_progress"
        state["messages"] = [
            {"role": "user", "stage": "question", "content": question, "created_at": now_ts()},
            {"role": "assistant", "stage": "hypothesis", "content": state.get("hypothesis", ""), "created_at": now_ts()}
        ]

        CHAT_SESSIONS[session_id] = state
        save_research_state(session_id, state)
        upsert_research_meta(
            session_id,
            title=make_research_title(question),
            kb_name=kb_name,
            question=question,
            current_stage="hypothesis",
            status="in_progress"
        )

        return {
            "success": True,
            "session_id": session_id,
            "hypothesis": state.get("hypothesis", "")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hypothesis/regenerate")
async def regenerate_hypothesis(req: RegenerateHypothesisRequest):
    state = get_session_state_or_404(req.session_id)

    try:
        result = build_hypothesis_node(state)
        state.update(result)

        CHAT_SESSIONS[req.session_id] = state
        state["current_stage"] = "hypothesis"
        state["status"] = "in_progress"
        save_research_state(req.session_id, state)
        upsert_research_meta(req.session_id, current_stage="hypothesis", status="in_progress")

        return {
            "success": True,
            "session_id": req.session_id,
            "hypothesis": state.get("hypothesis", "")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/hypothesis/apply-comment")
async def apply_hypothesis_comment(req: ApplyClarificationRequest):
    state = get_session_state_or_404(req.session_id)

    comment = (req.comment or "").strip()

    if not comment:
        raise HTTPException(status_code=400, detail="Комментарий пустой")

    old_hypothesis = state.get("hypothesis", "")

    prompt = f"""
У пользователя есть предварительная гипотеза:

{old_hypothesis}

Пользователь дал уточнение:

{comment}

Перепиши гипотезу с учетом уточнения пользователя.
Сохрани инженерный стиль, не добавляй неподтвержденные факты.
Верни только обновленную гипотезу.
"""

    try:
        from rag_core import ollama_chat

        new_hypothesis = ollama_chat(
            system="Ты помогаешь уточнять исследовательскую гипотезу.",
            user=prompt
        )

        state["hypothesis"] = new_hypothesis
        state["refined_hypothesis"] = new_hypothesis
        state["last_user_comment"] = comment
        state["current_stage"] = "hypothesis"
        state["completed_stages"] = ["question", "sources", "hypothesis"]

        save_research_state(req.session_id, state)

        upsert_research_meta(
            req.session_id,
            current_stage="hypothesis",
            status="draft"
        )

        CHAT_SESSIONS[req.session_id] = state

        return {
            "success": True,
            "hypothesis": new_hypothesis,
            "state": state
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/continue")
async def continue_answer(req: ContinueRequest):
    state = get_session_state_or_404(req.session_id)

    current_stage = state.get("current_stage")

    # ==========================================
    # ЭТАП PLAN -> ANALYSIS
    # ==========================================

    if current_stage == "plan":

        state["current_stage"] = "analysis"

        CHAT_SESSIONS[req.session_id] = state
        save_research_state(req.session_id, state)

        return {
            "success": True,
            "next_stage": "analysis"
        }

    # ==========================================
    # ЭТАП ANALYSIS -> FINAL
    # ==========================================

    if current_stage == "analysis":

        result = continue_debate_rag(state)
        
        result["used_fragments"] = (
            result.get("used_fragments")
            or state.get("used_fragments")
            or state.get("retrieved_items")
            or []
        )

        result["retrieved_items"] = state.get("retrieved_items", [])

        answer = result.get("final_answer", "Ответ не сформирован.")

        result["final_answer"] = answer

        answer = append_source_chunks_to_answer(
            answer,
            result.get("used_fragments", [])
        )

        result["id"] = req.session_id
        result["question"] = state.get("question", "")
        result["kb_name"] = state.get("kb_name", "default")

        result["hypothesis"] = state.get("hypothesis", "")
        result["refined_hypothesis"] = state.get("refined_hypothesis", "")
        result["plan"] = state.get("plan", "")
        result["analysis"] = state.get("analysis", "")
        result["final_answer"] = answer

        result["current_stage"] = "final"

        result["completed_stages"] = [
            "question",
            "sources",
            "hypothesis",
            "plan",
            "analysis",
            "final"
        ]

        result["status"] = "completed"

        CHAT_SESSIONS[req.session_id] = result

        save_research_state(
            req.session_id,
            result
        )

        upsert_research_meta(
            req.session_id,
            current_stage="final",
            status="completed"
        )

        return {
            "success": True,
            "answer": answer,
            "state": result
        }

    raise HTTPException(
        status_code=400,
        detail=f"Неверный этап: {current_stage}"
    )

@app.post("/api/sources/exclude")
async def exclude_source(req: ExcludeSourceRequest):
    state = get_session_state_or_404(req.session_id)

    fragments = state.get("used_fragments") or state.get("retrieved_items") or []

    if req.fragment_index < 0 or req.fragment_index >= len(fragments):
        raise HTTPException(status_code=400, detail="Источник не найден")

    excluded_fragment = fragments.pop(req.fragment_index)

    excluded = state.get("excluded_fragments") or []
    excluded.append(excluded_fragment)

    state["used_fragments"] = fragments
    state["excluded_fragments"] = excluded

    state["current_stage"] = "sources"

    # очищаем последующие этапы, потому что источник изменился
    state["used_fragments"] = fragments
    state["excluded_fragments"] = excluded

    state["sources_dirty"] = True
    state["needs_rebuild"] = True

    state["sources_dirty_message"] = (
        "Источники изменены. Гипотеза, план и анализ требуют обновления."
    )

    state["current_stage"] = "sources"

    append_stage_message(
        state,
        "user",
        "sources",
        f"Исключён источник №{req.fragment_index + 1}"
    )

    CHAT_SESSIONS[req.session_id] = state
    save_research_state(req.session_id, state)
    upsert_research_meta(
        req.session_id,
        current_stage="sources",
        status="in_progress"
    )

    return {
        "success": True,
        "state": state,
        "excluded_fragment": excluded_fragment
    }

@app.post("/api/plan")
async def generate_plan(req: PlanRequest):
    state = get_session_state_or_404(req.session_id)

    hypothesis = (req.hypothesis or state.get("refined_hypothesis") or state.get("hypothesis") or "").strip()

    if not hypothesis:
        raise HTTPException(status_code=400, detail="Гипотеза пустая")

    question = state.get("question", "")
    sources_text = format_research_fragments(state)

    system_prompt = """
Ты — инженерный аналитик.

Твоя задача — составить план проверки гипотезы.
Это НЕ финальный ответ и НЕ анализ.

План должен показать:
- что нужно проверить;
- какие данные нужно найти;
- какие расчёты нужны;
- какие ограничения нужно учесть;
- какой результат должен быть получен.

Не делай окончательных выводов.
Не выдумывай факты.
"""

    user_prompt = f"""
Вопрос пользователя:
{question}

Гипотеза:
{hypothesis}

Найденные фрагменты источников:
{sources_text}

Составь план исследования в понятной структуре.
"""

    plan = ollama_chat(system_prompt, user_prompt)

    state["hypothesis"] = hypothesis
    state["refined_hypothesis"] = hypothesis
    state["plan"] = plan
    state["current_stage"] = "plan"
    state["completed_stages"] = list(dict.fromkeys(
        (state.get("completed_stages") or []) + ["plan"]
    ))

    append_stage_message(state, "assistant", "plan", plan)

    CHAT_SESSIONS[req.session_id] = state
    save_research_state(req.session_id, state)
    upsert_research_meta(req.session_id, current_stage="plan", status="in_progress")

    return {
        "success": True,
        "plan": plan,
        "state": state
    }

@app.post("/api/analysis")
async def generate_analysis(req: AnalysisRequest):
    state = get_session_state_or_404(req.session_id)

    question = state.get("question", "")
    hypothesis = (state.get("refined_hypothesis") or state.get("hypothesis") or "").strip()
    plan = (state.get("plan") or "").strip()
    sources_text = format_research_fragments(state)

    if not hypothesis:
        raise HTTPException(status_code=400, detail="Гипотеза пустая")

    if not plan:
        raise HTTPException(status_code=400, detail="План исследования ещё не сформирован")

    system_prompt = """
Ты — инженерный аналитик.

Твоя задача — провести анализ гипотезы строго на основании найденных источников.

Важно:
- не делай финальный отчёт;
- не выдумывай факты;
- если данных нет, прямо напиши "данные не найдены";
- отделяй подтверждения от предположений;
- указывай ограничения и риски.
"""

    user_prompt = f"""
Вопрос пользователя:
{question}

Гипотеза:
{hypothesis}

План исследования:
{plan}

Найденные фрагменты источников:
{sources_text}

Проведи анализ по структуре:

1. Найденные подтверждения
2. Найденные противоречия
3. Недостающие данные
4. Технические ограничения
5. Риски
6. Предварительный вывод

Это именно анализ, а не финальный отчёт.
"""

    analysis = ollama_chat(system_prompt, user_prompt)

    state["analysis"] = analysis
    state["current_stage"] = "analysis"
    state["completed_stages"] = list(dict.fromkeys(
        (state.get("completed_stages") or []) + ["analysis"]
    ))

    append_stage_message(state, "assistant", "analysis", analysis)

    CHAT_SESSIONS[req.session_id] = state
    save_research_state(req.session_id, state)
    upsert_research_meta(req.session_id, current_stage="analysis", status="in_progress")

    return {
        "success": True,
        "analysis": analysis,
        "state": state
    }
    
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

    temp_dir = Path("data/temp") / kb_name
    temp_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(filename).name
    saved_path = temp_dir / safe_name
    book_name = Path(filename).stem

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

            save_kb_book(kb_name, book_name)
            saved_path.unlink(missing_ok=True)

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
            saved_path.unlink(missing_ok=True)

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

@app.post("/api/migrate-books-from-chunks/{kb_name}")
async def migrate_books_from_chunks(kb_name: str):
    import pickle

    chunks_path = Path("data/kb") / kb_name / "my_chunks.pkl"
    upload_dir = Path("data/uploads") / kb_name
    upload_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        raise HTTPException(status_code=404, detail="Файл чанков не найден")

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    books = set()

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        metadata = chunk.get("metadata", {})
        source = metadata.get("source")

        if source:
            books.add(str(source).strip())

    books = sorted(books)

    for book in books:
        save_kb_book(kb_name, book)

    return {
        "success": True,
        "kb_name": kb_name,
        "count": len(books),
        "books": books
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
