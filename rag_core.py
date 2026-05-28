import re
import time
import pickle
import requests
import tempfile
import numpy as np
import faiss
import fitz  
import os
from pathlib import Path
import subprocess
import json
import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from dotenv import load_dotenv
import io
from PIL import Image
import pytesseract
from pytesseract import Output
import pypandoc
from typing import Callable
import html

VERBOSE = True

def log(*args):
    if VERBOSE:
        print(*args)

KB_ROOT = Path("data/kb")
ACTIVE_KB_FILE = KB_ROOT / "_active_kb.txt"
DEFAULT_KB_NAME = "default"

ACTIVE_KB = DEFAULT_KB_NAME


def get_kb_dir(kb_name: str) -> Path:
    return KB_ROOT / kb_name


def get_kb_paths(kb_name: str):
    kb_dir = get_kb_dir(kb_name)
    kb_dir.mkdir(parents=True, exist_ok=True)

    return {
        "dir": kb_dir,
        "index": str(kb_dir / "my_index.faiss"),
        "chunks": str(kb_dir / "my_chunks.pkl"),
    }


def create_kb(kb_name: str):
    kb_dir = get_kb_dir(kb_name)
    kb_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ База готова: {kb_name}")
    print(f"📁 Папка: {kb_dir}")


def set_active_kb(kb_name: str):
    global ACTIVE_KB

    create_kb(kb_name)
    ACTIVE_KB = kb_name

    KB_ROOT.mkdir(parents=True, exist_ok=True)
    ACTIVE_KB_FILE.write_text(kb_name, encoding="utf-8")

    print(f"🎯 Активная база: {ACTIVE_KB}")


def get_active_kb():
    global ACTIVE_KB

    if ACTIVE_KB_FILE.exists():
        saved = ACTIVE_KB_FILE.read_text(encoding="utf-8").strip()
        if saved:
            ACTIVE_KB = saved

    return ACTIVE_KB


def list_kbs():
    KB_ROOT.mkdir(parents=True, exist_ok=True)

    bases = [p.name for p in KB_ROOT.iterdir() if p.is_dir()]
    if not bases:
        print("Баз пока нет")
        return

    current = get_active_kb()

    print("Список баз:")
    for name in sorted(bases):
        mark = " <-- active" if name == current else ""
        print(f"- {name}{mark}")

def connect_kb(kb_name: str):
    global GLOBAL_INDEX, GLOBAL_CHUNKS

    # 1. переключаем
    set_active_kb(kb_name)

    # 2. получаем пути
    paths = get_kb_paths(kb_name)

    # 3. загружаем
    if Path(paths["index"]).exists() and Path(paths["chunks"]).exists():
        GLOBAL_INDEX = faiss.read_index(paths["index"])

        with open(paths["chunks"], "rb") as f:
            GLOBAL_CHUNKS = pickle.load(f)

        print(f"📚 Загружена база: {kb_name}")
        print(f"📦 Чанков: {len(GLOBAL_CHUNKS)}")
    else:
        GLOBAL_INDEX = None
        GLOBAL_CHUNKS = []
        print(f"⚠️ База пустая: {kb_name}")

    return GLOBAL_INDEX, GLOBAL_CHUNKS

def load_active_kb():
    global GLOBAL_INDEX, GLOBAL_CHUNKS

    kb_name = get_active_kb()
    paths = get_kb_paths(kb_name)

    index_exists = os.path.exists(paths["index"])
    chunks_exists = os.path.exists(paths["chunks"])

    if not index_exists and not chunks_exists:
        print(f"⚠️ База '{kb_name}' пока пустая")
        GLOBAL_INDEX = None
        GLOBAL_CHUNKS = []
        return GLOBAL_INDEX, GLOBAL_CHUNKS

    if not chunks_exists:
        raise FileNotFoundError(f"Не найден файл чанков: {paths['chunks']}")

    if not index_exists:
        print(f"⚠️ У базы '{kb_name}' нет индекса, но чанки есть")
        with open(paths["chunks"], "rb") as f:
            GLOBAL_CHUNKS = pickle.load(f)
        GLOBAL_INDEX = None
        return GLOBAL_INDEX, GLOBAL_CHUNKS

    GLOBAL_INDEX, GLOBAL_CHUNKS = load_knowledge_base(
        index_path=paths["index"],
        chunks_path=paths["chunks"]
    )

    print(f"✅ Загружена база: {kb_name}")
    print("Чанков:", len(GLOBAL_CHUNKS))
    print("Размер индекса:", GLOBAL_INDEX.ntotal if GLOBAL_INDEX is not None else 0)

    return GLOBAL_INDEX, GLOBAL_CHUNKS


def save_active_kb():
    global GLOBAL_INDEX, GLOBAL_CHUNKS

    kb_name = get_active_kb()
    paths = get_kb_paths(kb_name)

    save_knowledge_base(
        GLOBAL_INDEX,
        GLOBAL_CHUNKS,
        index_path=paths["index"],
        chunks_path=paths["chunks"]
    )

    print(f"💾 Сохранена база: {kb_name}")

def delete_book_from_kb(kb_name: str, book_name: str):
    global GLOBAL_INDEX, GLOBAL_CHUNKS

    connect_kb(kb_name)

    before_count = len(GLOBAL_CHUNKS)

    kept_chunks = []
    removed_chunks = []

    for chunk in GLOBAL_CHUNKS:
        metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        source = str(metadata.get("source", "")).strip()

        if source == book_name:
            removed_chunks.append(chunk)
        else:
            kept_chunks.append(chunk)

    if not removed_chunks:
        return {
            "success": False,
            "message": "Книга не найдена в чанках",
            "kb_name": kb_name,
            "book_name": book_name,
            "removed_chunks": 0,
            "remaining_chunks": before_count
        }

    GLOBAL_CHUNKS = kept_chunks

    vectors = []

    for chunk in GLOBAL_CHUNKS:
        emb = chunk.get("embedding")

        if emb is not None:
            vectors.append(np.array(emb, dtype="float32"))

    if vectors:
        vectors = np.array(vectors, dtype="float32")
        GLOBAL_INDEX = build_faiss_index(vectors)
    else:
        GLOBAL_INDEX = None

    save_active_kb()

    return {
        "success": True,
        "kb_name": kb_name,
        "book_name": book_name,
        "removed_chunks": len(removed_chunks),
        "remaining_chunks": len(GLOBAL_CHUNKS)
    }    

def append_source_chunks_to_answer(answer: str, used_fragments: list) -> str:
    hidden_chunks = ['<div id="rag-source-chunks" style="display:none">']

    for n, frag in enumerate(used_fragments or [], start=1):
        real_id = html.escape(str(frag.get("fragment_id", "")))
        chunk_text = html.escape(str(frag.get("text", "")))

        aliases = [
            real_id,
            f"Фрагмент {n}",
            f"ФРАГМЕНТ {n}",
            f"fragment {n}",
            f"Fragment {n}",
            str(n),
        ]

        for alias in aliases:
            if alias:
                hidden_chunks.append(
                    f'<template data-fragment-id="{alias}">{chunk_text}</template>'
                )

    hidden_chunks.append('</div>')

    return answer + "\n\n" + "\n".join(hidden_chunks)

def load_knowledge_base(index_path, chunks_path):
    log("Загружаю FAISS индекс...")
    index = faiss.read_index(index_path)

    log("Загружаю чанки...")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    log("База загружена:")
    log("FAISS:", index_path)
    log("CHUNKS:", chunks_path)

    return index, chunks

GLOBAL_INDEX = None
GLOBAL_CHUNKS = []

debate_app_stage1 = None
debate_app_stage2 = None
debate_app_direct = None

current = get_active_kb()
print(f"Активная база при старте: {current}")

try:
    GLOBAL_INDEX, GLOBAL_CHUNKS = load_active_kb()
except Exception as e:
    print("❌ Ошибка загрузки базы:")
    raise

def save_knowledge_base(index, chunks, index_path, chunks_path):
    log("Сохраняю базу...")

    # создаём папки если нет
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(chunks_path).parent.mkdir(parents=True, exist_ok=True)

    if index is not None:
        log("Сохраняю FAISS индекс...")
        faiss.write_index(index, index_path)
    else:
        if os.path.exists(index_path):
            os.remove(index_path)
        log("FAISS индекс отсутствует")

    log("Сохраняю чанки...")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    log("База сохранена:")
    log("FAISS:", index_path)
    log("CHUNKS:", chunks_path)

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gpt-oss:120b-cloud"

DJVUTXT = 'djvutxt'  # r'C:\Users\a.kuznecov\DjVuLibre-3.5.17-win32\djvutxt.exe'
DJVUSED = 'djvused'  # r'C:\Users\a.kuznecov\DjVuLibre-3.5.17-win32\djvused.exe'

def count_pdf_text_chars(pdf_path: str, max_pages: int = 5) -> int:
    doc = fitz.open(pdf_path)
    limit = min(max_pages, len(doc))

    total_chars = 0

    for i in range(limit):
        text = doc[i].get_text("text") or ""
        total_chars += len(text.strip())

    doc.close()
    return total_chars


def detect_pdf_type(pdf_path: str, max_pages: int = 5) -> str:
    chars = count_pdf_text_chars(pdf_path, max_pages=max_pages)

    if chars > 0:
        return "pdf_text_layer"

    return "pdf_scanned_ocr"


def count_djvu_text_chars(djvu_path: str, max_pages: int = 5) -> int:
    total_pages = get_djvu_page_count(djvu_path)
    limit = min(max_pages, total_pages)

    total_chars = 0

    for page_num in range(1, limit + 1):
        text = extract_djvu_page_text(djvu_path, page_num) or ""
        total_chars += len(text.strip())

    return total_chars


def detect_djvu_type(djvu_path: str, max_pages: int = 5) -> str:
    chars = count_djvu_text_chars(djvu_path, max_pages=max_pages)

    if chars > 0:
        return "djvu_text_layer"

    return "djvu_scanned_ocr"


def detect_document_type(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return detect_pdf_type(file_path)

    if suffix == ".djvu":
        return detect_djvu_type(file_path)

    raise ValueError("Поддерживаются только PDF и DJVU")

def format_eta(start_time: float, percentage: int) -> str:
    elapsed = time.time() - start_time

    if percentage <= 0:
        return f"прошло {int(elapsed)} сек"

    total_estimated = elapsed / (percentage / 100)
    remaining = max(0, total_estimated - elapsed)

    if remaining < 60:
        return f"осталось ~{int(remaining)} сек"

    return f"осталось ~{round(remaining / 60, 1)} мин"


def emit_progress(
    progress_callback,
    percentage: int,
    status_text: str,
    book_name: str,
    start_time: float
):
    if not progress_callback:
        return

    progress_callback({
        "percentage": max(0, min(100, int(percentage))),
        "status_text": status_text,
        "book_name": book_name,
        "eta": format_eta(start_time, percentage),
    })

def add_document_to_kb(
    file_path: str,
    kb_name: str,
    source_name: str = None,
    progress_callback: Callable[[dict], None] = None
):
    start_time = time.time()

    if source_name is None:
        source_name = Path(file_path).stem

    emit_progress(progress_callback, 5, "Сохранение файла завершено", source_name, start_time)

    emit_progress(progress_callback, 10, "Определяю тип документа", source_name, start_time)
    doc_type = detect_document_type(file_path)

    emit_progress(progress_callback, 15, "Подключаю базу знаний", source_name, start_time)
    connect_kb(kb_name)

    before_chunks = len(GLOBAL_CHUNKS)

    if doc_type == "pdf_text_layer":
        build_knowledge_base_from_pdf(
            pdf_path=file_path,
            source_name=source_name,
            use_ocr_if_no_text=False,
            save=True,
            progress_callback=progress_callback,
            progress_start=20,
            progress_end=95,
            progress_started_at=start_time,
        )

    elif doc_type == "pdf_scanned_ocr":
        build_knowledge_base_from_pdf(
            pdf_path=file_path,
            source_name=source_name,
            use_ocr_if_no_text=True,
            ocr_lang="eng+rus",
            ocr_dpi=300,
            save=True,
            progress_callback=progress_callback,
            progress_start=20,
            progress_end=95,
            progress_started_at=start_time,
        )

    elif doc_type == "djvu_text_layer":
        build_knowledge_base_from_djvu(
            djvu_path=file_path,
            source_name=source_name,
            save=True,
            progress_callback=progress_callback,
            progress_start=20,
            progress_end=95,
            progress_started_at=start_time,
        )

    elif doc_type == "djvu_scanned_ocr":
        raise NotImplementedError(
            "DJVU без текстового слоя требует предварительной конвертации страниц в изображения/PDF для OCR."
        )

    after_chunks = len(GLOBAL_CHUNKS)

    emit_progress(progress_callback, 100, "Документ добавлен", source_name, start_time)

    return {
        "kb_name": kb_name,
        "source_name": source_name,
        "document_type": doc_type,
        "chunks_count": after_chunks - before_chunks,
        "total_chunks": after_chunks
    }

VERBOSE = True

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY не найден (будет работать только локальная модель)")
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ==================================================
#   ПРОМПТ: ИНСТРУКЦИЯ ПО ИЗГОТОВЛЕНИЮ
# ==================================================
PRACTICAL_INSTRUCTION_PROMPT = """
Ты — инженер-технолог и практик-производственник.

Твоя задача — на основе предоставленного контекста выдать ЧЁТКУЮ ИНСТРУКЦИЮ ПО ИЗГОТОВЛЕНИЮ
(созданию, сборке или реализации) продукта, технологии или системы.

⚠️ Ключевое требование:
Результат должен быть именно "КАК СДЕЛАТЬ", а не анализ, не обзор и не исследование.

Работай универсально (для любых областей: физика, IT, строительство, производство и др.)

---

Правила:
- Используй только предоставленный контекст
- Не выдумывай факты
- Не давай опасных или нереалистичных действий
- Если данных недостаточно — явно укажи, что шаг требует уточнения
- Исключи общие фразы (например: "провести анализ", "рассмотреть варианты")
- Каждый шаг должен быть физическим или логическим действием

---

❗ ВАЖНО:
Каждый шаг должен отвечать на вопросы:
- Что конкретно сделать?
- Чем (инструменты / материалы / код)?
- Как именно (операция / последовательность)?
- Как понять, что получилось?

---

Сформируй инструкцию в формате ТЕХПРОЦЕССА:

1. Подготовка материалов и инструментов  
2. Подготовка компонентов  
3. Изготовление (ключевые операции)  
4. Сборка / интеграция  
5. Завершение (обработка / настройка)  
6. Проверка работоспособности  
7. Использование / ввод в эксплуатацию  

---

Для каждого шага укажи:

- actions: последовательность КОНКРЕТНЫХ действий (глаголы: взять, подключить, нагреть, запустить, записать, собрать и т.д.)
- inputs: конкретные материалы, компоненты, данные или инструменты
- output: физический или измеримый результат

---

❗ Запрещено:
- писать абстрактно
- писать как исследование
- давать только теорию без действий

---

Верни СТРОГО JSON без пояснений.

Формат:
{
  "practical_guide": [
    {
      "step": 1,
      "title": "Название этапа",
      "actions": "...",
      "inputs": "...",
      "output": "..."
    }
  ]
}
"""

def ollama_chat(system: str, user: str, model: str = LLM_MODEL) -> str:

    prompt = f"""
{system}

{user}
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9
        }
    }

    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=300
    )

    r.raise_for_status()

    data = r.json()

    if "response" not in data:
        raise ValueError(f"Некорректный ответ Ollama: {data}")

    return data["response"].strip()

def clean_text_for_embedding(text: str) -> str:
    if not text:
        return ""

    # убираем управляющие символы
    text = re.sub(r'[\x00-\x1f\x7f]', ' ', text)

    # убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def get_embedding(text: str) -> np.ndarray:
    text = (text or "").strip()
    text = clean_text_for_embedding(text)

    if not text:
        raise ValueError("Пустой текст для embedding")

    payload = {
        "model": EMBED_MODEL,
        "prompt": text[:2000]
    }

    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json=payload,
        timeout=300
    )
    r.raise_for_status()

    data = r.json()

    if "embedding" not in data:
        raise ValueError(f"Нет embedding в ответе Ollama: {data}")

    return np.array(data["embedding"], dtype="float32")

def chatgpt_chat(system: str, user: str, model: str = "gpt-5.4") -> str:

    if client is None:
        raise RuntimeError("OPENAI_API_KEY не найден, ChatGPT-режим недоступен. Используйте локальный RAG через Ollama.")

    response = client.responses.create(
        model=model,
        instructions=system,
        input=user
    )

    return response.output_text.strip()

def is_first_chapter_page(text: str) -> bool:
    if not text:
        return False

    text = text.strip().lower()

    patterns = [
        r"\bchapter\s*1\b",
        r"\bглава\s*1\b",
        r"^\s*1[\.\s]+\w+",
        r"\bchapter one\b",
    ]

    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

def find_first_chapter_page(doc, max_pages_to_scan=None):
    if max_pages_to_scan is None:
        max_pages_to_scan = len(doc)

    limit = min(max_pages_to_scan, len(doc))
    log(f"Ищу первую главу в первых {limit} страницах...")

    for i in range(limit):
        text = doc[i].get_text("text")
        if (text):
            log(f"Первая глава найдена на странице: {i + 1}")
            return i

    log("Первая глава не найдена. Беру с начала документа.")
    return 0

def is_contents_page(text: str) -> bool:
    if not text:
        return False

    low = text.lower()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return False

    contents_keywords = [
        "contents",
        "table of contents",
        "оглавление",
        "содержание"
    ]
    if any(k in low for k in contents_keywords):
        return True

    section_only_count = 0
    page_only_count = 0
    dotted_count = 0
    short_title_count = 0
    section_title_count = 0
    long_paragraph_count = 0
    formula_number_count = 0
    math_symbol_count = 0

    for line in lines[:200]:
        if len(line) >= 90 and len(line.split()) >= 8:
            long_paragraph_count += 1

        if re.fullmatch(r"\(\d+(\.\d+)*\)", line):
            formula_number_count += 1

        if re.search(r"[=∑πωΩ±√∫]", line) or "exp(" in line.lower():
            math_symbol_count += 1

        if re.fullmatch(r"\d+(\.\d+)*", line):
            if "." in line:
                section_only_count += 1
            else:
                if int(line) <= 20:
                    section_only_count += 1
                else:
                    page_only_count += 1
            continue

        if re.fullmatch(r"\d{1,4}", line):
            page_only_count += 1
            continue

        if re.fullmatch(r"\.+", line) or re.fullmatch(r"(\.\s*){4,}\d*", line):
            dotted_count += 1
            continue

        if re.match(r"^\d+(\.\d+)*\s+\S+", line):
            section_title_count += 1
            continue

        words = line.split()
        if 1 <= len(words) <= 12 and len(line) <= 120:
            short_title_count += 1

    if long_paragraph_count >= 2:
        return False

    if formula_number_count >= 2:
        return False

    if math_symbol_count >= 2:
        return False

    if section_only_count >= 5 and page_only_count >= 5:
        return True

    if section_only_count >= 4 and dotted_count >= 4 and short_title_count >= 4:
        return True

    if page_only_count >= 5 and dotted_count >= 5:
        return True

    if section_title_count >= 5 and page_only_count >= 4:
        return True

    return False

def merge_toc_lines_from_blocks(blocks):
    items = []

    for b in blocks:
        if b.get("type", 0) != 0:
            continue

        if "lines" not in b:
            continue

        for line in b["lines"]:
            spans = line.get("spans", [])
            if not spans:
                continue

            line_text = " ".join(
                (span.get("text") or "").strip()
                for span in spans
                if (span.get("text") or "").strip()
            ).strip()

            if not line_text:
                continue

            bbox = line["bbox"] if "bbox" in line else b["bbox"]
            y0 = bbox[1]
            x0 = bbox[0]

            items.append((y0, x0, line_text))

    items.sort(key=lambda x: (round(x[0], 1), x[1]))
    return [text for _, _, text in items]

def is_index_page(blocks) -> bool:
    lines = merge_toc_lines_from_blocks(blocks)
    pattern = r"^[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9\-() /]+,\s*\d+(,\s*\d+)*$"
    index_score = sum(1 for text in lines if re.match(pattern, text.strip()))
    return index_score >= 5

def looks_like_math(text: str) -> bool:
    if not text:
        return False

    text = text.strip()

    if len(text) > 120:
        return False

    math_tokens = [
        "=", "∫", "π", "δ", "√", "∞",
        "sin", "cos", "tan", "exp",
        "dx", "dy", "dz",
        "∑", "±", "≤", "≥", "ω", "λ", "μ"
    ]

    if any(tok in text for tok in math_tokens):
        return True

    if re.search(r"\(\d+(\.\d+)*\)$", text):
        return True

    special_count = sum(ch in "=+-*/^_[]{}<>∫πδ√∞∑±≤≥ωλμ" for ch in text)
    if special_count >= 3:
        return True

    return False

def is_bold_font(font_name: str, flags: int = 0) -> bool:
    font_name = (font_name or "").lower()

    if "bold" in font_name:
        return True

    if isinstance(flags, int) and (flags & 16):
        return True

    return False

def classify_block(text: str, max_size: float, avg_size: float, bold_ratio: float) -> str:
    text = (text or "").strip()

    if not text:
        return "empty"

    if looks_like_math(text):
        return "formula"

    text_len = len(text)

    if max_size >= 16 and text_len <= 150:
        return "heading_1"

    if max_size >= 14 and text_len <= 180:
        return "heading_2"

    if max_size >= 12.5 and bold_ratio >= 0.5 and text_len <= 220:
        return "heading_3"

    return "text"

def extract_text_blocks(
    page,
    drop_headers=True,
    header_ratio=0.08,
    footer_ratio=0.06,
    use_ocr_if_no_text=True,
    ocr_lang="eng+rus",
    ocr_dpi=300,
    tesseract_config="--oem 3 --psm 4"
):
    page_height = page.rect.height
    page_width = page.rect.width
    raw_blocks = page.get_text("dict").get("blocks", [])

    blocks = []

    for b in raw_blocks:
        if b.get("type", 0) != 0:
            continue

        if "lines" not in b:
            continue

        x0, y0, x1, y1 = b["bbox"]

        if drop_headers:
            if y0 < page_height * header_ratio:
                continue
            if y1 > page_height * (1 - footer_ratio):
                continue

        text_parts = []
        spans_info = []

        for line in b["lines"]:
            for span in line.get("spans", []):
                span_text = (span.get("text") or "").strip()
                if not span_text:
                    continue

                size = span.get("size", 0)
                font = span.get("font", "")
                flags = span.get("flags", 0)
                bbox = span.get("bbox")

                text_parts.append(span_text)

                spans_info.append({
                    "text": span_text,
                    "size": size,
                    "font": font,
                    "flags": flags,
                    "bbox": bbox,
                    "is_bold": is_bold_font(font, flags),
                })

        text = " ".join(text_parts).strip()

        if not text:
            continue

        sizes = [s["size"] for s in spans_info if s["size"] is not None]
        max_size = max(sizes) if sizes else 0
        avg_size = sum(sizes) / len(sizes) if sizes else 0

        bold_count = sum(1 for s in spans_info if s["is_bold"])
        bold_ratio = bold_count / len(spans_info) if spans_info else 0

        block_type = classify_block(
            text=text,
            max_size=max_size,
            avg_size=avg_size,
            bold_ratio=bold_ratio
        )

        blocks.append({
            "text": text,
            "bbox": b["bbox"],
            "page_num": page.number + 1,
            "type": block_type,
            "max_size": max_size,
            "avg_size": avg_size,
            "bold_ratio": bold_ratio,
            "spans": spans_info,
        })
    
    if blocks or not use_ocr_if_no_text:
        return blocks
    
    log(f"OCR для страницы {page.number + 1}")

    pix = page.get_pixmap(dpi=ocr_dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))

    data = pytesseract.image_to_data(
        img,
        lang=ocr_lang,
        config=tesseract_config,
        output_type=Output.DICT
    )

    n = len(data["text"])
    if n == 0:
        return []

    lines_map = {}

    for i in range(n):
        word = (data["text"][i] or "").strip()
        conf_raw = data["conf"][i]

        try:
            conf = float(conf_raw)
        except:
            conf = -1

        if not word:
            continue

        if conf < 20:
            continue

        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])
        
        x0 = left * page_width / img.width
        y0 = top * page_height / img.height
        x1 = (left + width) * page_width / img.width
        y1 = (top + height) * page_height / img.height

        if drop_headers:
            if y0 < page_height * header_ratio:
                continue
            if y1 > page_height * (1 - footer_ratio):
                continue

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i]
        )

        if key not in lines_map:
            lines_map[key] = {
                "words": [],
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1
            }

        lines_map[key]["words"].append(word)
        lines_map[key]["x0"] = min(lines_map[key]["x0"], x0)
        lines_map[key]["y0"] = min(lines_map[key]["y0"], y0)
        lines_map[key]["x1"] = max(lines_map[key]["x1"], x1)
        lines_map[key]["y1"] = max(lines_map[key]["y1"], y1)

    if not lines_map:
        return []
    
    ocr_lines = []
    for _, item in lines_map.items():
        line_text = " ".join(item["words"]).strip()
        if not line_text:
            continue

        ocr_lines.append({
            "text": line_text,
            "bbox": (item["x0"], item["y0"], item["x1"], item["y1"])
        })
    
    ocr_lines.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))

    current_group = []
    current_bbox = None
    last_y1 = None

    for line in ocr_lines:
        text = line["text"]
        x0, y0, x1, y1 = line["bbox"]

        if current_group == []:
            current_group = [text]
            current_bbox = [x0, y0, x1, y1]
            last_y1 = y1
            continue

        vertical_gap = y0 - last_y1

        if vertical_gap <= 18:
            current_group.append(text)
            current_bbox[0] = min(current_bbox[0], x0)
            current_bbox[1] = min(current_bbox[1], y0)
            current_bbox[2] = max(current_bbox[2], x1)
            current_bbox[3] = max(current_bbox[3], y1)
            last_y1 = y1
        else:
            block_text = " ".join(current_group).strip()
            if block_text:
                block_type = classify_block(
                    text=block_text,
                    max_size=0,
                    avg_size=0,
                    bold_ratio=0
                )

                blocks.append({
                    "text": block_text,
                    "bbox": tuple(current_bbox),
                    "page_num": page.number + 1,
                    "type": block_type,
                    "max_size": 0,
                    "avg_size": 0,
                    "bold_ratio": 0,
                    "spans": [],
                })

            current_group = [text]
            current_bbox = [x0, y0, x1, y1]
            last_y1 = y1

    if current_group:
        block_text = " ".join(current_group).strip()
        if block_text:
            block_type = classify_block(
                text=block_text,
                max_size=0,
                avg_size=0,
                bold_ratio=0
            )

            blocks.append({
                "text": block_text,
                "bbox": tuple(current_bbox),
                "page_num": page.number + 1,
                "type": block_type,
                "max_size": 0,
                "avg_size": 0,
                "bold_ratio": 0,
                "spans": [],
            })

    return blocks

def get_working_page_numbers(doc, max_pages_to_scan=None):
    start_page = find_first_chapter_page(doc, max_pages_to_scan=max_pages_to_scan)

    keep_pages = []
    skipped_contents = []
    skipped_index = []

    log("Начинаю фильтрацию страниц...")

    for i in range(start_page, len(doc)):
        page = doc[i]

        text = page.get_text("text")
        blocks = page.get_text("dict")["blocks"]

        if is_contents_page(text):
            skipped_contents.append(i)
            log(f"Пропуск CONTENTS: стр. {i + 1}")
            continue

        if is_index_page(blocks):
            skipped_index.append(i)
            log(f"Пропуск INDEX:    стр. {i + 1}")
            continue

        keep_pages.append(i)

    log(f"Страниц оставлено: {len(keep_pages)}")
    log(f"Пропущено contents: {len(skipped_contents)}")
    log(f"Пропущено index:    {len(skipped_index)}")

    return keep_pages

def extract_blocks_from_working_pages(
    doc,
    working_pages,
    use_ocr_if_no_text=True,
    ocr_lang="eng+rus",
    ocr_dpi=300,
    tesseract_config="--oem 3 --psm 4",
    max_workers=None
):
    all_blocks = []

    log("Начинаю извлечение блоков...")

    if max_workers is None:
        max_workers = max(1, min(8, (os.cpu_count() or 4) - 1))

    def process_page(page_num):
        page = doc[page_num]
        page_blocks = extract_text_blocks(
            page,
            use_ocr_if_no_text=use_ocr_if_no_text,
            ocr_lang=ocr_lang,
            ocr_dpi=ocr_dpi,
            tesseract_config=tesseract_config
        )

        # for b in page_blocks:
        #     b["page_num"] = page_num

        return page_num, page_blocks

    page_results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {
            executor.submit(process_page, page_num): page_num
            for page_num in working_pages
        }

        for n, future in enumerate(as_completed(future_to_page), start=1):
            page_num = future_to_page[future]

            try:
                result_page_num, page_blocks = future.result()
                page_results[result_page_num] = page_blocks
                log(f"[{n}/{len(working_pages)}] стр. {result_page_num + 1} -> блоков: {len(page_blocks)}")
            except Exception as e:
                log(f"Ошибка на странице {page_num + 1}: {e}")
                page_results[page_num] = []

    # Собираем строго в порядке страниц
    for page_num in working_pages:
        all_blocks.extend(page_results.get(page_num, []))

    log(f"Всего извлечено блоков: {len(all_blocks)}")
    return all_blocks

def chunk_blocks_with_metadata(blocks, max_chars=1200, overlap=200, source_name="unknown_source"):
    chunks = []

    current_chapter = None
    current_section = None
    current_subsection = None

    current_text_parts = []
    current_pages = []
    current_formulas = []
    current_block_types = []

    def flush_chunk():
        nonlocal current_text_parts, current_pages, current_formulas, current_block_types

        text = "\n".join(part for part in current_text_parts if part.strip()).strip()
        if not text:
            return

        chunks.append({
            "text": text,
            "pages": sorted(set(current_pages)),
            "metadata": {
                "source": source_name,
                "chapter": current_chapter,
                "section": current_section,
                "subsection": current_subsection,
                "block_types": sorted(set(current_block_types)),
                "formulas": current_formulas.copy(),
            }
        })

        current_text_parts = []
        current_pages = []
        current_formulas = []
        current_block_types = []

    log("Начинаю сборку структурных чанков...")

    for i, b in enumerate(blocks, start=1):
        text = (b.get("text") or "").strip()
        block_type = b.get("type", "text")
        page_num = b.get("page_num")

        if not text:
            continue

        if block_type == "heading_1":
            flush_chunk()
            current_chapter = text
            current_section = None
            current_subsection = None
            log(f"heading_1 -> {text}")
            continue

        if block_type == "heading_2":
            flush_chunk()
            current_section = text
            current_subsection = None
            log(f"heading_2 -> {text}")
            continue

        if block_type == "heading_3":
            flush_chunk()
            current_subsection = text
            log(f"heading_3 -> {text}")
            continue

        if block_type == "formula":
            current_formulas.append(text)
            current_block_types.append("formula")

            formula_text = f"[FORMULA]\n{text}\n[/FORMULA]"
            candidate_text = "\n".join(current_text_parts + [formula_text]).strip()

            if len(candidate_text) > max_chars and current_text_parts:
                flush_chunk()

            current_text_parts.append(formula_text)
            current_pages.append(page_num)
            continue

        current_block_types.append(block_type)

        candidate_text = "\n".join(current_text_parts + [text]).strip()

        if len(candidate_text) <= max_chars:
            current_text_parts.append(text)
            current_pages.append(page_num)
        else:
            flush_chunk()

            if overlap > 0 and chunks:
                prev_text = chunks[-1]["text"]
                overlap_text = prev_text[-overlap:] if len(prev_text) > overlap else prev_text
                if overlap_text.strip():
                    current_text_parts.append(overlap_text)

            current_text_parts.append(text)
            current_pages.append(page_num)

        if i % 100 == 0:
            log(f"Обработано блоков: {i}/{len(blocks)} | чанков сейчас: {len(chunks)}")

    flush_chunk()
    log(f"Всего создано чанков: {len(chunks)}")

    return chunks

def build_chunks_from_pdf(
    doc, 
    source_name="book.pdf", 
    max_pages_to_scan=None, 
    max_chars=1200, 
    overlap=200,
    use_ocr_if_no_text=True,
    ocr_lang='eng+rus',
    ocr_dpi=300,
    tesseract_config='--oem 3 --psm 6'
    ):
    working_pages = get_working_page_numbers(doc, max_pages_to_scan=max_pages_to_scan)
    log("Страницы после очистки:", len(working_pages))

    all_blocks = extract_blocks_from_working_pages(doc, working_pages)
    log("Всего блоков:", len(all_blocks))

    all_blocks = extract_blocks_from_working_pages(
        doc,
        working_pages,
        use_ocr_if_no_text=use_ocr_if_no_text,
        ocr_lang=ocr_lang,
        ocr_dpi=ocr_dpi,
        tesseract_config=tesseract_config,
        max_workers=6
    )
    log('Всего блоков:', len(all_blocks))

    chunks = chunk_blocks_with_metadata(
        all_blocks,
        max_chars=max_chars,
        overlap=overlap,
        source_name=source_name
    )
    log("Всего чанков:", len(chunks))

    return chunks

def get_djvu_page_count(djvu_path):
    result = subprocess.run(
        [DJVUSED, '-e', 'n', djvu_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return int(result.stdout.strip())


def extract_djvu_page_text(djvu_path, page_num):
    result = subprocess.run(
        [DJVUTXT, f'--page={page_num}', djvu_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))

    return result.stdout.decode("utf-8", errors="ignore")

def _extract_one_page(args):
    djvu_path, page_num = args
    text = extract_djvu_page_text(djvu_path, page_num)
    return {
        "page": page_num,
        "text": text
    }


def extract_djvu_by_pages_fast(djvu_path: str, max_workers: int = 8):
    total_pages = get_djvu_page_count(djvu_path)
    pages = [None] * total_pages

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_extract_one_page, (djvu_path, page_num))
            for page_num in range(1, total_pages + 1)
        ]

        for future in as_completed(futures):
            item = future.result()
            pages[item["page"] - 1] = item    
    return pages

def djvu_pages_to_blocks(pages):
    blocks = []

    current_paragraph = []
    current_page = None

    current_chapter = None
    current_section = None
    current_subsection = None

    def flush_paragraph():
        nonlocal current_paragraph, current_page

        if not current_paragraph:
            return

        paragraph_text = " ".join(current_paragraph).strip()
        if not paragraph_text:
            current_paragraph = []
            return

        block_type = "formula" if looks_like_math(paragraph_text) else "text"

        blocks.append({
            "text": paragraph_text,
            "type": block_type,
            "page_num": current_page,
            "chapter": current_chapter,
            "section": current_section,
            "subsection": current_subsection
        })

        current_paragraph = []

    for page in pages:
        page_num = page["page"]
        text = page["text"].strip()

        if not text:
            continue

        lines = text.split("\n")
        current_page = page_num

        for line in lines:
            line = line.strip()
            if not line:
                flush_paragraph()
                continue
            
            if re.match(r"^chapter\s+\d+", line, flags=re.IGNORECASE):
                flush_paragraph()

                current_chapter = line
                current_section = None
                current_subsection = None

                blocks.append({
                    "text": line,
                    "type": "heading_1",
                    "page_num": page_num,
                    "chapter": current_chapter,
                    "section": None,
                    "subsection": None
                })
                continue
            
            if re.match(r"^\d+\.\d+\b", line):
                flush_paragraph()

                current_section = line
                current_subsection = None

                blocks.append({
                    "text": line,
                    "type": "heading_2",
                    "page_num": page_num,
                    "chapter": current_chapter,
                    "section": current_section,
                    "subsection": None
                })
                continue
            
            if re.match(r"^\d+\.\d+\.\d+\b", line):
                flush_paragraph()

                current_subsection = line

                blocks.append({
                    "text": line,
                    "type": "heading_3",
                    "page_num": page_num,
                    "chapter": current_chapter,
                    "section": current_section,
                    "subsection": current_subsection
                })
                continue
            
            if looks_like_math(line):
                flush_paragraph()

                blocks.append({
                    "text": line,
                    "type": "formula",
                    "page_num": page_num,
                    "chapter": current_chapter,
                    "section": current_section,
                    "subsection": current_subsection
                })
                continue

            current_paragraph.append(line)

        flush_paragraph()

    return blocks

def build_embeddings_for_chunks(
    chunks,
    progress_callback=None,
    source_name="Документ",
    progress_start=55,
    progress_end=85,
    progress_started_at=None
):
    vectors = []
    enriched_chunks = []

    if progress_started_at is None:
        progress_started_at = time.time()

    log("Начинаю создание эмбеддингов...")

    total = len(chunks)

    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        emb = get_embedding(text)

        vectors.append(emb)

        enriched_chunk = {
            "chunk_id": i,
            "text": text,
            "pages": chunk.get("pages", []),
            "metadata": chunk.get("metadata", {}),
            "embedding": emb.tolist() if hasattr(emb, "tolist") else emb
        }

        enriched_chunks.append(enriched_chunk)

        if progress_callback and total > 0:
            percent = progress_start + ((i + 1) / total) * (progress_end - progress_start)

            emit_progress(
                progress_callback,
                int(percent),
                f"Генерация эмбеддингов {i + 1}/{total}",
                source_name,
                progress_started_at
            )

        if (i + 1) % 10 == 0 or (i + 1) == total:
            log(f"Эмбеддинги: {i + 1}/{total}")

    vectors = np.array(vectors, dtype="float32")

    log("Создание эмбеддингов завершено.")
    log("Форма массива vectors:", vectors.shape)

    return enriched_chunks, vectors

def build_faiss_index(vectors: np.ndarray):
    if len(vectors.shape) != 2:
        raise ValueError(f"Ожидался 2D массив vectors, получено: {vectors.shape}")

    log("Начинаю построение FAISS индекса...")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    log(f"FAISS индекс готов. Векторов в базе: {index.ntotal}")
    return index

def search_in_db(query: str, index=None, all_chunks=None, k: int = 5):
    global GLOBAL_INDEX, GLOBAL_CHUNKS

    if index is None:
        index = GLOBAL_INDEX

    if all_chunks is None:
        all_chunks = GLOBAL_CHUNKS

    if index is None or all_chunks is None:
        print("База не загружена.")
        return []

    log(f"Поиск по базе. Запрос: {query}")

    q = get_embedding(query).astype("float32").reshape(1, -1)

    if q.shape[1] != index.d:
        print("Ошибка: размерность embedding запроса не совпадает с базой.")
        print("Размерность запроса:", q.shape[1])
        print("Размерность базы:", index.d)
        return []

    D, I = index.search(q, k)

    results = []

    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue

        chunk_data = all_chunks[idx]
        metadata = chunk_data.get("metadata", {})

        results.append({
            "idx": int(idx),
            "distance": float(dist),
            "text": chunk_data.get("text", ""),
            "pages": chunk_data.get("pages", []),
            "source": metadata.get("source"),
            "chapter": metadata.get("chapter"),
            "section": metadata.get("section"),
            "subsection": metadata.get("subsection"),
            "formulas": metadata.get("formulas", []),
            "block_types": metadata.get("block_types", []),
        })

    log(f"Найдено результатов: {len(results)}")
    return results

def build_knowledge_base_from_pdf(
    pdf_path: str,
    source_name: str = None,
    max_pages_to_scan: int = 80,
    max_chars: int = 1200,
    overlap: int = 200,
    save: bool = True,
    use_ocr_if_no_text: bool = True,
    ocr_lang: str = 'eng+rus',
    ocr_dpi: int = 300,
    tesseract_config: str = '--oem 3 --psm6',
    progress_callback=None,
    progress_start=20,
    progress_end=95,
    progress_started_at=None
):
    global GLOBAL_INDEX, GLOBAL_CHUNKS

    if source_name is None:
        from pathlib import Path
        source_name = Path(pdf_path).stem

    log("=" * 80)
    log("СТАРТ ДОБАВЛЕНИЯ PDF В БАЗУ")
    log("PDF:", pdf_path)
    log("SOURCE:", source_name)
    log("=" * 80)

    start_time = time.time()

    if progress_started_at is None:
        progress_started_at = start_time

    emit_progress(progress_callback, progress_start, "Чтение PDF", source_name, progress_started_at)

    doc = fitz.open(pdf_path)
    log("Всего страниц в PDF:", len(doc))

    chunks = build_chunks_from_pdf(
        doc,
        source_name=source_name,
        max_pages_to_scan=max_pages_to_scan,
        max_chars=max_chars,
        overlap=overlap,
        use_ocr_if_no_text=use_ocr_if_no_text,
        ocr_lang=ocr_lang,
        ocr_dpi=ocr_dpi,
        tesseract_config=tesseract_config
    )

    emit_progress(
        progress_callback,
        50,
        f"Создание чанков: {len(chunks)}",
        source_name,
        progress_started_at
    )

    if not chunks:
        log("Чанки не найдены.")
        return GLOBAL_INDEX, GLOBAL_CHUNKS

    if chunks:
        log("Пример первого чанка:")
        log("pages:", chunks[0].get("pages"))
        log("metadata:", chunks[0].get("metadata"))
        log("text preview:", chunks[0].get("text", "")[:300])

    enriched_chunks, vectors = build_embeddings_for_chunks(
        chunks,
        progress_callback=progress_callback,
        source_name=source_name,
        progress_start=55,
        progress_end=85,
        progress_started_at=progress_started_at
    )
    
    if GLOBAL_INDEX is None or len(GLOBAL_CHUNKS) == 0:
        GLOBAL_INDEX = build_faiss_index(vectors)
        GLOBAL_CHUNKS = enriched_chunks.copy()
    else:        
        GLOBAL_INDEX.add(vectors)
        GLOBAL_CHUNKS.extend(enriched_chunks)

    if save:
        emit_progress(
            progress_callback,
            92,
            "Сохранение в FAISS индекс",
            source_name,
            progress_started_at
        )

        save_active_kb()

    elapsed = time.time() - start_time

    log("=" * 80)
    log("БАЗА ГОТОВА")
    log("Всего чанков:", len(GLOBAL_CHUNKS))
    log("Векторов в индексе:", GLOBAL_INDEX.ntotal)
    log("Время:", round(elapsed / 60, 2), "мин")
    log("=" * 80)

    return GLOBAL_INDEX, GLOBAL_CHUNKS


def build_knowledge_base_from_djvu(
    djvu_path: str,
    source_name: str = None,
    max_chars: int = 1200,
    overlap: int = 200,
    save: bool = True,
    progress_callback=None,
    progress_start=20,
    progress_end=95,
    progress_started_at=None
):
    global GLOBAL_INDEX, GLOBAL_CHUNKS

    if source_name is None:
        source_name = Path(djvu_path).stem

    start_time = time.time()

    if progress_started_at is None:
        progress_started_at = start_time

    log("=" * 80)
    log("СТАРТ ДОБАВЛЕНИЯ DJVU В БАЗУ")
    log("DJVU:", djvu_path)
    log("SOURCE:", source_name)
    log("АКТИВНАЯ БАЗА:", get_active_kb())
    log("=" * 80)

    emit_progress(
        progress_callback,
        progress_start,
        "Чтение DJVU",
        source_name,
        progress_started_at
    )

    pages = extract_djvu_by_pages_fast(djvu_path)
    log("Всего страниц в DJVU:", len(pages))

    emit_progress(
        progress_callback,
        40,
        f"Извлечение текста DJVU: {len(pages)} стр.",
        source_name,
        progress_started_at
    )

    blocks = djvu_pages_to_blocks(pages)
    log("Всего блоков:", len(blocks))

    emit_progress(
        progress_callback,
        50,
        f"Создание блоков: {len(blocks)}",
        source_name,
        progress_started_at
    )

    chunks = chunk_blocks_with_metadata(
        blocks,
        max_chars=max_chars,
        overlap=overlap,
        source_name=source_name
    )
    log("Всего чанков:", len(chunks))

    emit_progress(
        progress_callback,
        55,
        f"Создание чанков: {len(chunks)}",
        source_name,
        progress_started_at
    )

    if not chunks:
        log("Чанки не найдены.")
        return GLOBAL_INDEX, GLOBAL_CHUNKS

    enriched_chunks, vectors = build_embeddings_for_chunks(
        chunks,
        progress_callback=progress_callback,
        source_name=source_name,
        progress_start=60,
        progress_end=85,
        progress_started_at=progress_started_at
    )

    if GLOBAL_INDEX is None or len(GLOBAL_CHUNKS) == 0:
        GLOBAL_INDEX = build_faiss_index(vectors)
        GLOBAL_CHUNKS = enriched_chunks.copy()
    else:
        GLOBAL_INDEX.add(vectors)
        GLOBAL_CHUNKS.extend(enriched_chunks)

    if save:
        emit_progress(
            progress_callback,
            92,
            "Сохранение в FAISS индекс",
            source_name,
            progress_started_at
        )

        save_active_kb()

    elapsed = time.time() - start_time

    log("=" * 80)
    log("DJVU ДОБАВЛЕНА В БАЗУ")
    log("АКТИВНАЯ БАЗА:", get_active_kb())
    log("Всего чанков:", len(GLOBAL_CHUNKS))
    log("Векторов в индексе:", GLOBAL_INDEX.ntotal if GLOBAL_INDEX is not None else 0)
    log("Время:", round(elapsed / 60, 2), "мин")
    log("=" * 80)

    return GLOBAL_INDEX, GLOBAL_CHUNKS

def list_books():
    global GLOBAL_CHUNKS

    if not GLOBAL_CHUNKS:
        print("База пустая")
        return

    books = set()

    for chunk in GLOBAL_CHUNKS:
        if not isinstance(chunk, dict):
            continue

        metadata = chunk.get("metadata", {})
        source = metadata.get("source")
        if source:
            books.add(source)
            continue

        if "book" in chunk and chunk["book"]:
            books.add(chunk["book"])

    books = sorted(books)

    print("\nКниги в базе:\n")
    for b in books:
        print("-", b)

    print("\nВсего книг:", len(books))

class DebateState(TypedDict):
    # === БАЗА ===
    question: str
    # === АНАЛИЗ ВОПРОСА ===
    question_type: List[str]
    answer_mode: str
    entities: List[dict]
    constraints: List[str]
    required_answer_elements: List[str]
    # === ПОДЗАДАЧИ ===
    subtasks: List[dict]
    selected_subtasks: List[dict]
    subtask_results: List[dict]
    # === КОНТЕКСТ ===
    retrieved_context: str
    retrieved_items: List[dict]
    additional_query: str
    additional_context: str
    subtask_contexts: List[dict]
    # === ОСНОВНОЕ РАССУЖДЕНИЕ ===
    hypothesis: str
    criticism: str
    evidence: str
    used_fragments: List[str]
    # === УТОЧНЕНИЕ ===
    clarification_aspects: List[dict]
    selected_aspects: List[dict]
    additional_user_input: str    
    refined_question: str 
    refined_hypothesis: str
    # === СИНТЕЗ ===
    synthesized_answer: str
    practical_guide: str        
    # === ФИНАЛ ===
    final_answer: str

def build_context_from_results(results: list[dict]) -> str:
    if not results:
        return ""

    parts = []

    for i, item in enumerate(results, start=1):
        source = item.get("source", "unknown_source")
        chapter = item.get("chapter") or "—"
        section = item.get("section") or "—"
        subsection = item.get("subsection") or "—"
        pages = item.get("pages", [])
        distance = item.get("distance", 0.0)
        text = item.get("text", "").strip()
        formulas = item.get("formulas", [])

        page_str = ", ".join(map(str, pages)) if pages else "—"

        formulas_block = "нет"
        if formulas:
            formulas_block = "\n".join(f"- {f}" for f in formulas[:5])

        part = (
            f"[ФРАГМЕНТ {i}]\n"
            f"Источник: {source}\n"
            f"Глава: {chapter}\n"
            f"Раздел: {section}\n"
            f"Подраздел: {subsection}\n"
            f"Страницы: {page_str}\n"
            f"Distance: {distance:.4f}\n"
            f"Формулы:\n{formulas_block}\n"
            f"Текст:\n{text}\n"
        )
        parts.append(part)

    return "\n\n".join(parts)

def retrieve_context_node(state: DebateState) -> dict:
    question = (state.get("refined_question") or state.get("question") or "").strip()

    print("\n=== RETRIEVE CONTEXT ===")
    print("Вопрос для retrieval:")
    print(question)

    results = search_in_db(question, k=5)
    context = build_context_from_results(results)

    return {
        "retrieved_items": results,
        "retrieved_context": context
    }

def retrieve_context_from_chatgpt_node(state: dict) -> dict:    
    question = state["question"]
    log("Debate node: retrieve_context_from_chatgpt")

    system = """
Ты аналитик-исследователь.

Твоя задача:
- прочитать вопрос пользователя;
- дать предварительный контекст по теме;
- выделить ключевые факты, идеи, ограничения и возможные подходы;
- не писать лишнюю воду;
- если уверенность низкая, прямо скажи об этом;
- пиши строго на русском языке.

Формат ответа:
КОНТЕКСТ
...
"""

    user = f"""
Вопрос:
{question}

Собери предварительный аналитический контекст по этому вопросу.
"""

    context = chatgpt_chat(system=system, user=user)

    return {
        "retrieved_items": [
            {
                "source": "ChatGPT",
                "text": context
            }
        ],
        "retrieved_context": context
    }

def extract_json_block(text: str) -> dict:
    if not text:
        return {}

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}

    return {}


def analyze_question_node(state: DebateState) -> dict:
    question = state.get("refined_question") or state["question"]

    system = """
Ты аналитик научных и технических вопросов.

Твоя задача:
1. определить тип вопроса;
2. определить режим итогового ответа;
3. выделить сущности вопроса;
4. выделить ограничения;
5. определить, что обязательно должно быть в хорошем ответе.

Верни СТРОГО JSON без пояснений.

Формат:
{
  "question_type": ["..."],
  "answer_mode": "...",
  "entities": [
    {
      "type": "...",
      "value": "...",
      "role": "..."
    }
  ],
  "constraints": [
    "..."
  ],
  "required_answer_elements": [
    "..."
  ]
}

Допустимые question_type:
- explanatory
- design/proposal
- comparative
- feasibility
- procedural

Допустимые answer_mode:
- explanation
- proposal
- comparison
- feasibility
- procedure
"""

    user = f"""
Вопрос:
{question}
"""

    raw = ollama_chat(system=system, user=user)
    data = extract_json_block(raw)

    return {
        "question_type": data.get("question_type", []),
        "answer_mode": data.get("answer_mode", "explanation"),
        "entities": data.get("entities", []),
        "constraints": data.get("constraints", []),
        "required_answer_elements": data.get("required_answer_elements", [])
    }

def build_subtasks_node(state: DebateState) -> dict:
    question = state.get("refined_question") or state["question"]
    question_type = state.get("question_type", [])
    answer_mode = state.get("answer_mode", "")
    entities = state.get("entities", [])
    constraints = state.get("constraints", [])
    required_answer_elements = state.get("required_answer_elements", [])

    system = """
Ты системный аналитик.

Твоя задача:
разбить вопрос на минимально достаточные подзадачи, чтобы итоговый ответ был полным.

Требования:
- подзадачи должны быть на русском языке;
- подзадачи должны покрывать смысл вопроса;
- подзадачи не должны дублировать друг друга;
- каждая подзадача должна быть полезна для финального ответа;
- подзадачи должны быть универсальными, без привязки только к технологии.

Верни СТРОГО JSON без пояснений.

Формат:
{
  "subtasks": [
    {
      "id": "subtask_1",
      "title": "...",
      "goal": "...",
      "why_needed": "...",
      "depends_on": []
    }
  ]
}
"""

    user = f"""
Вопрос:
{question}

Тип вопроса:
{question_type}

Режим ответа:
{answer_mode}

Сущности:
{json.dumps(entities, ensure_ascii=False, indent=2)}

Ограничения:
{json.dumps(constraints, ensure_ascii=False, indent=2)}

Обязательные элементы ответа:
{json.dumps(required_answer_elements, ensure_ascii=False, indent=2)}
"""

    raw = ollama_chat(system=system, user=user)
    data = extract_json_block(raw)

    subtasks = data.get("subtasks", [])

    return {
        "subtasks": subtasks
    }

def retrieve_context_for_subtasks_node(state: DebateState) -> dict:
    """
    Для каждой подзадачи делает отдельный retrieval.
    Использует refined_question, если он уже появился после уточнения.
    """
    question = (state.get("refined_question") or state.get("question") or "").strip()
    subtasks = state.get("subtasks", [])

    subtask_contexts = []

    print("\n=== RETRIEVE CONTEXT FOR SUBTASKS ===")
    print(f"Главный вопрос для подзадач: {question}")
    print(f"Количество подзадач: {len(subtasks)}")

    for i, subtask in enumerate(subtasks, start=1):
        subtask_id = subtask.get("id", f"subtask_{i}")
        title = (subtask.get("title") or "").strip()
        goal = (subtask.get("goal") or "").strip()

        # Короткий поисковый запрос под конкретную подзадачу
        subquery_parts = [
            f"Главный вопрос: {question}",
            f"Подзадача: {title}",
        ]
        if goal:
            subquery_parts.append(f"Цель подзадачи: {goal}")

        subquery = "\n".join(subquery_parts)

        print(f"\n--- Подзадача {i} ---")
        print("ID:", subtask_id)
        print("TITLE:", title)
        print("GOAL:", goal)
        print("SUBQUERY:")
        print(subquery)

        results = search_in_db(subquery, k=5)
        context = build_context_from_results(results)

        print(f"Найдено фрагментов: {len(results)}")

        subtask_contexts.append({
            "subtask_id": subtask_id,
            "title": title,
            "goal": goal,
            "query": subquery,
            "context": context,
            "items": results
        })

    return {"subtask_contexts": subtask_contexts}

def solve_subtasks_node(state: DebateState) -> dict:
    """
    Решает каждую подзадачу на основе ее собственного контекста.
    """
    question = state.get("refined_question") or state["question"]
    subtask_contexts = state.get("subtask_contexts", [])
    results = []

    print("\n=== SOLVE SUBTASKS ===")
    print(f"Количество подзадач для решения: {len(subtask_contexts)}")

    system = """
Ты аналитик RAG-системы.

Решай только одну подзадачу.
Используй только переданный контекст.
Не выдумывай факты вне контекста.

Верни строго JSON без пояснений.

Формат:
{
  "answer": "краткий, но содержательный ответ по подзадаче",
  "what_is_supported": ["что подтверждено источниками"],
  "what_is_missing": ["чего не хватает в источниках"],
  "confidence": "low | medium | high"
}
""".strip()

    for i, item in enumerate(subtask_contexts, start=1):
        title = item.get("title", "")
        goal = item.get("goal", "")
        context = item.get("context", "")
        subtask_id = item.get("subtask_id", f"subtask_{i}")
        sources = item.get("items", [])

        print(f"\n--- Решение подзадачи {i} ---")
        print("ID:", subtask_id)
        print("TITLE:", title)
        print("GOAL:", goal)
        print(f"Источников у подзадачи: {len(sources)}")

        user = f"""
Главный вопрос:
{question}

Подзадача:
{title}

Цель подзадачи:
{goal}

Контекст:
{context}
""".strip()

        raw = ollama_chat(system=system, user=user)
        data = extract_json_block(raw)

        if not isinstance(data, dict):
            data = {
                "answer": raw.strip(),
                "what_is_supported": [],
                "what_is_missing": ["LLM не вернула корректный JSON"],
                "confidence": "low"
            }

        results.append({
            "id": subtask_id,
            "title": title,
            "goal": goal,
            "answer": data.get("answer", ""),
            "what_is_supported": data.get("what_is_supported", []),
            "what_is_missing": data.get("what_is_missing", []),
            "confidence": data.get("confidence", "medium"),
            "sources": sources,
            "query": item.get("query", ""),
            "context": context
        })

    return {"subtask_results": results}

def synthesize_answer_node(state: DebateState) -> dict:
    question = state.get("refined_question") or state["question"]
    answer_mode = state.get("answer_mode", "explanation")
    subtask_results = state.get("subtask_results", [])
    constraints = state.get("constraints", [])
    required_answer_elements = state.get("required_answer_elements", [])

    system = """
Ты финальный аналитик, который синтезирует промежуточные результаты в единый ответ.

Твоя задача:
- объединить результаты подзадач;
- явно закрыть главный вопрос;
- сохранить честность: отделять подтверждённое от предположений;
- учитывать ограничения вопроса;
- не терять смысловые части вопроса.

Верни СТРОГО JSON без пояснений.

Формат:
{
  "synthesized_answer": "...",  
}
"""

    user = f"""
Главный вопрос:
{question}

Режим ответа:
{answer_mode}

Ограничения:
{json.dumps(constraints, ensure_ascii=False, indent=2)}

Обязательные элементы ответа:
{json.dumps(required_answer_elements, ensure_ascii=False, indent=2)}

Результаты по подзадачам:
{json.dumps(subtask_results, ensure_ascii=False, indent=2)}
"""

    raw = ollama_chat(system=system, user=user)
    data = extract_json_block(raw)

    return {
        "synthesized_answer": data.get("synthesized_answer", "")        
    }

def build_hypothesis_node(state: DebateState) -> dict:
    log("Debate node: build_hypothesis")

    question = state["question"]
    context = state.get("retrieved_context", "")

    if not context.strip():
        return {
            "hypothesis": "Гипотеза не может быть сформулирована, потому что в базе не найден релевантный контекст."
        }

    system = """
Ты аналитик RAG-системы.

Твоя задача:
- прочитать вопрос;
- прочитать контекст;
- сформулировать предварительную гипотезу;
- опираться только на контекст;
- не выдумывать факты;
- не использовать знания вне контекста;
- если данных мало, прямо скажи об этом.

Пиши на русском языке.
Выводи только текст гипотезы.
"""

    user = f"""
Вопрос:
{question}

Контекст:
{context}

Сформулируй предварительную гипотезу по вопросу строго на основе контекста.
"""

    hypothesis = ollama_chat(system=system, user=user)
    return {"hypothesis": hypothesis}

def critic_review_node(state: DebateState) -> dict:
    log("Debate node: critic_review")

    question = state["question"]
    context = state.get("retrieved_context", "")
    hypothesis = state.get("hypothesis", "")

    if not context.strip():
        return {
            "criticism": "Критика невозможна, потому что отсутствует найденный контекст."
        }

    system = """
Ты критик-аналитик.

Твоя задача:
- проверить гипотезу на слабые места;
- указать, что подтверждено хорошо;
- указать, что подтверждено слабо;
- отметить противоречия;
- отметить, каких данных не хватает;
- не выдумывать факты;
- использовать только переданный контекст.

Пиши на русском языке.
Выводи только критический разбор.
"""

    user = f"""
Вопрос:
{question}

Контекст:
{context}

Гипотеза:
{hypothesis}

Проведи критический анализ гипотезы.
"""

    criticism = ollama_chat(system=system, user=user)
    return {"criticism": criticism}

def propose_clarification_aspects_node(state: DebateState) -> dict:
    question = state["question"]
    context = state.get("retrieved_context", "")
    hypothesis = state.get("hypothesis", "")
    criticism = state.get("criticism", "")

    log("Debate node: propose_clarification_aspects")

    system = r"""
Ты научный ассистент исследовательской RAG-системы.

Твоя задача:
- проанализировать вопрос пользователя;
- проанализировать найденный контекст;
- проанализировать исходную гипотезу;
- проанализировать критику гипотезы;
- предложить несколько уточняющих аспектов вопроса, которые помогут сузить тему и улучшить дальнейшее рассуждение.

Что нужно сделать:
1. Выдели от 4 до 7 уточняющих аспектов темы.
2. Эти аспекты должны быть такими, чтобы пользователь мог выбрать НЕСКОЛЬКО сразу.
3. Аспекты должны быть короткими, понятными и разными по смыслу.
4. Не предлагай действия вроде "сделать поиск" или "уточнить гипотезу".
5. Предлагай именно тематические пункты уточнения.

Примеры хороших аспектов:
- для вопроса о погоде и связи: дождь, снег, туман, ветер, мороз
- для вопроса о строительстве: стоимость, сроки, материалы, нормативы, риски
- для вопроса о связи: качество сигнала, задержка, стабильность, покрытие, помехи

Требования:
1. Не придумывай факты, которых нет в контексте.
2. Если по вопросу уже явно видны естественные аспекты темы, выдели их.
3. Каждый аспект должен иметь:
   - id
   - title
   - description
4. Верни ответ СТРОГО в виде JSON-массива.
5. Не добавляй никакой текст вне JSON.

Формат ответа:
[
  {
    "id": 1,
    "title": "Дождь",
    "description": "Влияние дождя и повышенной влажности на качество сигнала и передачу данных."
  },
  {
    "id": 2,
    "title": "Снег",
    "description": "Влияние снегопада, налипания снега и зимних осадков на работу сети."
  }
]
"""

    user = f"""
ВОПРОС:
{question}

КОНТЕКСТ:
{context}

ИСХОДНАЯ ГИПОТЕЗА:
{hypothesis}

КРИТИКА:
{criticism}
"""

    raw = ollama_chat(system, user).strip()
    
    try:
        aspects = json.loads(raw)
        if isinstance(aspects, list):
            normalized = []
            for i, item in enumerate(aspects, start=1):
                if isinstance(item, dict):
                    normalized.append({
                        "id": item.get("id", i),
                        "title": str(item.get("title", "")).strip(),
                        "description": str(item.get("description", "")).strip()
                    })
            if normalized:
                return {"clarification_aspects": normalized}
    except Exception:
        pass
   
    try:
        match = re.search(r"\[\s*{.*}\s*\]", raw, flags=re.DOTALL)
        if match:
            aspects = json.loads(match.group(0))
            if isinstance(aspects, list):
                normalized = []
                for i, item in enumerate(aspects, start=1):
                    if isinstance(item, dict):
                        normalized.append({
                            "id": item.get("id", i),
                            "title": str(item.get("title", "")).strip(),
                            "description": str(item.get("description", "")).strip()
                        })
                if normalized:
                    return {"clarification_aspects": normalized}
    except Exception:
        pass
    
    fallback_aspects = [
        {
            "id": 1,
            "title": "Основные условия",
            "description": "Уточнить, какие именно ключевые условия или факторы нужно рассматривать в вопросе."
        },
        {
            "id": 2,
            "title": "Ограничения",
            "description": "Уточнить, есть ли ограничения, при которых вывод может измениться."
        },
        {
            "id": 3,
            "title": "Практические последствия",
            "description": "Уточнить, какие именно практические эффекты или результаты интересуют пользователя."
        },
        {
            "id": 4,
            "title": "Технические параметры",
            "description": "Уточнить, какие количественные или технические характеристики наиболее важны."
        },
        {
            "id": 5,
            "title": "Внешние факторы",
            "description": "Уточнить, какие внешние воздействия или условия нужно отдельно учесть."
        }
    ]

    return {"clarification_aspects": fallback_aspects}

def select_clarification_aspects_widget(state: DebateState) -> dict:
    aspects = state.get("clarification_aspects", [])

    if not aspects:
        print("Нет уточняющих аспектов для выбора.")
        return {"selected_aspects": []}

    title = widgets.HTML("<b>Выберите один или несколько уточняющих пунктов:</b>")
    output = widgets.Output()
    button = widgets.Button(description="Подтвердить выбор")

    checkboxes = []
    for aspect in aspects:
        cb = widgets.Checkbox(
            value=False,
            description=str(aspect.get("title", "")),
            indent=False
        )
        desc = widgets.HTML(
            value=(
                f"<div style='margin-left: 24px; color: gray;'>"
                f"{aspect.get('description', '')}"
                f"</div>"
            )
        )
        checkboxes.append((cb, aspect, desc))

    result = {"selected_aspects": []}

    def on_button_click(_):
        selected = [aspect for cb, aspect, _ in checkboxes if cb.value]

        with output:
            clear_output()

            if not selected:
                print("Нужно выбрать хотя бы один уточняющий пункт.")
                return

            result["selected_aspects"] = selected

            print("Вы выбрали:")
            for item in selected:
                print(f"- {item.get('title', '')}")

    ui_items = [title]
    for cb, _, desc in checkboxes:
        ui_items.extend([cb, desc])

    ui_items.extend([button, output])

    button.on_click(on_button_click)
    display(widgets.VBox(ui_items))

    return result

def apply_selected_clarification_node(state: DebateState) -> dict:
    question = state["question"]
    selected_aspects = state.get("selected_aspects", [])
    hypothesis = state.get("hypothesis", "")
    criticism = state.get("criticism", "")
    context = state.get("retrieved_context", "")
    additional_user_input = state.get("additional_user_input", "")

    log("Debate node: apply_selected_clarification")
    
    if not selected_aspects:
        return {
            "refined_question": question,
            "refined_hypothesis": hypothesis        }

    
    aspect_titles = [
        str(item.get("title", "")).strip()
        for item in selected_aspects
        if str(item.get("title", "")).strip()
    ]

    aspects_text = ", ".join(aspect_titles)

    system = r"""
Ты научный ассистент исследовательской RAG-системы.

Твоя задача:
1. На основе исходного вопроса и выбранных пользователем уточняющих аспектов
   сформулировать более точный уточнённый вопрос.
2. На основе уточнённого вопроса, исходной гипотезы, критики и контекста
   сформулировать более точную уточнённую гипотезу.

Требования:
- не придумывай новые факты;
- опирайся только на исходный вопрос, контекст, гипотезу, критику и выбранные аспекты;
- уточнённый вопрос должен быть коротким, ясным и естественным;
- уточнённая гипотеза должна быть осторожнее, точнее и логичнее исходной;
- если данных недостаточно, прямо отражай это в формулировке гипотезы;
- верни ответ строго в JSON-объекте с полями:
  - refined_question
  - refined_hypothesis

Пример формата:
{
  "refined_question": "Как дождь и туман влияют на качество сотовой связи?",
  "refined_hypothesis": "Дождь и туман могут ухудшать качество сотовой связи за счёт ослабления сигнала и роста помех, однако степень влияния зависит от частотного диапазона, плотности сети и погодной интенсивности."
}
"""

    user = f"""
ИСХОДНЫЙ ВОПРОС:
{question}

ВЫБРАННЫЕ УТОЧНЯЮЩИЕ АСПЕКТЫ:
{aspects_text}

ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ОТ ПОЛЬЗОВАТЕЛЯ:
{additional_user_input}

КОНТЕКСТ:
{context}

ИСХОДНАЯ ГИПОТЕЗА:
{hypothesis}

КРИТИКА:
{criticism}
"""

    raw = ollama_chat(system, user).strip()
    
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            refined_question = str(data.get("refined_question", "")).strip()
            refined_hypothesis = str(data.get("refined_hypothesis", "")).strip()

            return {
                "refined_question": refined_question or question,
                "refined_hypothesis": refined_hypothesis or hypothesis
            }
    except Exception:
        pass
    
    try:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                refined_question = str(data.get("refined_question", "")).strip()
                refined_hypothesis = str(data.get("refined_hypothesis", "")).strip()

                return {
                    "refined_question": refined_question or question,
                    "refined_hypothesis": refined_hypothesis or hypothesis
                }
    except Exception:
        pass
    
    fallback_refined_question = f"{question} (уточнение: {aspects_text})"

    fallback_refined_hypothesis = hypothesis
    if aspect_titles:
        fallback_refined_hypothesis = (
            f"{hypothesis}\n\n"
            f"Дополнительное уточнение: особое внимание следует уделить аспектам: {aspects_text}."
        ).strip()

    return {
        "refined_question": fallback_refined_question,
        "refined_hypothesis": fallback_refined_hypothesis
    }

def evidence_check_node(state: DebateState) -> dict:
    """
    Собирает доказательства по всем подзадачам на основе их собственных источников.
    """
    subtask_results = state.get("subtask_results", [])

    evidence_lines = []
    used_fragments = []

    print("\n=== EVIDENCE CHECK ===")
    print(f"Подзадач: {len(subtask_results)}")

    for idx, sub in enumerate(subtask_results, start=1):
        title = sub.get("title", "")
        answer = sub.get("answer", "")
        confidence = sub.get("confidence", "medium")
        sources = sub.get("sources", [])

        evidence_lines.append(f"Подзадача {idx}: {title}")
        evidence_lines.append(f"Ответ: {answer}")
        evidence_lines.append(f"Уверенность: {confidence}")

        if not sources:
            evidence_lines.append("Источники: не найдены")
            evidence_lines.append("")
            continue

        evidence_lines.append("Источники подзадачи:")

        for j, src in enumerate(sources, start=1):
            # Поддержка двух форматов источников:
            # 1) {"chunk": {...}}
            # 2) обычный результат search_in_db: {"text": ..., "source": ..., "pages": ...}
            chunk = src.get("chunk", src) if isinstance(src, dict) else {}
            text = (chunk.get("text") or "").strip()

            metadata = chunk.get("metadata", {}) or {}
            source_name = metadata.get("source") or chunk.get("source") or "unknown_source"
            pages = chunk.get("pages", []) or metadata.get("pages", [])
            page_str = ", ".join(map(str, pages)) if pages else "?"

            fragment_id = f"S{idx}.F{j}"

            short_text = text[:400].replace("\n", " ")
            evidence_lines.append(
                f"- [{fragment_id}] Источник: {source_name}, страницы: {page_str}, фрагмент: {short_text}"
            )

            used_fragments.append({
                "fragment_id": fragment_id,
                "subtask_title": title,
                "source": source_name,
                "pages": pages,
                "text": text
            })

        evidence_lines.append("")

    evidence_text = "\n".join(evidence_lines)

    return {
        "evidence": evidence_text,
        "used_fragments": used_fragments
    }

def practical_realization_node(state: DebateState) -> dict:
    question = state.get("refined_question") or state["question"]

    context = state.get("retrieved_context", "")
    hypothesis = state.get("refined_hypothesis") or state.get("hypothesis", "")
    criticism = state.get("criticism", "")
    evidence = state.get("evidence", "")

    synthesized_answer = state.get("synthesized_answer", "")
    subtask_results = state.get("subtask_results", [])
    required_answer_elements = state.get("required_answer_elements", [])
    constraints = state.get("constraints", [])
    question_type = state.get("question_type", [])
    answer_mode = state.get("answer_mode", "")

    system = PRACTICAL_INSTRUCTION_PROMPT

    user = f"""
Главный вопрос:
{question}

Тип вопроса:
{json.dumps(question_type, ensure_ascii=False, indent=2)}

Режим ответа:
{answer_mode}

Ограничения:
{json.dumps(constraints, ensure_ascii=False, indent=2)}

Обязательные элементы ответа:
{json.dumps(required_answer_elements, ensure_ascii=False, indent=2)}

Синтезированный ответ:
{synthesized_answer}

Ответы по подзадачам:
{json.dumps(subtask_results, ensure_ascii=False, indent=2)}

Гипотеза:
{hypothesis}

Критика:
{criticism}

Доказательства:
{evidence}

Контекст:
{context}
"""

    raw = ollama_chat(system=system, user=user)
    data = extract_json_block(raw)

    return {
        "practical_guide": data.get("practical_guide", "")
    }

def _format_practical_guide_to_md(practical_guide_data) -> str:
    """Преобразует JSON практической реализации в готовую Markdown-таблицу."""
    if not practical_guide_data:
        return "_Данные по практической реализации отсутствуют._"
    
    try:
        # Поддержка и строки, и готовых объектов
        if isinstance(practical_guide_data, str):
            data = json.loads(practical_guide_data)
        else:
            data = practical_guide_data
            
        steps = data.get("practical_guide", []) if isinstance(data, dict) else data
        if not isinstance(steps, list) or not steps:
            return "_Шаги практической реализации не найдены._"
            
        md = "| Шаг | Название | Действия | Входные данные / Материалы | Ожидаемый результат |\n"
        md += "|:---|:---|:---|:---|:---|\n"
        
        for step in steps:
            # Экранируем | и переносы строк, чтобы не ломать таблицу
            s = str(step.get("step", ""))
            title = str(step.get("title", ""))
            actions = str(step.get("actions", "")).replace("|", "\\|").replace("\n", " ")
            inputs = str(step.get("inputs", "")).replace("|", "\\|").replace("\n", " ")
            output = str(step.get("output", "")).replace("|", "\\|").replace("\n", " ")
            md += f"| {s} | {title} | {actions} | {inputs} | {output} |\n"
            
        return md
    except Exception as e:
        return f"_Ошибка парсинга практической реализации: {str(e)}_"


def final_conclusion_node(state: Dict[str, Any]) -> dict:
    question = state.get("refined_question") or state.get("question", "")
    context = state.get("retrieved_context", "")
    hypothesis = state.get("refined_hypothesis") or state.get("hypothesis", "")
    criticism = state.get("criticism", "")
    evidence = state.get("evidence", "")
    synthesized_answer = state.get("synthesized_answer", "")
    practical_guide_raw = state.get("practical_guide", "")

    # 1. Форматируем практическую реализацию в таблицу НА СТОРОНЕ PYTHON
    practical_guide_md = _format_practical_guide_to_md(practical_guide_raw)

    # 2. Компактный JSON для экономии контекста
    def safe_json(obj):
        return json.dumps(obj, ensure_ascii=False)

    system_template = """Ты финальный аналитик RAG-системы. Твоя задача — синтезировать полный, точный и структурированный ответ на основе предоставленных данных.

ПРАВИЛА:
1. Используй ТОЛЬКО предоставленные данные. Не выдумывай факты.
2. Форматируй ответ строго в Markdown. Все формулы оборачивай в $...$ или $$...$$.
3. Раздел "КРИТИКА ГИПОТЕЗЫ" оформи строго как Markdown-таблицу.
4. Раздел "ПРАКТИЧЕСКАЯ РЕАЛИЗАЦИЯ" уже отформатирован как таблица. Вставь его как есть, не меняй структуру.
5. Явно отделяй подтверждённые факты от предположений. Укажи уровень уверенности.
6. Если данных по разделу нет, напиши "Данных недостаточно".
7. Не добавляй вводные фразы вроде "Вот ответ:". Сразу начинай с заголовков.

СТРУКТУРА ОТВЕТА:
## СУЩНОСТИ ВОПРОСА
- ...

## ПОДЗАДАЧИ
- ...

## ГИПОТЕЗА
...

## КРИТИКА ГИПОТЕЗЫ
| Аспект | Подтверждено | Противоречие/Слабые места | Недостающие данные |
|---|---|---|---|
| ... | ... | ... | ... |

## ОТВЕТЫ ПО ПОДЗАДАЧАМ
- ...

## ДОКАЗАТЕЛЬСТВА
- Подтверждает: ...
- Ослабляет: ...

## ИСТОЧНИКИ
- [Фрагмент N] Источник: ... | Глава: ... | Страницы: ...

## СИНТЕЗИРОВАННЫЙ ОТВЕТ
...

## ПРАКТИЧЕСКАЯ РЕАЛИЗАЦИЯ
{{PRACTICAL_TABLE}}

## ИТОГОВЫЙ ВЫВОД
...

## УРОВЕНЬ УВЕРЕННОСТИ
высокий / средний / низкий
"""

    # Подставляем готовую таблицу в промпт
    system_prompt = system_template.replace("{{PRACTICAL_TABLE}}", practical_guide_md)

    user_prompt = f"""ВОПРОС: {question}
ТИП: {safe_json(state.get('question_type', []))}
СУЩНОСТИ: {safe_json(state.get('entities', []))}
ПОДЗАДАЧИ: {safe_json(state.get('subtasks', []))}
ОТВЕТЫ ПО ПОДЗАДАЧАМ: {safe_json(state.get('subtask_results', []))}
КОНТЕКСТ: {context}
ГИПОТЕЗА: {hypothesis}
КРИТИКА: {criticism}
ДОКАЗАТЕЛЬСТВА: {evidence}
СИНТЕЗ: {synthesized_answer}
ОБЯЗАТЕЛЬНЫЕ ЭЛЕМЕНТЫ: {safe_json(state.get('required_answer_elements', []))}

Сформируй ответ строго по структуре."""

    final_answer = ollama_chat(system=system_prompt, user=user_prompt)

    # Нормализация ответа (убираем экранированные переносы, которые часто шлёт Ollama)
    if not final_answer:
        return {"final_answer": "⚠️ Модель не вернула ответ."}
        
    final_answer = (
        final_answer.replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace("\\r", "")
                    .strip()
    )
    
    return {"final_answer": final_answer}

def build_debate_graph_stage1():
    graph = StateGraph(DebateState)

    graph.add_node("analyze_question", analyze_question_node)
    graph.add_node("build_subtasks", build_subtasks_node)
    graph.add_node("retrieve_context", retrieve_context_node)  # твоя текущая нода
    graph.add_node("build_hypothesis", build_hypothesis_node)
    graph.add_node("critic_review", critic_review_node)
    graph.add_node("propose_clarification_aspects", propose_clarification_aspects_node)

    graph.add_edge(START, "analyze_question")
    graph.add_edge("analyze_question", "build_subtasks")
    graph.add_edge("build_subtasks", "retrieve_context")
    graph.add_edge("retrieve_context", "build_hypothesis")
    graph.add_edge("build_hypothesis", "critic_review")
    graph.add_edge("critic_review", "propose_clarification_aspects")
    graph.add_edge("propose_clarification_aspects", END)

    return graph.compile()

def build_debate_graph_stage2():
    graph = StateGraph(DebateState)

    graph.add_node("apply_selected_clarification", apply_selected_clarification_node)
    graph.add_node("retrieve_context_for_subtasks", retrieve_context_for_subtasks_node)
    graph.add_node("solve_subtasks", solve_subtasks_node)
    graph.add_node("evidence_check", evidence_check_node)
    graph.add_node("synthesize_answer", synthesize_answer_node)
    graph.add_node("practical_realization", practical_realization_node)
    graph.add_node("final_conclusion", final_conclusion_node)

    graph.add_edge(START, "apply_selected_clarification")
    graph.add_edge("apply_selected_clarification", "retrieve_context_for_subtasks")
    graph.add_edge("retrieve_context_for_subtasks", "solve_subtasks")
    graph.add_edge("solve_subtasks", "evidence_check")
    graph.add_edge("evidence_check", "synthesize_answer")
    graph.add_edge("synthesize_answer", "practical_realization")
    graph.add_edge("practical_realization", "final_conclusion")
    graph.add_edge("final_conclusion", END)

    return graph.compile()

def build_debate_graph_direct():
    """
    Полный граф без этапа уточняющих аспектов.
    Вопрос сразу проходит весь путь до финального ответа.
    """
    graph = StateGraph(DebateState)

    graph.add_node("analyze_question", analyze_question_node)
    graph.add_node("build_subtasks", build_subtasks_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("build_hypothesis", build_hypothesis_node)
    graph.add_node("critic_review", critic_review_node)
    graph.add_node("retrieve_context_for_subtasks", retrieve_context_for_subtasks_node)
    graph.add_node("solve_subtasks", solve_subtasks_node)
    graph.add_node("evidence_check", evidence_check_node)
    graph.add_node("synthesize_answer", synthesize_answer_node)
    graph.add_node("practical_realization", practical_realization_node)
    graph.add_node("final_conclusion", final_conclusion_node)

    graph.add_edge(START, "analyze_question")
    graph.add_edge("analyze_question", "build_subtasks")
    graph.add_edge("build_subtasks", "retrieve_context")
    graph.add_edge("retrieve_context", "build_hypothesis")
    graph.add_edge("build_hypothesis", "critic_review")
    graph.add_edge("critic_review", "retrieve_context_for_subtasks")
    graph.add_edge("retrieve_context_for_subtasks", "solve_subtasks")
    graph.add_edge("solve_subtasks", "evidence_check")
    graph.add_edge("evidence_check", "synthesize_answer")
    graph.add_edge("synthesize_answer", "practical_realization")
    graph.add_edge("practical_realization", "final_conclusion")
    graph.add_edge("final_conclusion", END)

    return graph.compile()

def init_debate_apps():
    global debate_app_stage1, debate_app_stage2, debate_app_direct

    # Старые графы оставлены для совместимости, но основной сервер использует debate_app_direct.
    debate_app_stage1 = build_debate_graph_stage1()
    debate_app_stage2 = build_debate_graph_stage2()
    debate_app_direct = build_debate_graph_direct()

    log("Debate graphs собраны: stage1, stage2 и direct без уточнения аспектов.")

def start_debate_rag(question: str) -> DebateState:
    global debate_app_stage1

    if debate_app_stage1 is None:
        raise ValueError("Debate graph stage1 не собран. Сначала вызови init_debate_apps().")

    log("Запуск debate RAG stage1...")

    state: DebateState = {
        "question": question,

        "retrieved_context": "",
        "retrieved_items": [],

        "hypothesis": "",
        "criticism": "",
        "evidence": "",

        "clarification_aspects": [],
        "selected_aspects": [],

        "additional_user_input": "",
        "refined_question": question,
        "refined_hypothesis": "",

        "additional_query": "",
        "additional_context": "",

        "final_answer": ""
    }

    result = debate_app_stage1.invoke(state)
    return result

def continue_debate_rag(state: DebateState) -> DebateState:
    global debate_app_stage2

    if debate_app_stage2 is None:
        raise ValueError("Debate graph stage2 не собран. Сначала вызови init_debate_apps().")

    log("Запуск debate RAG stage2...")

    result = debate_app_stage2.invoke(state)
    return result

def ask_debate_rag_direct(question: str) -> str:
    """
    Основная функция для сайта: сразу ищет ответ без выбора аспектов.
    """
    global debate_app_direct

    if debate_app_direct is None:
        init_debate_apps()

    state: DebateState = {
        "question": question,
        "question_type": [],
        "answer_mode": "explanation",
        "entities": [],
        "constraints": [],
        "required_answer_elements": [],
        "subtasks": [],
        "selected_subtasks": [],
        "subtask_results": [],
        "retrieved_context": "",
        "retrieved_items": [],
        "additional_query": "",
        "additional_context": "",
        "subtask_contexts": [],
        "hypothesis": "",
        "criticism": "",
        "evidence": "",
        "clarification_aspects": [],
        "selected_aspects": [],
        "additional_user_input": "",
        "refined_question": question,
        "refined_hypothesis": "",
        "synthesized_answer": "",
        "practical_guide": "",
        "used_fragments": [],
        "final_answer": ""
    }

    result = debate_app_direct.invoke(state)

    answer = result.get("final_answer", "Ответ не сформирован.")
    used_fragments = result.get("used_fragments", [])

    hidden_chunks = ['<div id="rag-source-chunks" style="display:none">']

    for n, frag in enumerate(used_fragments, start=1):
        real_id = html.escape(str(frag.get("fragment_id", "")))
        chunk_text = html.escape(str(frag.get("text", "")))

        aliases = [
            real_id,                 # S1.F1
            f"Фрагмент {n}",          # Фрагмент 1
            f"ФРАГМЕНТ {n}",          # ФРАГМЕНТ 1
            f"fragment {n}",          # fragment 1
            f"Fragment {n}",          # Fragment 1
            str(n),                  # 1
        ]

        for alias in aliases:
            if alias:
                hidden_chunks.append(
                    f'<template data-fragment-id="{alias}">{chunk_text}</template>'
                )

    hidden_chunks.append('</div>')

    return answer + "\n\n" + "\n".join(hidden_chunks)

def ask_debate_chatgpt(question: str, show_result: bool = True) -> str:
    log("Запуск debate ChatGPT mode (без уточнений)...")

    def normalize_final_answer(text: str) -> str:
        if not text:
            return ""

        text = str(text).strip()

        # Если пришло как JSON-строка
        try:
            loaded = json.loads(text)
            if isinstance(loaded, str):
                text = loaded
        except Exception:
            pass

        # Мягкая нормализация, чтобы не ломать формулы
        text = text.replace("\\n", "\n")
        text = text.replace("\\t", "\t")
        text = text.replace("\\r", "")
        text = text.replace("\\u202f", " ")
        text = text.replace("\\u00a0", " ")

        return text.strip()

    state = {
        "question": question,
        "refined_question": question,

        "retrieved_context": "",
        "retrieved_items": [],

        "entities": [],
        "subtasks": [],

        "subtask_contexts": [],
        "subtask_results": [],

        "hypothesis": "",
        "criticism": "",
        "evidence": "",
        "used_fragments": [],
        "final_answer": ""
    }

    try:
        # 1. Общий поиск через ChatGPT
        state.update(retrieve_context_from_chatgpt_node({
            "question": question
        }))

        # 2. Декомпозиция вопроса
        state.update(build_subtasks_node(state))

        # 3. Отдельный поиск через ChatGPT по каждой подзадаче
        subtask_contexts = []

        for i, subtask in enumerate(state.get("subtasks", []), start=1):
            title = (subtask.get("title") or "").strip()
            goal = (subtask.get("goal") or "").strip()
            subtask_id = subtask.get("id", f"subtask_{i}")

            subquery = f"""
Главный вопрос:
{question}

Подзадача:
{title}

Цель подзадачи:
{goal}
""".strip()

            sub_state = {"question": subquery}
            sub_retrieved = retrieve_context_from_chatgpt_node(sub_state)

            subtask_contexts.append({
                "subtask_id": subtask_id,
                "title": title,
                "goal": goal,
                "query": subquery,
                "context": sub_retrieved.get("retrieved_context", ""),
                "items": sub_retrieved.get("retrieved_items", [])
            })

        state["subtask_contexts"] = subtask_contexts

        # 4. Решение подзадач
        state.update(solve_subtasks_node(state))

        # 5. Гипотеза и критика
        state.update(build_hypothesis_node(state))
        state.update(critic_review_node(state))

        # 6. Проверка доказательств
        state.update(evidence_check_node(state))

        # 7. Финальный вывод
        state.update(final_conclusion_node(state))

        final_answer = normalize_final_answer(state.get("final_answer", ""))
        
        if final_answer:
            final_answer = f"## ВОПРОС\n{question}\n\n" + final_answer

        if show_result:
            clear_output(wait=True)
            if final_answer:
                display(Markdown(final_answer))
            else:
                print("⚠️ final_answer пустой")
                print(state)

        return final_answer

    except Exception as e:
        if show_result:
            clear_output(wait=True)
            print("❌ Ошибка:", repr(e))
        raise

if __name__ == "__main__":
    init_debate_apps()
    print("✅ RAG Core инициализирован")