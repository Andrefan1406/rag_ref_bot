import httpx
from typing import Any, Dict, List

from ..config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL


def detect_lang(text: str) -> str:
    """
    Детектор: если есть кириллица — считаем RU, иначе EN.
    """
    t = (text or "").lower()
    for ch in t:
        if ("а" <= ch <= "я") or (ch == "ё"):
            return "ru"
    return "en"


def _lang_name(lang: str) -> str:
    return "русском" if lang == "ru" else "английском"


async def _ollama_chat(messages: List[Dict[str, str]], timeout: int = 180) -> str:
    """
    Единый вызов Ollama /api/chat.
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_LLM_MODEL,
        "messages": messages,
        "stream": False,
        # Можно подкрутить поведение тут
        "options": {
            "temperature": 0.2,
            "repeat_penalty": 1.2
        },
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()

    return (data.get("message") or {}).get("content", "").strip()


def _system_prompt_for_answer(answer_lang: str) -> str:
    lang_name = _lang_name(answer_lang)
    return f"""
Ты отвечаешь строго на {lang_name} языке — на языке запроса пользователя.
Если контекст на другом языке — используй его, но переводя смысл на {lang_name}.
Если в контексте недостаточно данных — напиши:
"В представленном контексте недостаточно данных для полного ответа на поставленный вопрос."
""".strip()


async def generate_answer(question: str, context: str) -> str:
    
    answer_lang = detect_lang(question)
    system_prompt = _system_prompt_for_answer(answer_lang)

    user_prompt = f"""
КОНТЕКСТ:
{context}

ВОПРОС:
{question}
""".strip()

    return await _ollama_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        timeout=180,
    )


# =========================
# ДВУХАГЕНТНЫЙ РЕЖИМ: ГИПОТЕЗА ↔ КРИТИК
# =========================

def _sources_prompt(context: str) -> str:
    
    return f"ИСТОЧНИКИ (фрагменты из базы):\n{context}".strip()


def _hypothesis_system(answer_lang: str) -> str:
    lang_name = _lang_name(answer_lang)
    return f"""
Ты — аналитический исследователь, работающий с RAG-контекстом.

ПРАВИЛА:
1. Пиши строго на {lang_name}. Не используй смешение языков.
2. Используй ТОЛЬКО информацию из источников ниже.
3. Нельзя добавлять знания вне источников.
4. Не повторяй одни и те же мысли разными словами.
5. Пиши кратко, логично и структурированно.
6. Не используй вводные рассуждения вроде "возможно", "можно предположить", если это не следует из источников.
7. Если данных недостаточно — прямо напиши:
   "Недостаточно данных в источниках."

СТИЛЬ ОТВЕТА:
- строгий научный
- короткие абзацы
- списки вместо длинных текстов
- избегай повторений

ЗАПРЕЩЕНО:
- домысливать
- использовать внешние знания
- смешивать русский и английский
""".strip()


def _critic_system(answer_lang: str, critic_style: str) -> str:
    lang_name = _lang_name(answer_lang)
    return f"""
Ты — критик научной аргументации ({critic_style}).

ПРАВИЛА:
1. Пиши строго на {lang_name}.
2. Проверяй утверждения ТОЛЬКО по источникам.
3. Не добавляй знания вне источников.
4. Не повторяй одинаковые замечания.
5. Пиши кратко и структурированно.

ЦЕЛЬ КРИТИКИ:
Найти реальные слабые места аргументации:

- утверждения без опоры на источники
- логические скачки
- противоречия между источниками
- двусмысленные формулировки
- недоказанные выводы

ФОРМАТ ОТВЕТА:
Нумерованный список.

Каждый пункт:
1. Название проблемы
2. Короткое объяснение (1–2 предложения)
3. Ссылка на фрагмент источника

Не повторяй одинаковые замечания.
""".strip()


def _hypothesis_user(question: str, sources_block: str) -> str:
    return f"""
ВОПРОС:
{question}

{sources_block}

Сформулируй:
1) Гипотезу/тезис (1–3 предложения).
2) 3–6 поддерживающих утверждений.
3) Для каждого утверждения укажи, на какой фрагмент(ы) источников ты опираешься (цитируй коротко 1–2 предложения).
Если данных недостаточно — напиши: "Недостаточно данных в источниках."
""".strip()


def _critic_user(question: str, sources_block: str, hypothesis_text: str) -> str:
    return f"""
ВОПРОС:
{question}

{sources_block}

ТЕКУЩАЯ ГИПОТЕЗА/ОБОСНОВАНИЕ:
{hypothesis_text}

Найди 3–8 слабых мест:
- где тезис не подтверждён источниками
- где есть логический скачок
- где двусмысленность
- где возможны альтернативные объяснения (если они прямо следуют из источников)
- где могут быть противоречия между фрагментами

Формат: маркированный список.
""".strip()


def _revise_user(question: str, sources_block: str, hypothesis_text: str, critique_text: str) -> str:
    return f"""
ВОПРОС:
{question}

{sources_block}

ТЕКУЩАЯ ГИПОТЕЗА:
{hypothesis_text}

КРИТИКА:
{critique_text}

Перепиши гипотезу так, чтобы она была:
- максимально строгая
- опиралась ТОЛЬКО на источники
- явно отмечала ограничения и нехватку данных

Выдай:
1) Обновлённую гипотезу (1–3 предложения)
2) 3–6 поддерживающих утверждений (с короткими цитатами/опорами на источники)
3) Блок "Ограничения" (1–5 пунктов)
""".strip()


def _final_user(question: str, sources_block: str, last_revision: str) -> str:
    return f"""
ВОПРОС:
{question}

{sources_block}

ФИНАЛЬНАЯ ВЕРСИЯ ГИПОТЕЗЫ:
{last_revision}

Собери итоговый ответ строго в формате:

ВЫВОД
Краткий ответ на вопрос (3–5 предложений).

ЧТО ПОДТВЕРЖДЕНО ИСТОЧНИКАМИ
- пункт
- пункт
- пункт

ЧТО НЕ ПОДТВЕРЖДЕНО / ЧЕГО НЕ ХВАТАЕТ
- пункт
- пункт
- пункт

ПРАВИЛА:
- не повторять одни и те же мысли
- не смешивать языки
- писать кратко
- не использовать знания вне источников
""".strip()


async def generate_debate(
    question: str,
    context: str,
    rounds: int = 2,
    critic_style: str = "строгий",
) -> Dict[str, Any]:
    """
    Возвращает:
    {
      "final_answer": str,
      "turns": [
          {"round": 1, "hypothesis": "...", "critique": "...", "revision": "..."},
          ...
      ]
    }
    """
    rounds = max(1, min(int(rounds or 2), 4))
    answer_lang = detect_lang(question)
    sources_block = _sources_prompt(context)

    # 1) стартовая гипотеза
    hypothesis = await _ollama_chat(
        messages=[
            {"role": "system", "content": _hypothesis_system(answer_lang)},
            {"role": "user", "content": _hypothesis_user(question, sources_block)},
        ],
        timeout=240,
    )

    turns: List[Dict[str, Any]] = []

    # 2) циклы критики/переписывания
    for r in range(1, rounds + 1):
        critique = await _ollama_chat(
            messages=[
                {"role": "system", "content": _critic_system(answer_lang, critic_style)},
                {"role": "user", "content": _critic_user(question, sources_block, hypothesis)},
            ],
            timeout=240,
        )

        revision = await _ollama_chat(
            messages=[
                {"role": "system", "content": _hypothesis_system(answer_lang)},
                {"role": "user", "content": _revise_user(question, sources_block, hypothesis, critique)},
            ],
            timeout=240,
        )

        turns.append(
            {
                "round": r,
                "hypothesis": hypothesis,
                "critique": critique,
                "revision": revision,
            }
        )

        hypothesis = revision

    # 3) финальная сборка ответа
    final_answer = await _ollama_chat(
        messages=[
            {"role": "system", "content": _hypothesis_system(answer_lang)},
            {"role": "user", "content": _final_user(question, sources_block, hypothesis)},
        ],
        timeout=240,
    )

    return {"final_answer": final_answer, "turns": turns}
