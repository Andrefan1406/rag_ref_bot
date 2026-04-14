from __future__ import annotations
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from ..services.vector_store import search as vs_search
from ..services.llm import generate_answer   # <-- добавь импорт


class RAGState(TypedDict, total=False):
    question: str
    top_k: int
    hits: List[Dict[str, Any]]
    answer: str
    sources: List[Dict[str, Any]]


async def node_retrieve(state: RAGState) -> Dict[str, Any]:
    q = state["question"]
    k = int(state.get("top_k", 5))
    hits = await vs_search(q, top_k=k)
    return {"hits": hits}


def route_has_hits(state: RAGState) -> str:
    return "generate" if state.get("hits") else "fallback"


def node_fallback(state: RAGState) -> Dict[str, Any]:
    return {"answer": "Ничего не найдено в базе.", "sources": []}


async def node_generate(state: RAGState) -> Dict[str, Any]:
    hits = state["hits"]

    # 1) убираем дубликаты (source_file + chunk_id)
    seen = set()
    uniq = []
    for h in hits:
        key = (h.get("source_file"), h.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(h)

    # 2) собираем контекст
    context = "\n\n".join([h.get("text", "") for h in uniq])

    # 3) генерируем ответ
    answer = await generate_answer(question=state["question"], context=context)

    return {"answer": answer, "sources": uniq}


def build_rag_graph():
    g = StateGraph(RAGState)

    g.add_node("retrieve", node_retrieve)
    g.add_node("fallback", node_fallback)
    g.add_node("generate", node_generate)   # <-- новый узел

    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route_has_hits, {
        "fallback": "fallback",
        "generate": "generate",
    })
    g.add_edge("fallback", END)
    g.add_edge("generate", END)

    return g.compile()


rag_graph = build_rag_graph()