"""Medicao da qualidade da busca.

Sem numero nao da para saber se um ajuste melhorou ou so mudou. O conjunto de
avaliacao lista perguntas com as paginas onde a resposta realmente esta
(conferidas no PDF), e as perguntas fora do assunto, que devem devolver zero
trecho.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import BASE_DIR
from .store import Hit, get_store

EVAL_DIR = Path(BASE_DIR / "data" / "eval")


def load_set(collection: str) -> dict:
    path = EVAL_DIR / f"{collection}.json"
    if not path.exists():
        raise FileNotFoundError(f"sem conjunto de avaliacao para '{collection}': {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _pages(hit: Hit) -> set[int]:
    """Paginas cobertas pelo trecho (um chunk pode atravessar a virada)."""
    start = hit.chunk.metadata.get("page")
    if start is None:
        return set()
    return set(range(start, hit.chunk.metadata.get("page_end", start) + 1))


def _is_relevant(hit: Hit, item: dict) -> bool:
    """Gabarito por pagina (PDF) ou por trecho literal que o chunk deve conter."""
    if "pages" in item:
        return bool(_pages(hit) & set(item["pages"]))
    return item["contains"].lower() in hit.chunk.text.lower()


def evaluate(collection: str, top_k: int | None = None, **search_kwargs) -> dict:
    """Roda o conjunto e devolve as metricas agregadas.

    - acerto@k : a resposta apareceu em algum lugar do top-K
    - MRR      : quao no topo ela apareceu (1.0 = sempre em primeiro)
    - precisao : fracao dos trechos devolvidos que eram de fato relevantes
    - ruido    : trechos devolvidos para perguntas fora do assunto (ideal: 0)
    """
    data = load_set(collection)
    store = get_store(collection)

    hits_at_k, reciprocal, relevant, returned = 0, 0.0, 0, 0
    detail = []
    for item in data["questions"]:
        results = store.search(item["q"], top_k=top_k, **search_kwargs)
        marks = [_is_relevant(h, item) for h in results]
        rank = marks.index(True) + 1 if any(marks) else 0

        hits_at_k += 1 if rank else 0
        reciprocal += 1 / rank if rank else 0.0
        relevant += sum(marks)
        returned += len(results)
        detail.append({"question": item["q"], "rank": rank, "returned": len(results)})

    noise = sum(
        len(store.search(q, top_k=top_k, **search_kwargs)) for q in data.get("off_topic", [])
    )
    total = len(data["questions"])
    return {
        "collection": collection,
        "questions": total,
        "hit_at_k": hits_at_k / total,
        "mrr": reciprocal / total,
        "precision": relevant / returned if returned else 0.0,
        "avg_returned": returned / total,
        "off_topic_noise": noise,
        "detail": detail,
    }
