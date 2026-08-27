"""Pipeline de RAG (etapa de recuperacao).

Fluxo: documento -> chunks -> embeddings -> indice
       pergunta  -> embedding -> busca por cosseno -> contexto montado

O contexto devolvido e exatamente o bloco de texto que seria injetado no prompt
de um LLM. Aqui a geracao NAO acontece: a demo termina no retrieval.
"""
from __future__ import annotations

from pathlib import Path

from .chunking import chunk_text
from .config import DOCS_DIR, MIN_SCORE, TOP_K
from .store import store

TEXT_SUFFIXES = {".md", ".txt", ".markdown"}


def ingest_text(source: str, text: str, metadata: dict | None = None) -> dict:
    """Indexa (ou reindexa) um texto identificado por `source`."""
    store.remove_source(source)
    chunks = chunk_text(text)
    added = store.add(source, chunks, metadata)
    store.save()
    return {"source": source, "chunks": added, "chars": len(text)}


def ingest_directory(directory: Path = DOCS_DIR) -> dict:
    """Indexa todos os arquivos de texto de uma pasta."""
    directory = Path(directory)
    if not directory.exists():
        return {"files": [], "chunks": 0, "error": f"pasta nao encontrada: {directory}"}

    results, total = [], 0
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        source = str(path.relative_to(directory))
        store.remove_source(source)
        chunks = chunk_text(text)
        total += store.add(source, chunks, {"path": str(path)})
        results.append({"source": source, "chunks": len(chunks)})

    store.save()
    return {"files": results, "chunks": total}


def build_context(question: str, top_k: int = TOP_K, min_score: float = MIN_SCORE,
                  mmr: float = 0.0) -> dict:
    """Recupera os trechos mais relevantes e monta o contexto do prompt."""
    hits = store.search(question, top_k=top_k, min_score=min_score, mmr=mmr)

    passages = [
        {
            "rank": i + 1,
            "score": round(score, 4),
            "source": chunk.source,
            "position": chunk.position,
            "text": chunk.text,
        }
        for i, (chunk, score) in enumerate(hits)
    ]

    context = "\n\n".join(
        f"[{p['rank']}] ({p['source']} #{p['position']}, score {p['score']})\n{p['text']}"
        for p in passages
    )

    prompt = (
        "Responda a pergunta usando apenas o contexto abaixo. "
        "Cite as fontes pelo numero entre colchetes. "
        "Se a resposta nao estiver no contexto, diga que nao sabe.\n\n"
        f"### Contexto\n{context or '(nenhum trecho relevante encontrado)'}\n\n"
        f"### Pergunta\n{question}\n\n### Resposta\n"
    )

    return {
        "question": question,
        "passages": passages,
        "context": context,
        "prompt": prompt,
        "note": "Etapa de geracao nao executada: esta demo cobre embedding + retrieval.",
    }
