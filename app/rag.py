"""Pipeline de RAG (etapa de recuperacao).

Fluxo: documento -> blocos -> chunks -> embeddings -> indice
       pergunta  -> embedding -> busca por cosseno -> contexto montado

O contexto devolvido e exatamente o bloco de texto que seria injetado no prompt
de um LLM. Aqui a geracao NAO acontece: a demo termina no retrieval.

Tudo acontece dentro de uma colecao ("modo"): os documentos didaticos e o PPC
em PDF vivem em indices separados e nunca se misturam num mesmo contexto.
"""
from __future__ import annotations

from pathlib import Path

from .chunking import chunk_blocks, chunk_text
from .config import collection_config, collection_docs_dir
from .loaders import SUPPORTED_SUFFIXES, load_blocks
from .store import Chunk, get_store


def ingest_text(
    source: str, text: str, metadata: dict | None = None, collection: str | None = None
) -> dict:
    """Indexa (ou reindexa) um texto identificado por `source`."""
    name, cfg = collection_config(collection)
    store = get_store(name)
    store.remove_source(source)
    chunks = chunk_text(text, cfg["chunk_size"], cfg["chunk_overlap"])
    added = store.add(source, chunks, metadata)
    store.save()
    return {"collection": name, "source": source, "chunks": added, "chars": len(text)}


def ingest_directory(collection: str | None = None) -> dict:
    """Indexa todos os documentos suportados da pasta da colecao."""
    name, cfg = collection_config(collection)
    directory = collection_docs_dir(name)
    store = get_store(name)

    if not directory.exists():
        return {
            "collection": name,
            "files": [],
            "chunks": 0,
            "error": f"pasta nao encontrada: {directory}",
        }

    results, total = [], 0
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        blocks = load_blocks(path)
        if not blocks:
            continue
        pieces = chunk_blocks(
            [(b.text, b.metadata) for b in blocks], cfg["chunk_size"], cfg["chunk_overlap"]
        )
        source = str(path.relative_to(directory))
        store.remove_source(source)
        total += store.add(
            source,
            [p.text for p in pieces],
            metadata={"path": str(path)},
            metadatas=[p.metadata for p in pieces],
        )
        results.append(
            {"source": source, "blocks": len(blocks), "chunks": len(pieces)}
        )

    store.save()
    return {"collection": name, "files": results, "chunks": total}


def citation(chunk: Chunk) -> str:
    """Como o trecho aparece citado no prompt: arquivo, pagina e chunk."""
    page = chunk.metadata.get("page")
    page_end = chunk.metadata.get("page_end")
    if page and page_end:
        return f"{chunk.source}, p. {page}-{page_end}"
    if page:
        return f"{chunk.source}, p. {page}"
    return f"{chunk.source} #{chunk.position}"


def build_context(
    question: str,
    top_k: int | None = None,
    min_score: float | None = None,
    mmr: float = 0.0,
    collection: str | None = None,
    alpha: float | None = None,
) -> dict:
    """Recupera os trechos mais relevantes e monta o contexto do prompt.

    Os parametros nao informados caem no ajuste da propria colecao (ver
    COLLECTIONS em config.py), que e onde a calibragem de cada corpus mora.
    """
    name, _ = collection_config(collection)
    hits = get_store(name).search(
        question, top_k=top_k, min_score=min_score, mmr=mmr, alpha=alpha
    )

    passages = [
        {
            "rank": i + 1,
            "score": round(h.score, 4),
            "dense": round(h.dense, 4),
            "lexical": round(h.lexical, 4),
            "source": h.chunk.source,
            "position": h.chunk.position,
            "page": h.chunk.metadata.get("page"),
            "page_end": h.chunk.metadata.get("page_end"),
            "citation": citation(h.chunk),
            "text": h.chunk.text,
        }
        for i, h in enumerate(hits)
    ]

    context = "\n\n".join(
        f"[{p['rank']}] ({p['citation']}, score {p['score']})\n{p['text']}"
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
        "collection": name,
        "question": question,
        "passages": passages,
        "context": context,
        "prompt": prompt,
        "note": "Etapa de geracao nao executada: esta demo cobre embedding + retrieval.",
    }
