"""API FastAPI da demo de embeddings + RAG local."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import (
    COLLECTIONS,
    DEFAULT_COLLECTION,
    HYBRID_ALPHA,
    MIN_SCORE,
    MODEL_NAME,
    TOP_K,
    WEB_DIR,
    collection_config,
    collection_docs_dir,
)
from .embeddings import cosine, embed
from .rag import build_context, citation, ingest_directory, ingest_text
from .store import get_store


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Indexa so a colecao padrao no startup: indexar o PPC inteiro levaria
    # dezenas de segundos e nem toda execucao vai usar esse modo. As outras
    # sao indexadas sob demanda (botao na UI, /api/ingest/docs ou a CLI).
    store = get_store(DEFAULT_COLLECTION)
    if not store.chunks:
        ingest_directory(DEFAULT_COLLECTION)
    yield


app = FastAPI(
    title="RAG Local",
    description="Demo de embeddings e retrieval 100% offline (sem LLM), "
    "com colecoes alternaveis: notas didaticas ou um PDF real de 132 paginas.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------- schemas
class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, examples=[["gato", "cachorro"]])
    preview: int = Field(8, description="Quantas dimensoes retornar do vetor (0 = todas)")


class CompareRequest(BaseModel):
    texts: list[str] = Field(..., min_length=2)


class IngestTextRequest(BaseModel):
    source: str
    text: str
    collection: str | None = Field(None, description="Colecao alvo; None = a padrao")


class IngestDocsRequest(BaseModel):
    collection: str | None = None


class SearchRequest(BaseModel):
    """Campos nao informados caem no ajuste calibrado da propria colecao."""

    query: str
    top_k: int | None = Field(None, description=f"Teto de trechos (padrao {TOP_K})")
    min_score: float | None = Field(None, description="Score minimo para entrar no contexto")
    mmr: float = Field(0.0, ge=0.0, le=1.0, description="0 = so relevancia, 0.5 = diversifica")
    alpha: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Peso do sinal denso na fusao: 1 = so embeddings, 0 = so BM25",
    )
    collection: str | None = Field(None, description="Em qual colecao buscar; None = a padrao")


def resolve(collection: str | None) -> str:
    """Valida o nome da colecao e devolve 404 se nao existir."""
    try:
        name, _ = collection_config(collection)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return name


# ---------------------------------------------------------------------- rotas
@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/collections")
def api_collections():
    """Os modos disponiveis e o estado do indice de cada um."""
    return {
        "default": DEFAULT_COLLECTION,
        "collections": [
            {
                "name": name,
                "label": cfg["label"],
                "description": cfg["description"],
                "sample_question": cfg["sample_question"],
                "chunk_size": cfg["chunk_size"],
                "chunk_overlap": cfg["chunk_overlap"],
                "hybrid_alpha": cfg.get("hybrid_alpha", HYBRID_ALPHA),
                "min_score": cfg.get("min_score", MIN_SCORE),
                "docs_dir": str(collection_docs_dir(name)),
                "chunks": get_store(name).stats()["chunks"],
            }
            for name, cfg in COLLECTIONS.items()
        ],
    }


@app.get("/api/stats")
def stats(collection: str | None = Query(None)):
    return {"model": MODEL_NAME, **get_store(resolve(collection)).stats()}


@app.post("/api/embed")
def api_embed(req: EmbedRequest):
    """Mostra o vetor gerado para cada texto."""
    started = time.perf_counter()
    vectors = embed(req.texts)
    elapsed = (time.perf_counter() - started) * 1000
    n = req.preview if req.preview > 0 else vectors.shape[1]
    return {
        "model": MODEL_NAME,
        "dimension": int(vectors.shape[1]),
        "elapsed_ms": round(elapsed, 1),
        "items": [
            {
                "text": text,
                "vector_preview": [round(float(v), 5) for v in vec[:n]],
                "norm": round(float(np.linalg.norm(vec)), 4),
            }
            for text, vec in zip(req.texts, vectors)
        ],
    }


@app.post("/api/compare")
def api_compare(req: CompareRequest):
    """Matriz de similaridade de cosseno entre os textos enviados."""
    vectors = embed(req.texts)
    matrix = [
        [round(cosine(a, b), 4) for b in vectors]
        for a in vectors
    ]
    pairs = sorted(
        (
            {
                "a": req.texts[i],
                "b": req.texts[j],
                "similarity": matrix[i][j],
            }
            for i in range(len(req.texts))
            for j in range(i + 1, len(req.texts))
        ),
        key=lambda p: -p["similarity"],
    )
    return {"texts": req.texts, "matrix": matrix, "pairs": pairs}


@app.post("/api/ingest/text")
def api_ingest_text(req: IngestTextRequest):
    if not req.text.strip():
        raise HTTPException(400, "texto vazio")
    return ingest_text(req.source, req.text, collection=resolve(req.collection))


@app.post("/api/ingest/docs")
def api_ingest_docs(req: IngestDocsRequest | None = None):
    """Reindexa a pasta de documentos da colecao (data/docs/<colecao>)."""
    name = resolve(req.collection if req else None)
    return ingest_directory(name)


@app.post("/api/search")
def api_search(req: SearchRequest):
    """Busca semantica pura: devolve os chunks mais proximos da pergunta."""
    name = resolve(req.collection)
    started = time.perf_counter()
    hits = get_store(name).search(
        req.query, top_k=req.top_k, min_score=req.min_score, mmr=req.mmr, alpha=req.alpha
    )
    return {
        "collection": name,
        "query": req.query,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "results": [
            {
                "score": round(h.score, 4),
                "dense": round(h.dense, 4),
                "lexical": round(h.lexical, 4),
                "source": h.chunk.source,
                "position": h.chunk.position,
                "page": h.chunk.metadata.get("page"),
                "citation": citation(h.chunk),
                "text": h.chunk.text,
            }
            for h in hits
        ],
    }


@app.post("/api/rag")
def api_rag(req: SearchRequest):
    """Retrieval + montagem do contexto/prompt (sem chamar LLM)."""
    started = time.perf_counter()
    result = build_context(
        req.query,
        top_k=req.top_k,
        min_score=req.min_score,
        mmr=req.mmr,
        alpha=req.alpha,
        collection=resolve(req.collection),
    )
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result


@app.delete("/api/index")
def api_clear(collection: str | None = Query(None)):
    store = get_store(resolve(collection))
    store.clear()
    store.save()
    return {"cleared": True, **store.stats()}
