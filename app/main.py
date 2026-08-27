"""API FastAPI da demo de embeddings + RAG local."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import DOCS_DIR, MIN_SCORE, MODEL_NAME, TOP_K, WEB_DIR
from .embeddings import cosine, embed
from .rag import build_context, ingest_directory, ingest_text
from .store import store


@asynccontextmanager
async def lifespan(_: FastAPI):
    # O indice ja e carregado do disco no import de app.store; se estiver vazio,
    # faz a primeira indexacao a partir de data/docs.
    if not store.chunks:
        ingest_directory(DOCS_DIR)
    yield


app = FastAPI(
    title="RAG Local",
    description="Demo de embeddings e retrieval 100% offline (sem LLM).",
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


class SearchRequest(BaseModel):
    query: str
    top_k: int = TOP_K
    min_score: float = MIN_SCORE
    mmr: float = Field(0.0, ge=0.0, le=1.0, description="0 = so relevancia, 0.5 = diversifica")


# ---------------------------------------------------------------------- rotas
@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/stats")
def stats():
    return {"model": MODEL_NAME, **store.stats()}


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
    return ingest_text(req.source, req.text)


@app.post("/api/ingest/docs")
def api_ingest_docs():
    """Reindexa a pasta data/docs."""
    return ingest_directory(DOCS_DIR)


@app.post("/api/search")
def api_search(req: SearchRequest):
    """Busca semantica pura: devolve os chunks mais proximos da pergunta."""
    started = time.perf_counter()
    hits = store.search(req.query, top_k=req.top_k, min_score=req.min_score, mmr=req.mmr)
    return {
        "query": req.query,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "results": [
            {
                "score": round(score, 4),
                "source": c.source,
                "position": c.position,
                "text": c.text,
            }
            for c, score in hits
        ],
    }


@app.post("/api/rag")
def api_rag(req: SearchRequest):
    """Retrieval + montagem do contexto/prompt (sem chamar LLM)."""
    started = time.perf_counter()
    result = build_context(req.query, top_k=req.top_k, min_score=req.min_score, mmr=req.mmr)
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result


@app.delete("/api/index")
def api_clear():
    store.clear()
    store.save()
    return {"cleared": True, **store.stats()}
