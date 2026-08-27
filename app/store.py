"""Vector store minimalista: matriz numpy em memoria + persistencia em disco.

Para poucos milhares de chunks a busca exaustiva (produto de matriz) e
instantanea e dispensa qualquer banco vetorial.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .config import INDEX_DIR
from .embeddings import dimension, embed

VECTORS_PATH = INDEX_DIR / "vectors.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.json"


@dataclass
class Chunk:
    id: int
    source: str
    position: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray = np.zeros((0, dimension()), dtype=np.float32)

    # ------------------------------------------------------------------ escrita
    def add(self, source: str, texts: list[str], metadata: dict | None = None) -> int:
        if not texts:
            return 0
        vectors = embed(texts)
        with self._lock:
            start = len(self.chunks)
            for i, text in enumerate(texts):
                self.chunks.append(
                    Chunk(
                        id=start + i,
                        source=source,
                        position=i,
                        text=text,
                        metadata=metadata or {},
                    )
                )
            self.vectors = (
                vectors if self.vectors.size == 0 else np.vstack([self.vectors, vectors])
            )
        return len(texts)

    def remove_source(self, source: str) -> int:
        with self._lock:
            keep = [i for i, c in enumerate(self.chunks) if c.source != source]
            removed = len(self.chunks) - len(keep)
            if removed:
                self.chunks = [self.chunks[i] for i in keep]
                self.vectors = self.vectors[keep] if keep else np.zeros(
                    (0, dimension()), dtype=np.float32
                )
                for new_id, chunk in enumerate(self.chunks):
                    chunk.id = new_id
        return removed

    def clear(self) -> None:
        with self._lock:
            self.chunks = []
            self.vectors = np.zeros((0, dimension()), dtype=np.float32)

    # -------------------------------------------------------------------- busca
    def search(
        self, query: str, top_k: int = 4, min_score: float = 0.0, mmr: float = 0.0
    ) -> list[tuple[Chunk, float]]:
        """Busca semantica por similaridade de cosseno.

        `mmr` (0..1) ativa Maximal Marginal Relevance: penaliza chunks parecidos
        entre si para diversificar o contexto devolvido.
        """
        if not self.chunks:
            return []

        q = embed([query])[0]
        scores = self.vectors @ q  # cosseno (vetores ja normalizados)

        candidates = [i for i in np.argsort(-scores) if scores[i] >= min_score]
        if not candidates:
            return []

        if mmr <= 0:
            selected = candidates[:top_k]
        else:
            pool = candidates[: max(top_k * 4, top_k)]
            selected: list[int] = []
            while pool and len(selected) < top_k:
                best_i, best_val = None, -np.inf
                for i in pool:
                    redundancy = (
                        max(float(self.vectors[i] @ self.vectors[j]) for j in selected)
                        if selected
                        else 0.0
                    )
                    val = (1 - mmr) * float(scores[i]) - mmr * redundancy
                    if val > best_val:
                        best_i, best_val = i, val
                selected.append(best_i)
                pool.remove(best_i)

        return [(self.chunks[i], float(scores[i])) for i in selected]

    # ------------------------------------------------------------- persistencia
    def save(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with self._lock:
            np.save(VECTORS_PATH, self.vectors)
            CHUNKS_PATH.write_text(
                json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def load(self) -> bool:
        if not (VECTORS_PATH.exists() and CHUNKS_PATH.exists()):
            return False
        vectors = np.load(VECTORS_PATH)
        raw = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        if vectors.shape[0] != len(raw):
            return False
        if vectors.size and vectors.shape[1] != dimension():
            # indice gerado por outro modelo: ignora e forca reindexacao
            return False
        with self._lock:
            self.vectors = vectors.astype(np.float32)
            self.chunks = [Chunk(**c) for c in raw]
        return True

    # ------------------------------------------------------------------- infos
    def stats(self) -> dict:
        sources: dict[str, int] = {}
        for c in self.chunks:
            sources[c.source] = sources.get(c.source, 0) + 1
        return {
            "chunks": len(self.chunks),
            "dimension": int(self.vectors.shape[1]) if self.vectors.size else dimension(),
            "sources": sources,
        }


store = VectorStore()
# Carrega o indice do disco assim que o modulo e importado, para que scripts e
# CLI funcionem sem depender do startup do servidor.
store.load()
