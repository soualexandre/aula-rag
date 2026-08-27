"""Vector store minimalista: matriz numpy em memoria + persistencia em disco.

Para poucos milhares de chunks a busca exaustiva (produto de matriz) e
instantanea e dispensa qualquer banco vetorial.

Cada colecao ("modo") tem a sua propria instancia e o seu proprio indice em
data/index/<colecao>/ -- alternar de modo e trocar de VectorStore, nada mais.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .config import (
    ABSTAIN_DENSE,
    ABSTAIN_STRONG_DENSE,
    DEDUP_THRESHOLD,
    HYBRID_ALPHA,
    MIN_SCORE,
    RELATIVE_CUTOFF,
    TOP_K,
    collection_config,
    collection_index_dir,
)
from .embeddings import dimension, embed
from .retrieval import BM25, cut_tail, drop_duplicates, fuse, saturate


@dataclass
class Chunk:
    id: int
    source: str
    position: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    """Um resultado com os dois sinais que o colocaram ali."""

    chunk: Chunk
    score: float      # pontuacao final (fusao)
    dense: float      # cosseno do embedding
    lexical: float    # BM25 normalizado (1.0 = melhor casamento lexico da consulta)


class VectorStore:
    def __init__(self, name: str) -> None:
        self.name = name
        _, self.config = collection_config(name)
        self.dir = collection_index_dir(name)
        self._lock = threading.Lock()
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray = np.zeros((0, dimension()), dtype=np.float32)
        self._bm25: BM25 | None = None   # construido sob demanda, invalidado a cada escrita

    @property
    def vectors_path(self):
        return self.dir / "vectors.npy"

    @property
    def chunks_path(self):
        return self.dir / "chunks.json"

    # ------------------------------------------------------------------ escrita
    def add(
        self,
        source: str,
        texts: list[str],
        metadata: dict | None = None,
        metadatas: list[dict] | None = None,
    ) -> int:
        """Indexa textos. `metadata` vale para todos; `metadatas` e por chunk."""
        if not texts:
            return 0
        base = metadata or {}
        per_chunk = metadatas or [{}] * len(texts)
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
                        metadata={**base, **per_chunk[i]},
                    )
                )
            self.vectors = (
                vectors if self.vectors.size == 0 else np.vstack([self.vectors, vectors])
            )
            self._bm25 = None
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
                self._bm25 = None
        return removed

    def clear(self) -> None:
        with self._lock:
            self.chunks = []
            self.vectors = np.zeros((0, dimension()), dtype=np.float32)
            self._bm25 = None

    # -------------------------------------------------------------------- busca
    @property
    def bm25(self) -> BM25:
        """Indice lexico, construido na primeira busca depois de cada escrita."""
        if self._bm25 is None:
            self._bm25 = BM25([c.text for c in self.chunks])
        return self._bm25

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        mmr: float = 0.0,
        alpha: float | None = None,
        relative_cutoff: float | None = None,
        dedup: float | None = None,
        abstain: float | None = None,
    ) -> list[Hit]:
        """Busca hibrida: cosseno denso + BM25 lexico, com filtros de precisao.

        A ordem importa. Primeiro funde os dois sinais, depois corta pelo score
        absoluto, entao pela distancia ao melhor resultado, e so no fim remove
        duplicata e aplica o teto do top-K -- filtrar antes de fundir deixaria
        o BM25 sem chance de resgatar o trecho que o denso enterrou.
        """
        if not self.chunks:
            return []

        # o que a colecao nao definir cai no padrao global
        cfg = self.config
        top_k = TOP_K if top_k is None else top_k
        min_score = cfg.get("min_score", MIN_SCORE) if min_score is None else min_score
        alpha = cfg.get("hybrid_alpha", HYBRID_ALPHA) if alpha is None else alpha
        if relative_cutoff is None:
            relative_cutoff = cfg.get("relative_cutoff", RELATIVE_CUTOFF)
        dedup = DEDUP_THRESHOLD if dedup is None else dedup
        if abstain is None:
            abstain = cfg.get("abstain_dense", ABSTAIN_DENSE)

        q = embed([query])[0]
        dense = self.vectors @ q                 # cosseno (vetores ja normalizados)
        lexical = self.bm25.scores(query)

        # Abstencao: a pergunta e sobre outro assunto, entao a resposta certa e
        # nenhum trecho. Os dois sinais precisam concordar que ha algo ali --
        # cada um sozinho se deixa enganar. O BM25 casa um numero solto ("copa
        # de 2002" acerta o ano no PDF); o denso acha vizinhanca tematica onde
        # nao ha assunto em comum ("capital da Australia" contra a pagina de
        # municipios). Exigir os dois separa os casos reais dos coincidentes.
        best_dense, best_lexical = float(dense.max()), float(lexical.max())
        if best_dense < abstain:
            return []
        if best_lexical <= 0 and best_dense < ABSTAIN_STRONG_DENSE:
            return []
        final = fuse(dense, lexical, alpha)
        lexical_norm = saturate(lexical)

        ranked = [(int(i), float(final[i])) for i in np.argsort(-final) if final[i] >= min_score]
        ranked = cut_tail(ranked, relative_cutoff)
        ranked = drop_duplicates(ranked, self.vectors, dedup)
        if not ranked:
            return []

        if mmr <= 0:
            selected = ranked[:top_k]
        else:
            pool = ranked[: max(top_k * 4, top_k)]
            selected = []
            while pool and len(selected) < top_k:
                best, best_val = None, -np.inf
                for i, score in pool:
                    redundancy = (
                        max(float(self.vectors[i] @ self.vectors[j]) for j, _ in selected)
                        if selected
                        else 0.0
                    )
                    val = (1 - mmr) * score - mmr * redundancy
                    if val > best_val:
                        best, best_val = (i, score), val
                selected.append(best)
                pool.remove(best)

        return [
            Hit(
                chunk=self.chunks[i],
                score=score,
                dense=float(dense[i]),
                lexical=float(lexical_norm[i]),
            )
            for i, score in selected
        ]

    # ------------------------------------------------------------- persistencia
    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            np.save(self.vectors_path, self.vectors)
            self.chunks_path.write_text(
                json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def load(self) -> bool:
        if not (self.vectors_path.exists() and self.chunks_path.exists()):
            return False
        vectors = np.load(self.vectors_path)
        raw = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        if vectors.shape[0] != len(raw):
            return False
        if vectors.size and vectors.shape[1] != dimension():
            # indice gerado por outro modelo: ignora e forca reindexacao
            return False
        with self._lock:
            self.vectors = vectors.astype(np.float32)
            self.chunks = [Chunk(**c) for c in raw]
            self._bm25 = None
        return True

    # ------------------------------------------------------------------- infos
    def stats(self) -> dict:
        sources: dict[str, int] = {}
        for c in self.chunks:
            sources[c.source] = sources.get(c.source, 0) + 1
        return {
            "collection": self.name,
            "chunks": len(self.chunks),
            "dimension": int(self.vectors.shape[1]) if self.vectors.size else dimension(),
            "sources": sources,
        }


# ------------------------------------------------------------------- registro
# Um store por colecao, criado sob demanda e carregado do disco na primeira vez.
# Assim a CLI e os scripts funcionam sem depender do startup do servidor, e
# nenhuma colecao paga o custo de carregar indice que ninguem pediu.
_stores: dict[str, VectorStore] = {}
_registry_lock = threading.Lock()


def get_store(collection: str | None = None) -> VectorStore:
    name, _ = collection_config(collection)
    with _registry_lock:
        st = _stores.get(name)
        if st is None:
            st = VectorStore(name)
            st.load()
            _stores[name] = st
        return st
