"""Divisao de texto em chunks.

Estrategia: agrupa paragrafos ate atingir CHUNK_SIZE caracteres, mantendo uma
sobreposicao (overlap) entre chunks vizinhos para nao cortar contexto no meio.

Os paragrafos chegam agrupados em blocos (uma pagina de PDF, por exemplo). O
metadado do bloco acompanha cada paragrafo, e o chunk final registra de onde
veio -- inclusive quando ele atravessa duas paginas.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import CHUNK_OVERLAP, CHUNK_SIZE


@dataclass
class Piece:
    """Um chunk pronto para virar embedding, com a origem que o gerou."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _split_paragraphs(text: str, size: int) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # paragrafo gigante: quebra por frase para nao estourar o chunk
        if len(p) > size * 2:
            out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip())
        else:
            out.append(p)
    return out


def _overlap_tail(text: str, overlap: int) -> str:
    """Ultimos `overlap` caracteres, alinhados ao inicio de uma frase ou palavra."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    sentence = re.search(r"(?<=[.!?])\s+(\S)", tail)
    if sentence:
        return tail[sentence.start(1):].strip()
    word = tail.find(" ")
    return tail[word + 1 :].strip() if word != -1 else tail.strip()


def _merge_metadata(metas: list[dict]) -> dict:
    """Funde os metadados dos paragrafos que compoem um chunk.

    Paginas viram um intervalo: um chunk que atravessa a virada de pagina
    guarda `page` (a primeira) e `page_end` (a ultima).
    """
    merged: dict[str, Any] = {}
    for meta in metas:
        for key, value in meta.items():
            if key != "page":
                merged.setdefault(key, value)
    pages = sorted({m["page"] for m in metas if "page" in m})
    if pages:
        merged["page"] = pages[0]
        if pages[-1] != pages[0]:
            merged["page_end"] = pages[-1]
    return merged


def chunk_blocks(
    blocks: list[tuple[str, dict]],
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Piece]:
    """Quebra blocos `(texto, metadado)` em chunks que preservam a origem."""
    # achata os blocos em paragrafos, cada um carregando o metadado do bloco
    paragraphs: list[tuple[str, dict]] = [
        (para, meta)
        for text, meta in blocks
        for para in _split_paragraphs(text, size)
    ]

    pieces: list[Piece] = []
    buf = ""
    metas: list[dict] = []

    def flush() -> None:
        if buf.strip():
            pieces.append(Piece(buf.strip(), _merge_metadata(metas)))

    for para, meta in paragraphs:
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) <= size:
            buf = candidate
            metas.append(meta)
            continue
        if buf:
            flush()
            tail = _overlap_tail(buf, overlap)
            # o trecho sobreposto vem do fim do chunk anterior: a origem dele
            # entra junto com a do paragrafo novo
            metas = [metas[-1], meta] if tail else [meta]
            buf = f"{tail}\n\n{para}".strip() if tail else para
        else:
            # paragrafo unico maior que o size: fatia em janelas deslizantes
            step = max(size - overlap, 1)
            for i in range(0, len(para), step):
                piece = para[i : i + size]
                if piece.strip():
                    pieces.append(Piece(piece.strip(), dict(meta)))
            buf, metas = "", []

    flush()
    return pieces


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Versao simples, para um texto avulso sem metadado de origem."""
    return [p.text for p in chunk_blocks([(text, {})], size, overlap)]
