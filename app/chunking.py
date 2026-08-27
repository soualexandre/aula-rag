"""Divisao de texto em chunks.

Estrategia: agrupa paragrafos ate atingir CHUNK_SIZE caracteres, mantendo uma
sobreposicao (overlap) entre chunks vizinhos para nao cortar contexto no meio.
"""
from __future__ import annotations

import re

from .config import CHUNK_OVERLAP, CHUNK_SIZE


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # paragrafo gigante: quebra por frase para nao estourar o chunk
        if len(p) > CHUNK_SIZE * 2:
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


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    buf = ""

    for para in paragraphs:
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            tail = _overlap_tail(buf, overlap)
            buf = f"{tail}\n\n{para}".strip() if tail else para
        else:
            # paragrafo unico maior que o size: fatia em janelas deslizantes
            step = max(size - overlap, 1)
            for i in range(0, len(para), step):
                piece = para[i : i + size]
                if piece.strip():
                    chunks.append(piece.strip())
            buf = ""

    if buf.strip():
        chunks.append(buf.strip())
    return chunks
