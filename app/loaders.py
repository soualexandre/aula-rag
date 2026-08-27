"""Leitura de documentos: texto puro e PDF.

Cada arquivo vira uma lista de blocos. Um bloco e um pedaco de documento com
sua propria origem (no PDF, a pagina), e essa informacao viaja junto com o
texto ate virar a citacao do trecho recuperado.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TEXT_SUFFIXES = {".md", ".txt", ".markdown"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES

# Uma linha que se repete em mais de 60% das paginas e cabecalho/rodape, nao
# conteudo. No PPC do IFTO sao 8 linhas (ministerio, campus, endereco) em 100%
# das paginas; o titulo de secao mais frequente aparece em 33%.
BOILERPLATE_RATIO = 0.6


@dataclass
class Block:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_blocks(path: Path) -> list[Block]:
    """Le um arquivo suportado e devolve seus blocos de texto."""
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return _load_pdf(path)
    if suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Block(text)] if text.strip() else []
    raise ValueError(f"formato nao suportado: {suffix}")


# ------------------------------------------------------------------------ pdf
def _load_pdf(path: Path) -> list[Block]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    boilerplate = _find_boilerplate(pages)

    blocks: list[Block] = []
    for number, raw in enumerate(pages, start=1):
        text = _clean_page(raw, boilerplate, number)
        if text:
            blocks.append(Block(text, {"page": number}))
    return blocks


def _find_boilerplate(pages: list[str]) -> set[str]:
    """Linhas que se repetem na maioria das paginas (cabecalho e rodape)."""
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        # set(): a mesma linha repetida numa pagina so conta uma vez
        counts.update({line.strip() for line in page.splitlines() if line.strip()})
    limit = len(pages) * BOILERPLATE_RATIO
    return {line for line, n in counts.items() if n >= limit}


def _clean_page(raw: str, boilerplate: set[str], number: int) -> str:
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line in boilerplate:
            continue
        # numero da pagina solto (fica entre o cabecalho e o rodape)
        if line == str(number):
            continue
        line = re.sub(r"[ \t]+", " ", line)
        lines.append(line)

    text = "\n".join(_join_wrapped(lines))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Inicios de linha que sempre abrem um bloco novo: marcador de lista, numero de
# secao ("3.2"), artigo de regulamento, inciso romano.
_STARTS_BLOCK = re.compile(r"^([\u2022\u25cf\u25aa\u2043\-\u2013*]|\d+(\.\d+)*[\.\)]?\s|Art\.|\u00a7|[IVX]+\s*[-\u2013])")


def _join_wrapped(lines: list[str]) -> list[str]:
    """Reconstroi paragrafos: o PDF quebra linha a cada linha impressa.

    Junta a linha atual com a anterior, a menos que ela pareca comeco de algo
    novo (lista, secao, titulo) ou que a anterior ja tenha terminado a frase.
    Linhas curtas viram bloco proprio: sao titulos e celulas de tabela.
    """
    out: list[str] = []
    for line in lines:
        joinable = (
            out
            and not _STARTS_BLOCK.match(line)
            and not line.isupper()
            and not out[-1].endswith((".", ":", ";", "!", "?"))
            and not out[-1].isupper()
            and len(out[-1]) >= 50          # linha curta = titulo ou celula
        )
        if joinable:
            # hifenizacao de fim de linha: "desen-" + "volvimento"
            if out[-1].endswith("-"):
                out[-1] = out[-1][:-1] + line
            else:
                out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return out
