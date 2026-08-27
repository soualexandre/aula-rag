"""Ranqueamento: BM25 lexico + similaridade densa, fundidos.

Busca so-densa erra numa classe inteira de pergunta: termo raro e literal.
"Fundamentos de Libras e optativa?" some no meio de dezenas de paginas que
falam de "disciplina" e "obrigatoriedade" em geral, porque o vetor de 384
dimensoes dilui um nome proprio que aparece em 3 dos 334 chunks.

BM25 faz o oposto: pesa exatamente o termo que quase ninguem usa. Os dois
juntos cobrem o que cada um perde -- o denso acha parafrase, o lexico ancora
no termo exato (numeros, siglas, nomes de disciplina, artigos de lei).
"""
from __future__ import annotations

import math
import re
import unicodedata

import numpy as np

# Palavras funcionais do portugues: aparecem em quase todo chunk, entao nao
# discriminam nada e so atrapalham o BM25.
STOPWORDS = {
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "do", "dos", "e",
    "ela", "elas", "ele", "eles", "em", "entre", "era", "essa", "esse", "esta",
    "este", "eu", "foi", "for", "isso", "ja", "la", "lhe", "mais", "mas", "me",
    "mesmo", "meu", "muito", "na", "nas", "nao", "no", "nos", "num", "numa",
    "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "per", "por",
    "qual", "quais", "quando", "que", "quem", "se", "sem", "ser", "seu", "seus",
    "so", "sua", "suas", "sao", "tem", "ter", "um", "uma", "voce", "à", "às",
}

_WORD = re.compile(r"[0-9a-z]+")
_stemmer = None


def _stem(words: list[str]) -> list[str]:
    """Radicaliza em portugues: "obrigatoria" e "obrigatoriamente" viram o mesmo termo."""
    global _stemmer
    if _stemmer is None:
        from py_rust_stemmers import SnowballStemmer

        _stemmer = SnowballStemmer("portuguese")
    return _stemmer.stem_words(words)


def tokenize(text: str) -> list[str]:
    """Minusculas, sem acento, sem palavra funcional, radicalizado.

    Numeros ficam: "3180", "400" e "5.626" sao exatamente o tipo de termo que
    a busca densa perde e que a pergunta costuma citar de forma literal.
    """
    flat = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    words = [w for w in _WORD.findall(flat) if len(w) > 1 and w not in STOPWORDS]
    return _stem(words) if words else []


class BM25:
    """BM25 Okapi sobre listas invertidas em memoria.

    Para alguns milhares de chunks isso e instantaneo e nao adiciona nenhuma
    dependencia: o stemmer ja vem junto do fastembed.
    """

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        tokenized = [tokenize(d) for d in documents]
        self.n = len(tokenized)
        self.lengths = np.array([len(t) for t in tokenized], dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if self.n else 0.0

        # termo -> (indices dos documentos, frequencia em cada um)
        postings: dict[str, dict[int, int]] = {}
        for i, tokens in enumerate(tokenized):
            for token in tokens:
                postings.setdefault(token, {}).setdefault(i, 0)
                postings[token][i] += 1

        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {
            term: (
                np.fromiter(docs, dtype=np.int32, count=len(docs)),
                np.fromiter(docs.values(), dtype=np.float32, count=len(docs)),
            )
            for term, docs in postings.items()
        }
        # idf de Robertson: termo em poucos documentos vale muito mais
        self.idf = {
            term: math.log(1 + (self.n - len(docs) + 0.5) / (len(docs) + 0.5))
            for term, docs in postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """Pontuacao BM25 de cada documento para a consulta (0 = nao casa nada)."""
        out = np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.lengths / (self.avg_length or 1.0))
        for term in tokenize(query):
            entry = self.postings.get(term)
            if entry is None:
                continue
            docs, freqs = entry
            out[docs] += self.idf[term] * (freqs * (self.k1 + 1)) / (freqs + norm[docs])
        return out


# Saturacao do BM25: metade da escala no valor tipico de um bom casamento.
BM25_HALF = 8.0


def saturate(scores: np.ndarray, half: float = BM25_HALF) -> np.ndarray:
    """Leva o BM25 (ilimitado) para 0..1 sem olhar para o resto da consulta.

    Dividir pelo maior da consulta seria mais simples, mas destroi justamente a
    informacao que interessa: numa pergunta fora do assunto o melhor casamento
    lexico e ruim, e mesmo assim viraria 1.0. Saturando contra uma constante, um
    resultado ruim continua com cara de ruim -- e o filtro de score consegue
    barrar a pergunta que nao tem resposta no corpus.
    """
    return scores / (scores + half)


def fuse(dense: np.ndarray, lexical: np.ndarray, alpha: float) -> np.ndarray:
    """Combina os dois sinais. `alpha` = peso do denso (1.0 = so denso, 0.0 = so BM25)."""
    if alpha >= 1.0:
        return dense
    if alpha <= 0.0:
        return saturate(lexical)
    return alpha * dense + (1.0 - alpha) * saturate(lexical)


def cut_tail(
    ranked: list[tuple[int, float]], relative_cutoff: float
) -> list[tuple[int, float]]:
    """Descarta a cauda muito abaixo do melhor resultado.

    E o que impede o top-K de completar a cota com ruido: se o melhor trecho
    pontua 0.80 e o quarto 0.30, o quarto nao e resposta -- e enchimento. Com
    isso o K vira um teto, nao uma cota a cumprir.
    """
    if not ranked or relative_cutoff <= 0:
        return ranked
    floor = ranked[0][1] * relative_cutoff
    return [(i, s) for i, s in ranked if s >= floor]


def drop_duplicates(
    ranked: list[tuple[int, float]], vectors: np.ndarray, threshold: float
) -> list[tuple[int, float]]:
    """Remove chunks quase identicos entre si (efeito colateral do overlap)."""
    if threshold >= 1.0:
        return ranked
    kept: list[tuple[int, float]] = []
    for i, score in ranked:
        if any(float(vectors[i] @ vectors[j]) > threshold for j, _ in kept):
            continue
        kept.append((i, score))
    return kept
