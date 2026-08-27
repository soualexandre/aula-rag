"""Configuracoes da aplicacao (tudo sobrescrivel por variavel de ambiente)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", BASE_DIR / "data" / "docs"))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", BASE_DIR / "data" / "index"))
WEB_DIR = BASE_DIR / "web"

# Modelo de embedding local (ONNX, roda em CPU, baixado uma vez para ~/.cache).
# Multilingue por padrao para funcionar bem com textos em portugues.
# Alternativa mais leve (so ingles, 67 MB): BAAI/bge-small-en-v1.5
MODEL_NAME = os.getenv(
    "RAG_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Chunking (padrao; cada colecao pode sobrescrever)
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 600))      # caracteres por chunk
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 120))  # sobreposicao

# Retrieval
TOP_K = int(os.getenv("RAG_TOP_K", 4))
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", 0.22))

# Busca hibrida: peso do sinal denso na fusao com o BM25 lexico.
# 1.0 = so embeddings, 0.0 = so BM25. O meio termo e o que acerta tanto
# parafrase quanto termo literal (numero, sigla, nome de disciplina).
HYBRID_ALPHA = float(os.getenv("RAG_HYBRID_ALPHA", 0.4))

# Corte relativo ao melhor resultado: descarta o que pontua abaixo de
# `melhor * RELATIVE_CUTOFF`. E o que faz o top-K devolver 1 trecho quando so
# existe 1 relevante, em vez de completar a cota com ruido.
RELATIVE_CUTOFF = float(os.getenv("RAG_RELATIVE_CUTOFF", 0.75))

# O limiar absoluto ficou baixo de proposito: quem barra pergunta fora do
# assunto e a abstencao abaixo, e nao ele. Consulta curta ("MMR", "Libras")
# produz cosseno naturalmente menor, e um limiar alto puniria justamente ela.

# Abstencao: se nem o melhor chunk chega a esta similaridade densa, o corpus
# nao fala do assunto e a busca devolve zero. E um teste sobre a CONSULTA, nao
# sobre cada resultado -- e por isso pega ate a pergunta fora do assunto que
# casa um termo por acaso ("copa de 2002" acerta o ano "2002" no PDF).
ABSTAIN_DENSE = float(os.getenv("RAG_ABSTAIN_DENSE", 0.34))

# Acima disso dois chunks sao considerados o mesmo trecho (efeito do overlap).
DEDUP_THRESHOLD = float(os.getenv("RAG_DEDUP", 0.95))

# ----------------------------------------------------------------- colecoes
# Cada colecao ("modo") tem sua propria pasta de documentos e seu proprio
# indice em data/index/<nome>/. Da para alternar entre elas na UI, na API
# (parametro `collection`) e na CLI (-c).
COLLECTIONS: dict[str, dict] = {
    "demo": {
        "label": "Didatico",
        "description": "Notas curtas sobre embeddings, RAG e bancos vetoriais.",
        "sample_question": "Por que dividir os documentos em pedacos?",
        # textos curtos: chunk pequeno mantem cada trecho em um so assunto
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        # prosa didatica e escrita com as palavras do leitor: o denso acerta a
        # parafrase, e o corpus tem poucos termos raros para o BM25 explorar
        "hybrid_alpha": 0.45,
        "min_score": 0.22,
        "relative_cutoff": 0.85,
    },
    "ppc": {
        "label": "PPC - Sistemas de Informacao (IFTO Paraiso)",
        "description": "Projeto Pedagogico do Curso: 132 paginas em PDF, documento real.",
        "sample_question": "Qual e a carga horaria total do curso?",
        # documento formal e prolixo: chunk maior evita cortar um artigo/tabela
        # (medido: 900 bate 500, 600, 750 e 1200 no conjunto de avaliacao)
        "chunk_size": int(os.getenv("RAG_PPC_CHUNK_SIZE", 900)),
        "chunk_overlap": int(os.getenv("RAG_PPC_CHUNK_OVERLAP", 180)),
        # documento cheio de termo literal (siglas, numeros de resolucao, nomes
        # de disciplina): o BM25 pesa mais que o denso
        "hybrid_alpha": 0.30,
        "min_score": 0.25,
    },
}

DEFAULT_COLLECTION = os.getenv("RAG_COLLECTION", "demo")


def collection_config(name: str | None = None) -> tuple[str, dict]:
    """Resolve o nome da colecao e devolve (nome, config). Erra se nao existir."""
    name = name or DEFAULT_COLLECTION
    if name not in COLLECTIONS:
        known = ", ".join(COLLECTIONS)
        raise KeyError(f"colecao desconhecida: {name!r} (disponiveis: {known})")
    return name, COLLECTIONS[name]


def collection_docs_dir(name: str) -> Path:
    return DOCS_DIR / name


def collection_index_dir(name: str) -> Path:
    return INDEX_DIR / name
