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

# Chunking
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 600))      # caracteres por chunk
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 120))  # sobreposicao

# Retrieval
TOP_K = int(os.getenv("RAG_TOP_K", 4))
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", 0.15))
