"""Wrapper do modelo de embedding local.

Usa fastembed (ONNX Runtime) -> sem PyTorch, sem GPU, sem chamada de rede em
tempo de execucao. O modelo e baixado uma unica vez para o cache local do
HuggingFace e depois tudo roda offline.
"""
from __future__ import annotations

import threading
import warnings

import numpy as np

from .config import MODEL_NAME

_model = None
_lock = threading.Lock()

_DLL_HELP = """Nao foi possivel carregar o ONNX Runtime, o motor que roda o modelo
de embedding (fastembed -> onnxruntime -> DLLs nativas).

No Windows isso quase sempre e uma destas causas, nesta ordem:

1) Falta o Microsoft Visual C++ Redistributable (x64). Baixe, instale e
   reabra o terminal:  https://aka.ms/vs17/release/vc_redist.x64.exe

2) A venv foi criada a partir do Python do Anaconda (o traceback mostra
   caminhos em ...\\anaconda3\\...). As DLLs do Anaconda conflitam com as do
   onnxruntime. Apague a pasta .venv e recrie com o Python do python.org:
       rmdir /s /q .venv
       py -3 -m venv .venv
       run.bat

3) A versao do onnxruntime e mais nova que o runtime do Windows da maquina.
   Instale uma anterior dentro da venv:
       .venv\\Scripts\\python -m pip install "onnxruntime==1.19.2"

Erro original: {err}"""


def _import_text_embedding():
    """Importa o fastembed traduzindo a falha de DLL em instrucao acionavel."""
    try:
        from fastembed import TextEmbedding
    except (ImportError, OSError) as err:  # DLL load failed, .so ausente, etc.
        raise RuntimeError(_DLL_HELP.format(err=err)) from err
    return TextEmbedding


def get_model():
    """Carrega o modelo sob demanda (lazy) e reaproveita a instancia."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                TextEmbedding = _import_text_embedding()

                # fastembed avisa sobre a troca de CLS para mean pooling; o
                # comportamento atual e o correto para este modelo.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Converte textos em vetores L2-normalizados (shape: n x dim).

    Normalizar permite usar produto escalar como similaridade de cosseno.
    """
    if not texts:
        return np.zeros((0, dimension()), dtype=np.float32)
    vectors = np.array(list(get_model().embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]


def dimension() -> int:
    TextEmbedding = _import_text_embedding()

    for meta in TextEmbedding.list_supported_models():
        if meta["model"] == MODEL_NAME:
            return int(meta["dim"])
    return len(embed_one("dim probe"))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Similaridade de cosseno entre dois vetores ja normalizados."""
    return float(np.dot(a, b))
