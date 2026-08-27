# RAG Local

Demonstração de **embeddings** e **RAG (etapa de recuperação)** rodando
inteiramente na máquina: sem API externa, sem LLM, sem GPU, sem banco vetorial.

O modelo de embedding é baixado uma única vez (~220 MB) para o cache do
HuggingFace. Depois disso a aplicação funciona 100% offline.

## Stack

| Peça | Escolha | Por quê |
|---|---|---|
| API | FastAPI + Uvicorn | docs automáticas em `/docs` |
| Embeddings | `fastembed` (ONNX Runtime) | roda em CPU, **sem PyTorch** |
| Modelo | `paraphrase-multilingual-MiniLM-L12-v2` (384 dim) | multilíngue, funciona em PT |
| Índice | matriz NumPy + produto escalar | busca exata em ms, sem dependência extra |
| UI | um HTML sem build | zero toolchain |

## Como rodar

```bash
./run.sh                 # cria a venv, instala e sobe em http://127.0.0.1:8000
```

Ou manualmente:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn app.main:app --reload
```

Abra <http://127.0.0.1:8000> para a interface e <http://127.0.0.1:8000/docs>
para o Swagger. Na primeira execução a pasta `data/docs/` é indexada
automaticamente.

## Pelo terminal

```bash
./.venv/bin/python -m app.cli index
./.venv/bin/python -m app.cli search  "como escolher o tamanho do chunk"
./.venv/bin/python -m app.cli context "como escolher o tamanho do chunk"
./.venv/bin/python -m app.cli compare "gato" "felino doméstico" "carro esportivo"
```

```
0.6602  'gato' <-> 'felino doméstico'
0.2128  'gato' <-> 'carro esportivo'
```

## Endpoints

| Método | Rota | O que faz |
|---|---|---|
| `POST` | `/api/embed` | mostra o vetor gerado para cada texto |
| `POST` | `/api/compare` | matriz de similaridade de cosseno entre textos |
| `POST` | `/api/search` | busca semântica nos chunks indexados |
| `POST` | `/api/rag` | busca + monta o contexto/prompt final |
| `POST` | `/api/ingest/text` | indexa um texto avulso |
| `POST` | `/api/ingest/docs` | reindexa `data/docs/` |
| `GET`  | `/api/stats` | modelo, dimensões, chunks por fonte |
| `DELETE` | `/api/index` | apaga o índice |

## O pipeline

```
INDEXAÇÃO (uma vez)
  documento → chunks (600 chars, 120 de overlap) → embeddings → matriz NumPy → disco

CONSULTA (a cada pergunta)
  pergunta → embedding → cosseno contra todos os chunks → top-K → contexto montado
```

`/api/rag` devolve o **prompt pronto**, com os trechos numerados e citados —
exatamente o que seria enviado a um LLM. **A geração não acontece**: a demo
termina no retrieval, de propósito.

## Detalhes que valem notar

- **Vetores normalizados.** Com norma 1, similaridade de cosseno vira produto
  escalar, e a busca inteira é uma multiplicação de matriz (`vectors @ q`).
- **Overlap alinhado.** A sobreposição entre chunks é cortada em fronteira de
  frase ou palavra, nunca no meio de uma.
- **MMR opcional.** O parâmetro `mmr` (0 a 1) penaliza chunks parecidos entre si,
  evitando um contexto com quatro variações do mesmo trecho.
- **Score mínimo.** Resultados abaixo de `min_score` (0.15) são descartados —
  melhor não devolver nada do que devolver ruído.
- **Trocar de modelo invalida o índice.** `app/store.py` detecta a mudança de
  dimensão e força reindexação.

## Configuração

Tudo por variável de ambiente (ver `app/config.py`):

```bash
RAG_MODEL="BAAI/bge-small-en-v1.5"   # alternativa mais leve: 67 MB, só inglês
RAG_CHUNK_SIZE=600
RAG_CHUNK_OVERLAP=120
RAG_TOP_K=4
RAG_MIN_SCORE=0.15
```

## Estrutura

```
app/
  config.py      configuração via env
  chunking.py    quebra de texto em chunks com overlap
  embeddings.py  wrapper do modelo ONNX (lazy, singleton)
  store.py       vector store NumPy + persistência
  rag.py         pipeline: ingestão e montagem de contexto
  main.py        API FastAPI
  cli.py         interface de terminal
web/index.html   UI sem build
data/docs/       documentos de exemplo (sobre embeddings e RAG)
data/index/      índice gerado (vectors.npy + chunks.json)
```
