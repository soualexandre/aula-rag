# RAG Local

Aula de demonstração de **embeddings** e **RAG (etapa de recuperação)**, rodando
inteiramente na máquina: sem API externa, sem LLM, sem GPU, sem banco vetorial.

A ideia é deixar cada etapa do pipeline visível e manipulável — dá para ver o
vetor que um texto gera, comparar a similaridade entre frases, buscar nos
documentos indexados e ler o prompt final já montado com as citações.

O modelo de embedding é baixado uma única vez (~220 MB) para o cache do
HuggingFace. Depois disso a aplicação funciona 100% offline.

## Os dois modos

O projeto vem com **duas coleções**, alternáveis na interface, na API e na CLI:

| Modo | Corpus | Chunk |
|---|---|---|
| `demo` | três notas curtas em Markdown sobre embeddings, RAG e bancos vetoriais | 600 / 120 |
| `ppc` | o PPC do Bacharelado em Sistemas de Informação do IFTO Paraíso — PDF real, 132 páginas | 900 / 180 |

Cada modo tem pasta e índice próprios e nunca se misturam num mesmo contexto.
O `demo` mostra o mecanismo com um corpus pequeno e legível; o `ppc` mostra o
mesmo pipeline sobre um documento institucional de verdade, com tabelas,
resoluções e citação por página.

## Stack

| Peça | Escolha | Por quê |
|---|---|---|
| API | FastAPI + Uvicorn | docs automáticas em `/docs` |
| Embeddings | `fastembed` (ONNX Runtime) | roda em CPU, **sem PyTorch** |
| Modelo | `paraphrase-multilingual-MiniLM-L12-v2` (384 dim) | multilíngue, funciona em PT |
| Índice | matriz NumPy + produto escalar | busca exata em ms, sem dependência extra |
| Ranqueamento | híbrido: denso + BM25 léxico | acerta paráfrase **e** termo literal |
| PDF | `pypdf` | extração de texto por página, sem OCR |
| UI | um HTML sem build | zero toolchain |

## Como rodar

**Linux / macOS**

```bash
./run.sh                 # cria a venv, instala e sobe em http://127.0.0.1:8000
```

**Windows**

```bat
run.bat
```

No PowerShell, `.\run.bat`. O script já resolve a diferença de nome do
interpretador: no Windows não existe `python3` — o comando é `python`, ou o
launcher `py -3`, que é o mais confiável quando há mais de uma versão instalada.
Se nenhum dos dois responder, o Python não está no PATH: reinstale pelo
[python.org](https://www.python.org/downloads/) marcando **"Add python.exe to
PATH"** (a versão da Microsoft Store costuma dar dor de cabeça com venv).

Ou manualmente:

```bash
# Linux / macOS
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn app.main:app --reload
```

```bat
:: Windows
py -3 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn app.main:app --reload
```

A venv do Windows guarda os executáveis em `.venv\Scripts\` (e não em
`.venv/bin/`). Chamar `.venv\Scripts\python -m <modulo>` funciona sem precisar
ativar a venv — é o que os exemplos deste README fazem, trocando o prefixo
`./.venv/bin/python` por `.venv\Scripts\python`.

Abra <http://127.0.0.1:8000> para a interface e <http://127.0.0.1:8000/docs>
para o Swagger.

No startup só o modo padrão (`demo`) é indexado — indexar o PPC inteiro leva
alguns segundos e nem toda execução precisa dele. Ao trocar para um modo ainda
não indexado a interface mostra um botão **Indexar agora**. Para deixar tudo
pronto de uma vez:

```bash
./.venv/bin/python -m app.cli index --all
```

## Pelo terminal

`-c/--collection` escolhe o modo e vem **antes** do comando (padrão: `demo`):

```bash
./.venv/bin/python -m app.cli collections            # lista os modos e o estado de cada índice
./.venv/bin/python -m app.cli index --all            # indexa todos
./.venv/bin/python -m app.cli stats                  # modelo, dimensões, chunks por fonte
./.venv/bin/python -m app.cli search  "como escolher o tamanho do chunk"
./.venv/bin/python -m app.cli -c ppc search  "carga horária total do curso"
./.venv/bin/python -m app.cli -c ppc context "como funciona o estágio supervisionado"
./.venv/bin/python -m app.cli -c ppc eval            # métricas da busca no gabarito
./.venv/bin/python -m app.cli compare "gato" "felino doméstico" "carro esportivo"
```

`compare` mostra a similaridade de cosseno entre os textos:

```
0.6602  'gato' <-> 'felino doméstico'
0.2128  'gato' <-> 'carro esportivo'
```

`search` no PPC mostra os dois sinais do ranqueamento e a página de onde o
trecho saiu:

```
[1] 0.6362  (denso 0.67 | léxico 0.62)  ppc-...pdf, p. 33-34
    ... Quadro 13: Total da carga horária do curso e total de aulas ...
```

## O pipeline

```
INDEXAÇÃO (uma vez por modo)
  arquivo → blocos (página do PDF) → chunks → embeddings → matriz NumPy → disco

CONSULTA (a cada pergunta)
  pergunta ─┬→ embedding → cosseno ──┐
            └→ BM25 léxico ──────────┴→ fusão → filtros → contexto montado
```

O bloco é a unidade de origem: no Markdown é o arquivo inteiro, no PDF é uma
página. O metadado do bloco viaja junto com o texto e vira a citação — inclusive
quando um chunk atravessa a virada de página (`p. 33-34`).

`/api/rag` devolve o **prompt pronto**, com os trechos numerados e citados —
exatamente o que seria enviado a um LLM. **A geração não acontece**: a demo
termina no retrieval, de propósito.

## Como a busca funciona

A busca combina dois sinais que se complementam. O **denso** (embeddings)
encontra paráfrase: acha o trecho certo mesmo quando nenhuma palavra coincide.
O **léxico** (BM25) pesa o termo exato e raro — sigla, número de resolução, nome
de disciplina — que num vetor de 384 dimensões acabaria diluído.

O ranqueamento passa por estas etapas, nesta ordem:

1. **Abstenção** — antes de ranquear, decide se o corpus fala do assunto. Se nem
   o melhor chunk chega a `abstain_dense` de cosseno, devolve **zero**. Se
   nenhum termo da pergunta aparece no corpus, exige um casamento semântico
   forte para não abster. É um teste sobre a *pergunta*, não sobre cada
   resultado, e exige que os dois sinais concordem: o BM25 sozinho casa um
   número solto (*"copa de 2002"* acerta o ano "2002" no PDF) e o denso sozinho
   acha vizinhança temática onde não há assunto em comum (*"capital da
   Austrália"* contra a página que lista municípios).
2. **Fusão** `alpha · denso + (1-alpha) · léxico`. A fusão vem antes de
   qualquer filtro, para o BM25 poder resgatar o trecho que o denso enterrou.
3. **Limiar absoluto** (`min_score`) — descarta o resultado fraco individual.
4. **Corte relativo ao melhor** (`RELATIVE_CUTOFF`) — se o melhor pontua 0.80 e
   o quarto 0.30, o quarto não é resposta, é enchimento para cumprir a cota.
   **O top-K é um teto, não uma cota.**
5. **Deduplicação** — o overlap entre chunks vizinhos gera quase-duplicatas.

O BM25 é normalizado por saturação (`x/(x+8)`), não pelo maior da consulta:
dividir pelo maior faria o melhor casamento virar `1.00` mesmo numa pergunta
fora do assunto, e o limiar perderia o sentido.

Dá para ver o efeito ao vivo: o campo **Denso × léxico** na interface é o
`alpha`, e cada trecho mostra os dois sinais que o colocaram ali. Com `alpha`
em 1.0 a busca vira puramente semântica; em 0.0, puramente lexical.

### Cada modo tem a sua calibragem

| | `demo` | `ppc` |
|---|---|---|
| `hybrid_alpha` (peso do denso) | 0.45 | 0.30 |
| `min_score` | 0.22 | 0.34 |
| `abstain_dense` | 0.20 | 0.34 |
| chunk | 600 | 900 |

Prosa didática é escrita com as palavras do leitor, então o denso rende; o PPC é
cheio de sigla, número de resolução e nome de disciplina, onde o léxico ganha.

Os limiares diferem porque a **escala do cosseno depende do corpus**: com poucos
chunks curtos o `demo` pontua naturalmente mais baixo, e ali quem separa o que é
do assunto é a âncora léxica. Com 132 páginas, quase todo termo comum aparece em
algum lugar, a âncora léxica sozinha não basta e o piso denso precisa subir.
Um limiar global serviria mal aos dois — por isso ele mora na coleção.

### Medindo a busca

`data/eval/*.json` traz perguntas com o gabarito conferido no documento (página,
no PDF; trecho literal, no Markdown), mais perguntas fora do assunto que devem
devolver **zero**:

```bash
./.venv/bin/python -m app.cli -c ppc eval
```

As métricas são **acerto@k** (a resposta apareceu no top-K), **MRR** (quão no
topo ela apareceu), **precisão** (fração dos trechos devolvidos que era
relevante), **devolvidos** (média por pergunta) e **ruído** (trechos devolvidos
para perguntas fora do assunto — o ideal é 0).

## Endpoints

Todas as rotas de busca e ingestão aceitam `collection` (no corpo JSON ou como
query string); omitir usa o modo padrão. Um nome inexistente devolve 404.
`/api/search` e `/api/rag` aceitam ainda `top_k`, `min_score`, `alpha` e `mmr` —
omitidos, valem os da coleção. Cada resultado devolve `score` (final), `dense` e
`lexical`, dá para ver qual sinal sustentou o trecho.

| Método | Rota | O que faz |
|---|---|---|
| `GET`  | `/api/collections` | modos disponíveis e chunks indexados em cada um |
| `POST` | `/api/embed` | mostra o vetor gerado para cada texto |
| `POST` | `/api/compare` | matriz de similaridade de cosseno entre textos |
| `POST` | `/api/search` | busca híbrida nos chunks indexados |
| `POST` | `/api/rag` | busca + monta o contexto/prompt final |
| `POST` | `/api/ingest/text` | indexa um texto avulso |
| `POST` | `/api/ingest/docs` | reindexa `data/docs/<modo>/` |
| `GET`  | `/api/stats` | modelo, dimensões, chunks por fonte |
| `DELETE` | `/api/index` | apaga o índice do modo |

## Detalhes que valem notar

- **Vetores normalizados.** Com norma 1, similaridade de cosseno vira produto
  escalar, e a busca inteira é uma multiplicação de matriz (`vectors @ q`).
- **Overlap alinhado.** A sobreposição entre chunks é cortada em fronteira de
  frase ou palavra, nunca no meio de uma.
- **MMR opcional.** O parâmetro `mmr` (0 a 1) penaliza chunks parecidos entre si,
  evitando um contexto com quatro variações do mesmo trecho.
- **Stemmer em português.** O BM25 radicaliza os termos, então "obrigatória",
  "obrigatoriamente" e "obrigatório" contam como o mesmo. Números ficam
  inteiros: "3180" e "5.626" são exatamente o que a pergunta cita literalmente.
- **Trocar de modelo invalida o índice.** `app/store.py` detecta a mudança de
  dimensão e força reindexação.
- **PDF sem OCR.** `pypdf` extrai o texto embutido; um PDF escaneado não
  funcionaria. As 132 páginas do PPC saem em ~7 s (334 chunks).
- **Cabeçalho é ruído.** Toda página do PPC repete 8 linhas de timbre
  institucional. O loader descarta qualquer linha presente em ≥60% das páginas —
  no PPC isso pega exatamente o timbre (100%) e preserva os títulos de seção
  reais, cujo mais frequente aparece em 33%.
- **Linha de PDF não é parágrafo.** O texto extraído quebra a cada linha
  impressa; o loader remonta os parágrafos, mantendo separados os títulos, os
  itens de lista, os artigos de regulamento e as células de tabela.
- **Chunk por modo.** Nota didática pede chunk pequeno (600) para não misturar
  assuntos; documento formal pede chunk maior (900) para não cortar um artigo ou
  uma tabela ao meio.

## Configuração

Tudo por variável de ambiente (ver `app/config.py`):

```bash
RAG_MODEL="BAAI/bge-small-en-v1.5"   # alternativa mais leve: 67 MB, só inglês
RAG_COLLECTION=demo                  # modo padrão
RAG_CHUNK_SIZE=600
RAG_CHUNK_OVERLAP=120
RAG_PPC_CHUNK_SIZE=900
RAG_PPC_CHUNK_OVERLAP=180
RAG_TOP_K=4                          # teto de trechos, não cota
RAG_MIN_SCORE=0.22                   # padrão; cada modo pode ter o seu
RAG_HYBRID_ALPHA=0.4                 # 1 = só embeddings, 0 = só BM25
RAG_RELATIVE_CUTOFF=0.75             # corta o que fica abaixo de 75% do melhor
RAG_ABSTAIN_DENSE=0.34               # abaixo disso, a busca devolve zero
RAG_ABSTAIN_STRONG_DENSE=0.55        # dispensa âncora léxica se o denso for forte
RAG_DEDUP=0.95
```

Para adicionar um terceiro modo, basta uma entrada em `COLLECTIONS`
(`app/config.py`) e uma pasta `data/docs/<nome>/` com os arquivos.

## Estrutura

```
app/
  config.py      configuração via env + registro das coleções (modos)
  loaders.py     leitura de .md/.txt e de PDF (limpeza de timbre, parágrafos)
  chunking.py    quebra em chunks com overlap, preservando a origem
  embeddings.py  wrapper do modelo ONNX (lazy, singleton)
  retrieval.py   BM25 léxico, fusão com o denso e filtros de precisão
  evaluation.py  métricas contra o gabarito (acerto@k, MRR, precisão, ruído)
  store.py       vector store NumPy + persistência, um por coleção
  rag.py         pipeline: ingestão e montagem de contexto
  main.py        API FastAPI
  cli.py         interface de terminal
web/index.html   UI sem build
data/docs/demo/  notas didáticas em Markdown
data/docs/ppc/   o PPC em PDF (132 páginas)
data/eval/       gabaritos conferidos no documento, por modo
data/index/demo/ índice do modo demo (vectors.npy + chunks.json)
data/index/ppc/  índice do modo ppc
```
