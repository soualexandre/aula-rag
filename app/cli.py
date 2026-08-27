"""CLI da demo, para usar sem subir o servidor.

    python -m app.cli collections
    python -m app.cli index --all
    python -m app.cli -c ppc index
    python -m app.cli -c ppc search "carga horaria total do curso"
    python -m app.cli -c ppc context "como funciona o estagio supervisionado"
    python -m app.cli compare "gato" "felino domestico" "carro"

A opcao -c/--collection escolhe o modo (padrao: RAG_COLLECTION ou "demo").
"""
from __future__ import annotations

import argparse
import sys

from .config import COLLECTIONS, DEFAULT_COLLECTION, MODEL_NAME
from .embeddings import cosine, embed
from .rag import build_context, citation, ingest_directory
from .store import get_store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-local", description=__doc__)
    parser.add_argument(
        "-c", "--collection", default=None, choices=list(COLLECTIONS),
        help=f"colecao/modo a usar (padrao: {DEFAULT_COLLECTION})",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="indexa data/docs/<colecao>")
    p_index.add_argument("--all", action="store_true", help="indexa todas as colecoes")
    sub.add_parser("stats", help="mostra o estado do indice")
    sub.add_parser("collections", help="lista os modos disponiveis")

    for name, help_text in [("search", "busca semantica"), ("context", "monta o contexto do prompt")]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("query")
        p.add_argument("-k", type=int, default=None, help="teto de trechos")
        p.add_argument("--min-score", type=float, default=None)
        p.add_argument("--mmr", type=float, default=0.0)
        p.add_argument("--alpha", type=float, default=None,
                       help="peso do denso na fusao (1 = so embeddings, 0 = so BM25)")

    p = sub.add_parser("compare", help="similaridade entre textos")
    p.add_argument("texts", nargs="+")

    p = sub.add_parser("eval", help="mede a qualidade da busca no conjunto de avaliacao")
    p.add_argument("-k", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--min-score", type=float, default=None)

    args = parser.parse_args(argv)
    store = get_store(args.collection)

    if args.cmd == "index":
        targets = list(COLLECTIONS) if args.all else [store.name]
        for target in targets:
            result = ingest_directory(target)
            print(f"[{target}]")
            if result.get("error"):
                print(f"  {result['error']}")
                continue
            for f in result["files"]:
                print(f"  {f['source']:<44} {f['chunks']} chunks")
            print(f"  total: {result['chunks']} chunks indexados")

    elif args.cmd == "collections":
        for name, cfg in COLLECTIONS.items():
            mark = "*" if name == store.name else " "
            n = get_store(name).stats()["chunks"]
            print(f"{mark} {name:<6} {cfg['label']}")
            print(f"           {cfg['description']}")
            print(f"           chunk {cfg['chunk_size']}/{cfg['chunk_overlap']} | {n} chunks indexados")

    elif args.cmd == "stats":
        s = store.stats()
        print(f"colecao   : {s['collection']}")
        print(f"modelo    : {MODEL_NAME}")
        print(f"dimensoes : {s['dimension']}")
        print(f"chunks    : {s['chunks']}")
        for src, n in s["sources"].items():
            print(f"  {src:<44} {n}")

    elif args.cmd == "search":
        hits = store.search(args.query, top_k=args.k, min_score=args.min_score,
                            mmr=args.mmr, alpha=args.alpha)
        if not hits:
            print("nenhum resultado acima do score minimo")
        for i, h in enumerate(hits, 1):
            snippet = " ".join(h.chunk.text.split())[:220]
            print(f"\n[{i}] {h.score:.4f}  (denso {h.dense:.2f} | lexico {h.lexical:.2f})  "
                  f"{citation(h.chunk)}\n    {snippet}...")

    elif args.cmd == "context":
        result = build_context(
            args.query, top_k=args.k, min_score=args.min_score,
            mmr=args.mmr, alpha=args.alpha, collection=store.name,
        )
        print(result["prompt"])
        print(f"--- {result['note']}")

    elif args.cmd == "eval":
        from .evaluation import evaluate

        kwargs = {k: v for k, v in
                  (("alpha", args.alpha), ("min_score", args.min_score)) if v is not None}
        r = evaluate(store.name, top_k=args.k, **kwargs)
        for d in r["detail"]:
            mark = f"#{d['rank']}" if d["rank"] else "--"
            print(f"  {mark:>3}  ({d['returned']} trechos)  {d['question']}")
        print(f"\ncolecao   : {r['collection']}  ({r['questions']} perguntas)")
        print(f"acerto@k  : {r['hit_at_k']:.2f}   (a resposta apareceu no top-K)")
        print(f"MRR       : {r['mrr']:.3f}  (1.0 = sempre em primeiro)")
        print(f"precisao  : {r['precision']:.3f}  (fracao dos trechos devolvidos que era relevante)")
        print(f"devolvidos: {r['avg_returned']:.1f}   (media por pergunta, teto {args.k or 4})")
        print(f"ruido     : {r['off_topic_noise']}      (trechos devolvidos fora do assunto; ideal 0)")

    elif args.cmd == "compare":
        vectors = embed(args.texts)
        for i in range(len(args.texts)):
            for j in range(i + 1, len(args.texts)):
                print(f"{cosine(vectors[i], vectors[j]):.4f}  {args.texts[i]!r} <-> {args.texts[j]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
