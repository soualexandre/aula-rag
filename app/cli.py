"""CLI da demo, para usar sem subir o servidor.

    python -m app.cli index
    python -m app.cli search "como funciona o chunking"
    python -m app.cli context "como funciona o chunking"
    python -m app.cli compare "gato" "felino domestico" "carro"
"""
from __future__ import annotations

import argparse
import sys

from .config import MIN_SCORE, MODEL_NAME, TOP_K
from .embeddings import cosine, embed
from .rag import build_context, ingest_directory
from .store import store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-local", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("index", help="indexa data/docs")
    sub.add_parser("stats", help="mostra o estado do indice")

    for name, help_text in [("search", "busca semantica"), ("context", "monta o contexto do prompt")]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("query")
        p.add_argument("-k", type=int, default=TOP_K)
        p.add_argument("--min-score", type=float, default=MIN_SCORE)
        p.add_argument("--mmr", type=float, default=0.0)

    p = sub.add_parser("compare", help="similaridade entre textos")
    p.add_argument("texts", nargs="+")

    args = parser.parse_args(argv)

    if args.cmd == "index":
        result = ingest_directory()
        for f in result["files"]:
            print(f"  {f['source']:<28} {f['chunks']} chunks")
        print(f"total: {result['chunks']} chunks indexados")

    elif args.cmd == "stats":
        s = store.stats()
        print(f"modelo    : {MODEL_NAME}")
        print(f"dimensoes : {s['dimension']}")
        print(f"chunks    : {s['chunks']}")
        for src, n in s["sources"].items():
            print(f"  {src:<28} {n}")

    elif args.cmd == "search":
        hits = store.search(args.query, top_k=args.k, min_score=args.min_score, mmr=args.mmr)
        if not hits:
            print("nenhum resultado acima do score minimo")
        for i, (chunk, score) in enumerate(hits, 1):
            snippet = " ".join(chunk.text.split())[:220]
            print(f"\n[{i}] {score:.4f}  {chunk.source} #{chunk.position}\n    {snippet}...")

    elif args.cmd == "context":
        result = build_context(args.query, top_k=args.k, min_score=args.min_score, mmr=args.mmr)
        print(result["prompt"])
        print(f"--- {result['note']}")

    elif args.cmd == "compare":
        vectors = embed(args.texts)
        for i in range(len(args.texts)):
            for j in range(i + 1, len(args.texts)):
                print(f"{cosine(vectors[i], vectors[j]):.4f}  {args.texts[i]!r} <-> {args.texts[j]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
