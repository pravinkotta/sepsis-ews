"""Retrieval regression suite over a 20-query golden set.

This measures retrieval only. RAGAS faithfulness / answer relevancy require an
LLM and stay out of `eval_scores.json` until a keyed run is performed; the
grounding gate correctly reports SKIP without them.

    python -m src.eval_rag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ARTIFACTS, RAG_TOP_K
from .rag import Retriever

GOLDEN = Path(__file__).resolve().parents[1] / "guidelines" / "golden_queries.json"


def evaluate(retriever, queries: list[dict], k: int = RAG_TOP_K) -> dict:
    rows = []
    hits = 0
    reciprocal_rank = 0.0
    for item in queries:
        results = retriever.search(item["query"], k)
        texts = [result.text.lower() for result in results]
        terms = [term.lower() for term in item["expected_terms"]]
        hit = any(term in text for term in terms for text in texts)
        rank = None
        for position, text in enumerate(texts, start=1):
            if any(term in text for term in terms):
                rank = position
                break
        hits += int(hit)
        reciprocal_rank += (1.0 / rank) if rank else 0.0
        rows.append(
            {
                "id": item["id"],
                "query": item["query"],
                "hit_at_k": bool(hit),
                "rank": rank,
                "top_chunks": [
                    {"citation_id": r.citation_id, "chunk_id": r.chunk_id, "score": r.score}
                    for r in results
                ],
            }
        )
    n = len(queries)
    return {
        "n_queries": n,
        "k": k,
        "recall_at_k": round(hits / n, 6) if n else 0.0,
        "mrr": round(reciprocal_rank / n, 6) if n else 0.0,
        "rows": rows,
        "note": (
            "Authored regression suite, not an unbiased benchmark. Retrieval "
            "only; RAGAS metrics require a keyed LLM run."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate guideline retrieval")
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--k", type=int, default=RAG_TOP_K)
    args = parser.parse_args(argv)

    queries = json.loads(args.golden.read_text(encoding="utf-8"))
    retriever = Retriever(args.artifacts)
    result = evaluate(retriever, queries, args.k)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "retrieval_scores.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"recall@{args.k}={result['recall_at_k']:.3f} "
        f"MRR={result['mrr']:.3f} n={result['n_queries']}"
    )
    misses = [row["id"] for row in result["rows"] if not row["hit_at_k"]]
    if misses:
        print(f"misses: {misses}")
    print(f"scores -> {args.artifacts / 'retrieval_scores.json'}")
    return 0 if result["recall_at_k"] >= 0.90 else 1


if __name__ == "__main__":
    sys.exit(main())
