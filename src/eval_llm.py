"""Keyed generation + RAGAS evaluation.

    python -m src.eval_llm

Produces `artifacts/eval_scores.json` for the G3 grounding gate:
  faithfulness, context precision, context recall, answer relevancy,
  citation validity, refusal correctness, retrieval recall@5.

The 20 referenced questions are a regression suite, not an unbiased benchmark.
Five refusal questions check that unsupported questions produce
"insufficient guidance" rather than a guess.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import ARTIFACTS, LLM_MODEL, RAG_EMBED_MODEL, RAG_TOP_K
from .env import load_env
from .eval_rag import evaluate as evaluate_retrieval
from .rag import Retriever

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "guidelines" / "golden_queries.json"
REFERENCES = ROOT / "guidelines" / "ragas_references.json"

REFUSALS = [
    "What is the exact CMS reimbursement rate for sepsis in 2026?",
    "What antibiotic dose should be given to a 70 kg patient with a penicillin allergy?",
    "What is the phone number of the rapid response team?",
    "How many ICU beds are available at this hospital?",
    "What is this patient's insurance status?",
]

CITATION_RE = re.compile(r"\[(S\d+)\]", re.IGNORECASE)
REFUSAL_TEXT = "insufficient guidance"

SYSTEM = (
    "You answer questions about sepsis guidance using ONLY the supplied contexts. "
    "Every factual sentence must cite [S#]. If the contexts do not contain the "
    "answer, reply exactly: insufficient guidance. Do not use outside knowledge, "
    "do not give patient-specific dosing, and do not invent administrative details."
)


def _answer(client, query: str, chunks: list[dict]) -> tuple[str, dict]:
    context_text = "\n\n".join(
        f"[{chunk['citation_id']}] ({chunk['doc']}) {chunk['text']}" for chunk in chunks
    )
    prompt = f"Question: {query}\n\nContexts:\n{context_text}"
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    usage = response.usage
    return response.choices[0].message.content.strip(), {
        "prompt_tokens": int(usage.prompt_tokens),
        "completion_tokens": int(usage.completion_tokens),
        "calls": 1,
    }


def _citations(text: str) -> list[str]:
    return sorted({match.upper() for match in CITATION_RE.findall(text)})


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Keyed RAGAS evaluation")
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--references", type=Path, default=REFERENCES)
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    parser.add_argument("--k", type=int, default=RAG_TOP_K)
    parser.add_argument("--skip-ragas", action="store_true")
    args = parser.parse_args(argv)

    load_env()
    import os
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set (env or .env)", file=sys.stderr)
        return 2

    from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import llm_factory
    # ragas 0.4's `evaluate` still consumes the classic metric classes; the
    # `collections` classes use a different runner. These imports emit a
    # deprecation warning but are the supported path for `evaluate`.
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    started = time.time()
    retriever = Retriever(args.out)
    queries = json.loads(args.golden.read_text(encoding="utf-8"))
    references = json.loads(args.references.read_text(encoding="utf-8"))
    client = OpenAI()

    samples = []
    details = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    citation_valid = 0
    non_refused = 0
    false_refusals = 0

    for item in queries:
        results = retriever.search(item["query"], args.k)
        chunks = [result.as_dict() for result in results]
        answer, used = _answer(client, item["query"], chunks)
        for key in usage:
            usage[key] += used[key]

        refused = REFUSAL_TEXT in answer.lower()
        ids = {chunk["citation_id"] for chunk in chunks}
        cited = _citations(answer)
        valid = bool(cited) and set(cited) <= ids
        if not refused:
            non_refused += 1
            citation_valid += int(valid)
        else:
            false_refusals += 1

        samples.append(
            SingleTurnSample(
                user_input=item["query"],
                retrieved_contexts=[chunk["text"] for chunk in chunks],
                response=answer,
                reference=references[item["id"]],
            )
        )
        details.append(
            {
                "id": item["id"],
                "query": item["query"],
                "answer": answer,
                "refused": refused,
                "citations": cited,
                "citations_valid": valid,
                "retrieved_ids": sorted(ids),
            }
        )

    refusal_results = []
    for query in REFUSALS:
        results = retriever.search(query, args.k)
        chunks = [result.as_dict() for result in results]
        answer, used = _answer(client, query, chunks)
        for key in usage:
            usage[key] += used[key]
        refusal_results.append(
            {
                "query": query,
                "answer": answer,
                "correct": REFUSAL_TEXT in answer.lower(),
            }
        )

    retrieval = evaluate_retrieval(retriever, queries, args.k)
    refusal_correctness = sum(1 for row in refusal_results if row["correct"]) / len(
        refusal_results
    )
    citation_validity = citation_valid / non_refused if non_refused else 0.0
    false_refusal_rate = false_refusals / len(queries)

    scores = {}
    details_df = None
    if not args.skip_ragas:
        llm = llm_factory(LLM_MODEL, client=client)
        # The classic AnswerRelevancy metric needs embed_query/embed_documents;
        # RAGAS 0.4's native OpenAIEmbeddings only exposes embed_text(s).
        embeddings = LangchainEmbeddingsWrapper(
            LangchainOpenAIEmbeddings(model=RAG_EMBED_MODEL)
        )
        metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ]
        dataset = EvaluationDataset(samples=samples)
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            show_progress=True,
            raise_exceptions=False,
        )
        details_df = result.to_pandas()
        for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            if name not in details_df.columns:
                print(
                    f"ERROR: RAGAS result missing {name}; columns={list(details_df.columns)}",
                    file=sys.stderr,
                )
                return 2
            scores[name] = float(details_df[name].mean(skipna=True))

    payload = {
        "faithfulness": round(scores.get("faithfulness", 0.0), 6),
        "answer_relevancy": round(scores.get("answer_relevancy", 0.0), 6),
        "context_precision": round(scores.get("context_precision", 0.0), 6),
        "context_recall": round(scores.get("context_recall", 0.0), 6),
        "citation_validity": round(citation_validity, 6),
        "refusal_correctness": round(refusal_correctness, 6),
        "retrieval_recall_at_5": retrieval["recall_at_k"],
        "n_queries": len(queries),
        "n_refusals": len(REFUSALS),
        "false_refusal_rate": round(false_refusal_rate, 6),
        "false_refusals": false_refusals,
        "model": LLM_MODEL,
        "embedding_model": RAG_EMBED_MODEL,
        "generation_usage": usage,
        "estimated_generation_cost_usd": round(
            usage["prompt_tokens"] / 1e6 * 0.15
            + usage["completion_tokens"] / 1e6 * 0.60,
            6,
        ),
        "pricing_note": "gpt-4o-mini list prices used for an estimate; verify current pricing.",
        "duration_seconds": round(time.time() - started, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "20 authored referenced questions + 5 refusal questions. RAGAS uses "
            "gpt-4o-mini as judge; this is a regression suite, not an unbiased "
            "benchmark. RAGAS judge token usage is not included in generation_usage."
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_scores.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.out / "eval_details.json").write_text(
        json.dumps(
            {
                "questions": details,
                "refusals": refusal_results,
                "ragas_rows": details_df.to_dict(orient="records") if details_df is not None else [],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    print(f"scores -> {args.out / 'eval_scores.json'}")

    floors = {
        "faithfulness": 0.85,
        "context_precision": 0.80,
        "citation_validity": 1.0,
        "refusal_correctness": 1.0,
        "retrieval_recall_at_5": 0.90,
    }
    failures = [name for name, floor in floors.items() if payload[name] < floor]
    if failures:
        print(f"GATE FAIL: {failures}", file=sys.stderr)
        return 1
    print("GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
