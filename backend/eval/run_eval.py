"""
Evaluation harness. Run with:

    python -m eval.run_eval

Implements the core principle from guide file 04, section 7: retrieval and
generation are SEPARATE failure surfaces and must be measured separately.
This script computes:

1. Retrieval metrics (recall@k) — does the expected source document appear
   in the top-k retrieved chunks, independent of whether generation used
   it well?
2. Generation metrics (faithfulness) — using the SAME output guardrail
   groundedness checker that runs in production (app/guardrails/
   output_guardrails.py), so eval and production use one code path, not
   two divergent implementations that could silently drift apart.
3. The deliberate out-of-scope query is scored separately as a "refusal"
   check, not folded into recall@k (it has no expected source by design).
"""

import json
from pathlib import Path

from app.retrieval.hybrid_search import hybrid_search, build_bm25_index
from app.retrieval.reranker import rerank
from app.generation.generate import generate_answer
from app.guardrails.output_guardrails import check_groundedness
from app.config import settings

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text())


def evaluate_retrieval(item: dict, retrieved_chunks: list[dict]) -> dict:
    """Recall@k for a single eval item — did the expected source appear
    anywhere in the retrieved (post-rerank) set?"""
    if item["expected_source"] is None:
        return {"applicable": False}

    retrieved_sources = [c.get("metadata", {}).get("source_file", "") for c in retrieved_chunks]
    hit = item["expected_source"] in retrieved_sources

    section_hit = None
    if item.get("expected_section_contains"):
        section_hit = any(
            item["expected_section_contains"].lower() in c.get("metadata", {}).get("section", "").lower()
            for c in retrieved_chunks
        )

    return {"applicable": True, "source_hit": hit, "section_hit": section_hit}


def evaluate_refusal(item: dict, answer: str) -> dict:
    """For out-of-scope queries, check the model correctly declined rather
    than answering from parametric knowledge — this is the exact failure
    mode the original POC guide (file 04, section 8) flagged as worth
    testing explicitly."""
    refusal_phrases = ["don't know", "do not know", "not contain", "doesn't contain",
                        "cannot answer", "can't answer", "no information", "not enough information"]
    correctly_refused = any(p in answer.lower() for p in refusal_phrases)
    return {"correctly_refused": correctly_refused}


def run_eval():
    print("Building BM25 index for eval run...")
    build_bm25_index()

    eval_set = load_eval_set()
    retrieval_results = []
    refusal_results = []
    groundedness_results = []

    for i, item in enumerate(eval_set):
        print(f"\n[{i+1}/{len(eval_set)}] {item['query']}")

        candidates = hybrid_search(item["query"], top_k=settings.TOP_K_RETRIEVE)
        top_chunks = rerank(item["query"], candidates, top_k=settings.TOP_K_FINAL)

        if item["expected_source"] is not None:
            r = evaluate_retrieval(item, top_chunks)
            retrieval_results.append(r)
            print(f"  Retrieval: source_hit={r['source_hit']}, section_hit={r['section_hit']}")

        answer = generate_answer(item["query"], top_chunks)

        if item["expected_source"] is None:
            ref = evaluate_refusal(item, answer)
            refusal_results.append(ref)
            print(f"  Refusal check: correctly_refused={ref['correctly_refused']}")
        else:
            g = check_groundedness(answer, top_chunks)
            groundedness_results.append(g)
            print(f"  Groundedness: {g['verdict']}")

    # --- Aggregate ---
    applicable_retrieval = [r for r in retrieval_results if r["applicable"]]
    recall_at_k = sum(r["source_hit"] for r in applicable_retrieval) / len(applicable_retrieval)
    section_recall = sum(r["section_hit"] for r in applicable_retrieval if r["section_hit"] is not None) / len(applicable_retrieval)

    grounded_count = sum(1 for g in groundedness_results if g["verdict"] == "GROUNDED")
    faithfulness_rate = grounded_count / len(groundedness_results) if groundedness_results else None

    refusal_accuracy = (
        sum(r["correctly_refused"] for r in refusal_results) / len(refusal_results)
        if refusal_results else None
    )

    print("\n" + "=" * 50)
    print("EVAL SUMMARY")
    print("=" * 50)
    print(f"Retrieval recall@{settings.TOP_K_FINAL} (document-level): {recall_at_k:.2%}")
    print(f"Retrieval recall@{settings.TOP_K_FINAL} (section-level):  {section_recall:.2%}")
    print(f"Generation faithfulness rate (GROUNDED verdicts):         {faithfulness_rate:.2%}" if faithfulness_rate is not None else "Generation faithfulness rate: N/A")
    print(f"Out-of-scope refusal accuracy:                            {refusal_accuracy:.2%}" if refusal_accuracy is not None else "Refusal accuracy: N/A")


if __name__ == "__main__":
    run_eval()
