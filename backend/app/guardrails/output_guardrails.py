"""
Output guardrails. See guide file 07, section 3.

Two checks here:
1. Groundedness/faithfulness check via LLM-as-judge — verifies every claim
   in the generated answer is actually supported by the retrieved context,
   catching hallucination that prompt instructions alone don't prevent
   (this is the exact failure mode demonstrated by the "capital of France"
   out-of-scope test in the original POC guide, file 04 section 8).
2. Citation verification — checks that every [Source: X] citation in the
   answer actually corresponds to a document that was in the retrieved
   set, not a fabricated or hallucinated source name.

Honesty note on local models: a small local judge model (llama3.2 by default)
is less reliable at strictly following the VERDICT/UNSUPPORTED_CLAIMS output
format than a frontier API model would be — the regex parsing below falls
back to "UNKNOWN" if the format isn't followed, which itself is useful
signal (a real production system would alert on a rising UNKNOWN rate as a
sign the judge model needs a stronger prompt, few-shot examples, or a
larger model).
"""

import re
import ollama
from app.config import settings

_client = ollama.Client(host=settings.OLLAMA_HOST)

JUDGE_PROMPT_TEMPLATE = """You are a strict fact-checker reviewing an AI assistant's answer.

CONTEXT (the only information the assistant was allowed to use):
{context}

ANSWER (what the assistant said):
{answer}

Check whether every factual claim in the ANSWER is directly supported by the CONTEXT.
Respond in EXACTLY this format, nothing else:
VERDICT: GROUNDED or UNGROUNDED or PARTIALLY_GROUNDED
UNSUPPORTED_CLAIMS: <comma-separated list of unsupported claims, or "none">"""

CITATION_PATTERN = re.compile(r"\[Source:\s*([^\]]+)\]")


def check_groundedness(answer: str, retrieved_chunks: list[dict]) -> dict:
    context = "\n\n".join(c["text"] for c in retrieved_chunks)
    prompt = JUDGE_PROMPT_TEMPLATE.format(context=context, answer=answer)

    response = _client.chat(
        model=settings.JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},  # judge calls should be maximally deterministic
    )
    raw = response["message"]["content"]

    verdict_match = re.search(r"VERDICT:\s*(\w+)", raw)
    claims_match = re.search(r"UNSUPPORTED_CLAIMS:\s*(.+)", raw, re.DOTALL)

    verdict = verdict_match.group(1) if verdict_match else "UNKNOWN"
    unsupported_raw = claims_match.group(1).strip() if claims_match else ""
    has_listed_claims = unsupported_raw.lower() not in ("none", "")

    # Consistency check, added after hands-on testing surfaced this exact
    # failure mode with llama3.2: the judge sometimes returns a verdict of
    # UNGROUNDED/PARTIALLY_GROUNDED while ALSO reporting zero unsupported
    # claims — a direct self-contradiction (the verdict and its own stated
    # evidence disagree). That's a judge instruction-following failure, not
    # a real grounding problem with the answer, so we don't trust it as-is.
    # Reclassifying to UNKNOWN is more honest than either silently trusting
    # a contradictory verdict, or silently downgrading it to GROUNDED — both
    # of those would hide the judge's actual reliability gap instead of
    # surfacing it. A real production system would track UNKNOWN rate as a
    # signal that the judge model/prompt needs improvement.
    if verdict in ("UNGROUNDED", "PARTIALLY_GROUNDED") and not has_listed_claims:
        verdict = "UNKNOWN"

    return {
        "verdict": verdict,
        "unsupported_claims": unsupported_raw if has_listed_claims else None,
        "raw_judge_output": raw,
    }


def verify_citations(answer: str, retrieved_chunks: list[dict]) -> dict:
    """
    Confirms every [Source: X] cited in the answer corresponds to a source
    file actually present in the retrieved set — catches the model citing
    a plausible-sounding but fabricated source.

    Explicitly excludes "None"/"none"/"N/A" style citations from this check.
    Found via hands-on testing: when correctly refusing an out-of-scope
    question, the model sometimes writes "[Source: None]" to indicate no
    source applies — which is the CORRECT behavior (it's not citing a real
    document because there isn't one), not a fabrication. Treating that
    phrasing as a fabricated citation was a false positive that penalized
    exactly the refusal behavior the system is supposed to encourage.
    """
    NO_SOURCE_PLACEHOLDERS = {"none", "n/a", "na", "not applicable"}

    cited_sources = set(CITATION_PATTERN.findall(answer))
    real_cited_sources = {
        s for s in cited_sources if s.strip().lower() not in NO_SOURCE_PLACEHOLDERS
    }
    actual_sources = {
        c.get("metadata", {}).get("source_file", "") for c in retrieved_chunks
    }

    fabricated = real_cited_sources - actual_sources
    return {
        "cited_sources": list(cited_sources),
        "fabricated_citations": list(fabricated),
        "all_citations_valid": len(fabricated) == 0,
    }


def run_output_guardrails(answer: str, retrieved_chunks: list[dict]) -> dict:
    groundedness = check_groundedness(answer, retrieved_chunks)
    citations = verify_citations(answer, retrieved_chunks)

    # Design decision (guide file 07, section 3 — "what happens when
    # groundedness fails is itself a design decision"): we don't hard-block
    # here, we surface a visible caveat. For a higher-risk deployment this
    # could instead trigger a regeneration attempt or full block — that's a
    # config/policy decision, not a technical limitation of this code.
    # UNKNOWN is included here too: a verdict we couldn't reliably parse or
    # had to discard for self-contradiction still means we can't CONFIRM
    # groundedness, even though it's not the same as a confirmed failure —
    # the caveat text in routes/chat.py is written to be accurate either way.
    needs_caveat = (
        groundedness["verdict"] in ("UNGROUNDED", "PARTIALLY_GROUNDED", "UNKNOWN")
        or not citations["all_citations_valid"]
    )

    return {
        "groundedness": groundedness,
        "citations": citations,
        "needs_caveat": needs_caveat,
    }