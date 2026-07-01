# Findings Log

Real issues found through hands-on testing of the deployed system, and the
fixes applied. Kept as a record because this — not the original build — is
the strongest material in this project: evidence of testing your
own system adversarially and finding genuine, specific calibration gaps.

---

## Finding 1: Self-contradictory groundedness verdicts from the local judge model

**How it was found:** Querying "What hours can a collector call a consumer?"
against the live system. The answer was correct and properly cited (verified
manually against `collections_call_practices_policy.md` section 1), and
retrieval/reranking correctly surfaced the right chunk as the top result
(rerank score 8.10). Despite this, the output groundedness guardrail flagged
the answer as `UNGROUNDED`.

**Root cause, confirmed via `raw_judge_output`:**

VERDICT: UNGROUNDED
UNSUPPORTED_CLAIMS: None


The judge model (`llama3.2`, a small local model) returned a verdict that
contradicts its own stated evidence — UNGROUNDED, but zero unsupported
claims listed. This is a judge instruction-following reliability gap, not a
real grounding failure in the answer.

**Reproduced a second time** on a different query (the deliberate
out-of-scope "What is the capital of France?" test), confirming this wasn't
a one-off — same contradiction pattern (`UNGROUNDED` / `UNSUPPORTED_CLAIMS:
none`) on a completely different answer.

**Fix:** `app/guardrails/output_guardrails.py`, `check_groundedness()` — when
the verdict is `UNGROUNDED` or `PARTIALLY_GROUNDED` but no unsupported
claims are actually listed, reclassify the verdict to `UNKNOWN` rather than
trusting the contradictory result. This is more honest than either silently
trusting a broken verdict or silently downgrading it to `GROUNDED` — both
would hide the judge's actual reliability gap. A production system should
track `UNKNOWN` rate as a signal that the judge model or prompt needs
improvement (larger model, few-shot examples, or a more constrained output
format).

**Verified the fix doesn't mask genuine failures:** added a companion test
confirming that when the judge *does* list real unsupported claims alongside
an `UNGROUNDED` verdict, that verdict is preserved as-is, not reclassified.

---

## Finding 2: "[Source: None]" on correct refusal flagged as a fabricated citation

**How it was found:** Same out-of-scope test ("What is the capital of
France?"). The model did exactly the right thing — refused to answer,
correctly noted the context only covers collections policy topics, and
wrote `[Source: None]` to indicate no source applies. The citation
verification guardrail then flagged `"None"` as a fabricated source, since
it isn't a real filename in the retrieved set.

**Root cause:** `verify_citations()` treated every `[Source: X]` match as a
literal filename claim to check, with no allowance for the model's own
"no source applies" phrasing on a correct refusal.

**Fix:** `app/guardrails/output_guardrails.py`, `verify_citations()` — added
an explicit exclusion list (`none`, `n/a`, `na`, `not applicable`,
case-insensitive) so these placeholder citations are never checked against
the real source list, and therefore never flagged as fabricated.

**Verified the fix doesn't mask genuine failures:** added a companion test
confirming an actual fabricated filename (not a placeholder) is still
correctly caught.

---

## What this demonstrates

Both bugs were found by deliberately stress-testing the system's own
guardrails rather than just confirming the happy path worked — exactly the
practice the original learning guide (file 07, section 6) recommends:
"deliberately test a prompt injection," "check the out-of-scope query."
Neither bug was in retrieval, reranking, or generation — both were in the
guardrail layer itself, which is a useful, honest finding: guardrails need
their own testing and calibration, they aren't infallible just because
they're a safety mechanism.
