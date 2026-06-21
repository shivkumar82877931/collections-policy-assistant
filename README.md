# Collections Policy Assistant

A retrieval-augmented generation (RAG) system that answers collections-agent policy questions — calling hours, required disclosures, hardship programs, data retention — grounded in a versioned internal policy corpus, with citations and a live groundedness check on every answer.

Built as an applied demonstration of production RAG architecture decisions for a compliance-sensitive domain: hybrid retrieval, cross-encoder reranking, structure-aware chunking, layered input/output guardrails, and a separated retrieval/generation evaluation harness.

**Runs entirely locally via Ollama — no API key, no cost, no data leaves your machine.**

---

## Why this exists

Most RAG demos are "embed some PDFs, ask a question." This one is built around the failure modes that actually show up once you treat a RAG system as production infrastructure rather than a tutorial: tables getting split mid-row during chunking, superseded policy versions getting retrieved as if current, embeddings missing exact regulatory citations, and models hallucinating confidently when context is insufficient. Each of those failure modes has a specific, testable fix implemented in this repo — see [Architecture Decisions](#architecture-decisions) below.

## Architecture

```
User query
  → Input guardrails (PII redaction, prompt-injection pattern check)
  → Hybrid retrieval (dense vector search + BM25, fused via reciprocal rank fusion)
  → Cross-encoder reranking (top-10 candidates → top-4 final context)
  → Generation (llama3.2 via Ollama, low temperature, citation-required prompt)
  → Output guardrails (LLM-as-judge groundedness check, citation verification)
  → Response, with sources and a groundedness verdict surfaced to the user
```

| Layer | Choice | Why |
|---|---|---|
| Chunking | Structure-aware (markdown headers) + table-atomic | Policy documents have real section structure; the fee schedule table must never be split mid-row |
| Metadata | Deterministic regex extraction (doc type, version, effective date, status) | Cheapest reliable method for a consistent header format — no LLM call needed at ingestion |
| Embeddings | `nomic-embed-text` via Ollama, with explicit query/document prefixing | nomic-embed-text is asymmetric — it requires a `search_query:`/`search_document:` prefix depending on role; getting this wrong silently degrades retrieval quality |
| Retrieval | Hybrid: dense (Ollama embeddings + Chroma/HNSW) + BM25, fused with RRF | Dense retrieval alone misses exact regulatory citations (e.g. "§1692c(a)(1)") and dollar figures — known embedding failure modes |
| Reranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`), local, no extra cost | Retrieve broad and cheap, rerank narrow and precise — embedding similarity is a relevance proxy, not ground truth |
| Filtering | Metadata pre-filter on `is_current` | Stops superseded policy versions from being retrieved and presented as current guidance |
| Generation | Low temperature, explicit context-only instruction, required inline citations | Compliance domain — minimize creative variance, force traceability |
| Guardrails | Input: PII regex + injection pattern detection. Output: LLM-as-judge groundedness + citation verification against the actually-retrieved set | Defense in depth — neither layer alone catches everything |
| Evaluation | Labeled eval set, recall@k computed separately from faithfulness rate | You cannot fix what you cannot separate: a wrong answer is either a retrieval failure or a generation failure, and they require different fixes |

## Architecture Decisions

A few specific things this repo deliberately tests, not just implements:

- **`_superseded_call_practices_v3.1.md`** exists specifically to verify the `is_current` metadata filter actually excludes outdated policy from default retrieval — toggle "include superseded versions" in the UI to see the filter turned off.
- The same file contains an embedded prompt-injection test string. If the assistant's behavior changes based on that text rather than answering the user's actual question, the structural defenses in `generation/generate.py` (delimited context blocks + explicit system-prompt rule) have failed and need revisiting.
- `hardship_loss_mitigation_guide.md` contains a 5-column markdown fee table specifically to stress-test the table-atomic chunking logic in `ingestion/chunking.py` — verified in `tests/test_ingestion.py` with a deliberately tiny chunk size to force the split decision.
- The eval set (`eval/eval_set.json`) includes one deliberately out-of-scope query ("What is the capital of France?") to test refusal behavior — a system that answers this from parametric knowledge despite an explicit "context only" instruction has a real faithfulness gap that prompting alone didn't close, which is exactly why the output groundedness check exists as a separate layer.

## Honest note on local model quality

`llama3.2` (the default small local model) is noticeably less reliable than a frontier API model at strictly following instructions like the citation format or "say so if context is insufficient." You will likely see this yourself while testing — that's expected, not a bug, and it's exactly the kind of gap the output groundedness guardrail exists to catch rather than relying on prompting alone. If you want stronger generation quality later, this is a one-line config change away from a frontier API model (see "Switching to an API model" below) — the rest of the architecture doesn't change.

## Project structure

```
backend/
  app/
    ingestion/        chunking, metadata extraction, ingestion script
    retrieval/         embeddings, vector store, hybrid search, reranker
    generation/         prompt construction, LLM calls
    guardrails/         input (PII, injection) and output (groundedness, citations)
    routes/             FastAPI endpoints
  data/sample_docs/    the policy corpus (4 documents)
  eval/                labeled eval set + recall@k / faithfulness harness
  tests/               unit tests for deterministic logic (no API calls)
frontend/
  index.html           single-file chat UI, no build step
```

## Setup

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```
(macOS: you can also download the app directly from [ollama.com](https://ollama.com))

### 2. Pull the models this project uses

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Ollama runs as a background service after install — no need to manually start it each time (`ollama serve` only if it's not already running, e.g. `ollama list` returns a connection error).

### 3. Set up the Python environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

cp .env.example .env
# Defaults in .env.example already point at Ollama — no key needed.
```

### 4. Build the index

```bash
python -m app.ingestion.ingest
```

### 5. Run the API

```bash
uvicorn app.main:app --reload
```

Open `frontend/index.html` directly in a browser (it points at `http://localhost:8000` by default).

### Run the tests

```bash
pytest tests/ -v
```

Unit tests cover chunking and guardrail logic deterministically — table-atomicity, metadata extraction, PII detection, and citation verification — none of these require Ollama running.

### Run the evaluation harness

```bash
python -m eval.run_eval
```

Requires Ollama running (it calls the embedding and chat models). Outputs recall@k (retrieval quality, independent of generation), faithfulness rate (generation quality, via the same groundedness checker used in production), and out-of-scope refusal accuracy — computed as three separate numbers, not one blended "accuracy" score, deliberately.

## Switching to an API model later

Everything in `retrieval/embeddings.py`, `generation/generate.py`, and `guardrails/output_guardrails.py` talks to Ollama through a thin client wrapper — swapping to OpenAI, Anthropic, or another provider means rewriting those three files' API calls (same function signatures, different client), not redesigning the pipeline. This is a deliberate benefit of keeping the orchestration layer (`routes/chat.py`, `retrieval/hybrid_search.py`, `retrieval/reranker.py`) provider-agnostic from the start.

## What I'd change for actual production scale

Documented honestly rather than glossed over:

- BM25 is rebuilt in-memory at startup from whatever's in Chroma — fine for a handful of documents, would move to a real search engine (OpenSearch/Elasticsearch) at real document volume.
- A local model this size is not production-grade for a regulated domain — this setup is for learning/demo purposes; a real deployment would use a larger, more reliable model (local or API) and a stronger judge model for groundedness checking.
- Groundedness checking adds a full extra LLM call per request — for high-throughput production use, I'd sample-check a percentage of live traffic rather than checking every single response synchronously in the request path.
- No conversation memory / multi-turn context yet — each query is independent. Multi-turn would need explicit state management (a real LangGraph use case).
- No structured-data fusion layer yet — a real collections assistant would also need to query live case status from a case management system, not just retrieve policy documents.
- No deployed/hosted version — this is local-only by design to keep cost at zero. A cloud deployment would need either an API-based model (Ollama doesn't run well on most free-tier hosts, which lack persistent GPU/long-running processes) or a self-hosted GPU instance.
