# NoseKnows — Implementation Plan (v0 → v1)

> Migrating from the sketch (linear LangChain pipeline, no DB, single-page UI)  
> to the full graded system (LangGraph agent, PostgreSQL, multi-page frontend, fine-tuned SLM, LLM-judge eval).

---

## Overview

| Phase | What | Touches |
|-------|------|---------|
| **1** | PostgreSQL + schema | infra |
| **2** | LangGraph agent | `src/agent/` |
| **3** | New frontend | `frontend/` |
| **4** | Ingestion UI + API | `backend/` + `frontend/` |
| **5** | Guardrails (upgraded) | `src/tools/guard.py` |
| **6** | RLHF data pipeline | `src/rlhf/` |
| **7** | Fine-tuning | `src/finetune/` |
| **8** | Evaluation (LLM-judge) | `src/eval/` + `frontend/` |

All phases build on each other in order. Each phase ends with the system in a runnable state.

---

## Phase 1 — PostgreSQL + Core Schema

**Goal:** replace the flat `.jsonl` feedback file and add all stateful data (chats, messages, feedback) to a proper Postgres DB. This is the foundation for chat history, multi-session support, and RLHF data management.

### 1.1 Add PostgreSQL to Docker Compose

Add a `postgres` service to `docker-compose.yml`:
```yaml
postgres:
  image: postgres:16-alpine
  environment:
    POSTGRES_DB: nosknows
    POSTGRES_USER: nosknows
    POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-nosknows}
  ports:
    - "5432:5432"
  volumes:
    - pg-data:/var/lib/postgresql/data
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U nosknows"]
    interval: 10s
    retries: 5
```

Backend depends on `postgres: condition: service_healthy`.

New env vars:
```
POSTGRES_URL=postgresql://nosknows:nosknows@postgres:5432/nosknows
```

### 1.2 Schema

Use **SQLAlchemy** (async, with `asyncpg`) + **Alembic** for migrations.

```sql
-- sessions: one row per browser/user session
CREATE TABLE sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title       TEXT,                        -- auto-generated from first query
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- messages: all chat turns
CREATE TABLE messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,               -- 'user' | 'assistant'
    content     TEXT NOT NULL,
    intent      TEXT,                        -- classify_intent output
    hyde_doc    TEXT,                        -- HyDE hypothetical doc
    perfumes    JSONB,                       -- list of PerfumeCard objects
    latency_ms  INTEGER,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- feedback: per-message thumbs rating
CREATE TABLE feedback (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id  UUID REFERENCES messages(id) ON DELETE CASCADE,
    session_id  UUID REFERENCES sessions(id) ON DELETE CASCADE,
    score       SMALLINT NOT NULL CHECK (score IN (-1, 1)),  -- -1=down, 1=up
    comment     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);
```

### 1.3 New API Endpoints

Replace the old `/chat` (stateless) and `/feedback` (jsonl) with:

```
POST   /sessions                  → create new chat session
GET    /sessions                  → list all sessions (id, title, updated_at)
DELETE /sessions/{id}             → delete session + cascade messages + feedback
GET    /sessions/{id}/messages    → full message history for a session
POST   /sessions/{id}/chat        → send message (replaces /chat)
POST   /messages/{id}/feedback    → rate a specific message (replaces /feedback)
GET    /health                    → unchanged
```

The old `/chat` and `/feedback` can remain as deprecated aliases during transition.

### 1.4 Backend Deps

Add to `requirements.txt`:
```
sqlalchemy[asyncio]>=2.0
asyncpg
alembic
```

**Files to create:**
- `src/db/models.py` — SQLAlchemy ORM models
- `src/db/session.py` — async engine + session factory
- `alembic/` — migration directory (`alembic init`)

**Files to modify:**
- `backend/main.py` — wire up new routes, remove flat-file feedback logic
- `docker-compose.yml` — add postgres service + volume
- `.env.example` — add `POSTGRES_URL`, `POSTGRES_PASSWORD`

---

## Phase 2 — LangGraph Agent

**Goal:** replace the hand-rolled ReAct loop (`src/agent/agent.py`) with a proper LangGraph `StateGraph`. The graph has typed state, explicit nodes for each tool, conditional edges for routing, and a loop-back mechanism for retries. This makes the agent debuggable, extensible, and more reliable with small models.

### 2.1 Graph Design

```
START
  │
  ▼
[classify_intent]
  │
  ├─ note_based ──────────────────────────────────┐
  │                                               │
  ├─ mood_based / occasion_based                 │
  │       │                                       │
  │       ▼                                       │
  │  [generate_hyde]                              │
  │       │                                       │
  ▼       ▼                                       ▼
[retrieve]  ◄──────────────────────────────────────
  │
  ▼
[generate_recommendation]
  │
  ▼
[validate]
  │
  ├─ PASS ──► END (return response)
  │
  └─ FAIL ──► [regenerate] (max 2 retries, then safe fallback)
                │
                └──► [validate] (loop)
```

### 2.2 State Schema

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    session_id: str
    query: str
    filters: dict
    intent: str | None
    hyde_doc: str | None
    retrieved: str | None          # formatted text block from ChromaDB
    recommendation: str | None
    validation: str | None         # "PASS: ..." or "FAIL: ..."
    final_answer: str | None
    retry_count: int
    messages: Annotated[list, add_messages]   # LangGraph message history
```

### 2.3 Node Implementations

Each node is a pure function `(state: AgentState) -> dict` that returns only the fields it updates.

```python
# src/agent/nodes.py

def node_classify_intent(state): ...
def node_generate_hyde(state): ...
def node_retrieve(state): ...
def node_recommend(state): ...
def node_validate(state): ...
def node_regenerate(state): ...    # bumps retry_count, tweaks prompt

# Routing functions (used by add_conditional_edges):
def route_after_intent(state) -> str: ...   # "hyde" | "retrieve"
def route_after_validate(state) -> str: ... # "end" | "regenerate" | "fallback"
```

### 2.4 Memory — PostgreSQL Checkpointer

LangGraph supports custom checkpointers. Implement `PostgresCheckpointer` (or use the community `langgraph-checkpoint-postgres` package if available) so that graph state is persisted per `session_id`. This gives the agent full chat history across turns for free.

The `messages` field in `AgentState` accumulates the full conversation so follow-up queries work correctly ("something lighter than the last one").

### 2.5 Tool Reuse

All existing tool functions in `src/tools/` stay unchanged — they're plain Python functions. The nodes just call them directly. **No LangChain tool wrappers needed.**

### 2.6 LLM Interface

Keep `src/model.py → OllamaLLM` as-is. Nodes call `llm.invoke(prompt)` directly.  
For fine-tuned phase: the same interface will point at the LoRA-loaded HF model instead of Ollama.

### 2.7 Dependencies

```
langgraph>=0.2
langgraph-checkpoint-postgres   # or implement custom
```

**Files to create:**
- `src/agent/graph.py` — `StateGraph` definition + compiled graph
- `src/agent/nodes.py` — all node functions + routing functions
- `src/agent/state.py` — `AgentState` TypedDict

**Files to replace/remove:**
- `src/agent/agent.py` — delete; `build_agent` / `run_agent` replaced by graph

**Files to modify:**
- `backend/main.py` — call `graph.invoke(state)` instead of `run_agent`; pass `session_id` as LangGraph thread ID for checkpointing.

---

## Phase 3 — New Frontend

**Goal:** rebuild `frontend/` to match the 5-page design in `desing/`. Keep Next.js + TypeScript + Tailwind. The design's vanilla React + CSS is the reference for layout/UX; translate it to Next.js App Router components.

### 3.1 Pages

| Route | Component | What it does |
|-------|-----------|-------------|
| `/` | `app/page.tsx` → redirect to `/chat` | |
| `/chat` | `app/chat/page.tsx` | Multi-session chat (sidebar with session list) |
| `/explorer` | `app/explorer/page.tsx` | Searchable perfume dataset browser |
| `/ingest` | `app/ingest/page.tsx` | Upload CSV + trigger re-indexing |
| `/eval` | `app/eval/page.tsx` | LLM-judge metrics dashboard |
| `/feedback` | `app/feedback/page.tsx` | Feedback log browser |

### 3.2 Chat Page (most complex)

**Left sidebar:** session list (title, timestamp). New chat button. Delete session button.  
**Main area:** message stream — user bubbles right, agent bubbles left.  
**Agent bubble:** intent badge, HyDE indicator, latency, text body, perfume cards grid, thumbs up/down.  
**Perfume card:** name, brand, note chips (top/heart/base), accord badges, rating.  
**Input bar:** textarea + send button (Enter to send, Shift+Enter newline).

State managed with `useState` + `useReducer` for the session list. No global state library needed.

API calls:
- `GET /sessions` — load session list on mount
- `POST /sessions` — create new session
- `DELETE /sessions/{id}` — delete session
- `GET /sessions/{id}/messages` — load history when switching sessions
- `POST /sessions/{id}/chat` — send message (streaming response if available)
- `POST /messages/{id}/feedback` — rate message

### 3.3 Assets

Copy logo files from `desing/assets/` to `frontend/public/`. Use in header.

### 3.4 Styling

Port the design's CSS custom properties to Tailwind config or a global CSS file. The design uses a `--clr-*` variable system and a fixed sidebar layout — replicate with Tailwind's `grid` layout.

### 3.5 Explorer Page

Simple table/grid of perfumes. Filters: text search, gender, accord, brand. Calls a new backend endpoint:
```
GET /perfumes?q=&brand=&accord=&gender=&limit=50&offset=0
```
Backend reads from ChromaDB metadata or from a Postgres-cached copy of chunks.

### 3.6 Files to Create

```
frontend/app/
  chat/page.tsx
  explorer/page.tsx
  ingest/page.tsx
  eval/page.tsx
  feedback/page.tsx
frontend/components/
  SessionSidebar.tsx
  MessageBubble.tsx
  PerfumeCard.tsx          (rewrite of existing)
  FilterBar.tsx
  MetricsChart.tsx
  FeedbackTable.tsx
frontend/lib/
  api.ts                   (expand with new endpoints)
  types.ts                 (expand with Session, Message types)
```

**Files to delete/replace:**
- `frontend/app/page.tsx` (single-page version)
- `frontend/components/Sidebar.tsx` (search options only — split into SessionSidebar + FilterBar)
- `frontend/components/ChatMessage.tsx` (rewrite as MessageBubble)

---

## Phase 4 — Ingestion API + UI

**Goal:** let users upload a new CSV through the UI and trigger re-indexing. This satisfies the "support data ingestion of new datasets" requirement.

### 4.1 Backend Endpoints

```
POST /ingest/upload       → multipart CSV upload; validates format; saves to disk
POST /ingest/run          → triggers data_pipeline + build_index as background task
GET  /ingest/status       → poll status of the running ingest job
GET  /ingest/datasets     → list uploaded datasets
```

Use FastAPI's `BackgroundTasks` or a simple `asyncio.Task` for the ingest job. Store status in a Postgres `ingest_jobs` table (or in-memory dict for simplicity).

### 4.2 Validation

When a CSV is uploaded, check that it has the required columns (`Perfume`, `Brand`, top/heart/base note columns, accord columns). Return a preview of the first 5 rows.

### 4.3 Pipeline

`POST /ingest/run` calls:
1. `src/data_pipeline.py` — clean CSV → chunks.jsonl (appends or rebuilds)
2. `src/rag/build_index.py` — re-embed and upsert into ChromaDB

For incremental ingestion: ChromaDB supports `upsert`, so new perfumes can be added without rebuilding the full index.

### 4.4 Frontend (Ingestion Page)

- Drag-and-drop CSV upload zone
- Column preview table after upload
- "Run Ingestion" button → polls `/ingest/status` → shows progress bar
- List of past ingestion jobs with row counts

---

## Phase 5 — Upgraded Guardrails

**Goal:** make the hallucination and toxicity guards robust enough to be relied upon with a small, sometimes unreliable SLM.

### 5.1 Toxicity (keep + upgrade)

Keep `detoxify` as primary. Add a lightweight regex blocklist for obvious off-topic abuse as a fast pre-filter (runs before the model call, saving latency).

### 5.2 Hallucination Guard (upgrade)

The current heuristic (capitalised noun phrase matching) is too coarse. Replace with:

1. **Entity extraction:** use spaCy's `en_core_web_sm` NER to extract `PRODUCT` and `ORG` entities from the response (perfume names + brands).
2. **Context grounding:** check each extracted entity against the retrieved perfume list.
3. **Confidence score:** `grounded / total_entities`. Threshold: 0.5.
4. **Fallback:** if score < threshold → strip the response, return only the top-1 retrieved perfume name with a template explanation.

### 5.3 Output Length Guard

SLMs sometimes produce excessively long outputs or repetitive loops. Add a max-token post-processor: if response > 600 tokens, truncate at the last complete sentence.

### 5.4 Domain Relevance — DO NOT IMPLEMENT

The v0 keyword-based domain relevance check (`_has_domain_content`) must be **removed**, not upgraded.

The problem: NoseKnows is specifically designed to handle subjective, metaphorical, and indirect queries. A query like *"a perfume for travelling to Paris"* or *"something that smells like pizza"* contains zero fragrance vocabulary, yet is completely valid input. Any keyword gate will reject these — exactly the queries the HyDE step is designed to handle.

The correct model: **the guard's job is to validate the response, not the query**. A valid response to "perfume for Paris" should mention notes, accords, or perfume names — so the toxicity check and hallucination grounding check are sufficient to catch off-topic responses. There is no need to check the input at all.

**Remove `_has_domain_content` from `src/tools/guard.py` entirely.**

### 5.5 Files to Modify

- `src/tools/guard.py` — all changes land here; keep the same `validate_response(response, context)` signature.

Add dep:
```
spacy>=3.7
# python -m spacy download en_core_web_sm
```

---

## Phase 6 — RLHF Data Pipeline

**Goal:** turn the accumulated feedback in Postgres into a usable DPO training set, and build the tooling to export and use it.

### 6.1 Feedback Collection (already in Phase 1)

Each thumbs up/down is stored in the `feedback` table with `message_id` linkage, so we can retrieve the full (query, response, score) triple.

### 6.2 DPO Pair Generation

For DPO (Direct Preference Optimisation) we need **(chosen, rejected)** pairs for the same prompt.

Strategy:
- For each session, find queries that received both a positive and a negative response (or use score > 0 as chosen and score < 0 as rejected relative to the same query across sessions).
- Export pairs as a JSONL file:
  ```json
  {"prompt": "...", "chosen": "...", "rejected": "..."}
  ```

Script: `src/rlhf/export_dpo_pairs.py`

### 6.3 Reward Model (lightweight alternative)

If DPO pairs are insufficient (low feedback volume), use a simple **reward model** approach:
- Train a binary classifier (logistic regression or small transformer head) on (query + response, score) pairs.
- Use it to filter synthetic training data: only include triples that score above threshold.

Script: `src/rlhf/train_reward_model.py`

### 6.4 Integration with Fine-tuning

The DPO pairs feed into Phase 7. The reward model can also be used at inference time as an additional guardrail.

### 6.5 Files to Create

```
src/rlhf/
  __init__.py
  export_dpo_pairs.py    reads from Postgres, outputs dpo_pairs.jsonl
  train_reward_model.py  optional lightweight reward model
```

**Files to modify:**
- `backend/main.py` — feedback now writes to Postgres (done in Phase 1); no change here.

---

## Phase 7 — Fine-tuning (Final Step, Desktop Only)

**Goal:** produce a LoRA-adapted version of the base SLM trained on the synthetic triples from `data/synthetic_triples.jsonl` (and optionally DPO-refined with RLHF pairs). This is what pushes the grade to 10/10.

### 7.1 Pre-conditions

- `data/synthetic_triples.jsonl` must exist (generated by `src/finetune/generate_synthetic.py`).
- Optionally: `data/dpo_pairs.jsonl` from Phase 6.
- GTX 1070 TI on the desktop (8GB VRAM) — quantise to 4-bit (QLoRA) to fit.

### 7.2 SFT (Supervised Fine-Tuning)

Keep existing `src/finetune/train.py` but upgrade:
- Switch to **QLoRA** (4-bit NF4 quantisation via `bitsandbytes`) to fit on 1070 TI.
- Format: `<|system|>\n{system}\n<|user|>\n{input}\n<|assistant|>\n{output}` (Gemma / LLaMA instruct template).
- Target modules: `q_proj`, `v_proj` (LoRA rank 16, alpha 32).
- Training: 3 epochs, batch size 4, gradient accumulation 4, cosine LR schedule.

```bash
python -m src.finetune.train \
  --base-model google/gemma-2-2b-it \
  --data data/synthetic_triples.jsonl \
  --output models/nosknows-lora \
  --epochs 3 \
  --use-4bit
```

### 7.3 DPO Fine-tuning (Phase 2 of training)

If DPO pairs exist, run a second training pass with `trl.DPOTrainer` on top of the SFT adapter:

```bash
python -m src.finetune.train_dpo \
  --model models/nosknows-lora \
  --data data/dpo_pairs.jsonl \
  --output models/nosknows-dpo
```

### 7.4 Switching the Inference Backend

When the adapter is ready, switch `src/model.py` from Ollama to direct HuggingFace inference:

```python
# src/model.py — add HFAdapterLLM class
class HFAdapterLLM:
    """Load a base model + LoRA adapter via HuggingFace transformers."""
    def __init__(self, base_model_id, adapter_path, use_4bit=True): ...
    def invoke(self, prompt: str) -> str: ...
```

Controlled by env var: `USE_HF_ADAPTER=true` / `LORA_ADAPTER_PATH=models/nosknows-dpo`.  
When `USE_HF_ADAPTER=false`, falls back to Ollama (for CPU-only laptop use).

### 7.5 Files to Create/Modify

- `src/finetune/train.py` — add QLoRA support (`--use-4bit` flag)
- `src/finetune/train_dpo.py` — new DPO training script
- `src/model.py` — add `HFAdapterLLM`; `load_llm()` checks env and returns appropriate backend
- `backend/Dockerfile.finetune` — ensure `bitsandbytes` + `flash-attn` installed

---

## Phase 8 — Evaluation (LLM-Judge)

**Goal:** offline batch evaluation using GPT-4o-mini as an LLM judge, run manually (or via CI) — **not on every live request**. Results are stored in Postgres and displayed in the Evaluation Lab page. This satisfies the course "Evaluation" requirement.

The judge is a tool you run deliberately: before a demo, after fine-tuning, when comparing two model configs. It is completely decoupled from the live agent pipeline.

### 8.1 Eval Set

Create `data/eval_queries.jsonl` once — 50–100 hand-crafted queries spanning all intent types and edge cases:

```jsonl
{"id": "e001", "query": "Something warm and cozy for a winter date night", "intent": "occasion_based"}
{"id": "e002", "query": "A perfume that smells like a rainy forest", "intent": "mood_based"}
{"id": "e003", "query": "Something with oud and rose", "intent": "note_based"}
{"id": "e004", "query": "A perfume for travelling to Paris", "intent": "occasion_based"}
{"id": "e005", "query": "Something that smells like pizza", "intent": "mood_based"}
...
```

This file is static and version-controlled. It never changes between runs so results are comparable.

### 8.2 Evaluation Script

`src/eval/run_eval.py` — run manually from the CLI:

```bash
# Run eval against the current configured model (baseline or fine-tuned)
python -m src.eval.run_eval \
  --queries data/eval_queries.jsonl \
  --model gpt-4o-mini \
  --tag "baseline-gemma2-2b" \
  --output data/eval_results.jsonl

# Run again after fine-tuning to compare
python -m src.eval.run_eval \
  --queries data/eval_queries.jsonl \
  --model gpt-4o-mini \
  --tag "finetuned-lora-v1" \
  --output data/eval_results.jsonl
```

**What it does:**
1. Reads queries from the eval set.
2. For each query: calls the local agent (same code path as production, but with no DB writes).
3. Sends the `(query, response, retrieved_perfumes)` triple to GPT-4o-mini with a judge prompt.
4. Aggregates scores across all queries.
5. Appends results to `data/eval_results.jsonl` and writes a run record to Postgres.

**Judge prompt:**
```
You are an expert evaluating an AI fragrance recommendation system.

User query: {query}
Agent response: {response}
Retrieved perfume list: {perfume_list}

Rate the response on three dimensions (integer 1–5):
- relevance: do the recommended perfumes match the user's description?
- groundedness: is the explanation based on the actual notes of the recommended perfumes?
- helpfulness: would a non-expert find this response useful and clear?

Respond with only valid JSON:
{"relevance": N, "groundedness": N, "helpfulness": N, "reasoning": "<one sentence>"}
```

### 8.3 Metrics Tracked

| Metric | Source |
|--------|--------|
| **Relevance** (1–5 avg ± std) | LLM judge, per eval run |
| **Groundedness** (1–5 avg ± std) | LLM judge, per eval run |
| **Helpfulness** (1–5 avg ± std) | LLM judge, per eval run |
| **Toxicity rate** | Guard logs in Postgres (from live traffic) |
| **Hallucination rate** | Guard logs in Postgres (from live traffic) |
| **User satisfaction** | Avg thumbs score from `feedback` table |
| **Latency P50/P95** | `messages.latency_ms` from live traffic |

Guard/satisfaction/latency metrics come from the live Postgres tables — no judge call needed for those. The judge is only for the quality dimensions.

### 8.4 Comparative Evaluation

Run the script twice with different `--tag` values (before and after fine-tuning). The Eval Lab page displays both runs side-by-side. Tags are stored in the Postgres `eval_runs` table.

### 8.5 Backend Endpoints

```
POST /eval/run             → triggers run_eval as a background task, returns job_id
GET  /eval/runs            → list past eval runs (tag, scores summary, created_at)
GET  /eval/runs/{id}       → full per-query results for a run
GET  /eval/live-metrics    → live stats from Postgres (toxicity, satisfaction, latency)
```

The `POST /eval/run` endpoint is optional — the CLI script is sufficient. The endpoint exists so the Eval Lab "Run Evaluation" button works without needing a terminal.

### 8.6 Frontend (Evaluation Lab)

- **Live metrics panel** (always visible, no run needed): toxicity rate, hallucination rate, avg user satisfaction, latency P50/P95 — pulled from `/eval/live-metrics`.
- **Past runs table:** list of eval runs with tag, date, avg scores. Click to expand per-query results.
- **"Run Evaluation" button:** calls `POST /eval/run`, polls status, refreshes the runs table when done.
- **Comparison view:** select two runs → bar chart of Relevance / Groundedness / Helpfulness side-by-side.

### 8.7 Files to Create

```
src/eval/
  __init__.py
  run_eval.py          CLI entry point + orchestration
  judge.py             OpenAI judge call + prompt template
  metrics.py           aggregation (mean, std, per-intent breakdown)
data/
  eval_queries.jsonl   static eval set (create once, commit to repo)
```

New env var:
```
OPENAI_API_KEY=sk-...
```

New Postgres table:
```sql
CREATE TABLE eval_runs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tag         TEXT NOT NULL,            -- e.g. "baseline-gemma2-2b"
    config      JSONB,                    -- model, n_samples, etc.
    scores      JSONB,                    -- aggregated metrics
    results     JSONB,                    -- per-query detail array
    created_at  TIMESTAMPTZ DEFAULT now()
);
```

---

## Dependency Summary

### New Python deps (add to requirements.txt)

```
# Phase 1 — DB
sqlalchemy[asyncio]>=2.0
asyncpg>=0.29
alembic>=1.13
psycopg2-binary>=2.9          # for Alembic sync ops

# Phase 2 — LangGraph
langgraph>=0.2
langgraph-checkpoint-postgres  # community package, may need manual install

# Phase 5 — Guardrails
spacy>=3.7

# Phase 7 — Fine-tuning (desktop only)
bitsandbytes>=0.43
flash-attn>=2.5               # optional, speeds up training

# Phase 8 — Eval
openai>=1.30
```

### Docker Compose additions

- `postgres:16-alpine` service
- New volume `pg-data`
- Backend env vars for `POSTGRES_URL`

---

## What Does NOT Change

- **ChromaDB** — kept as vector store (no migration to pgvector needed for this scope).
- **Ollama** — kept as the default inference backend for development/CPU use.
- **`src/tools/*.py`** — all tool functions stay; only their callers change (nodes instead of agent loop).
- **`src/rag/`** — `HyDERetriever` and `build_index.py` stay as-is.
- **`src/data_pipeline.py`** — unchanged.
- **Dataset files** — unchanged.
- **`src/finetune/generate_synthetic.py`** — unchanged.

---

## Suggested Work Order

```
Phase 1 (DB)      → Phase 2 (LangGraph)  →  Phase 3 (Frontend)
       ↘                                          ↙
        Phase 4 (Ingestion UI)  ←─────────────────
              ↓
        Phase 5 (Guardrails)
              ↓
        Phase 6 (RLHF pipeline)
              ↓
        Phase 7 (Fine-tuning)   ← desktop-only, do last
              ↓
        Phase 8 (Evaluation)
```

Phases 3 and 4 can be done in parallel with Phases 2 and 5 if the team splits work. Phase 7 is blocked until the desktop GPU setup is ready. Phase 8 is blocked until Phase 7 (needs both baseline and fine-tuned to compare).
