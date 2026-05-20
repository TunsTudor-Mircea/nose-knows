# NoseKnows — AI Fragrance Consultant

NoseKnows is an LLM-powered fragrance recommendation system. Describe a mood, occasion, or vibe in plain English ("something warm and cozy for a winter date night") and it returns ranked real perfume recommendations with note-level explanations — powered by a small language model, HyDE-augmented RAG over ~24 000 Fragrantica perfumes, and a full agentic pipeline.

**Team CTRL+Z** · 2025-2026 LLMs course · [GitHub](https://github.com/TunsTudor-Mircea/nose-knows)

---

## What It Can Do

| Feature | Details |
|---|---|
| **Natural-language chat** | Ask for perfumes by mood, occasion, notes, or follow-up questions across persistent sessions |
| **HyDE RAG** | Generates a hypothetical note profile from your query, embeds it, and retrieves the closest real perfumes from ChromaDB (~24k indexed) |
| **Intent classification** | Automatically routes queries: `mood_based`, `occasion_based`, `note_based`, `follow_up`, `guard` |
| **Perfume cards** | Each recommendation shows top/heart/base notes, accords, rating, and a link to Fragrantica |
| **Hallucination guard** | spaCy NER checks that named perfumes in the response are actually in the retrieved context |
| **Toxicity filter** | Regex blocklist + `detoxify` model screens queries and responses |
| **Multi-session chat** | Chat history persisted in PostgreSQL; switch between conversations via the sidebar |
| **Perfume explorer** | Browse and search all 24k indexed perfumes with gender and text filters |
| **Dataset ingestion UI** | Upload a new Fragrantica CSV and re-index the vector store from the browser |
| **Feedback collection** | Thumbs up/down on every assistant message; data stored for RLHF/DPO fine-tuning |
| **RLHF export** | One-click DPO pair export from the API when enough contrasting feedback is collected |
| **Eval dashboard** | View evaluation run results in the browser |

---

## Architecture

```
User query
    │
    ▼
Next.js frontend (port 3000)
    │  POST /sessions/{id}/chat
    ▼
FastAPI backend (port 8000)
    │
    ▼
LangGraph StateGraph
    ├── classify_intent   → mood_based | occasion_based | note_based | follow_up | guard
    │                        guard → immediate refusal (no retrieval or LLM call)
    ├── generate_hyde     → hypothetical note profile (embedded for retrieval)
    │                        skipped for note_based and follow_up intents
    ├── retrieve          → ChromaDB cosine search (all-MiniLM-L6-v2, 384-dim)
    ├── recommend         → Ollama SLM generates final answer
    └── validate          → toxicity + hallucination guard; retry loop on failure
    │
    ▼
PostgreSQL — sessions · messages · feedback · ingest_jobs
ChromaDB   — fragrances collection (~24k perfumes)
Ollama     — serves gemma2:2b (or fine-tuned nosknows-gemma2, see below)
```

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Docker Compose v2)
- ~6 GB free disk space (model + vector index + database)

---

## Setup (standard — gemma2:2b via Ollama)

This is the primary way to run NoseKnows. It uses `docker-compose-raw.yml`, which pulls `gemma2:2b` directly from Ollama — no local model files needed.

### Step 1 — Clone the repo

```bash
git clone https://github.com/TunsTudor-Mircea/nose-knows.git
cd nose-knows
```

### Step 2 — Get the Fragrantica dataset

The dataset is not included in the repository (too large for git).

1. Go to Kaggle and download the **Fragrantica Perfumes Dataset**  
   Search for: `fragrantica perfumes dataset` — look for the one with columns `Perfume`, `Brand`, `Top Notes`, `Heart Notes`, `Base Notes`, `Accords`, `Rating`.
2. Create the `dataset/` directory and place the CSV there:

```bash
mkdir dataset
cp /path/to/fragrantica.csv dataset/fra_cleaned.csv
```

### Step 3 — Configure environment

```bash
cp .env.example .env
```

The defaults work out of the box for Docker Compose. Only edit if you need to change ports.

### Step 4 — GPU / CPU: check the compose file

Open `docker-compose-raw.yml` and find the `ollama` service. There is a commented-out `deploy` block for NVIDIA GPU access:

```yaml
  ollama:
    ...
    # #only with gpu, don't use it on cpu because it is very slow
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [ gpu ]
```

- **No GPU (CPU only):** leave it commented out as-is. Inference will be slower (~30–90 s per response).
- **NVIDIA GPU available:** uncomment those lines to enable GPU pass-through. Responses will be significantly faster.

### Step 5 — Pull the model (first time only)

Start Ollama, then use the `model-init` helper to pull `gemma2:2b` from the Ollama registry:

```bash
# Start Ollama in the background
docker compose -f docker-compose-raw.yml up ollama -d

# Wait ~15 s for Ollama to become healthy, then pull the model
docker compose -f docker-compose-raw.yml --profile setup run --rm model-init
```

This pulls `gemma2:2b` into the `ollama-data` Docker volume. You only need to do this once.

### Step 6 — Ingest the dataset (first time only)

Build the ChromaDB vector index from the CSV:

```bash
docker compose -f docker-compose-raw.yml --profile tools run --rm ingest
```

This runs `src/data_pipeline.py` (CSV → `chunks.jsonl`) followed by `src/rag/build_index.py` (embed + index into ChromaDB). Takes 5–20 minutes depending on hardware. Progress is logged to stdout.

### Step 7 — Start everything

```bash
docker compose -f docker-compose-raw.yml up --build
```

Services start in dependency order: PostgreSQL → Ollama → backend → frontend.

| Service | URL |
|---|---|
| **Chat app** | http://localhost:3000 |
| **API docs** | http://localhost:8000/docs |
| **Backend health** | http://localhost:8000/health |

---

## Option: Run with the Fine-tuned Model

If you have the fine-tuned GGUF (`nosknows-gemma2-2b.Q4_K_M.gguf`) provided by the team, you can run the fine-tuned variant instead. This uses `docker-compose.yml`.

### Step A — Add the model files

The fine-tuned model is distributed as `models.zip` (provided separately by the team).

```bash
# Unzip into the repo root — creates the models/ directory automatically
unzip models.zip

# Verify:
ls models/
# nosknows-gemma2-2b.Q4_K_M.gguf
# Modelfile
```

### Step B — GPU / CPU: check the compose file

Same as above — open `docker-compose.yml`, find the `ollama` service, and uncomment the `deploy` block only if you have an NVIDIA GPU.

### Step C — Register the model with Ollama (first time only)

```bash
docker compose up ollama -d
docker compose --profile setup run --rm model-init
```

This reads `models/Modelfile` (which points to the GGUF) and registers it as `nosknows-gemma2` in Ollama.

### Step D — Ingest the dataset (first time only, skip if already done)

```bash
docker compose --profile tools run --rm ingest
```

### Step E — Start everything

```bash
docker compose up --build
```

---

## Evaluation (LLM-as-Judge)

NoseKnows includes a built-in eval pipeline that runs the live agent over a static query set and scores every response with an OpenAI LLM judge on three dimensions: **relevance**, **groundedness**, and **helpfulness** (each 1–5).

The judge is blind to rubric hints — a generic baseline scores ~3, a well-tuned model scores 4–5, making the delta meaningful for comparing runs.

### Prerequisites

The judge calls the OpenAI API. Add your key to `.env`:

```
OPENAI_API_KEY=sk-...
```

The default judge model is `gpt-4o-mini` (cheap, fast). You can swap to `gpt-4o` with `--model gpt-4o` for stricter scoring.

### Run the eval

Make sure the backend dependencies are installed and Ollama is running, then:

```bash
# Full eval over all queries in data/eval_queries.jsonl
python -m src.eval.run_eval \
    --queries data/eval_queries.jsonl \
    --output data/eval_results.jsonl \
    --tag baseline-gemma2-2b

# Smoke test — first 5 queries only
python -m src.eval.run_eval --limit 5 --tag smoke

# Use a different judge model
python -m src.eval.run_eval --tag finetuned-v1 --model gpt-4o

# Suppress the agent's verbose ReAct trace
python -m src.eval.run_eval --tag baseline-gemma2-2b --quiet
```

Results are **appended** to `--output` (default `data/eval_results.jsonl`), so multiple tagged runs accumulate in the same file. Each row contains the query, agent response, retrieved candidates, per-dimension scores, and the judge's one-sentence reasoning.

### What gets scored

| Dimension | What it measures |
|---|---|
| **Relevance** | Does the response match the query? Correct refusals on unanswerable queries score 5. |
| **Groundedness** | Are named perfumes and notes drawn from the retrieved candidates, not hallucinated? |
| **Helpfulness** | Does it read like a real fragrance expert — specific names, brands, note-level rationale? |

### Console output

```
[1/20] mood_001  intent=mood_based
  → rel=4 grd=5 hlp=4   Specific perfumes cited with accurate notes, slight generic framing

=== Eval summary — tag: baseline-gemma2-2b ===
Total: 20  scored: 19  failed: 1

Overall (mean ± std, n):
  relevance       3.84 ± 0.69  (n=19)
  groundedness    4.21 ± 0.71  (n=19)
  helpfulness     3.63 ± 0.76  (n=19)

Per intent:
  follow_up        relevance=3.50  groundedness=4.00  helpfulness=3.25   (n=4)
  mood_based       relevance=4.00  groundedness=4.25  helpfulness=3.75   (n=4)
  ...
```

### View results in the browser

Saved eval results are served by the API and visible in the frontend at **http://localhost:3000/eval**.

To re-run the backend eval inside Docker instead of locally:

```bash
docker compose -f docker-compose-raw.yml run --rm backend \
    python -m src.eval.run_eval --tag docker-run --quiet
```

(Requires `OPENAI_API_KEY` to be set in `.env`.)

---

## Fine-tuning from scratch (optional, requires CUDA GPU)

```bash
# Smoke-test on 50 examples first
docker compose --profile tools run --rm finetune \
  python -m src.finetune.train --max-samples 50 --epochs 1

# Full run
docker compose --profile tools run --rm finetune
```

Set `HUGGINGFACE_TOKEN` in `.env` for gated HuggingFace models (Gemma, LLaMA).

---

## Pages

| Route | What it does |
|---|---|
| `/chat` | Main chat interface — multi-session with sidebar history, top-k control |
| `/explorer` | Browse and text-search all indexed perfumes; filter by gender |
| `/ingest` | Upload a new Fragrantica CSV and trigger a live re-index |
| `/feedback` | View all collected thumbs-up/down ratings with query context |
| `/eval` | Display saved evaluation run results |

---

## Project Layout

```
nose-knows/
├── dataset/                  fra_cleaned.csv (gitignored — download from Kaggle)
├── data/                     chunks.jsonl, feedback_log.jsonl, dpo_pairs.jsonl
├── models/                   nosknows-gemma2-2b.Q4_K_M.gguf + Modelfile (gitignored)
├── chroma_db/                ChromaDB vector store (generated by ingest, gitignored)
├── src/
│   ├── data_pipeline.py      CSV → chunks.jsonl normaliser
│   ├── model.py              Ollama LLM wrapper
│   ├── agent/
│   │   ├── graph.py          LangGraph StateGraph definition
│   │   └── nodes.py          Intent, HyDE, retrieve, recommend, validate nodes
│   ├── rag/
│   │   ├── build_index.py    Embed chunks and write to ChromaDB
│   │   └── retriever.py      HyDERetriever (hypothetical document → cosine search)
│   ├── tools/
│   │   └── guard.py          Toxicity (detoxify) + hallucination (spaCy NER) checks
│   ├── db/
│   │   ├── models.py         SQLAlchemy ORM: Session, Message, Feedback, IngestJob
│   │   └── session.py        Async engine + get_db dependency
│   ├── finetune/
│   │   ├── generate_synthetic.py   Produce (input, reasoning, output) SFT triples
│   │   └── train.py                LoRA SFT with trl + peft
│   └── rlhf/
│       └── export_dpo_pairs.py     Export contrasting feedback as DPO pairs
├── backend/
│   ├── main.py               FastAPI app — all endpoints
│   ├── Dockerfile            Production image
│   └── Dockerfile.finetune   Fine-tuning image (CUDA)
├── frontend/                 Next.js 14 (TypeScript, Tailwind, App Router)
│   ├── app/
│   │   ├── chat/page.tsx     Chat UI with session sidebar
│   │   ├── explorer/page.tsx Perfume browser
│   │   ├── ingest/page.tsx   CSV upload + re-index
│   │   ├── feedback/page.tsx Feedback log viewer
│   │   └── eval/page.tsx     Eval results dashboard
│   └── components/
│       ├── ChatMessage.tsx   Renders markdown responses + thumbs feedback
│       ├── PerfumeCard.tsx   Note chips, accord badges, rating display
│       └── Sidebar.tsx       Session list + chat controls
├── dataset_creation/         Jupyter notebook for synthetic dataset generation
├── fine-tune/                Fine-tuning notebook + docs
├── alembic/                  Database migrations
├── docker-compose-raw.yml    Standard setup (gemma2:2b pulled from Ollama)
├── docker-compose.yml        Fine-tuned setup (local GGUF)
└── .env.example
```

---

## API Reference

### Sessions

| Method | Path | Description |
|---|---|---|
| `POST` | `/sessions` | Create a new chat session |
| `GET` | `/sessions` | List all sessions |
| `DELETE` | `/sessions/{id}` | Delete session and all its messages |
| `GET` | `/sessions/{id}/messages` | Full message history for a session |
| `POST` | `/sessions/{id}/chat` | Send a message and get a recommendation |

### Chat request body

```json
{
  "query": "something warm and spicy for autumn evenings",
  "filters": { "top_k": 5 }
}
```

### Chat response

```json
{
  "message_id": "uuid",
  "response": "For warm autumn evenings, I'd suggest…",
  "perfumes": [
    {
      "perfume": "Black Opium",
      "brand": "Yves Saint Laurent",
      "top_notes": ["pink pepper", "orange blossom"],
      "heart_notes": ["jasmine", "coffee"],
      "base_notes": ["vanilla", "patchouli", "cedarwood"],
      "accords": ["sweet", "warm spicy", "floral"],
      "rating": 4.2,
      "url": "https://www.fragrantica.com/…"
    }
  ],
  "hyde_doc": "Top notes: cinnamon, cardamom…",
  "intent": "occasion_based"
}
```

### Feedback

```
POST /messages/{message_id}/feedback
{ "score": 1 }   // thumbs up
{ "score": -1 }  // thumbs down
```

### Other endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/perfumes` | Browse indexed perfumes (`?search=&gender=&limit=&offset=`) |
| `GET` | `/brands` | Brand autocomplete (`?q=`) |
| `POST` | `/ingest/upload` | Upload a CSV file |
| `POST` | `/ingest/run` | Trigger re-index from uploaded file |
| `GET` | `/ingest/status/{job_id}` | Poll ingest job status |
| `GET` | `/rlhf/stats` | Feedback stats + DPO readiness |
| `POST` | `/rlhf/export` | Export DPO pairs to `data/dpo_pairs.jsonl` |
| `GET` | `/eval/results` | Return saved eval run rows |
| `GET` | `/health` | `{"status": "ok"}` liveness probe |

Full interactive docs: **http://localhost:8000/docs**

---

## Stopping / resetting

```bash
# Stop all containers
docker compose -f docker-compose-raw.yml down   # or docker compose down for fine-tuned

# Stop and delete all data (database, vector index, Ollama model cache)
docker compose -f docker-compose-raw.yml down -v
```

After `down -v` you will need to repeat the model pull (step 5) and ingest (step 6).
