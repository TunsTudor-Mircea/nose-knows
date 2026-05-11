# NoseKnows — AI Fragrance Consultant

An LLM/SLM-based fragrance recommendation agent that maps natural-language scent descriptions to real perfumes using HyDE-powered RAG, a fine-tuned SLM, and an agentic tool pipeline.

**Team CTRL+Z** · [GitHub](https://github.com/TunsTudor-Mircea/nose-knows)

---

## Architecture

```
User query
    │
    ▼
FastAPI backend (/chat)
    │
    ▼
LangChain ReAct Agent
    ├── classify_intent        → mood_based | occasion_based | note_based | follow_up
    ├── generate_hyde_document → hypothetical structured note profile
    ├── retrieve_fragrances    → ChromaDB (all-MiniLM-L6-v2 embeddings, ~24k perfumes)
    ├── generate_recommendation→ final answer with note rationale
    └── validate_response      → toxicity + hallucination guard
    │
    ▼
Next.js UI (chat + perfume cards + thumbs feedback)
    │
    ▼
/feedback → feedback_log.jsonl (RLHF seed data)
```

---

## Quick Start

### Docker (recommended)

```bash
# 1. Copy env file (no secrets needed for basic use)
cp .env.example .env

# 2. Start Ollama, then pull the model (first time only)
docker compose up ollama -d
docker compose --profile setup run --rm model-init   # pulls gemma2:2b

# 3. First-time data ingestion
docker compose --profile tools run --rm ingest

# 4. Start everything
docker compose up --build

# Open http://localhost:3000
```

**CPU note:** this stack is configured to run without GPU access. Expect the
model to be slower on CPU than on a CUDA-enabled machine.

**Swap the model:**
```bash
# In .env — any model available at https://ollama.com/library
OLLAMA_MODEL=gemma2:2b
```

**One-shot fine-tuning inside Docker (uses CUDA):**
```bash
# Smoke-test (50 examples)
docker compose run --rm --profile tools finetune python -m src.finetune.train --max-samples 50 --epochs 1

# Full run
docker compose run --rm --profile tools finetune
```

---

### Manual (no Docker)

### 1. Python environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
# Edit .env — set HUGGINGFACE_TOKEN for gated models (Gemma / LLaMA)
```

### 3. Data pipeline (one-time)

```bash
# Step 1: clean CSV → chunks.jsonl
python -m src.data_pipeline

# Step 2: embed and index into ChromaDB (~5–15 min on first run)
python -m src.rag.build_index
```

### 4. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

---

## Fine-tuning (optional — improves recommendations)

```bash
# Generate synthetic training triples (overnight job, ~5 000 examples)
python -m src.finetune.generate_synthetic

# Smoke-test training on 50 examples first
python -m src.finetune.train --max-samples 50 --epochs 1

# Full fine-tuning run
python -m src.finetune.train

# Point the backend at the adapter
# Set LORA_ADAPTER_PATH=models/nosknows-lora in .env
```

---

## Models

| Role | Default model | Alternative |
|------|---------------|-------------|
| SLM (backbone) | `google/gemma-2-2b-it` | `meta-llama/Llama-3.2-3B-Instruct` |
| Embeddings | `all-MiniLM-L6-v2` | — |

Both Gemma and LLaMA require a HuggingFace account with gated model access.  
Run `huggingface-cli login` or set `HUGGINGFACE_TOKEN` in `.env`.

---

## Project layout

```
nose-knows/
├── dataset/              Raw Fragrantica CSVs
├── data/                 Generated: chunks.jsonl, synthetic_triples.jsonl, feedback_log.jsonl
├── chroma_db/            ChromaDB vector store (generated, not committed)
├── models/               LoRA adapter weights (generated, not committed)
├── src/
│   ├── data_pipeline.py  CSV → chunks.jsonl
│   ├── model.py          SLM loader (HuggingFace + LoRA)
│   ├── agent/
│   │   └── agent.py      LangChain ReAct AgentExecutor
│   ├── rag/
│   │   ├── build_index.py  Embed + ingest into ChromaDB
│   │   └── retriever.py    HyDERetriever
│   ├── tools/
│   │   ├── intent.py       classify_intent
│   │   ├── hyde.py         generate_hyde_document
│   │   ├── retriever.py    retrieve_fragrances
│   │   ├── recommender.py  generate_recommendation
│   │   └── guard.py        validate_response
│   └── finetune/
│       ├── generate_synthetic.py  Produce (input, reasoning, output) triples
│       └── train.py               LoRA SFT with trl + peft
├── backend/
│   └── main.py           FastAPI: /health, /chat, /feedback
└── frontend/             Next.js (TypeScript, Tailwind, App Router)
    ├── app/page.tsx       Main chat view
    ├── components/
    │   ├── ChatMessage.tsx Renders agent responses + feedback buttons
    │   ├── PerfumeCard.tsx Note chips, accord badges, rating
    │   └── Sidebar.tsx     HyDE toggle, top-k, gender/accord filters
    └── lib/
        ├── api.ts          sendChat / sendFeedback / checkHealth
        └── types.ts        Shared TypeScript interfaces
```

---

## API reference

### `POST /chat`

```json
{
  "query": "something cozy for a winter date night",
  "filters": {
    "gender": "women",
    "accord": "amber",
    "top_k": 5,
    "use_hyde": true
  }
}
```

Response:
```json
{
  "response": "For a cozy winter evening…",
  "perfumes": [{ "perfume": "Black Opium", "brand": "YSL", … }],
  "hyde_doc": "Top notes: cinnamon…",
  "intent": "occasion_based"
}
```

### `POST /feedback`

```json
{ "query": "…", "response": "…", "score": 5 }
```

### `GET /health`

```json
{ "status": "ok", "service": "nosknows-api" }
```
