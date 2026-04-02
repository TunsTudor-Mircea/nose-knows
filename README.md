# 👃 NoseKnows

**Team:** CTRL+Z

**Members:**
Zaharie Nikolass-Rafael, Tuns Tudor-Mircea, Turcu Alexia-Cristiana, Paraschiv Tudor-Costin, Bogdan Ana, Ravas Adrian-Georgel

---

## Overview

NoseKnows is an AI-powered conversational fragrance recommendation system that translates natural language scent descriptions into structured perfume characteristics. Rather than browsing by note or brand, users describe a mood, occasion, or feeling — and the system reasons over a structured fragrance database to surface relevant recommendations.

---

## Architecture

The system follows an **agent architecture** in which a central fine-tuned SLM orchestrates a set of tools and makes decisions dynamically at each step.

### Dataset

- **Primary:** [Fragrantica Fragrance Dataset](https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset) — structured perfume records with top, heart, and base notes, accords, and community ratings.
- **Synthetic:** LLM-generated (input, reasoning, output) triples derived from the Fragrantica notes, used for fine-tuning the model to associate subjective language with olfactory profiles.

### Chunking Strategy

Each perfume is stored as a single atomic chunk containing its name, brand, and all note layers. This avoids cross-chunk reasoning at retrieval time and keeps embedding quality high.

### Vector Database

**ChromaDB** — chosen for its embedded/local mode, native Python integration, metadata filtering support, and open-source licensing.

### Models

| Role | Model |
|---|---|
| Core SLM (fine-tuned) | TBD |
| Embeddings | TBD |
| Reward Model (RLHF) | Trained on collected user preference pairs |

### RAG Strategy

Primary strategy is **HyDE (Hypothetical Document Embeddings)** — the agent generates a hypothetical structured note description from the user query, embeds it, and retrieves the top-k most similar perfumes from ChromaDB.

### Agentic Pipeline

The agent has access to the following tools and invokes them conditionally based on query context:

1. **Intent Analysis** — classifies the query type to determine the tool execution path.
2. **HyDE Generator** *(conditional)* — translates subjective/metaphorical queries into a hypothetical fragrance profile. Skipped for explicit note queries.
3. **Fragrance Retriever** — queries ChromaDB; retries with a reformulated query if confidence is below threshold.
5. **Recommendation Generator** — synthesises retrieved records and user intent into a final response.
6. **Toxicity / Hallucination Guard** — validates output before delivery; triggers regeneration or a safe fallback if needed.
7. **RLHF Feedback Loop** — user ratings are collected post-delivery and used to periodically update the SLM via DPO (Direct Preference Optimisation), steering future outputs toward higher-quality recommendations.

---

## Roadmap

| Week | Milestone |
|---|---|
| 2 | Team formation & role assignment |
| 4 | RFC — overview, motivation, goals |
| 5–6 | Architecture document (this) |
| 8–9 | Progress check & timeline update |
| 11–12 | Final demo, documentation & source code |
