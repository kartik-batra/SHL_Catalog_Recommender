# SHL Assessment Recommender

Conversational agent that takes a hiring manager from a vague intent
("I'm hiring a Java developer") to a grounded shortlist of SHL assessments
through multi-turn dialogue.

Built for the **SHL Labs AI Intern take-home assignment**.

---

## Quick start

### 1 — Clone & install

```bash
git clone <your-repo-url>
cd shl-recommender
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 — Configure

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=<your key from console.groq.com>
```

### 3 — Build the catalog + index  *(one-time, ~20 min with detail pages)*

```bash
python scripts/build_catalog.py
```

This scrapes all 32 pages of **Individual Test Solutions** from
`https://www.shl.com/products/product-catalog/`, enriches each item with
its detail page, and builds a FAISS index with FastEmbed embeddings.

Flags:
```
--no-details   Skip detail pages (faster, less context for the LLM)
--index-only   Rebuild index from an existing catalog.json
```

### 4 — Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5 — Test it

```bash
# Health check
curl http://localhost:8000/health

# Single-turn chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need to hire a mid-level Java developer who works with stakeholders"}]}'
```

---

## API specification

### `GET /health`

```json
{ "status": "ok" }
```

Returns `503` while the index is still loading (cold start ≤ 2 min on Render free tier).

---

### `POST /chat`

**Request**

```json
{
  "messages": [
    { "role": "user",      "content": "Hiring a Java developer who works with stakeholders" },
    { "role": "assistant", "content": "Sure. What is the seniority level?" },
    { "role": "user",      "content": "Mid-level, around 4 years" }
  ]
}
```

The full conversation history must be sent on every call. The service is **stateless**.

**Response**

```json
{
  "reply": "Here are 5 assessments that fit a mid-level Java developer with stakeholder needs.",
  "recommendations": [
    { "name": "Java 8 (New)",  "url": "https://www.shl.com/...", "test_type": "K" },
    { "name": "OPQ32r",        "url": "https://www.shl.com/...", "test_type": "P" }
  ],
  "end_of_conversation": false
}
```

- `recommendations` is **`[]`** when the agent is still clarifying or refusing.
- `recommendations` contains **1–10 items** once the agent commits to a shortlist.
- `end_of_conversation` is `true` only when the agent considers the task complete.

---

## Architecture

```
Offline (one-time)
  scraper.py  ──►  catalog.json  ──►  embedder.py  ──►  catalog.index + catalog_meta.pkl

Online (per request)
  POST /chat
    │
    ├── retriever.py  ──  FastEmbed query embedding  ──►  FAISS top-15
    │
    └── agent.py
          ├── Build system prompt  (static rules + top-12 catalog chunks)
          ├── Groq API call        (meta-llama/llama-4-scout-17b-16e-instruct, json_object mode, T=0.15)
          ├── Parse JSON response
          └── URL guard            (drop any URL not in catalog.json)
```

### Component decisions

| Concern | Choice | Rationale |
|---|---|---|
| LLM | `meta-llama/llama-4-scout-17b-16e-instruct` on Groq | 500+ t/s, JSON mode, function calling, free tier |
| Embeddings | FastEmbed `BAAI/bge-small-en-v1.5` | Fully local, 384-dim, ~50ms/query, no API cost |
| Vector store | FAISS `IndexFlatIP` | Zero infra, exact search, ~400 vectors fits in RAM |
| Framework | FastAPI + Pydantic v2 | Async-native, strict schema validation, fast |
| Deployment | Render free tier | Cold-start ≤ 2 min (within evaluator allowance) |

### Agent intents

| Intent | Trigger | `recommendations` | `end_of_conversation` |
|---|---|---|---|
| `clarify` | Vague query / missing role | `[]` | `false` |
| `recommend` | Sufficient context gathered | 1–10 items | `false` → `true` after confirmation |
| `compare` | "Difference between X and Y?" | `[]` | `false` |
| `refuse` | Off-topic / injection attempt | `[]` | `false` |

### URL hallucination guard

Every URL produced by the LLM is validated against the scraped `catalog.json`.
URLs not present in the catalog are dropped before the response is serialised.
If all recommendations are dropped, the intent is downgraded to `clarify`.

### Prompt-caching

Groq applies automatic 50% prompt-caching on meta-llama/llama-4-scout-17b-16e-instruct models for identical token
prefixes. Since the static portion of the system prompt (rules + catalog context)
is the same across turns in a conversation, caching kicks in from turn 2 onward —
reducing cost and latency.

---

## Evaluation

```bash
# Run against local service with the sample traces
python scripts/evaluate.py --traces data/traces/ --url http://localhost:8000
```

Metrics reported:
- **Mean Recall@10** across all traces
- **Schema compliance** — every response has `reply`, `recommendations`, `end_of_conversation`
- **Turn cap** — conversations never exceed 8 turns
- **Behavior probes** — no recommendation on turn 1 for vague queries; off-topic refusal

---

## Deployment (Render)

1. Push this repo to GitHub.
2. Create a new **Web Service** on [render.com](https://render.com), pointing at your repo.
3. Render auto-detects `render.yaml`. Set the `GROQ_API_KEY` environment variable
   in the dashboard (it is marked `sync: false` so it is never committed).
4. First deploy runs `scripts/build_catalog.py` (~20 min) then starts uvicorn.
5. Subsequent deploys reuse the cached index (add `data/catalog.json` to your repo
   and use `--index-only` in `buildCommand` after the first deploy).

> **Tip for faster re-deploys**: commit `catalog.json` to git but keep
> `catalog.index` and `catalog_meta.pkl` in `.gitignore`.
> Change the build command to `pip install -r requirements.txt && python scripts/build_catalog.py --index-only`
> after your first successful deploy.

---

## Project structure

```
shl-recommender/
├── data/
│   ├── catalog.json          # generated by scraper (commit this)
│   ├── catalog.index         # FAISS binary (gitignored, rebuilt on deploy)
│   ├── catalog_meta.pkl      # catalog metadata (gitignored, rebuilt on deploy)
│   └── traces/               # sample evaluation traces
│       ├── trace_001_java_dev.json
│       └── trace_002_sales_manager.json
├── scripts/
│   ├── build_catalog.py      # offline pipeline: scrape → embed → index
│   └── evaluate.py           # local evaluation harness
├── src/
│   ├── agent.py              # LLM pipeline, intent routing, URL guard
│   ├── config.py             # pydantic-settings from .env
│   ├── embedder.py           # FastEmbed + FAISS index builder
│   ├── models.py             # Pydantic request/response schemas
│   ├── retriever.py          # FAISS search, catalog lookup, prompt formatting
│   └── scraper.py            # async SHL catalog scraper
├── main.py                   # FastAPI app (lifespan, /health, /chat)
├── requirements.txt
├── render.yaml
├── Procfile
└── .env.example
```
