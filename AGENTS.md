# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project

IK ML Engineering course — Case Study 2. Build an AI-powered IMDb movie chatbot using RAG (retrieval-augmented generation). The primary deliverable is `wei-wong.ipynb` — a graded Jupyter notebook submission. **Do NOT post this notebook publicly** (course policy).

- `wei-wong.ipynb` — **THE graded submission** (Wei's own work)
- `imdb-notebook.ipynb` — Reference template (Codex-generated, not the submission)
- `imdb_langgraph_app.py` — Standalone LangGraph refactor (~950 lines, bonus work)

## Environment & Setup

Python 3.14, virtual environment at `.venv/`. Activate before running anything locally:

```bash
source .venv/bin/activate
```

`OPENAI_API_KEY` must be set in `.env` (see `.env.example`). The notebook also supports Google Colab Secrets as a fallback.

## Commands

```bash
# Launch the submission notebook
jupyter notebook wei-wong.ipynb

# Run the LangGraph standalone app (requires OPENAI_API_KEY in .env)
python imdb_langgraph_app.py
```

There are no automated tests or lint scripts. The notebook contains an inline evaluation harness (`run_evaluation_harness()`) in Part 7 that runs 15 test cases and reports KPIs.

## Architecture

### Data Pipeline (Part 1–2)

Raw `imdb_dataset.csv` (3173 rows) → dedup (2762 unique movies) → cast parsing (concatenated names like `LeonardoDiCaprioKateWinslet` split via regex) → `search_text` feature (labeled concatenation of all fields) → LangChain `Document` objects → FAISS index (in-memory, 1536-dim `text-embedding-3-small` embeddings). No chunking — each movie is a single document.

### RAG Pipeline (Part 3–4)

```
User query
  → LLM query rewriter (understand_query: resolve follow-ups, check topic relevance)
  → regex intent detection (detect_intent)
  → regex constraint extraction (extract_query_constraints + fuzzy name matching)
  → FAISS semantic retrieval (k=30 candidates)
  → reranking (72% semantic score + 28% preference bonuses, hard-constraint filter first)
  → top-5 docs → LLM generation (gpt-4o-mini, temp=0.2)
  → formatted response
```

Over-fetching (k=30) then filtering is intentional — pure semantic search cannot enforce numerical constraints (e.g., `rating > 8.0`).

### Multi-Agent Orchestration (Part 5)

Intent detection routes to one of five agents:
- **search** — general discovery: FAISS + reranking + LLM
- **recommendation** — preference-weighted: FAISS + preference-weighted reranking + LLM
- **catalog** — exhaustive lists: bypasses FAISS entirely, filters `clean_df` directly (DataFrame filter, no LLM call), sorted by IMDb rating desc
- **clarification** — ambiguous/too-short queries: static response
- **fallback** — off-topic/greetings: static response, no API calls

Catalog intentionally bypasses FAISS because "give me all X" queries need every match, not top-k semantic hits.

### Guardrails (Part 6)

`safe_chatbot()` is the 4-layer entry point: input validation → topic filter (`is_probably_movie_related()`, 5-signal check) → agent pipeline → exception wrapper. Returns a `status` field (`ok`, `invalid_input`, `off_topic`, `no_match`, `error`). `_gradio_history_to_tuples()` converts Gradio dict-format history to tuple pairs expected by the pipeline.

### Gradio UI (Part 7)

`gr.ChatInterface` with movie poster thumbnails (80px CSS), 4 example queries, and structured movie cards. The LLM writes only a conversational summary (3–5 sentences) — the UI renders all structured data. `format_movie_matches()`, `format_catalog_matches()`, `format_final_response()` handle display.

### LangGraph Refactor (`imdb_langgraph_app.py`)

`IMDbLangGraphApp` wraps the same logic as a `StateGraph` with typed `GraphState`. Nodes: `prepare → route → [search_agent | recommendation_agent | catalog_agent | clarification_agent] → finalize`. Runtime components (embeddings, FAISS, LLM) are lazily initialized on first query via `_ensure_runtime_components()`.

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding model | `text-embedding-3-small` | Fast, cheap, 1536 dims — sufficient for ~2762 movies |
| LLM | `gpt-4o-mini`, temp=0.2 | Cost-effective; low temp for consistent movie facts |
| Vector store | FAISS in-memory | No infra; dataset fits in RAM |
| Retrieval k | 30 | Over-fetch then rerank for better precision |
| Catalog retrieval | DataFrame filter | FAISS top-k misses exhaustive matches |
| Chunking | None | Median `search_text` ~212 chars; each movie is one document |

## Dataset Quick Facts

- Source: `imdb_dataset.csv` — 3173 raw rows → 2762 after dedup
- 10 columns: Title, IMDb Rating, Year, Certificates, Genre, Director, Star Cast, MetaScore, Poster-src, Duration (minutes)
- `Star Cast` has no delimiters — names are concatenated (requires regex parsing)
- 4 intentional title-year collisions kept (different films with same title/year)
- `movie_id` format: `dune_2021` (slugified title + year)

## Reference Notebook Cell Map

When working on `wei-wong.ipynb`, the reference notebook `imdb-notebook.ipynb` has working implementations:
- Cast parsing: Cell 7
- EDA + `split_cast_heuristic()`: Cell 10
- `build_search_text()`: Cell 12
- `row_to_document()`: Cell 16
- Embeddings, FAISS, LLM, prompt: Cells 17–20
- `history_to_text()`, `docs_to_context()`, `generate_answer()`: Cell 21
- Constraint/preference/reranking functions (15 total): Cell 22
- `rewrite_query_for_retrieval()`, `run_chatbot_query()`: Cell 23
- Formatting functions: Cell 24
- Agent functions: Cell 30
- `detect_intent()`, `orchestrate_agents()`: Cell 31
- Guardrails (`is_probably_movie_related`, `safe_chatbot`): Cell 32
- Gradio UI: Cell 33
- Evaluation harness: Cell 36
