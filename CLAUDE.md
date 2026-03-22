# CLAUDE.md — IMDb Movie Chatbot (Case Study 2)

## Project Overview

IK ML Engineering course — Case Study 2. Build an AI-powered IMDb movie chatbot that uses RAG (retrieval-augmented generation) to answer natural language movie queries.

**Business framing:** An intelligent content discovery assistant that replaces manual search with conversational AI. Positioned as a product demo, not just a notebook exercise.

## Project Status: SUBMITTED (2026-03-21)

Wei submitted `Wei_Wong_IMDb_Chatbot.ipynb` as her graded case study. The notebook runs end-to-end in Google Colab.

## How We Worked

Wei built `wei-wong.ipynb` as the working copy; `Wei_Wong_IMDb_Chatbot.ipynb` is the Colab-only submission copy (local paths and `.env` loading removed). `imdb-notebook.ipynb` was a Codex 5.3 reference template — not the submission.

**My role:** Mentor, coach, and pair programmer. Explain before executing. Wei drove decisions; I explained tradeoffs and helped write code. Each part has strong markdown narrative showing the grader she understands *why*, not just *what*.

## Notebook Structure — wei-wong.ipynb (ALL PARTS COMPLETE)

### Part 1: Data Understanding & Preparation — COMPLETE
- Header, Project Overview, Part 1 Objectives ✓
- Environment setup, imports, dataset loading (3173 → 2762 after dedup) ✓
- Data quality snapshot, duplicate inspection, exact dedup, movie_id ✓
- Cast parsing (lowercase→uppercase split + particle merge for Di/Mc/etc.) ✓
- Cast quality audit (reliable vs low-confidence flag) ✓
- 6-panel EDA (ratings, years, genres, certificates, duration, directors) ✓
- `search_text` feature (labeled format for embeddings) ✓
- Keyword spot-checks (Spielberg, Nolan, Documentary, DiCaprio, 2020) ✓
- Data integrity assertions (6 checks, all passing) ✓
- Data contract summary for Part 2 ✓

### Part 2: Embeddings & Vector Store — COMPLETE
- Section header with business framing + objectives ✓
- Install dependencies (langchain, faiss-cpu, etc.) ✓
- Imports + API key loading (.env → Colab Secrets → error) ✓
- Document construction explanation (chunking decision explained) ✓
- `row_to_document()` + build 2,762 Documents with full metadata ✓
- Embeddings explanation (text-embedding-3-small, 1536 dims) ✓
- FAISS index construction (2,762 vectors) ✓
- Retrieval validation: 5 test queries (2 title-based, 3 descriptive) ✓
- Retrieval insights + Part 2 data contract for Part 3 ✓
### Part 3: Retrieval & Reranking — COMPLETE
- Part 3 header with business framing + architecture diagram ✓
- Imports, lookup tables (genre tokens, titles, directors, actors) ✓
- Hard constraints vs soft preferences explanation ✓
- `extract_query_constraints()` — 9 constraint types via regex (genre, cert, actor, rating, year, decade, duration) ✓
- `extract_preference_profile()` — soft signals from query + chat history (titles, directors, moods, era) ✓
- Reranking formula explanation (72% semantic + 28% bonuses) ✓
- `rerank_with_constraints()` — hybrid scoring with hard filter + fallback ✓
- Debug panel: raw semantic vs reranked side-by-side on 3 queries ✓
- Query rewriting explanation (lightweight alternative to HyDE) ✓
- `rewrite_query_for_retrieval()` + `run_retrieval_pipeline()` — full Part 3 orchestrator ✓
- Retrieval k experiment (k=5/15/30/50 comparison) ✓
- Part 3 summary + data contract for Part 4 ✓
### Part 4: LLM Integration & Prompt Engineering — COMPLETE
- LLM-only vs RAG comparison (Ben Affleck queries — hallucination vs grounding) ✓
- `ChatPromptTemplate` with movie concierge persona + 4-part output structure ✓
- `history_to_text()`, `docs_to_context()`, `generate_answer()` helpers ✓
- Empty-docs edge case guard in `generate_answer()` ✓
- Demo: 3 query types (title lookup, preference-based, follow-up with history) ✓
- Ablation: temperature (0.0/0.2/0.5/0.7), prompt style (minimal/balanced/verbose), context window (3/5/10 docs) ✓
- Part 4 summary + data contract for Part 5 ✓
### Part 5: Multi-Agent Orchestration — COMPLETE
- Part 5 header with business framing + architecture diagram ✓
- `run_chatbot_query()` — unified retrieval + generation wrapper (combines Parts 3 & 4) ✓
- 5 specialized agents: search, recommendation, catalog, clarification, fallback ✓
- `fallback_agent()` — handles off-topic/greetings without retrieval or LLM calls (Edge Cases rubric) ✓
- `detect_intent()` — rule-based keyword/regex classifier with greeting detection ({0,2} trailing words) ✓
- `orchestrate_agents()` — top-level router; intent-first ordering so "hi" routes to fallback, not clarification ✓
- Demo: 7 queries routing to all 5 agents (search, recommendation, catalog, clarification, 2× fallback) ✓
- Part 5 summary + data contract for Part 6 ✓
### Part 6: Guardrails & Safety — COMPLETE
- Part 6 header with business framing + 6-layer architecture diagram ✓
- Defense-in-depth explanation (fallback_agent blocklist vs is_probably_movie_related allowlist) ✓
- `is_probably_movie_related()` — 5 signal checks (keywords, genres, certs, years, entities) ✓
- `evaluate_constraint_compliance()` — post-hoc audit using existing hard-constraint helpers ✓
- `safe_chatbot()` design explanation + status value documentation ✓
- `safe_chatbot()` — 6-layer entry point (input validation → greeting short-circuit → history normalization → LLM query rewriting → deterministic guards → agent pipeline + exception wrapper) ✓
- `_is_ranking_followup()` — detects ranking/comparison follow-ups; sets `suppress_movie_section` so answer-only (no result grid) ✓
- `_gradio_history_to_tuples()` — boundary converter so Gradio dict-format history flows through tuple-based pipeline ✓
- Stress test demo: 9 queries covering all status paths (invalid_input, off_topic, ok + compliance, fallback) ✓
- Multi-turn stress test: Gradio dict-format history + follow-up query with preference profile assertion ✓
- Part 6 summary + data contract for Part 7 ✓
### Part 7: UI (Gradio) & Evaluation Harness — COMPLETE
- Part 7 header with business framing + architecture diagram ✓
- `format_movie_matches()`, `format_catalog_matches()`, `format_final_response()` — 3 formatting functions with poster thumbnails ✓
- Gradio `Blocks` UI: two-column layout (chat + tabbed input panel with Type/Voice tabs) ✓
- `gradio_chat_fn()` adapter, `submit_text_turn()`, `submit_voice_turn()`, `clear_demo()` event handlers ✓
- `transcribe_audio_to_text()` — OpenAI `gpt-4o-mini-transcribe` with movie-domain transcription prompt ✓
- Transcript preview: voice input shows what was heard before sending to pipeline ✓
- LLM prompt updated: conversational summary only (no movie lists — UI handles structured cards) ✓
- `catalog_agent()` rewritten: bypasses FAISS, filters `clean_df` directly, sorts by IMDb rating desc, no LLM call ✓
- Genre plural normalization in `extract_query_constraints()`: "documentaries" → "Documentary" etc. ✓
- `EVAL_TEST_CASES` — 15 test cases (title lookup, constraints, follow-up, typo, edge, off-topic, no-match, surname matching, mood recommendation, excluded titles) ✓
- `_title_hit()`, `evaluate_single_case()`, `run_evaluation_harness()` — evaluation pipeline ✓
- KPI summary: retrieval rate, first-answer success, compliance, follow-up resolution, fallback rate, latency p50/p95 ✓
- Part 7 summary ✓
- Project summary + future enhancements (separate cell) ✓

### Part 8: Hybrid Query Understanding — COMPLETE
**Architecture pivot:** `understand_query()` simplified from full query analyzer to query rewriter only. LLM resolves follow-ups and checks topic relevance; regex stays as source of truth for intent detection (`detect_intent()`) and constraint extraction (`extract_query_constraints()`). This reduces regression risk from prompt drift.

- `understand_query()` — gpt-4o-mini call returning resolved_query + is_movie_related only (no intent/constraints) ✓
- Resolves follow-up references using chat history ("these" → previous results, 800-char context window) ✓
- History-aware topic filtering (follow-ups in movie conversations aren't blocked) ✓
- Fuzzy name matching: `_match_person_name()` — 3-pass resolution (exact → unique surname alias → typo recovery via difflib, cutoff 0.84) ✓
- `DIRECTOR_LAST_NAME_LOOKUP` (1,594 surnames), `ACTOR_LAST_NAME_LOOKUP` (3,849 surnames) ✓
- `extract_query_constraints()` rewritten to use `_match_person_name()` for director/actor ✓
- `detect_intent()` enhanced: mood-first recommendation routing ("something suspenseful tonight") ✓
- IMDb 7.0 floor for recommendation mode (raised from 6.0) ✓
- Title injection in retrieval: exact title matches from resolved query supplement FAISS results ✓
- Placeholder data penalty (-0.15) for movies with MetaScore=66.0 + Duration=116.3 (unscraped fields) ✓
- Mentioned-title exclusion: "I loved Inception" won't recommend Inception back ✓
- Broken poster fallback: onerror handler hides broken CDN images gracefully ✓
- Evaluation harness expanded: 9 → 15 test cases (added surname matching, mood recommendation, catalog constraints, excluded titles) ✓
- `format_final_response()` respects `suppress_movie_section` — ranking follow-ups return prose only, recommendation follow-ups still render cards ✓
- Part 8 write-up: synthetic_description enrichment documented as future enhancement (not implemented) ✓
- Topic filter fix: genre plurals added to `MOVIE_HINT_TERMS`; LLM rewrite always adopted for typo recovery ✓
- Footer cell with name, email, program ✓

## Architecture

```
User Input (typed text or voice → gpt-4o-mini-transcribe)
  → LLM Query Rewriter (understand_query: resolve follow-ups, check topic)
  → Regex Intent Detection (detect_intent) → Agent Routing
  → Regex Constraint Extraction (extract_query_constraints) + Fuzzy Name Matching
  → FAISS Retrieval (k=30) → Reranking → LLM Generation (gpt-4o-mini)
  → Formatted Response → Gradio Blocks UI
```

### Key Components

| Component | Tech | Notes |
|-----------|------|-------|
| Embeddings | OpenAI `text-embedding-3-small` | 1536-dim vectors |
| Vector store | FAISS | In-memory, ~2762 movies after dedup |
| LLM | OpenAI `gpt-4o-mini` (temp=0.2) | Cost-effective, consistent |
| Framework | LangChain | For notebook submission |
| UI | Gradio `Blocks` | Portfolio-ready demo |

### Multi-Agent Design

Five agents routed by intent detection:
- **search** — general movie discovery via FAISS semantic search + reranking + LLM (default)
- **recommendation** — taste/preference-based via FAISS + preference-weighted reranking + LLM
- **catalog** — exhaustive filtered lists via DataFrame filter (bypasses FAISS, no LLM), sorted by IMDb rating desc
- **clarification** — handles ambiguous or too-short queries (< 3 chars), static response
- **fallback** — handles off-topic queries (greetings, non-movie topics), static response

## Key Files

| File | Purpose |
|------|---------|
| `wei-wong.ipynb` | Working copy — Wei's own work |
| `Wei_Wong_IMDb_Chatbot.ipynb` | **Submitted to grader** — Colab-only copy (no local paths) |
| `imdb-notebook.ipynb` | Reference template (Codex 5.3 generated, 36 cells, 50 functions) |
| `imdb_langgraph_app.py` | Standalone LangGraph refactor (~950 lines, bonus) |
| `imdb_dataset.csv` | Source dataset (3173 raw rows → 2762 after dedup) |
| `case_study_instructions.md` | Assignment brief |
| `scoring_rubric.json` | 8 grading dimensions |
| `NOTES.md` | Wei's learning notes (blog-style) |
| `PLAN.md` | Implementation plan with design decisions |
| `.env` | OpenAI API key (DO NOT commit) |

## Grading Rubric (8 Dimensions)

1. **LLM Integration & Prompt Engineering** — Seamless API integration; well-crafted prompts
2. **Retrieval & Search Efficiency** — FAISS/vector search; title-based and descriptive queries
3. **Conversational Flow & Query Handling** — Multi-turn, follow-ups, context retention
4. **Movie Data Representation & Formatting** — Structured, user-friendly movie details
5. **Handling of Edge Cases & Error Responses** — Graceful handling of invalid/unsupported inputs
6. **User Interface & Deployment** — Intuitive Gradio chatbot interface
7. **Code Structure & Documentation** — Clean, modular, well-commented
8. **Creativity & Feature Enhancement** — Multi-agent workflows, advanced features

## Dataset Quick Facts (after dedup)

- 2762 unique movies, 2762 unique movie_ids
- 10 original columns: Title, IMDb Rating, Year, Certificates, Genre, Director, Star Cast, MetaScore, Poster-src, Duration (minutes)
- Star Cast has NO delimiters — names are concatenated (e.g., "LeonardoDiCaprioKateWinslet")
- 4 remaining title-year collisions (Sacrifice 2010 × 2, The Message 1976 × 2) — kept intentionally as different films
- No nulls in any column

## Reference Notebook Structure (imdb-notebook.ipynb)

When building wei-wong.ipynb, reference these cells for working code:
- **Cell 7:** Cast parsing functions (`normalize_text`, `parse_star_cast`, `_should_merge_cast_parts`)
- **Cell 10:** EDA 6-panel visualization + `split_cast_heuristic()`
- **Cell 12:** `build_search_text()`
- **Cell 16:** `row_to_document()` for LangChain Document objects
- **Cell 17-20:** OpenAI embeddings, FAISS init, ChatOpenAI, prompt template
- **Cell 21:** `history_to_text()`, `docs_to_context()`, `generate_answer()`
- **Cell 22:** 15 retrieval/reranking functions (constraints, preferences, filtering)
- **Cell 23:** `rewrite_query_for_retrieval()`, `run_chatbot_query()`, `build_catalog_response()`
- **Cell 24:** Formatting functions
- **Cell 30:** 4 agent functions
- **Cell 31:** `detect_intent()`, `orchestrate_agents()`
- **Cell 32:** Guardrails (`is_probably_movie_related`, `safe_chatbot`, `evaluate_constraint_compliance`)
- **Cell 33:** Gradio UI
- **Cell 36:** Evaluation harness

## Working Conventions

- Python environment: `.venv/` — do not commit
- API key loaded from `.env` via `python-dotenv` (fallback: Colab Secrets)
- Notebook is the graded submission — keep it well-commented with markdown explanations
- Do NOT post this notebook publicly (course policy)
- Submission filename: `Wei_Wong_IMDb_Chatbot.ipynb`

## Commands

```bash
# Run the notebook
jupyter notebook wei-wong.ipynb

# Run the LangGraph app standalone (if wired up)
python imdb_langgraph_app.py
```
