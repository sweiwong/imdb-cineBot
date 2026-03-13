# PLAN.md — IMDb Movie Chatbot Implementation Plan

## Status: In Progress

Last updated: 2026-03-12

---

## Part 1: Data Understanding & Preparation — DONE

- [x] Load `imdb_dataset.csv` (~1000 movies)
- [x] Build data quality snapshot (nulls, duplicates, ranges)
- [x] Text normalization and cast parsing (fix concatenated actor names)
- [x] Create `movie_id` (year-aware, e.g., `dune_2021`)
- [x] Data integrity assertions (no duplicates, unique IDs, no nulls in search fields)
- [x] EDA: 6 visualizations (rating distribution, year trends, top genres, certificates, duration, directors)
- [x] Build `search_text` feature (concatenated field for retrieval)
- [x] Keyword filter spot-checks to validate `search_text`
- [x] Define data contract for Part 2 (`clean_df` with `movie_id` + `search_text`)

## Part 2: Embeddings & Vector Store — DONE

- [x] Convert DataFrame rows to LangChain `Document` objects with rich metadata
- [x] Skip chunking (search_text median ~212 chars, all under 350 — chunking is no-op or harmful)
- [x] Generate embeddings (OpenAI `text-embedding-3-small`, 1536 dims)
- [x] Build FAISS index (2,762 vectors)
- [x] Retrieval validation: 5 test queries (2 title-based, 3 descriptive)

## Part 3: Retrieval & Reranking — DONE

- [x] Semantic similarity retrieval (k=30 initial candidates)
- [x] Constraint extraction (duration, rating, year, decade, genre, certificate, actor)
- [x] Preference profile extraction (mood, era, titles, directors, actors)
- [x] Genre matching fix: split multi-genre strings into individual tokens for reliable matching
- [x] Filter hard constraints + rerank with soft preference scoring (72/28 hybrid formula)
- [x] Debug panel: raw semantic vs. constraint-aware reranked results (3 queries)
- [x] Query rewriting (append preference hints before embedding)
- [x] `run_retrieval_pipeline()` — full Part 3 orchestrator (no LLM)
- [x] **Experiment: Retrieval k** — compare k=5 vs k=15 vs k=30 vs k=50 on same queries

## Part 4: LLM Integration & Prompt Engineering — DONE

### Part 4a: Why RAG? (LLM-only vs RAG comparison)

- [x] Naked LLM vs RAG comparison using Ben Affleck queries (director vs actor role disambiguation)
- [x] Side-by-side analysis: hallucination, catalog specificity, role disambiguation, verifiability

### Part 4b: Core LLM Pipeline

- [x] Initialize ChatOpenAI (gpt-4o-mini, temp=0.2)
- [x] Design `ChatPromptTemplate` (system persona: movie concierge, 4-part output structure)
- [x] `history_to_text()` — compress chat history (last 6 turns)
- [x] `docs_to_context()` — format retrieved docs as numbered blocks with metadata
- [x] `generate_answer()` — orchestrate prompt formatting + LLM call, with empty-docs guard
- [x] Demo: 3 diverse queries (title lookup, preference-based, follow-up with simulated history)

### Part 4c: Parameter Tuning & Ablation

- [x] **Temperature comparison** — temp=0.0, 0.2, 0.5, 0.7 on action thriller query. Justified temp=0.2.
- [x] **Prompt variations** — minimal vs balanced vs verbose (CinemaBot 3000). Balanced wins.
- [x] **Context window size** — top-3 vs top-5 vs top-10 docs. Validated max_docs=5.

**Skipped (not worth the effort):**
- Model comparison (`gpt-4o-mini` vs `gpt-4o`) — cost difference is clear, quality delta marginal for this use case
- Embedding model comparison (`small` vs `large`) — requires re-embedding entire dataset twice
- search_text format comparison (labeled vs unlabeled) — same re-embedding issue; markdown explanation is enough

## Part 5: Multi-Agent Orchestration — NOT STARTED

- [ ] Intent detection (search / recommendation / catalog)
- [ ] Four specialized agents with different retrieval and generation behaviors
- [ ] Dynamic routing via `orchestrate_agents()`

## Part 6: Guardrails & Safety — NOT STARTED

- [ ] Movie-relatedness heuristic check
- [ ] Constraint compliance evaluation
- [ ] `safe_chatbot()` wrapper with error handling and timeouts

## Part 7: UI & Evaluation — NOT STARTED

### Part 7a: Core UI
- [ ] Gradio `ChatInterface` with example queries and custom styling
- [ ] Movie poster images in responses (use `poster_src` URLs from metadata)
- [ ] Evaluation harness with 6+ test cases (title lookup, constraints, follow-ups, edge cases)
- [ ] Automated scorecard (title hits, constraint compliance, follow-up quality)

### Part 7b: Multimodal Features *(Rubric: Creativity & Feature Enhancement)*

These map directly to the rubric language: *"voice-based search, multimodal input (text + images/video trailers)"*

- [ ] **Speech-to-text input** — Gradio `Audio` component + OpenAI Whisper API. User speaks a query, it gets transcribed, then fed into the same pipeline. ~15 lines. *(Rubric: Creativity)*
- [ ] **Image upload → visual search** — Gradio `Image` upload + GPT-4o vision. User uploads a movie poster/screenshot → GPT-4o describes it → description becomes the search query → FAISS retrieves matches. ~25 lines. *(Rubric: Creativity, multimodal input)*
- [ ] **Text-to-speech output** (optional) — OpenAI TTS API reads back the chatbot response. ~10 lines. Lower priority.

**Skip:**
- Video trailer lookup — requires YouTube Data API, adds key management complexity, low rubric value for the effort

## Part 8: LangGraph Refactor (Bonus) — NOT STARTED

- [ ] Extract notebook logic into `IMDbLangGraphApp` class (~950 lines)
- [ ] Implement as proper LangGraph `StateGraph` with typed state
- [ ] Node-based architecture: prepare → route → agent → finalize

---

## Open Items / Polish

- [ ] Final review of notebook markdown explanations for grading clarity
- [ ] Test evaluation harness end-to-end with live API
- [ ] Decide: wire LangGraph app into Gradio UI or keep as separate demo
- [ ] Add submission header cell (Full Name, Uplevel Email, Problem Statement)
- [ ] Rename notebook for submission (`Wei_Wong_IMDb_Chatbot.ipynb`)

---

## Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Embedding model | `text-embedding-3-small` | Fast, cheap, good enough for ~1000 movies |
| LLM | `gpt-4o-mini` | Cost-effective for chatbot; temp=0.2 for consistency |
| Vector store | FAISS (in-memory) | No infra needed; dataset fits in memory |
| Chunk size | 350 tokens / 40 overlap | Movies are short documents; small chunks preserve specificity |
| Initial retrieval k | 30 | Over-fetch then rerank — better precision after constraint filtering |
| Framework | LangChain (notebook) + LangGraph (refactor) | LangChain for rapid prototyping; LangGraph shows production thinking |
| UI | Gradio | Fastest path to demo-able chatbot; integrates with notebooks |
