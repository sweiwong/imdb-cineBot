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

## Part 4: LLM Integration & Prompt Engineering — NOT STARTED

### Part 4a: Why RAG? (LLM-only vs RAG comparison)

Opens Part 4 by demonstrating *why* the entire RAG architecture is needed. ~3 cells.

Uses Ben Affleck as the test case — he's both a director and actor in our dataset, which forces the system to distinguish roles. Two queries:
- **"What are the highest rated movies directed by Ben Affleck?"**
- **"What are the top 5 Ben Affleck movies?"**

| Step | What happens | Expected outcome |
|------|-------------|-----------------|
| Naked LLM (no RAG) | Ask GPT-4o-mini directly, no context from our dataset | Answers from internet knowledge — may hallucinate titles not in our catalog, can't verify director vs actor role, doesn't know our dataset boundaries |
| With RAG | Retrieve from FAISS first, pass results as context | Answers grounded in our actual 2,762 movies, with metadata to distinguish director vs star_cast |

**Teaching points this demonstrates:**
- Hallucination risk without grounding
- Catalog specificity — we want answers from *our* dataset, not all of IMDb
- Role disambiguation — semantic search returns all Ben Affleck movies, but metadata distinguishes director vs actor (tees up Part 3 reranking)

### Part 4c: Core LLM Pipeline

- [ ] Initialize ChatOpenAI (gpt-4o-mini, temp=0.2)
- [ ] Design prompt template (system persona: movie concierge)
- [ ] Response structure: Quick Take → Why These Fit → Movies → Follow-up Question
- [ ] Chat history compression (last 6 turns)
- [ ] Query rewriting for better retrieval

### Part 4b: Parameter Tuning & Ablation

Dedicated subsection at the end of Part 4 to show engineering rigor. ~3-4 cells.

**Must-include (high rubric impact):**
- [ ] **Temperature comparison** — Run 3-4 queries at temp=0.0, 0.2, 0.5, 0.7. Compare response quality and consistency. Shows creativity-vs-consistency tradeoff. *(Rubric: LLM Integration & Prompt Engineering)*
- [ ] **Prompt variations** — Test 2-3 system prompt versions (concise vs detailed persona vs structured output). Highest-leverage experiment for the rubric. *(Rubric: LLM Integration & Prompt Engineering)*
- [ ] **Retrieval k** — Compare k=5 vs k=15 vs k=30 vs k=50. Too few = miss relevant movies, too many = flood LLM with noise. *(Rubric: Retrieval & Search Efficiency)*

**Nice-to-have (impressive, moderate effort):**
- [ ] **Context window size** — Send top-3 vs top-5 vs top-10 docs to LLM. More context = more info but higher cost and distraction risk. *(Rubric: LLM Integration & Prompt Engineering)*
- [ ] **Model comparison** — `gpt-4o-mini` vs `gpt-4o` on same 3 queries. Compare quality, cost, latency. Deliberate model selection. *(Rubric: LLM Integration & Prompt Engineering, Creativity)*

**Skip (not worth the effort):**
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
