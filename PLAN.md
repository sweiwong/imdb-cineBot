# PLAN.md — IMDb Movie Chatbot Implementation Plan

## Status: In Progress

Last updated: 2026-03-08

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
- [x] Chunk documents (RecursiveCharacterTextSplitter: 350 tokens, 40 overlap)
- [x] Generate embeddings (OpenAI `text-embedding-3-small`)
- [x] Build FAISS index from chunked documents

## Part 3: Retrieval & Reranking — DONE

- [x] Semantic similarity retrieval (k=30 initial candidates)
- [x] Constraint extraction (duration, rating, year, genre, certificate, actor, director, title)
- [x] Preference profile extraction (mood, era, pacing)
- [x] Filter hard constraints + rerank with soft preference scoring
- [x] Debug panel: raw semantic vs. constraint-aware reranked results

## Part 4: LLM Integration & Prompt Engineering — DONE

- [x] Initialize ChatOpenAI (gpt-4o-mini, temp=0.2)
- [x] Design prompt template (system persona: movie concierge)
- [x] Response structure: Quick Take → Why These Fit → Movies → Follow-up Question
- [x] Chat history compression (last 6 turns)
- [x] Query rewriting for better retrieval

## Part 5: Multi-Agent Orchestration — DONE

- [x] Intent detection (search / recommendation / catalog)
- [x] Four specialized agents with different retrieval and generation behaviors
- [x] Dynamic routing via `orchestrate_agents()`

## Part 6: Guardrails & Safety — DONE

- [x] Movie-relatedness heuristic check
- [x] Constraint compliance evaluation
- [x] `safe_chatbot()` wrapper with error handling and timeouts

## Part 7: UI & Evaluation — DONE

- [x] Gradio `ChatInterface` with example queries and custom styling
- [x] Evaluation harness with 6+ test cases (title lookup, constraints, follow-ups, edge cases)
- [x] Automated scorecard (title hits, constraint compliance, follow-up quality)

## Part 8: LangGraph Refactor (Bonus) — DONE

- [x] Extract notebook logic into `IMDbLangGraphApp` class (~950 lines)
- [x] Implement as proper LangGraph `StateGraph` with typed state
- [x] Node-based architecture: prepare → route → agent → finalize

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
