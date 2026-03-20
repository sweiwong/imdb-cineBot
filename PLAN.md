# PLAN.md — IMDb Movie Chatbot Implementation Plan

## Status: In Progress

Last updated: 2026-03-20

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

## Part 5: Multi-Agent Orchestration — DONE

- [x] `run_chatbot_query()` — unified wrapper combining retrieval pipeline + LLM generation
- [x] Five specialized agents: search, recommendation, catalog, clarification, fallback
- [x] `fallback_agent()` — static off-topic handler (no retrieval/LLM cost); addresses Edge Cases rubric
- [x] `detect_intent()` — rule-based keyword/regex classifier (catalog, recommendation, fallback, search)
- [x] Greeting regex with `{0,2}` trailing words — catches "hey there!" but not "hey recommend a thriller"
- [x] `orchestrate_agents()` — intent-first routing (fixed: greetings like "hi" checked before length guard)
- [x] Demo: 7 queries across all 5 agents

**Future enhancement (noted, not built):**
- Fan-faves vs critically-acclaimed reranking (IMDb Rating vs MetaScore) — better as a Part 3 reranking enhancement

## Part 6: Guardrails & Safety — DONE

- [x] Defense-in-depth explanation (fallback_agent blocklist vs is_probably_movie_related allowlist)
- [x] `is_probably_movie_related()` — 5-signal topic classifier (keywords, genres, certs, years, entities)
- [x] `evaluate_constraint_compliance()` — post-hoc audit reusing hard-constraint extractors
- [x] `safe_chatbot()` — 4-layer entry point (input validation → topic filter → agent pipeline → exception wrapper)
- [x] `_gradio_history_to_tuples()` — boundary converter: Gradio dict-format → tuple pairs for downstream pipeline
- [x] `safe_chatbot()` type hint updated to `list[dict] | list[tuple[str, str]] | None`
- [x] Stress test: 9 queries covering all status paths (invalid_input, off_topic, ok + compliance, fallback)
- [x] Multi-turn stress test: Gradio dict-format history + follow-up with preference profile assertion

## Part 7: UI & Evaluation — DONE

### Part 7a: Core UI
- [x] Gradio `ChatInterface` with 4 example queries, placeholder text, custom CSS
- [x] Movie poster thumbnails in responses (80px, floated left via CSS)
- [x] Response formatting: `format_movie_matches()`, `format_catalog_matches()`, `format_final_response()`
- [x] LLM prompt updated: conversational summary only (3-5 sentences), no movie lists — UI handles structured cards
- [x] Catalog agent rewritten: bypasses FAISS, filters `clean_df` directly, sorts by IMDb rating desc, no LLM call
- [x] Genre plural normalization: "documentaries" → "Documentary", "comedies" → "Comedy", etc.
- [x] Evaluation harness: 15 test cases (title lookup, constraints, follow-up, typo, edge, off-topic, no-match, surname matching, mood recommendation, catalog constraints, excluded titles)
- [x] KPI summary: retrieval usefulness, first-answer success, constraint compliance, follow-up resolution, fallback rate, latency p50/p95
- [x] Part 7 summary + full project summary across all 7 parts

### Part 7b: Multimodal Features *(Rubric: Creativity & Feature Enhancement)*

These map directly to the rubric language: *"voice-based search, multimodal input (text + images/video trailers)"*

**Deferred:** Multi-agent architecture + guardrails already satisfy Creativity rubric. Can add after core submission if time allows.

- [ ] **Speech-to-text input** — Gradio `Audio` component + OpenAI Whisper API. ~15 lines.
- [ ] **Image upload → visual search** — Gradio `Image` upload + GPT-4o vision. ~25 lines.
- [ ] **Text-to-speech output** (optional) — OpenAI TTS API. ~10 lines. Lowest priority.

## Part 8: Hybrid Query Understanding — DONE

**Architecture pivot:** Originally planned to replace all regex with a single LLM call. After testing, pivoted to a hybrid approach: LLM handles query rewriting (follow-up resolution, topic relevance), regex stays as source of truth for intent detection and constraint extraction. This reduces regression risk from prompt drift while getting LLM benefits where they matter most.

**Motivation:** Testing revealed follow-up queries failing (no chat history context in regex) and natural language synonyms missed ("funny" ≠ Comedy). But full LLM replacement introduced its own regressions — LLM intent/constraint extraction was less predictable than regex for well-structured queries.

### Phase 1: LLM orchestration (initial)
- [x] `understand_query()` — single LLM call returning structured JSON: resolved_query, intent, is_movie_related, constraints
- [x] Wire into pipeline: `safe_chatbot()`, `orchestrate_agents()`, `catalog_agent()`, `run_retrieval_pipeline()` consume LLM output
- [x] Title injection: exact title matches from resolved query supplement FAISS results
- [x] Placeholder data penalty (-0.15) for MetaScore=66.0 + Duration=116.3 (~815 rows)
- [x] Mentioned-title exclusion in recommendation mode
- [x] Broken poster fallback: `onerror` handler hides broken CDN images

### Phase 2: Hybrid pivot (current)
- [x] Simplify `understand_query()` to query rewriter only (resolved_query + is_movie_related)
- [x] Restore regex `detect_intent()` as primary intent classifier
- [x] Restore regex `extract_query_constraints()` as primary constraint extractor
- [x] Fuzzy name matching: `_match_person_name()` — 3-pass (exact → surname alias → typo recovery, difflib cutoff 0.84)
- [x] `_build_unique_last_name_lookup()` — surname-to-full-name aliases (1,594 directors, 3,849 actors)
- [x] `extract_query_constraints()` rewritten to use `_match_person_name()` for director/actor
- [x] `detect_intent()` enhanced: mood-first recommendation routing ("something suspenseful tonight")
- [x] `extract_preference_profile()` — fallback to `_match_person_name()`, added "suspenseful" mood
- [x] IMDb 7.0 floor for recommendation mode (raised from 6.0)
- [x] Evaluation harness expanded: 9 → 15 test cases (surname matching, mood recommendation, catalog constraints, excluded titles)
- [x] Bug fix: "R-rated" no longer false-matches actor "R."

**Detailed plan:** `docs/superpowers/plans/2026-03-19-llm-orchestration-rebuild.md`

**What stays:** FAISS, reranking, generate_answer(), formatting, Gradio UI, 5 agent design
**LLM role:** Query rewriting (follow-up resolution) + topic relevance check only
**Regex role:** Intent detection, constraint extraction, preference profiling (primary)
**What's new:** Fuzzy name matching infrastructure (surname aliases + typo recovery)

## Part 9: LangGraph Refactor (Bonus) — NOT STARTED

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
| Chunk size | No chunking (median search_text ~212 chars) | Movies are single documents; chunking is no-op or harmful |
| Catalog retrieval | DataFrame filter (not FAISS) | "Give me all" needs every match, not top-k semantic hits |
| Initial retrieval k | 30 | Over-fetch then rerank — better precision after constraint filtering |
| Framework | LangChain (notebook) + LangGraph (refactor) | LangChain for rapid prototyping; LangGraph shows production thinking |
| UI | Gradio | Fastest path to demo-able chatbot; integrates with notebooks |
