# NOTES.md — What I Learned Building the IMDb Movie Chatbot

*Technical learnings from Case Study 2. Written as I build, blog-style.*

---

## The Big Picture

This project taught me how RAG (retrieval-augmented generation) actually works end-to-end — not as an abstract concept, but as a pipeline I had to wire together piece by piece: clean data → embeddings → vector search → reranking → LLM generation → UI.

The most surprising lesson: **the LLM is the easy part.** Most of the work is everything that happens *before* the model sees a prompt.

---

## Data Preparation: The Unglamorous Foundation

### Cast Parsing Was Harder Than Expected

The `Star Cast` column had concatenated actor names with no clear delimiter — names like `LeonardoDiCaprioKateWinslet`. I had to write regex-based parsing to split them back into individual names. This is the kind of messy real-world data work that tutorials skip over.

**Takeaway:** Always inspect your raw data cell by cell, not just `.describe()`. The schema tells you what *should* be there; eyeballing tells you what *actually* is.

### The `search_text` Design Decision

I concatenated Title, Year, Genre, Director, Cast, Rating, Certificate, and Duration into a single `search_text` field for embedding. This means the vector search can match on *any* of those dimensions from a single query.

**Why not embed each field separately?** For ~1000 movies, the simplicity of one embedding per movie outweighs the precision gain of multi-field search. Keep it simple until the data demands complexity.

---

## Embeddings & FAISS: How Semantic Search Actually Works

### What Embeddings Do (My Mental Model)

An embedding converts text into a point in high-dimensional space (1536 dimensions for `text-embedding-3-small`). Similar texts end up near each other. FAISS finds the nearest neighbors fast.

Think of it like this: if every movie is a dot on a giant map, the embedding decides *where* each dot goes, and FAISS tells you which dots are closest to your query.

### Chunking Strategy

I used 350 tokens with 40-token overlap. Since each movie's `search_text` is short (a few sentences), most movies are a single chunk. The chunking is there as future-proofing — if I added plot summaries, the pipeline wouldn't break.

---

## The Retrieval-Reranking Pattern

This was my biggest "aha" moment: **semantic search alone isn't enough for structured queries.**

If someone asks "action movies rated above 8.0 from the 2010s," pure vector search will find movies *semantically similar* to that phrase — but it won't reliably enforce the rating > 8.0 or year range constraints. Embeddings encode meaning, not math.

### The Two-Stage Pattern

1. **Stage 1 — Semantic retrieval (k=30):** Cast a wide net. Get 30 semantically relevant candidates.
2. **Stage 2 — Constraint extraction + reranking:** Parse the query for hard constraints (rating, year, duration, genre, certificate). Filter out movies that don't meet them. Rerank the rest by how well they match soft preferences (mood, era, pacing).

This pattern — over-fetch then filter — is how production RAG systems work. It's the same idea as a database query planner: get candidates fast, then refine.

### Constraint Extraction

Originally regex-based parsers pulling structured constraints from natural language:
- "under 2 hours" → `duration_max: 120`
- "rated above 8" → `min_rating: 8.0`
- "from the 90s" → `year_min: 1990, year_max: 1999`
- "PG-13" → `certificate: PG-13`

These worked for literal patterns but failed on semantic language ("funny" ≠ "Comedy" to a regex). Now handled by `understand_query()` — the LLM maps natural language to structured constraints. The old regex extractors remain as fallbacks if the LLM call fails.

---

## Multi-Agent Routing: When One Prompt Isn't Enough

### Why Multiple Agents?

Different queries need different retrieval strategies:
- **"Tell me about Inception"** → Direct lookup. Search by title.
- **"I want something like Interstellar but shorter"** → Recommendation. Semantic similarity + constraints.
- **"List all Christopher Nolan films"** → Catalog. Exhaustive filter, not top-k.
- **"I'm bored"** → Clarification. Ask follow-up questions.

One-size-fits-all retrieval would underserve at least two of these. Intent detection lets me route each query to the right strategy.

### Intent Detection: From Regex to LLM

I initially used keyword heuristics to detect intent. Fast, cheap, transparent. But testing exposed two fatal flaws:
1. **Natural language synonyms fail.** "Show me all Brad Pitt movies that are funny" — the regex couldn't map "funny" to Comedy genre because it only matched literal genre tokens.
2. **Follow-ups are context-blind.** "Which of these are funny?" after a Brad Pitt query was flagged as off-topic because the regex topic filter had no chat history context. It just saw 5 words with no movie keywords.

**The fix:** Replace three regex functions (`detect_intent`, `extract_query_constraints`, `is_probably_movie_related`) with a single `gpt-4o-mini` call (`understand_query()`) that sees the query + last 3 turns of chat history and returns structured JSON: intent, constraints, topic relevance, and a resolved query with pronouns/references expanded.

**Cost:** ~$0.001 and ~300ms per query. Worth it. A fast wrong answer is worse than a slightly slower right one.

**Lesson:** Regex-based NLP works for well-scoped keyword patterns. The moment you need semantic understanding or conversational context, you need an LLM. Don't fight the problem with more regexes.

---

## LangGraph Refactor: From Notebook to Production Shape

### Why Refactor?

The notebook works, but it's 1400+ lines of code spread across 24 cells with shared global state. That's fine for exploration but terrible for:
- Testing individual components
- Deploying as a service
- Debugging multi-step flows

### What LangGraph Adds

LangGraph models the chatbot as a **state machine**:
```
prepare_request → route → [search | recommendation | catalog | clarification] → finalize_response
```

Each node is a function that reads from and writes to a typed `GraphState` dictionary. The graph handles:
- State passing between nodes
- Conditional routing (which agent to use)
- Clear boundaries between components

**What I learned:** The jump from "working notebook" to "structured application" isn't about adding features — it's about making the *existing* logic explicit. Every implicit dependency (shared variables, execution order) becomes an explicit edge in the graph.

---

## Prompt Engineering: Subtler Than It Looks

### The Prompt Structure

The prompt evolved significantly. The initial version asked the LLM to output a 4-part structured response (Quick Take, Why These Fit, Movies, Follow-up). But once we added formatted movie cards in the UI (Part 7), the LLM was duplicating information — listing movies with ratings and runtimes that the UI already displayed in structured cards below.

**Final approach:** The LLM writes *only* a conversational summary (3-5 sentences) explaining what was found and why the picks fit. No headers, no numbered sections, no movie lists. The UI handles all structured data display. This separation of concerns — LLM for narrative, UI for data — makes the output cleaner and avoids redundancy.

### What I Learned About Prompts

- **Temperature 0.2** keeps responses focused. Higher temperatures gave creative but less reliable movie details.
- **Providing `search_text` as context** (not raw data) means the LLM doesn't hallucinate movie facts — it can only reference what's in the retrieved documents.
- **Chat history compression** (last 6 turns) prevents context overflow while maintaining conversational continuity.
- **Tell the LLM what the UI does.** If the interface renders structured data, the prompt must say "don't repeat this" or the LLM will duplicate it. The prompt and UI are co-designed.

---

## Guardrails: The Boring Stuff That Matters

### Movie-Relatedness Check

A simple heuristic that checks if the query contains movie-related terms before sending it to the LLM. This prevents the chatbot from becoming a general-purpose assistant.

### Constraint Compliance Evaluation

After generating results, I check: did the returned movies actually satisfy the user's constraints? This surfaces retrieval failures before they reach the user.

### Error Handling

The `safe_chatbot()` wrapper catches exceptions and returns graceful error messages instead of stack traces. Boring but essential for a demo.

---

## Evaluation: How Do You Know It Works?

### The Test Suite Approach

I built an evaluation harness with 9 test cases across categories:
- Title lookup accuracy (T1: Dune)
- Constraint satisfaction (T2-T4: action/rating, documentary/duration, multi-constraint)
- Follow-up context retention (T5: simulated chat history)
- Typo resilience (T6: "cristopher noln")
- Edge cases (T7: empty input, T9: impossible constraints → no_match)
- Off-topic handling (T8: banana bread recipe)

### The Hard Part About Evaluating RAG

You can't just check "is the answer correct?" because:
1. There may be multiple valid answers
2. The quality of the *explanation* matters, not just the movie list
3. Follow-up handling is subjective

My scorecard tracks concrete, measurable things: Did the right title appear? Were constraints satisfied? Did the system ask a reasonable follow-up? It's not perfect, but it's better than eyeballing.

### What "Useful Retrieval" Actually Means

The harness uses a binary hit metric — did at least one returned movie match the expected criteria? This is intentionally simple. A graded relevance score (NDCG, MRR) would be more rigorous, but requires manual relevance judgments for every query-movie pair. Binary hit is the right tradeoff for a case study: measurable, automatable, and honest about its limitations.

---

## Key Takeaways

1. **RAG is a pipeline, not a model.** The LLM is one component. Data cleaning, embedding strategy, retrieval, reranking, and formatting do most of the heavy lifting.

2. **Over-fetch then filter.** Retrieve more candidates than you need, then apply structured constraints. Pure semantic search can't enforce numerical or categorical conditions.

3. **Intent detection enables specialization.** Different query types need different strategies. Routing is cheap; wrong answers are expensive.

4. **Notebook → application is a mindset shift.** The code doesn't change much, but making state and control flow explicit transforms maintainability.

5. **Evaluation is harder than building.** Defining "good" for a conversational system requires concrete, measurable criteria — not vibes.

---

## Part 7: The UI Changes Everything

### Catalog Agent: FAISS Was the Wrong Tool

The biggest Part 7 learning: catalog queries ("give me all documentaries rated 8+") are fundamentally different from search/recommendation queries. They need *every* matching row, not the top-k most semantically similar.

FAISS caps at `candidate_k` vectors (30 in our case), so "give me all" could never return more than 30 results — and many valid matches were excluded because they weren't semantically closest. The fix was obvious in retrospect: **catalog bypasses FAISS entirely and filters the DataFrame directly** using the same constraint extractor from Part 3. Results sorted by IMDb rating descending. No LLM call needed.

This is a pattern worth remembering: not every query in a RAG system needs vector search. Structured queries deserve structured retrieval.

### Genre Plurals: A Dumb Bug That Broke Everything

"Give me all documentaries rated 8.0 and above" returned 277 results instead of 46. The rating constraint *was* being extracted correctly (`imdb_min=8.0`). The bug: the genre regex `\bdocumentary\b` doesn't match "documentaries" because the plural form extends past the word boundary.

Fix: two regex substitutions before genre matching — `ies` → `y` and trailing `s` → stripped. Simple, but it affected every plural genre query (comedies, thrillers, mysteries, biographies, musicals, fantasies, westerns).

### Separation of Concerns: LLM vs UI

The initial prompt asked the LLM to list movies with ratings and runtimes in a structured format. But the UI already renders movie cards with posters, metadata, and "why it fits" tags. The result was duplicate information — the LLM listing the same movies the UI was about to display.

The fix: tell the LLM explicitly that the UI handles structured data, and its job is *only* conversational — a 1-2 sentence summary and a follow-up question. This made the output much cleaner.

---

## Key Takeaways (Updated)

1. **RAG is a pipeline, not a model.** The LLM is one component. Data cleaning, embedding strategy, retrieval, reranking, and formatting do most of the heavy lifting.

2. **Over-fetch then filter.** Retrieve more candidates than you need, then apply structured constraints. Pure semantic search can't enforce numerical or categorical conditions.

3. **Intent detection enables specialization.** Different query types need different strategies. Routing is cheap; wrong answers are expensive.

4. **Not every query needs vector search.** Catalog/filter queries should hit the DataFrame directly. FAISS is for fuzzy semantic matching, not exhaustive structured lookups.

5. **The LLM and UI are co-designed.** If the UI renders structured data, the prompt must account for that or you get duplication. Separation of concerns applies to AI outputs too.

6. **Evaluation is harder than building.** Defining "good" for a conversational system requires concrete, measurable criteria — not vibes.

---

## Part 8: LLM-Powered Orchestration — The Biggest Lesson

### Regex is Brittle; LLMs Understand

This was the most important refactor in the project. After completing all 7 parts, I started testing the chatbot with real queries and every single follow-up failed. The regex-based orchestration layer couldn't handle:
- Semantic synonyms ("funny" → Comedy)
- Follow-up references ("these", "that one") — no chat history context
- Natural language intent that didn't match keyword patterns

Replacing three regex functions with one LLM call (`understand_query()`) fixed all of this. The LLM sees the query + recent chat history and returns structured JSON with intent, constraints, and a resolved query. Cost: ~$0.001/query, ~300ms latency. Every penny worth it.

### Data Quality Matters More Than You Think

~815 movies (~30% of the dataset) had placeholder MetaScore (66.0) and Duration (116.3) values — unscraped fields filled with defaults. These movies were polluting recommendation results because FAISS ranked them by semantic similarity regardless of data quality. A -0.15 reranking penalty for placeholder rows and a 6.0 IMDb floor for recommendations cleaned this up.

### Title Injection: When FAISS Isn't Enough

Follow-up queries like "tell me about these 4" get resolved to specific movie titles by the LLM. But FAISS embeds the multi-title query as a single vector, so it might miss some titles. Direct title lookup from `TITLE_LOOKUP` supplements FAISS results — if the resolved query mentions known titles, inject them at distance 0.0.

### Don't Recommend What They Already Know

"I loved Inception and The Dark Knight — what should I watch next?" was returning Inception and The Dark Knight in the results. Fixed by excluding `mentioned_titles` (already tracked by `extract_preference_profile`) from recommendation results.

---

*Last updated: 2026-03-19. All 8 parts complete.*
