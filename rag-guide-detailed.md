# The RAG Guide for the IMDb Movie Chatbot

**Personal reference — not for submission. Concepts mapped to the IMDb chatbot architecture.**

---

## What Is RAG?

Retrieval-Augmented Generation (RAG) is a pattern where you **retrieve** relevant information from a knowledge base, then **augment** a prompt to a language model with that information, so the model can **generate** an informed answer.

Without RAG, an LLM can only answer from its training data. With RAG, the LLM answers from *your* data — in our case, 2,762 movies from the IMDb dataset.

The pipeline looks like this:

```
User query
   → Embed query into a vector
   → Search FAISS index for similar movie vectors
   → Retrieve top-k movie records
   → Build a prompt: "Given these movies, answer the user's question..."
   → Send to GPT
   → Return response to user
```

Every decision in this pipeline — how you build the vectors, how you search, how you assemble the prompt — affects the quality of the final answer.

---

## Why Naive RAG Fails

"Naive RAG" means: take your documents, embed them, search by cosine similarity, stuff results into a prompt, done. It's the default tutorial approach, and it breaks in predictable ways.

### Failure Mode 1: Embeddings Can't Do Precise Filtering

A user asks: *"Show me action movies from 2020 with a rating above 8.0"*

This query has three **hard constraints**: genre = Action, year = 2020, rating > 8.0. Embeddings encode *meaning*, not *logic*. The embedding for "rating above 8.0" won't reliably exclude a 7.9-rated movie — the vectors are too close in semantic space. You'll get results that are *about* action movies and *about* high ratings, but there's no guarantee they satisfy all three filters.

**The fix:** Use structured metadata filters to handle the precise constraints (genre, year, rating range), then use embeddings only for the fuzzy/semantic part. This is exactly the hybrid architecture our project uses.

### Failure Mode 2: Role Ambiguity (The Ben Affleck Problem)

Consider *Argo* (2012). Ben Affleck **directed** it and **starred** in it. In our dataset, this shows up as:

- `Director`: Ben Affleck
- `Star Cast`: Ben AffleckBryan CranstonJohn Goodman

If you naively concatenate all fields into a single string without labels:

```
Argo 2012 Biography Ben Affleck Ben Affleck Bryan Cranston John Goodman 7.7
```

Now a user asks: *"movies directed by Ben Affleck"*

The embedding model sees "Ben Affleck" in the text but has **no idea whether he directed or acted**. It would score this movie the same as any film where Affleck just acted (Good Will Hunting, The Last Duel, Dogma — there are 9 such films in our dataset). The queries "directed by Affleck" and "starring Affleck" would return nearly identical results, because the underlying text is the same.

With **labeled format** (what our `search_text` uses):

```
Title: Argo. Year: 2012. Genre: Biography. Director: Ben Affleck.
Star Cast: Ben Affleck, Bryan Cranston, John Goodman. IMDb Rating: 7.7.
```

Now the embedding model encodes Affleck in **two distinct semantic contexts**: `Director:` and `Star Cast:`. The query "directed by Affleck" will have higher cosine similarity to the `Director: Ben Affleck` portion of the embedding than to the cast mention. It's not perfect — embeddings are still fuzzy — but it's measurably better than the naive approach.

**Why this matters across the dataset:**

| Query | Without labels | With labels |
|-------|---------------|-------------|
| "directed by Ben Affleck" | Returns all 9 Affleck films equally | Boosts Argo (director match) |
| "starring Ben Affleck" | Returns all 9 Affleck films equally | Correctly returns all 9 (cast match) |
| "highly rated sci-fi" | "8.8" has no context — could be duration | `IMDb Rating: 8.8` anchors the number's meaning |
| "long movies" | No way to connect query to duration | `Duration (minutes): 148` gives semantic signal |

### Failure Mode 3: No Conversation Memory

Naive RAG treats every query independently. A user says:

1. "Recommend a Christopher Nolan film" → returns Inception
2. "What about something longer?" → has no idea what "something" refers to

**The fix:** Maintain conversation history and pass it as context to both the retrieval step (so the search knows what "something" means) and the LLM step (so it can reference prior answers). This is what our conversational flow layer will handle.

---

## How Embeddings Work (Intuition, Not Math)

An embedding model converts text into a vector — a list of numbers (e.g., 1,536 numbers for `text-embedding-3-small`). These numbers encode the *meaning* of the text in a high-dimensional space.

Key intuitions:

- **Similar meanings → nearby vectors.** "Action-packed thriller" and "exciting suspense movie" will have vectors that are close together (high cosine similarity).
- **Different meanings → distant vectors.** "Action-packed thriller" and "slow-paced documentary about flowers" will be far apart.
- **Context matters.** The word "bank" in "river bank" and "bank account" will embed differently because the surrounding words shift the meaning.
- **Labels create context.** `Director: Christopher Nolan` embeds differently from `Star Cast: Christopher Nolan` because the label word changes the semantic neighborhood.

### What Embeddings Are Good At

- Fuzzy matching: "something like Inception but scarier" → finds Sci-Fi thrillers
- Synonym handling: "funny movies" matches genres tagged as "Comedy"
- Conceptual similarity: "feel-good movie for a rainy day" → finds uplifting dramas even if no movie literally mentions rain

### What Embeddings Are Bad At

- Exact matches: searching "Inception" might not rank the actual movie Inception #1
- Numerical comparisons: "rated above 8.0" has no reliable semantic boundary
- Negation: "not a horror movie" often retrieves horror movies (the word "horror" dominates the embedding)
- Boolean logic: "action AND comedy but NOT romance" can't be decomposed by vectors

This is why a hybrid approach — metadata filters for the structured stuff, embeddings for the fuzzy stuff — is the expert-level architecture.

---

## Metadata Filtering (Pre-Filtering)

Before running vector similarity search, narrow the candidate set using structured metadata. This is the most impactful upgrade over naive RAG for a structured dataset like ours.

### How It Works in Our Project

```
User: "Best action movies from the 2010s"

Step 1 — Extract constraints:
  genre = "Action"
  year_range = 2010–2019

Step 2 — Pre-filter clean_df:
  candidates = clean_df[
      (clean_df["Genre"].str.contains("Action")) &
      (clean_df["Year"].between(2010, 2019))
  ]
  → Reduces 2,762 movies to ~50 candidates

Step 3 — Embed query + search FAISS only within candidates:
  → Returns top-5 from those ~50

Step 4 — LLM generates response using those 5 movies as context
```

### Constraint Extraction

The constraint extractor takes a natural language query and pulls out structured filters. For our project, there are a few approaches (from simple to advanced):

**Regex-based (simplest):** Pattern match for years (`\b(19|20)\d{2}\b`), known genres, rating phrases ("above 8", "highly rated").

**LLM-based (more robust):** Ask GPT to parse the query into structured fields before searching. E.g., prompt: "Extract any genre, year, rating, director, or actor constraints from this query. Return JSON."

**Hybrid:** Use regex for obvious patterns (years, numbers), fall back to LLM for ambiguous cases ("old movies" → year < 1980? 1990? Let the LLM decide).

### Filterable Fields in Our Dataset

| Field | Filter type | Example query trigger |
|-------|------------|----------------------|
| Genre | Exact/contains match | "action movies", "documentaries" |
| Year | Range | "from the 90s", "recent", "2020" |
| IMDb Rating | Threshold | "highly rated", "above 8" |
| MetaScore | Threshold | "critically acclaimed" |
| Certificates | Exact match | "family-friendly", "R-rated" |
| Duration | Range | "short films", "long movies" |
| Director | Exact match | "directed by Nolan" |
| Star Cast | Contains (parsed) | "starring DiCaprio" |

---

## Hybrid Search

Hybrid search combines **keyword/lexical search** (exact term matching) with **semantic/vector search** (meaning-based matching). Each has strengths the other lacks.

| Search type | Good at | Bad at |
|------------|---------|--------|
| Keyword (BM25) | Exact titles, names, specific terms | Fuzzy queries, synonyms, conceptual similarity |
| Semantic (FAISS) | "Movies like Inception", vague intent | Exact title lookup, precise names |

### When Each Wins in Our Chatbot

- User types **"Inception"** → keyword search instantly finds the exact title. Semantic search might rank it #1, but also might not if another movie's description happens to be semantically closer.
- User types **"mind-bending sci-fi about dreams"** → semantic search nails this. Keyword search would fail because no movie literally contains the phrase "mind-bending sci-fi about dreams."
- User types **"DiCaprio Scorsese"** → keyword search finds all movies where both names appear. Semantic search might drift toward thematically similar films that don't actually involve either person.

### Implementation Options

**Simple approach (good for our project scope):**
1. Try exact title match first (fast, deterministic)
2. If no exact match, fall through to FAISS semantic search
3. Apply metadata filters in either path

**Advanced approach (if targeting Excellent on Creativity):**
1. Run both keyword and semantic search in parallel
2. Merge results with weighted scoring (e.g., 0.6 × semantic score + 0.4 × keyword score)
3. De-duplicate and rank

---

## Re-Ranking

After retrieving the initial top-k results from FAISS, a re-ranking step re-scores them using a different (usually more expensive) method. The idea: the first retrieval casts a wide net quickly, and re-ranking picks the best fish.

### Why Bother?

FAISS uses bi-encoder similarity: the query and each document are embedded *independently*, then compared. This is fast but shallow — it can't model fine-grained interactions between the query and document.

A re-ranker (cross-encoder) takes the query and a candidate document *together* as input and scores them jointly. This captures deeper relevance signals but is too slow to run on all 2,762 movies — hence the two-stage approach.

### Practical Re-Ranking for Our Project

You don't need a cross-encoder model to get re-ranking benefits. Simple heuristic re-ranking can work well:

```python
def rerank(query, candidates):
    for movie in candidates:
        score = movie["faiss_score"]  # base semantic score

        # Boost exact title matches
        if query.lower() in movie["Title"].lower():
            score += 0.3

        # Boost if query mentions a genre that matches
        if detected_genre and detected_genre in movie["Genre"]:
            score += 0.1

        # Boost higher-rated movies for vague queries
        if is_vague_query:
            score += movie["IMDb Rating"] / 100

        movie["final_score"] = score

    return sorted(candidates, key=lambda x: x["final_score"], reverse=True)
```

This is lightweight, interpretable, and directly tied to product quality — users get better results without a heavy ML model.

---

## Query Rewriting

Query rewriting uses the LLM to reformulate a user's vague or conversational query into something more searchable before retrieval.

### Examples

| User says | Rewritten query |
|-----------|----------------|
| "something like Inception but scarier" | "Sci-Fi Thriller psychological mind-bending suspense" |
| "a good date night movie" | "Romance Comedy Drama highly rated feel-good" |
| "that movie where the boat sinks" | "Titanic" |
| "DiCaprio's best" | "Leonardo DiCaprio highest rated" |

### How to Implement

Send the user's raw query to GPT with a prompt like:

```
You are a movie search assistant. Rewrite the user's query to
maximize retrieval quality. Extract any specific filters (genre,
year, rating, director, actor) as structured fields. Expand vague
intent into descriptive keywords.

User query: "something like Inception but scarier"

Return JSON:
{
  "rewritten_query": "psychological sci-fi thriller suspense mind-bending",
  "filters": {"genre": "Sci-Fi, Thriller"},
  "similar_to": "Inception"
}
```

This gives you both the semantic search query AND the metadata filters in one LLM call. The rewritten query goes to FAISS; the filters go to the pre-filter step.

### Conversation-Aware Rewriting

For follow-up queries, include conversation history:

```
Previous turn: User asked about Christopher Nolan films. You recommended Inception.
Current query: "What about something longer?"

Rewritten: "Christopher Nolan films longer than 148 minutes"
Filters: {"director": "Christopher Nolan", "duration_min": 148}
```

This is where multi-turn conversation support and query rewriting intersect — and it maps to the **Conversational Flow & Query Handling** rubric criterion.

---

## Agentic RAG / Multi-Agent Workflows

Standard RAG is a single pipeline: retrieve → generate. Agentic RAG adds a **decision layer** where the LLM acts as an orchestrator, choosing which tools or retrieval strategies to use based on the query.

### Why This Matters for the Rubric

The **Creativity & Feature Enhancement** criterion specifically mentions "multi-agent workflows (e.g., separate agents for movie search, review analysis, and genre-based recommendations)." This is the clearest path to an Excellent rating.

### Agent Architecture for Our Chatbot

Instead of one monolithic pipeline, split responsibilities:

```
User query → Router Agent (decides which specialist to call)
  ├─ Search Agent:          "find me movies about X"      → FAISS semantic search + reranking + LLM
  ├─ Recommendation Agent:  "suggest based on my taste"   → FAISS + preference-weighted reranking + LLM
  ├─ Catalog Agent:         "give me all R-rated horror"  → DataFrame filter (no FAISS, no LLM)
  ├─ Clarification Agent:   "..." (too short/vague)       → static prompt for more detail
  └─ Fallback Agent:        greetings, off-topic          → static redirect (no API calls)
```

Each agent has its own retrieval strategy. The key insight: **catalog bypasses FAISS entirely** because "give me all" queries need exhaustive structured filtering, not top-k semantic similarity. The catalog agent filters `clean_df` directly using the same constraint extractor, then sorts by IMDb rating descending. No LLM call — the results speak for themselves.

### Practical Implementation

You don't need a framework like LangGraph or CrewAI. A simple function-based approach works:

```python
def route_query(query, history):
    """Use GPT to classify the query intent."""
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": "Classify the user's movie query into one of: "
                       "SEARCH, COMPARE, RECOMMEND, DETAIL, QUIZ. "
                       "Return only the category."
        }, {
            "role": "user",
            "content": query
        }]
    )
    intent = response.choices[0].message.content.strip()
    return intent

# Then dispatch:
if intent == "SEARCH":
    return search_agent(query, filters)
elif intent == "COMPARE":
    return compare_agent(query, movie_a, movie_b)
# ... etc.
```

### The Critic vs. Crowd Feature (MetaScore vs IMDb Rating)

This is a natural fit for the Recommendation Agent. When a user asks for "good movies," the agent can ask:

> "Good according to audiences or critics? Our dataset shows they disagree on about 15% of films — for example, *Patch Adams* has a 6.8 from audiences but only 26 from critics, while *My Left Foot* scores 7.8 from audiences but 97 from critics."

This turns a simple retrieval into a product experience — exactly what the rubric is looking for.

---

## Chunking: Why It Matters Less for Our Project

Most RAG guides spend a lot of time on chunking — splitting long documents into smaller pieces so each chunk fits in a context window and embeds coherently. Strategies include:

- **Fixed-size chunking:** Split every N tokens
- **By title/heading:** Split at section boundaries
- **Semantic chunking:** Split where the topic shifts
- **Parent-document retrieval:** Store small chunks for search but retrieve the full parent document

**For our project, chunking is mostly irrelevant.** Each movie is already a single, self-contained record. Our `search_text` strings are ~150-200 characters — well within embedding limits. There's nothing to chunk.

Where chunking *would* matter: if we were ingesting full movie reviews, plot synopses from Wikipedia, or transcript data. That's a future enhancement, not a current concern.

---

## Putting It All Together: Our Architecture

```
User query
   │
   ▼
┌──────────────┐
│ Router Agent │  ← Classifies intent (search/compare/recommend/detail/quiz)
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Query Rewriter   │  ← Expands vague queries, extracts structured filters
│ (LLM call)       │     Input: raw query + conversation history
│                  │     Output: rewritten_query + filters JSON
└──────┬───────────┘
       │
       ├─── filters ──────────────┐
       │                          ▼
       │                  ┌──────────────────┐
       │                  │ Metadata Filter   │  ← Pre-filters clean_df
       │                  │ (Pandas boolean)  │     by genre, year, rating, etc.
       │                  └──────┬───────────┘
       │                         │
       │                         ▼ candidate movie_ids
       ▼                         │
┌──────────────┐                 │
│ Embed query  │                 │
│ (OpenAI API) │                 │
└──────┬───────┘                 │
       │                         │
       ▼                         │
┌──────────────┐                 │
│ FAISS search │  ← Search only within filtered candidate set
│ (top-k)      │◄────────────────┘
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Re-ranker    │  ← Heuristic boost (title match, rating, genre alignment)
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Prompt Assembly  │  ← System prompt + retrieved movies + conversation history
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│ GPT Response │  ← Final answer to user
└──────┬───────┘
       │
       ▼
   Chatbot UI (Gradio/Streamlit)
```

### How This Maps to the Rubric

| Rubric Criterion | Where It's Addressed |
|-----------------|---------------------|
| LLM Integration & Prompt Engineering | Query Rewriter + Prompt Assembly + GPT Response |
| Retrieval & Search Efficiency | Metadata Filter + FAISS + Re-ranker |
| Conversational Flow & Query Handling | Router Agent + conversation history in Query Rewriter |
| Movie Data Representation & Formatting | Prompt Assembly (structured movie cards in response) |
| Handling of Edge Cases & Error Responses | Router Agent (fallback paths) + metadata filter (empty result handling) |
| User Interface & Deployment | Gradio/Streamlit layer |
| Code Structure & Documentation | Modular agent functions + clear notebook sections |
| Creativity & Feature Enhancement | Multi-agent routing + Critic vs Crowd + Quiz Agent |

---

## Key Takeaways

1. **Naive RAG fails on structured queries.** Metadata filtering is the fix, not a nice-to-have.
2. **Labeled `search_text` > raw concatenation.** Field labels give embeddings role-level context (the Ben Affleck problem).
3. **Hybrid search covers both precise and fuzzy queries.** Keyword for exact titles/names, semantic for vague intent.
4. **Re-ranking is cheap and high-impact.** Even simple heuristic boosts improve result quality.
5. **Query rewriting is the bridge between conversational UX and search.** The LLM translates human intent into search-friendly form.
6. **Multi-agent routing is the creativity differentiator.** It's what separates a chatbot from a search box.
7. **Not every query needs vector search.** Catalog/filter queries should bypass FAISS and filter the DataFrame directly — "give me all" needs every match, not top-k semantic hits.
8. **Chunking doesn't apply to our structured dataset** — but understanding why is part of the learning.
