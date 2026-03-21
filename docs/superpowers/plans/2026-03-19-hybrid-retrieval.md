# Hybrid Retrieval Implementation Plan

**Goal:** Make constrained search/recommendation queries find ALL compliant movies by pre-filtering the DataFrame before FAISS scoring, eliminating the k=30 bottleneck.

**Architecture:** When `run_retrieval_pipeline()` detects hard constraints, it first filters `clean_df` to get compliant movie_ids, then queries FAISS with k=full-index, intersects by movie_id, and feeds compliant candidates into the existing reranker. Unconstrained queries use the existing FAISS k=30 path unchanged.

**Tech Stack:** pandas (DataFrame filtering), FAISS (semantic scoring), LangChain (Document objects), existing notebook infrastructure

**Spec:** `docs/superpowers/specs/2026-03-19-hybrid-retrieval-design.md`

---

## Status Update — 2026-03-21

This plan is now implemented in `wei-wong.ipynb`.

### Completed

- `_normalize_certificate()`, `_has_hard_constraints()`, `_doc_satisfies_hard_constraints()`, and `_filter_compliant_movie_ids()` are in place
- `run_retrieval_pipeline()` uses the constrained hybrid branch for hard-filtered queries and keeps the original FAISS `k=30` path for unconstrained queries
- `catalog_agent()` reuses `_filter_compliant_movie_ids()` so catalog and search modes apply the same hard-constraint logic

### Validation Notes

- Hard-constrained zero-hit queries now correctly return no matches when the dataset truly has none
- Example: `Best PG-13 sci-fi movies rated above 8.0` has zero exact matches in this catalog, so the retrieval logic is behaving correctly
- To make that case feel better in the UI, `safe_chatbot()` now suggests nearby relaxations instead of stopping at a generic no-match message

### What This Means

The hybrid retrieval work is effectively complete. The next highest-value
feature is the interactive quiz, while any further retrieval work should focus
on descriptive-query quality rather than constrained-query coverage.

---

## File Map

All changes are in `wei-wong.ipynb`:

| Cell (id) | What lives there | What changes |
| --- | --- | --- |
| Cell 54 (`7locvswa3z`) | `_has_hard_constraints()`, `_doc_satisfies_hard_constraints()`, reranking helpers | Add `_filter_compliant_movie_ids()`, revert `_has_hard_constraints()` so `prefer_high_rating` alone doesn't trigger constrained path, add certificate normalization to `_doc_satisfies_hard_constraints()` |
| Cell 57 (`2svnpeqc29o`) | `rewrite_query_for_retrieval()`, `run_retrieval_pipeline()` | Add constrained branch inside `run_retrieval_pipeline()` |
| Cell 82 (`qcdrb3z7lae`) | `catalog_agent()`, `movie_search_agent()`, `recommendation_agent()` | Refactor `catalog_agent()` to use shared `_filter_compliant_movie_ids()` |

No other cells change. `understand_query()`, `orchestrate_agents()`, `safe_chatbot()`, `generate_answer()`, the LLM system prompt, and Gradio UI are all untouched.

---

### Task 1: Add `_filter_compliant_movie_ids()` helper and align certificate normalization

**Cell:** 54 (`7locvswa3z`) — reranking helpers cell

Add the new helper function after `_doc_satisfies_hard_constraints()` and before `_semantic_similarity()`.

- [ ] **Step 1: Write `_filter_compliant_movie_ids()`**

Add this function to cell 54. It builds a boolean mask on `clean_df` and returns the set of compliant movie_ids. Uses lowercase `set` type hint (Python 3.9+, no import needed).

```python
def _normalize_certificate(cert: str) -> str:
    """Normalize certificate string: strip 'Rated ' prefix and '-rated' suffix."""
    cert = re.sub(r"(?i)^rated\s+", "", cert)
    cert = re.sub(r"(?i)-rated$", "", cert)
    return cert.strip().upper()


def _filter_compliant_movie_ids(constraints: Dict) -> set:
    """Filter clean_df by hard constraints, return set of compliant movie_ids.

    Used by both run_retrieval_pipeline() (hybrid path) and catalog_agent()
    to ensure consistent filtering logic across all retrieval modes.
    """
    mask = pd.Series(True, index=clean_df.index)

    if constraints.get("genre"):
        genre = constraints["genre"].lower()
        mask &= clean_df["Genre"].str.lower().str.contains(genre, na=False)

    if constraints.get("certificate"):
        cert = _normalize_certificate(constraints["certificate"])
        mask &= clean_df["Certificates"].str.upper() == cert

    if constraints.get("actor_name"):
        actor = constraints["actor_name"].lower()
        mask &= clean_df["parsed_star_cast"].apply(
            lambda names: any(actor == name.lower() for name in names)
        )

    if constraints.get("director_name"):
        director = constraints["director_name"].lower()
        mask &= clean_df["Director"].str.lower() == director

    if constraints.get("imdb_min") is not None:
        mask &= clean_df["IMDb Rating"] >= constraints["imdb_min"]
    if constraints.get("imdb_max") is not None:
        mask &= clean_df["IMDb Rating"] <= constraints["imdb_max"]
    if constraints.get("year_min") is not None:
        mask &= clean_df["Year"] >= constraints["year_min"]
    if constraints.get("year_max") is not None:
        mask &= clean_df["Year"] <= constraints["year_max"]
    if constraints.get("duration_min") is not None:
        mask &= clean_df["Duration (minutes)"] >= constraints["duration_min"]
    if constraints.get("duration_max") is not None:
        mask &= clean_df["Duration (minutes)"] <= constraints["duration_max"]

    # prefer_high_rating applies IMDb >= 7.0 floor only when other
    # hard constraints are present (checked by caller via _has_hard_constraints)
    if constraints.get("prefer_high_rating"):
        mask &= clean_df["IMDb Rating"] >= 7.0

    return set(clean_df.loc[mask, "movie_id"])
```

- [ ] **Step 2: Update `_doc_satisfies_hard_constraints()` — align certificate normalization**

In the same cell 54, update the certificate check in `_doc_satisfies_hard_constraints()` to use the same `_normalize_certificate()` helper. This ensures the reranker's safety-net filter matches the DataFrame pre-filter. Change this line:

```python
# OLD:
if constraints.get("certificate") and str(m.get("certificate", "")).lower() != constraints["certificate"].lower():
    return False

# NEW:
if constraints.get("certificate") and str(m.get("certificate", "")).upper() != _normalize_certificate(constraints["certificate"]):
    return False
```

- [ ] **Step 3: Update `_has_hard_constraints()` — remove `prefer_high_rating` as standalone trigger**

In the same cell 54, revert `_has_hard_constraints()` so that `prefer_high_rating` alone does NOT return True. It should only check the standard hard keys:

```python
def _has_hard_constraints(constraints: Dict) -> bool:
    """Check if the constraints dict contains any non-None hard constraint."""
    hard_keys = [
        "genre", "certificate", "actor_name", "director_name",
        "imdb_min", "imdb_max", "year_min", "year_max",
        "duration_min", "duration_max",
    ]
    return any(constraints.get(k) is not None for k in hard_keys)
```

This reverts the earlier change that added `prefer_high_rating` as a trigger. With the hybrid approach, `prefer_high_rating` is applied as a filter inside `_filter_compliant_movie_ids()` when other hard constraints are present, but doesn't by itself trigger the constrained path.

- [ ] **Step 4: Verify cell runs without errors**

Re-run cell 54 in the notebook. Expected output:
```
Reranking pipeline ready: _has_hard_constraints, _doc_satisfies_hard_constraints,
_semantic_similarity, summarize_constraints, summarize_preferences, rerank_with_constraints
```

- [ ] **Step 5: Commit**

```bash
git add wei-wong.ipynb
git commit -m "Add _filter_compliant_movie_ids() helper for hybrid retrieval"
```

---

### Task 2: Add constrained branch to `run_retrieval_pipeline()`

**Cell:** 57 (`2svnpeqc29o`) — retrieval pipeline cell

Replace the single FAISS retrieval path with a branching strategy: constrained queries pre-filter via DataFrame, unconstrained queries use FAISS k=30 as before.

- [ ] **Step 1: Modify `run_retrieval_pipeline()` — add constrained branch**

Replace the existing Step 4 (FAISS retrieval) and Step 4b (title injection) with the branching logic. The full updated function:

```python
def run_retrieval_pipeline(
    query: str,
    chat_history: Optional[List[Tuple[str, str]]] = None,
    assistant_mode: str = "search",
    top_k: int = 5,
    candidate_k: int = 30,
    constraints: Optional[Dict] = None,
) -> Dict:
    """Complete retrieval orchestrator (no LLM call).

    Pipeline:
    1. Extract hard constraints from query (or use pre-extracted constraints)
    2. Extract soft preferences from query + chat history
    3. Rewrite query with preference hints
    4. Retrieve candidates:
       - If hard constraints detected: filter DataFrame first, then score ALL
         compliant movies via FAISS for semantic relevance (hybrid path)
       - If no constraints: retrieve candidate_k candidates from FAISS (original path)
    5. Rerank with hybrid scoring
    6. Return top_k results with full diagnostics
    """
    if chat_history is None:
        chat_history = []

    # Step 1-2: Parse the query — use pre-extracted constraints if available
    if constraints is None:
        constraints = extract_query_constraints(query)
    preferences = extract_preference_profile(query, chat_history=chat_history)

    # Step 3: Rewrite query for better retrieval
    retrieval_query = rewrite_query_for_retrieval(query, assistant_mode, preferences)

    # Step 4: Retrieve candidates — branch on whether hard constraints exist
    compliant_ids = None  # initialized here for safe scope in title injection

    if _has_hard_constraints(constraints):
        # ── Constrained path (hybrid retrieval) ──────────────────────
        # Pre-filter DataFrame to find ALL compliant movies, then score
        # them semantically via FAISS. This guarantees no compliant movie
        # is missed due to the k=30 bottleneck.
        compliant_ids = _filter_compliant_movie_ids(constraints)

        if not compliant_ids:
            # No movies match the constraints — return early
            return {
                "docs": [],
                "retrieval_query": retrieval_query,
                "constraints": constraints,
                "constraint_summary": summarize_constraints(constraints),
                "preference_profile": preferences,
                "preference_summary": summarize_preferences(preferences),
                "ranking_details": {},
                "used_constraint_fallback": False,
                "assistant_mode": assistant_mode,
            }

        # Query FAISS with full index to get semantic scores for every movie.
        # FAISS returns (doc, L2_distance) tuples — lower distance = better.
        # At 2,762 movies this is trivially fast (milliseconds). For larger
        # datasets, consider a filtered FAISS search or batched approach.
        all_results = vector_store.similarity_search_with_score(
            retrieval_query, k=vector_store.index.ntotal
        )

        # Intersect: keep only FAISS results whose movie_id is in the
        # compliant set. This preserves the (doc, distance) tuple format
        # so the reranker's _semantic_similarity() conversion works correctly.
        candidates_with_scores = [
            (doc, dist) for doc, dist in all_results
            if doc.metadata.get("movie_id") in compliant_ids
        ]
    else:
        # ── Unconstrained path (original FAISS retrieval) ────────────
        candidates_with_scores = vector_store.similarity_search_with_score(
            retrieval_query, k=candidate_k
        )

    # Step 4b: Inject exact title matches — follow-up queries often reference
    # specific movies by name. Direct title lookup fills the gap.
    # Title-injected docs must also pass the compliance filter if constraints exist.
    _existing_ids = {doc.metadata.get("movie_id") for doc, _ in candidates_with_scores}
    _q_lower = query.lower()
    for _title_key, _record in TITLE_LOOKUP.items():
        if len(_title_key) >= 4 and _title_key in _q_lower:
            for _md in movie_docs:
                mid = _md.metadata.get("movie_id")
                if (_md.metadata.get("title", "").lower() == _record.get("Title", "").lower()
                        and mid not in _existing_ids):
                    # If constrained path, only inject if the title is compliant
                    if compliant_ids is not None and mid not in compliant_ids:
                        continue
                    candidates_with_scores.append((_md, 0.0))
                    _existing_ids.add(mid)
                    break

    # Step 5: Rerank
    reranked = rerank_with_constraints(
        candidates_with_scores,
        constraints=constraints,
        preference_profile=preferences,
        top_k=top_k,
        assistant_mode=assistant_mode,
    )

    # Step 6: Return results with diagnostics
    return {
        "docs": reranked["docs"],
        "retrieval_query": retrieval_query,
        "constraints": reranked["constraints"],
        "constraint_summary": reranked["constraint_summary"],
        "preference_profile": reranked["preference_profile"],
        "preference_summary": reranked["preference_summary"],
        "ranking_details": reranked["ranking_details"],
        "used_constraint_fallback": reranked["used_constraint_fallback"],
        "assistant_mode": assistant_mode,
    }
```

- [ ] **Step 2: Update the demo queries at the bottom of the cell**

Replace the demo queries with ones that exercise both paths:

```python
# --- Demo: constrained vs unconstrained retrieval ---
demo_queries = [
    ("best R-rated action movies rated above 8.0", "search",
     {"genre": "Action", "certificate": "R", "imdb_min": 8.0, "prefer_high_rating": True}),
    ("I loved Inception and Interstellar. Something mind bending.", "recommendation", None),
]

for query, mode, demo_constraints in demo_queries:
    result = run_retrieval_pipeline(query, assistant_mode=mode, constraints=demo_constraints)
    path = "CONSTRAINED (DataFrame + FAISS)" if demo_constraints and _has_hard_constraints(demo_constraints) else "UNCONSTRAINED (FAISS k=30)"
    print(f"QUERY: \"{query}\" (mode={mode})")
    print(f"Path: {path}")
    print(f"Constraints: {result['constraint_summary']}")
    print(f"Preferences: {result['preference_summary']}")
    if result["used_constraint_fallback"]:
        print("⚠ Constraint fallback used")
    print(f"Top {len(result['docs'])} results:")
    for i, doc in enumerate(result["docs"]):
        m = doc.metadata
        detail = result["ranking_details"].get(m["movie_id"], {})
        score = detail.get("final_score", "?")
        tags = detail.get("reason_tags", [])
        print(f"  {i+1}. {m['title']} ({m['year']}) — {m['genre']} — Rating: {m['imdb_rating']} — Score: {score}")
        if tags:
            print(f"     Why: {'; '.join(tags)}")
    print()
```

- [ ] **Step 3: Re-run cell 57 and verify**

Re-run the cell. Expected behavior:
- First query ("best R-rated action movies rated above 8.0") should use the constrained path and return movies like The Matrix (8.7), Terminator 2 (8.6), Gladiator (8.5) — NOT Fast Charlie (6.0)
- Second query ("I loved Inception...") should use the unconstrained path (no constraints) and work as before

- [ ] **Step 4: Commit**

```bash
git add wei-wong.ipynb
git commit -m "Add hybrid retrieval: DataFrame pre-filter for constrained queries"
```

---

### Task 3: Refactor `catalog_agent()` to use shared helper

**Cell:** 82 (`qcdrb3z7lae`) — agents cell

Replace the inline filtering logic in `catalog_agent()` with the shared `_filter_compliant_movie_ids()` helper for consistency.

**Behavior change note:** The old `catalog_agent()` did not check `prefer_high_rating`. After this refactor, catalog queries with `prefer_high_rating=True` will apply an IMDb >= 7.0 floor, excluding lower-rated movies that were previously returned. This is the desired behavior — it's more consistent with the user's stated intent ("best documentaries" should not return 5.0-rated documentaries).

- [ ] **Step 1: Refactor `catalog_agent()`**

Replace the inline mask-building with a call to the shared helper, then use the compliant ids to filter `clean_df`:

```python
def catalog_agent(query: str, chat_history: List[Tuple[str, str]], constraints: Optional[Dict] = None) -> Dict:
    """Exhaustive filtered lists — bypasses FAISS and filters clean_df directly.

    Why not FAISS? Catalog queries like "give me all documentaries rated 8+"
    need every matching row, not the top-k most semantically similar. FAISS
    caps at candidate_k vectors and misses valid results. DataFrame filtering
    returns the full set, sorted by IMDb rating descending.

    No LLM call — catalog results speak for themselves.
    """
    # Use pre-extracted constraints from understand_query() if available,
    # otherwise fall back to regex extraction
    if constraints is None:
        constraints = extract_query_constraints(query)
    constraint_summary = summarize_constraints(constraints)

    # Filter using the shared helper (same logic as hybrid retrieval path)
    compliant_ids = _filter_compliant_movie_ids(constraints)

    if not compliant_ids:
        return {
            "status": "no_match",
            "answer": f"I searched the full catalog but couldn't find any movies matching: {constraint_summary}. Try loosening a filter.",
            "docs": [],
            "constraints": constraints,
            "constraint_summary": constraint_summary,
            "preference_profile": {},
            "preference_summary": "none",
            "ranking_details": {},
            "used_constraint_fallback": False,
            "assistant_mode": "catalog",
        }

    # Filter clean_df to compliant rows, sort by rating descending
    filtered = clean_df[clean_df["movie_id"].isin(compliant_ids)]
    filtered = filtered.sort_values("IMDb Rating", ascending=False)

    # Build Document objects for the formatting layer
    docs = [row_to_document(row) for _, row in filtered.iterrows()]

    return {
        "status": "ok",
        "answer": "",  # no LLM commentary for catalog
        "docs": docs,
        "constraints": constraints,
        "constraint_summary": constraint_summary,
        "preference_profile": {},
        "preference_summary": "none",
        "ranking_details": {},
        "used_constraint_fallback": False,
        "assistant_mode": "catalog",
    }
```

`movie_search_agent()`, `recommendation_agent()`, `clarification_agent()`, and `fallback_agent()` remain unchanged.

- [ ] **Step 2: Re-run cell 82 and verify**

Re-run the cell. No output expected (agents don't print on definition).

- [ ] **Step 3: Quick smoke test in notebook**

Run this in a new cell to verify catalog still works:

```python
# Smoke test: catalog agent should return all R-rated action movies above 8.0
test = catalog_agent("R-rated action above 8", [], constraints={
    "genre": "Action", "certificate": "R", "imdb_min": 8.0,
    "prefer_high_rating": True,
})
print(f"Catalog found {len(test['docs'])} movies")
for doc in test["docs"][:5]:
    m = doc.metadata
    print(f"  {m['title']} ({m['year']}) — {m['imdb_rating']} — {m.get('certificate', '?')}")
```

Expected: ~20+ R-rated Action movies rated above 8.0, sorted by rating descending, with The Matrix near the top.

- [ ] **Step 4: Commit**

```bash
git add wei-wong.ipynb
git commit -m "Refactor catalog_agent to use shared _filter_compliant_movie_ids helper"
```

---

### Task 4: End-to-end validation via Gradio

No code changes — this is manual testing through the running Gradio UI.

- [ ] **Step 1: Restart kernel and Run All**

Restart the notebook kernel and run all cells top-to-bottom. Fix any errors.

- [ ] **Step 2: Test constrained recommendation query**

In Gradio, enter: "I'm in the mood for a dark R-rated thriller"
- Should return well-known R-rated thrillers (Se7en, Silence of the Lambs, Zodiac, etc.)
- Should NOT return non-R-rated or non-thriller movies

- [ ] **Step 3: Test constrained search query**

In Gradio, enter: "Best PG-13 sci-fi movies rated above 8.0"
- Should return PG-13 sci-fi movies above 8.0 (Inception, Interstellar, etc.)
- Should NOT return movies below 8.0

- [ ] **Step 4: Test unconstrained recommendation query**

In Gradio, enter: "Something suspenseful and mind-bending for tonight"
- Should use the unconstrained FAISS path (no hard constraints)
- Should return thematically relevant movies based on semantic similarity

- [ ] **Step 5: Test catalog query**

In Gradio, enter: "Show me all Brad Pitt movies"
- Should return a comprehensive list sorted by rating
- Should match the previous catalog behavior

- [ ] **Step 6: Test the problematic query from earlier**

In Gradio, enter: "Recommend a fast-paced action thriller that is highly rated and critically acclaimed"
- Should return high-quality action thrillers (IMDb 7.0+)
- Should NOT return Fast Charlie (6.0) or similar low-rated movies
