# Hybrid Retrieval Architecture — Design Spec

**Date:** 2026-03-19
**Status:** Approved
**Scope:** Rework retrieval in `wei-wong.ipynb` so constrained queries never miss compliant movies

## Problem

The current retrieval pipeline uses FAISS semantic search (k=30) as the sole entry point for search and recommendation queries. When a query has hard constraints (genre, certificate, rating, actor, etc.), many compliant movies are never retrieved because they aren't semantically similar to the query text.

**Example:** "Best R-rated action thriller rated above 8.0" — FAISS returns 30 semantically similar movies, but only 3 of those are R-rated Action movies above 8.0. The dataset has 27 matching movies (The Matrix, Terminator 2, Gladiator, etc.) that the user never sees.

This creates a whack-a-mole pattern: fixing one query's routing or reranking breaks another, because the root cause is FAISS acting as a bottleneck that filters out compliant movies before reranking can help.

## Solution

**When constraints are detected, filter the DataFrame first to get all compliant movies, then use FAISS to score them for semantic relevance.**

### Current flow (all queries)

```
Query → FAISS (k=30) → Hard filter (remove non-compliant) → Rerank → Top 5
```

### New flow (when constraints detected)

```
Query → DataFrame filter (all compliant movies) → FAISS (k=full index) → Intersect by movie_id → Rerank → Top 5
```

### New flow (no constraints)

```
Query → FAISS (k=30) → Rerank → Top 5  (unchanged)
```

## What Changes

**One function:** `run_retrieval_pipeline()`
**One new helper:** `_filter_compliant_movie_ids(constraints)`

### New logic inside `run_retrieval_pipeline()`

1. Check for constraints using existing `_has_hard_constraints(constraints)` — note: `prefer_high_rating` alone does NOT trigger the constrained path (see below)
2. **If constraints detected:**
   - Call `_filter_compliant_movie_ids(constraints)` to get the set of compliant movie_ids from `clean_df`
   - Query FAISS with `k=vector_store.index.ntotal` (full index — trivially fast at this dataset size) to get semantic distance scores for every movie
   - Keep only FAISS results whose `doc.metadata["movie_id"]` is in the compliant set
   - Title injection: if title-injected docs exist, they must also pass the compliance filter
   - Feed filtered `(doc, distance)` pairs into existing `rerank_with_constraints()`
3. **If no constraints:**
   - Query FAISS with `k=30` (unchanged)
   - Feed into existing `rerank_with_constraints()`

### `_has_hard_constraints()` update

`prefer_high_rating` alone should NOT trigger the constrained path. "Best movies" doesn't need full DataFrame scanning — the reranker's rating bonus handles it fine. But `prefer_high_rating` combined with other constraints (genre, cert, etc.) should still apply the 7.0 floor during DataFrame filtering.

Updated logic:
- `_has_hard_constraints()` checks the standard hard keys (genre, certificate, actor, director, imdb_min/max, year_min/max, duration_min/max)
- `prefer_high_rating` is applied as a filter within `_filter_compliant_movie_ids()` when other hard constraints are present, but does not by itself trigger the constrained path

### `_filter_compliant_movie_ids(constraints)` — new helper

Build a boolean mask on `clean_df` checking each non-null constraint:
- `genre`: case-insensitive substring match on Genre column
- `certificate`: case-insensitive exact match on Certificates column
- `actor_name`: case-insensitive match against `parsed_star_cast` list
- `director_name`: case-insensitive substring match on Director column (consistent with genre matching)
- `imdb_min` / `imdb_max`: numeric range on IMDb Rating
- `year_min` / `year_max`: numeric range on Year
- `duration_min` / `duration_max`: numeric range on Duration (minutes)
- `prefer_high_rating`: if true AND other hard constraints present, apply IMDb >= 7.0 floor

Returns: `Set[str]` of compliant movie_ids.

Note: `catalog_agent()` should be updated to use this same helper for consistency (currently has its own inline filtering logic with exact match for director instead of substring).

### Constraints source

The constrained/unconstrained branch uses constraints from `understand_query()` (LLM-extracted), passed through `orchestrate_agents()` → agent → `run_retrieval_pipeline(constraints=...)`. The regex fallback path (`extract_query_constraints()`) is only used if the LLM call fails.

### Edge cases

- **0 compliant movies:** return empty docs, `no_match` status (handled downstream)
- **< top_k compliant movies:** return all of them
- **Reranking hard filter:** becomes a safety net — shouldn't trigger since candidates are pre-filtered, but kept for defense
- **Constraint-only queries with no semantic content** (e.g., "R-rated movies above 8"): FAISS scores will be near-random across compliant movies; reranker's rating/genre bonuses will dominate ranking, which is the desired behavior
- **Multi-value genre constraints:** `understand_query()` returns a single genre string; if the LLM returns multiple, use the first one (existing behavior)

## What Does NOT Change

- `understand_query()` — still extracts constraints and intent via LLM
- `orchestrate_agents()` — still routes to agents by intent
- `catalog_agent()` — pure DataFrame filter, returns all matches sorted by rating (will share the new `_filter_compliant_movie_ids` helper)
- `rerank_with_constraints()` — still applies hybrid scoring (72% semantic + 28% bonuses)
- `safe_chatbot()` — still ties it all together
- Gradio UI — unchanged
- Fallback/clarification agents — unchanged

## Why This Works

Intent classification accuracy matters much less. Whether "Best R-rated action thriller" routes to search, recommendation, or catalog:
- **Catalog:** returns all matches sorted by rating (as today)
- **Search/Recommendation:** finds all compliant movies via DataFrame, scores them semantically, returns top 5 most relevant — no movies lost

## Tradeoffs

**Pros:**
- Eliminates the entire class of "FAISS missed compliant movies" bugs
- Any constraint combination works without per-query tuning
- Minimal code change (one function, one new helper)

**Cons:**
- Slightly slower for constrained queries (FAISS k=full vs k=30) — negligible at this dataset size
- Less semantic "surprise" — a PG-13 movie that's a perfect vibe match for an R-rated query gets filtered out. But that's what the user asked for.
- Unconstrained queries ("something dark and atmospheric") unchanged — still use FAISS k=30, which is where FAISS excels

## Files Modified

| File | Change |
|------|--------|
| `wei-wong.ipynb` — reranking helpers cell | Add `_filter_compliant_movie_ids(constraints)` helper; update `_has_hard_constraints()` to exclude `prefer_high_rating` alone |
| `wei-wong.ipynb` — `run_retrieval_pipeline()` cell | Add DataFrame pre-filtering branch using the new helper |
| `wei-wong.ipynb` — `catalog_agent()` cell | Refactor to use shared `_filter_compliant_movie_ids()` helper |
| `wei-wong.ipynb` — debug panel cell (if exists) | Update debug output to reflect constrained vs unconstrained path |
