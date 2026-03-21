# Retrieval Improvement Options (Without Synthetic Descriptions)

## Goal

Improve the **Retrieval & Search Efficiency** rubric category without doing a
full dataset enrichment / re-embedding project.

This plan is intentionally **low-risk** and **incremental**. The current
retriever already works well for:

- title lookup
- actor / director queries
- hard constraints like rating, runtime, year, certificate
- catalog-style exhaustive lists

The main weakness is **descriptive semantic retrieval** for abstract themes or
plot concepts such as:

- `time travel`
- `feel-good`
- `mind-bending`
- `revenge`
- `courtroom drama`

Because `search_text` is still built from structured metadata, not plot or
synopsis text, we should focus on sharpening the **current hybrid retriever**
rather than pretending it can fully understand narrative concepts.

---

## Status Update — 2026-03-21

Some upstream retrieval-quality fixes have already been completed in
`wei-wong.ipynb` since this note was drafted.

### Completed Since Draft

- `_match_person_name()` now blocks surname-only matches unless the query shows real person-context signals
- Tokens inside detected movie titles are excluded from surname and fuzzy person matching, which prevents false positives like `Toy Story` -> `Anthony Story`
- Zero-hit constrained queries now offer nearby relaxation suggestions in the UI instead of a generic no-match

### Implication

The most urgent retrieval bug is no longer false actor/director matching. The
remaining retrieval opportunity is mainly about improving descriptive recall and
precision for abstract thematic asks.

---

## Option 1: Hybrid Lexical + FAISS Retrieval

### What

Add a lightweight lexical retrieval lane alongside FAISS, then merge both
candidate pools before reranking.

### Why

FAISS is strong for semantic similarity, but exact entity queries often benefit
from direct string matching. This gives the system a better chance of catching:

- exact title matches
- title substring matches
- actor / director name matches
- mild title or name typos

### Suggested implementation

Inside `run_retrieval_pipeline()`:

1. Run FAISS retrieval as usual
2. In parallel, collect lexical candidates from `clean_df` using:
   - exact normalized title match
   - title substring match
   - exact actor / director match
   - mild fuzzy match for title / person names
3. Union the lexical + FAISS candidates
4. Pass the combined candidate set into `rerank_with_constraints()`

### Why this is a good first improvement

- Low architectural risk
- No re-embedding required
- Improves title-based and structured search immediately
- Easy to explain in the notebook as a pragmatic hybrid retrieval design

---

## Option 2: Stronger Query Expansion for Metadata Proxies

### What

Improve `rewrite_query_for_retrieval()` so vague user language maps more
consistently to the metadata signals the dataset actually contains.

### Why

Without plot text, descriptive queries only work when we translate them into
useful proxies such as:

- genre
- era / year
- certificate
- runtime
- tone-adjacent genre clusters

### Examples

- `mind-bending` -> boost `sci-fi`, `mystery`, `thriller`
- `feel-good` -> boost `comedy`, `family`, `romance`
- `darker` -> boost `crime`, `thriller`, `mystery`
- `suspenseful` -> boost `thriller`, `crime`, `mystery`

### Why this helps

This improves retrieval **without** manually tagging every movie. We enrich the
query, not the full dataset.

---

## Option 3: Stricter Reranking for Weak-Fit Results

### What

Adjust `rerank_with_constraints()` so weak or only-adjacent candidates are less
likely to survive recommendation mode.

### Why

Some current misses are not catastrophic retrieval failures; they are
reranking-quality issues. The pipeline sometimes keeps movies that are vaguely
adjacent, highly rated, or era-matched even when they do not strongly satisfy
the main theme.

### Suggested implementation

- Increase bonuses for direct genre / entity matches
- Reduce the relative influence of weak soft signals like era-only alignment
- Penalize candidates with no evidence for the core request theme
- Keep recommendation mode quality thresholds high enough to avoid filler titles

### Example benefit

This should reduce results like a generally popular sci-fi/action title being
returned for a query that implies a more specific narrative concept.

---

## Recommended Order

If we decide to improve retrieval later, do it in this order:

1. **Hybrid lexical + FAISS retrieval**
2. **Stronger query expansion**
3. **Stricter reranking**

This order gives the best quality gain for the lowest engineering risk.

---

## What We Are Explicitly Not Doing Here

- No synthetic descriptions
- No full dataset re-embedding
- No large ontology / manual theme-tagging project
- No major architecture rewrite

Those may still be valid future enhancements, but they are not required to
meaningfully improve the current retriever.

---

## Suggested Notebook Framing

If this is mentioned in the final submission, the framing should be:

> A practical next step would be to strengthen the current hybrid retriever by
> combining FAISS with lightweight lexical matching, improving query expansion
> for metadata-friendly proxies, and tightening reranking so weak-fit results
> are less likely to survive. This would improve retrieval quality without
> requiring a full synthetic-description enrichment pipeline.
