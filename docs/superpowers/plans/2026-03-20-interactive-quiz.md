# Interactive Quiz Plan

## Goal

Add a lightweight **interactive movie quiz** to the existing Gradio demo so
users can get recommendations through a guided flow, not just free-text input.

This is meant to strengthen the **Creativity / Advanced Features** rubric
category without rewriting the recommendation system.

---

## Status Update — 2026-03-21

This is still the next planned feature and has not been implemented yet.

### Prerequisites Now Completed

- Gradio chat history normalization is fixed, so the UI boundary is stable again
- `_match_person_name()` no longer overfires on surname-only false positives like `Toy Story` -> `Anthony Story`
- Zero-result queries now return a helpful relaxation message instead of a dead-end no-match

### Recommended Next Implementation Order

1. Add a deterministic `build_quiz_query(...)` helper
2. Add `submit_quiz_turn(...)` so quiz answers flow through the existing chat path
3. Add a `Quiz` tab to the Part 7 Gradio UI
4. Smoke test a few quiz-generated recommendation queries

That sequencing keeps the quiz as a thin UX layer on top of the already-working
retrieval and recommendation pipeline.

---

## Product Idea

Instead of asking the user to type a full request, the quiz asks a small set of
guided preference questions, then converts the answers into a natural-language
movie query and sends that query through the **existing** chatbot pipeline.

That means:

- no new retrieval architecture
- no separate recommendation engine
- no new model prompts required

The quiz is just another **input mode** that feeds into:

`quiz answers -> quiz-to-query builder -> gradio_chat_fn() -> safe_chatbot() -> format_final_response()`

---

## Why This Approach

### What

Use the current RAG + recommendation stack as-is, and build a thin quiz layer
on top of it.

### Why

This is the safest way to add an advanced feature late in the project:

- low engineering risk
- minimal regression risk
- easy to explain in the notebook
- clearly visible to a grader

### How

The quiz collects a few high-signal preferences and translates them into a
plain-English request that the current chatbot already knows how to handle.

---

## Recommended Quiz Questions

Keep the quiz short. Four questions is enough.

### 1. Mood

Examples:

- Light / fun
- Dark / intense
- Mind-bending
- Emotional / heartfelt

### 2. Genre

Examples:

- Any
- Action / Adventure
- Comedy
- Drama
- Thriller / Mystery
- Sci-fi / Fantasy
- Family / Animation

### 3. Runtime

Examples:

- Under 2 hours
- Any length
- Long / epic

### 4. Era

Examples:

- Any era
- Recent (2010s+)
- 2000s
- 1990s and earlier

Optional fifth question if we want slightly more control later:

### 5. Content Rating

- Any
- Family-friendly / PG
- PG-13
- R

---

## Query Builder Strategy

### Goal

Turn the quiz choices into a clean natural-language recommendation query.

### Example output

If the user chooses:

- Mood: Mind-bending
- Genre: Sci-fi / Fantasy
- Runtime: Under 2 hours
- Era: Recent

Then the generated query could be:

> Recommend a mind-bending sci-fi movie from the 2010s or later under 120 minutes.

Another example:

- Mood: Light / fun
- Genre: Comedy
- Runtime: Any length
- Era: Any era

Query:

> Recommend a light, fun comedy.

### Important rule

The query builder should be **simple and deterministic**. It should not call an
LLM. Its job is just to compose a sensible sentence from the selected options.

---

## UI Integration

## Recommended placement

Add a third input mode to the current Gradio side panel:

- `Type`
- `Voice`
- `Quiz`

### Quiz tab contents

- short helper text
- 4 dropdowns or radios
- one primary button, e.g. `Get Quiz Picks`

### On submit

1. Build the natural-language query from the selected answers
2. Send it through `gradio_chat_fn()`
3. Append the generated query as the user's message in chat history
4. Show the assistant response normally

This keeps the quiz feeling like part of the same conversation, not a separate
tool bolted onto the page.

---

## Implementation Notes

### Notebook areas likely to change

- `wei-wong.ipynb`
  - Gradio UI cell
  - possibly Part 7 summary markdown

### Likely helper functions to add

- `build_quiz_query(...)`
- `submit_quiz_turn(...)`

### Existing functions to reuse

- `gradio_chat_fn(...)`
- `_append_chat_turn(...)`
- `clear_demo(...)`

No changes should be required to:

- `safe_chatbot(...)`
- retrieval / reranking
- agent routing
- formatting logic

---

## Recommended UX Copy

### Quiz helper text

> Answer a few quick questions and I’ll turn them into a movie recommendation request.

### Quiz button

> Get Quiz Picks

### Chat behavior

The chat should show the generated request as the user message. That makes the
flow transparent and helps debugging.

Example user-side message in chat:

> Quiz request: Recommend a dark thriller under 2 hours from the 2010s or later.

---

## Risks / Gotchas

### 1. Too many options

If the quiz has too many controls, it starts feeling like a form instead of a
fun interaction. Keep it short.

### 2. Over-constraining the query

If every quiz answer becomes a hard constraint, results may get too narrow.
The builder should phrase most quiz choices as **preferences**, not strict
filters, unless they are naturally hard constraints like runtime or rating.

### 3. UI crowding

The current Gradio layout is already fuller than it was earlier. The quiz
should live in the existing tab structure rather than creating a new section
elsewhere on the page.

### 4. Empty / default selections

If the user leaves answers on `Any`, those fields should be omitted from the
generated query rather than inserted awkwardly.

---

## Testing Plan

### Manual smoke tests

1. Pick `Mind-bending` + `Sci-fi / Fantasy` + `Under 2 hours` + `Recent`
   Expected: the generated query appears in chat and returns recommendation-mode results

2. Pick `Light / fun` + `Comedy` + `Any length` + `Any era`
   Expected: broad comedy recommendations

3. Pick `Dark / intense` + `Thriller / Mystery` + `Under 2 hours` + `Recent`
   Expected: thriller-heavy recommendations

4. Clear the app after a quiz turn
   Expected: chat, text box, transcript preview, audio input, and quiz controls reset cleanly

### Nice-to-have automated coverage

Add one or two quiz-generated query cases to the evaluation harness later by
hard-coding the output of `build_quiz_query(...)`.

---

## Estimated Effort

### Minimal version

~45-90 minutes

Includes:

- 4 quiz controls
- query builder
- submit handler
- chat integration
- light UI polish

### Slightly nicer version

~1.5-2.5 hours

Includes:

- better copy
- reset behavior for quiz controls
- notebook markdown updates
- evaluation-harness coverage

---

## Recommendation

When we come back to this, implement the **minimal version first**:

1. add `Quiz` tab
2. add `build_quiz_query(...)`
3. add `submit_quiz_turn(...)`
4. wire it into the same chat history as text/voice

That gets the rubric win with very little architecture risk.
