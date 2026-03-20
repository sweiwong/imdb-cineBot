# Speech-to-Text Implementation Plan

> **Scope:** Add a visible, rubric-friendly voice-based search feature to the notebook demo without changing the core retrieval, reranking, or agent logic.

**Goal:** Let a user speak a movie request through the microphone, transcribe it with OpenAI speech-to-text, and send the transcript through the existing `safe_chatbot()` pipeline.

**Why this path:** The rubric explicitly names voice-based search under Creativity & Feature Enhancement. This is a high-visibility demo feature and a lower-risk addition than a late-stage retrieval architecture change.

**Chosen architecture:** Keep the chatbot brain unchanged. Add one new boundary layer:

`audio file from microphone -> OpenAI transcription -> transcript string -> safe_chatbot() -> format_final_response()`

That means all the existing work stays intact:
- intent detection
- follow-up handling
- retrieval + reranking
- guardrails
- response formatting

Only the input path changes.

---

## Implementation Principles

- **Do not change retrieval logic.** Speech-to-text is an input adapter, not a ranking project.
- **Do not replace the current chatbot pipeline.** Reuse `safe_chatbot()` and `format_final_response()`.
- **Prefer a simple microphone upload flow over realtime streaming.** Realtime audio is unnecessary complexity for a course demo.
- **Show the transcript back to the user.** This makes transcription errors transparent and easier to debug.
- **Fail gracefully.** If audio is empty, too noisy, or transcription fails, return a friendly message instead of crashing.

---

## Official API Notes

Use OpenAI's Transcriptions API with a speech-to-text model supported by `/v1/audio/transcriptions`.

Recommended model:
- `gpt-4o-mini-transcribe` for cost-efficient speech-to-text

Useful references:
- OpenAI speech-to-text guide: https://platform.openai.com/docs/guides/speech-to-text
- Audio transcription API reference: https://platform.openai.com/docs/api-reference/audio/createTranscription
- GPT-4o mini transcribe model page: https://developers.openai.com/api/docs/models/gpt-4o-mini-transcribe

As of the current docs, file uploads are limited to 25 MB and support formats like `wav`, `mp3`, `m4a`, and `webm`.

---

## Environment Prerequisites

The shell environment in this repo currently does **not** have `gradio`, `openai`, or `langchain_openai` installed, so confirm the notebook kernel environment before implementation.

If needed, install:

```bash
pip install gradio openai
```

If the notebook kernel already has Gradio working, only `openai` may need to be added for direct transcription calls.

---

## File Map

All implementation work should stay in `wei-wong.ipynb`.

| Area | What changes |
| --- | --- |
| OpenAI client setup cell | Add a lightweight transcription client for the audio API |
| Part 7 UI section | Add microphone input support to the demo |
| Response adapter layer | Add helper that transcribes audio and forwards transcript into `safe_chatbot()` |
| Final summary markdown | Add a note that the UI now supports voice-based search |

No changes should be required in:
- `run_retrieval_pipeline()`
- `rerank_with_constraints()`
- `detect_intent()`
- `safe_chatbot()` core logic
- evaluation harness logic

---

## Recommended UI Strategy

### Choose `gr.Blocks` instead of extending `gr.ChatInterface`

**Reason:** The current `gr.ChatInterface` is great for text-only chat, but voice input needs:
- a microphone component
- a transcript preview
- a dedicated "Transcribe & Send" action
- shared chat history state

That is much easier to manage explicitly in `gr.Blocks`.

### Proposed UI structure

- `gr.Chatbot(type="messages")`
- `gr.Textbox` for typed requests
- `gr.Audio(sources=["microphone"], type="filepath")`
- `gr.Markdown` or `gr.Textbox` for transcript preview
- `Send` button
- `Transcribe & Send` button
- `Clear` button
- `gr.Examples` for the 4 demo prompts

This keeps the existing text UX while adding voice as a second input mode.

---

## Task Breakdown

### Task 1: Add a transcription helper

Create a small helper function dedicated to audio transcription.

Suggested shape:

```python
from openai import OpenAI

transcription_client = OpenAI(api_key=OPENAI_API_KEY)


def transcribe_audio_to_text(audio_path: str) -> str:
    """Turn a recorded audio file into text using OpenAI speech-to-text."""
    if not audio_path:
        return ""

    with open(audio_path, "rb") as audio_file:
        transcript = transcription_client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
        )
    return transcript.text.strip()
```

Guardrails:
- If `audio_path` is missing, return empty string
- If transcription fails, catch the exception and return a friendly UI message
- Optionally add a short prompt like `"Movie titles, actor names, directors, ratings, and genres may appear."` if title spelling becomes a problem

---

### Task 2: Add a voice-aware chat adapter

Create a helper that:
1. transcribes the uploaded/recorded audio
2. sends the transcript to `safe_chatbot()`
3. formats the response using `format_final_response()`
4. returns both the transcript and the chatbot reply

Suggested shape:

```python
def gradio_voice_chat_fn(audio_path: str, history):
    transcript = transcribe_audio_to_text(audio_path)
    if not transcript:
        return history, "I couldn't hear anything clearly. Please try again.", ""

    payload = safe_chatbot(transcript, chat_history=history)
    reply = format_final_response(payload)
    return history + [
        {"role": "user", "content": transcript},
        {"role": "assistant", "content": reply},
    ], transcript, ""
```

Exact return types may vary depending on the final `gr.Blocks` wiring, but the design principle is:
- transcript is visible
- response uses the existing chatbot pipeline
- chat history stays shared between typed and voice turns

---

### Task 3: Replace the text-only demo with a `gr.Blocks` app

Build a custom Blocks app that preserves the current look and examples while supporting two input paths.

Minimum requirements:
- typed message path still works exactly as before
- microphone path transcribes then sends into the same conversation
- transcript is shown to the user
- examples still populate the text box
- clear button resets the conversation state

Important: keep the existing 4 example prompts unchanged unless there is a deliberate reason to revise them.

---

### Task 4: Handle voice-specific edge cases

Add friendly handling for:
- empty or missing audio
- transcription failure
- transcript that resolves to empty text
- obvious non-movie audio requests after transcription

These should flow into existing guardrails where possible.

Examples:
- blank recording -> `"I couldn't hear anything clearly. Please try recording again."`
- transcript succeeds but is off-topic -> existing off-topic message from `safe_chatbot()`

---

### Task 5: Manual smoke test checklist

Test both typed and spoken paths.

Text path:
- `I loved Inception and The Dark Knight. What should I watch next?`
- `Show me all Brad Pitt movies`

Voice path:
- speak: `"Recommend something suspenseful and mind-bending for tonight"`
- speak: `"Best PG-13 adventure movies rated above 8.0"`
- speak: `"Which of these is the most original?"` after a prior recommendation turn

Expected behavior:
- transcript appears and is readable
- chatbot response matches the text-input behavior
- follow-up logic still works
- no crashes when audio is missing or noisy

---

## What Not To Do

- Do not add realtime streaming audio in this submission
- Do not change the retrieval architecture
- Do not add text-to-speech in the same pass
- Do not refactor Part 7 more broadly than needed

This should stay a focused input feature, not a UI redesign project.

---

## Success Criteria

The feature is done when:

- a user can record a voice query from the demo UI
- the app shows the transcript
- the transcript is processed by the existing chatbot pipeline
- the response quality is comparable to typed input
- the notebook remains stable and demo-friendly

---

## Recommended Commit Strategy

1. Commit the current notebook state before starting
2. Implement transcription helper only
3. Commit
4. Implement UI wiring
5. Commit
6. Run smoke tests and polish final markdown summary

Suggested commit messages:
- `Add OpenAI speech-to-text helper for voice input`
- `Add Gradio voice search flow to notebook demo`
- `Polish voice search UX and transcript handling`
