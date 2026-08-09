# AI Interview Agent

A conversational technical interviewer that reads a candidate's actual
curriculum mission history (passes, failures, attempt counts, skipped days)
and runs an adaptive interview grounded in **their real record**, not a
generic question bank.

## Endpoint

```
POST /api/interview
```

Matches `technical-spec.md` exactly:

- **Turn 1** — `{"sessionId": "...", "candidate": {...}}` → starts a session, returns the opening question.
- **Turn 2+** — `{"sessionId": "...", "message": "..."}` → advances the conversation.
- **Final turn** — `{"reply": "...", "done": true, "feedback": {"summary", "strengths", "gaps", "next"}}`

No auth, no database — an in-memory session store, matching the stated
out-of-scope list (no accounts, no persistent history).

## Run it

```bash
pip install -r requirements.txt
python3 main.py
# → http://localhost:8000
```

Try it end-to-end with the included demo client (in a second terminal):

```bash
python3 test_client.py CAND-006        # any candidate id from data/candidates.json
```

Or by hand:

```bash
curl -X POST http://localhost:8000/api/interview \
  -H "Content-Type: application/json" \
  -d '{"sessionId":"demo-1","candidate": <one entry from data/candidates.json>}'

curl -X POST http://localhost:8000/api/interview \
  -H "Content-Type: application/json" \
  -d '{"sessionId":"demo-1","message":"I used ChromaDB for local dev and Pinecone for the managed option."}'
```

### Optional: enable the LLM

The agent runs fully functional **without any API key** — see
"Architecture" below. To get richer, more adaptive phrasing, set:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export ANTHROPIC_MODEL=claude-sonnet-4-5   # optional, this is the default
python3 main.py
```

Every LLM call site (`llm.py`) is wrapped in a try/except that returns
`None` on any failure — missing key, network error, bad JSON — so the
engine transparently falls back to its deterministic logic. Nothing about
the API contract or conversation flow changes based on whether a key is
present.

## Architecture

```
main.py     FastAPI app — the one required HTTP endpoint + a /health check
engine.py   Core logic: candidate analysis, question selection, follow-up
            decisions, response scoring, feedback synthesis, session state
llm.py      Anthropic API wrapper, "best effort" — every call site has a
            deterministic fallback (see engine.py)
data/       curriculum.json + candidates.json (as supplied)
```

**Why plain FastAPI, in-memory state, no agent framework.** The task is a
single well-defined endpoint with short-lived, per-session state — a
LangChain/CrewAI-style agent graph would add orchestration overhead
without buying anything, since there's exactly one "tool" (the LLM) and
one state machine (ask → score → follow-up-or-advance → feedback). A
plain state machine is easier to reason about, test, and grade.

### How a candidate becomes an interview plan

`CandidateProfile.build_interview_plan()` in `engine.py`:

1. Every mission is scored for "how much attention does this deserve":
   skipped days score highest (untested — worth probing), failed missions
   next, then passed missions with 4+ attempts (shaky pass). Comfortably
   passed missions (1 attempt) score zero.
2. The plan opens with **one warm-up day** — the candidate's earliest
   comfortably-passed mission — so the interview doesn't open on their
   weakest spot.
3. It then walks the weak/untested days in severity order.
4. It mixes in **1–2 strong days near the end** as a cross-check: a
   candidate who's genuinely strong should still look strong here, which
   makes the weak-day findings more trustworthy by contrast.
5. If a candidate's own mission history is too sparse to reach 4 distinct
   days (e.g. an early-exit candidate with mostly `skipped` entries and no
   passes), the plan backfills from a fixed set of core, broadly-applicable
   curriculum days (Embeddings, Retrieval, Prompting, Backend, Agents,
   Capstone) — so **every** candidate, however incomplete their record,
   still gets a full interview.

This is the one deliberately "thoughtful" design choice in the brief's
sense: the interview isn't the same 8 questions for everyone — it's
targeted at each candidate's actual demonstrated weak points, with a
built-in sanity check against their strengths.

### How follow-ups work

After every answer, `score_response()` computes a lightweight signal —
word count, presence of hedging language ("not sure", "don't know", ...),
and keyword overlap against that day's `tools` + `objectives` from
`curriculum.json`. A thin or hedged answer triggers exactly one follow-up
question that names a specific tool from that day and asks the candidate
to go one level deeper. A solid answer moves straight to the next topic.
This keeps the interview length predictable (capped at 12 questions) while
still feeling reactive rather than scripted.

### Feedback

The same per-answer scores double as the evidence base for the final
feedback: days with consistently strong answers become `strengths`, days
with consistently thin/hedged answers become `gaps`, and `next` maps each
gap back to that day's actual curriculum objectives — so feedback is
always traceable to something the candidate actually said, not a generic
template.

## Testing notes

`test_client.py` runs a full 13-turn conversation against a live server
using a deliberately mixed set of strong/weak/hedged canned answers, to
exercise both the follow-up branch and the advance-to-next-day branch, and
prints the final structured feedback. It was used during development to
confirm: 8+ questions asked, 4+ distinct curriculum days covered, at least
one follow-up triggered, and a well-formed `feedback` object on
completion — against multiple candidates (`CAND-006`, a mixed performer;
`CAND-003`, near-perfect; `CAND-011`, a sparse/early-exit profile).
