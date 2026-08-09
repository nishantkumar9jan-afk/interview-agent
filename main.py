"""
main.py — FastAPI app exposing the single required endpoint:

    POST /api/interview

Session lifecycle (matches technical-spec.md exactly):
  1. First request:  {"sessionId": "...", "candidate": {...}}
     -> starts a new interview, returns the opening question.
  2. Every following request: {"sessionId": "...", "message": "..."}
     -> advances the conversation.
  3. Final response includes "done": true and a structured "feedback" object.

No auth, no persistent storage — an in-memory SessionStore is sufficient
per the spec's out-of-scope list (no accounts, no long-term history).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional

from engine import SESSIONS, MIN_QUESTIONS, MIN_DAYS

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="AI Interview Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class InterviewRequest(BaseModel):
    sessionId: str
    candidate: Optional[dict] = None
    message: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "min_questions": MIN_QUESTIONS, "min_days": MIN_DAYS}


@app.post("/api/interview")
def interview(req: InterviewRequest):
    session = SESSIONS.get(req.sessionId)

    # ---- Turn 1: start a new interview ----
    if session is None:
        if not req.candidate:
            raise HTTPException(
                status_code=400,
                detail="No session found for this sessionId. Include a 'candidate' object to start a new interview.",
            )
        session = SESSIONS.create(req.sessionId, req.candidate)
        opening_question = session.start()
        return {"reply": opening_question, "done": False}

    # ---- Interview already finished: stay idempotent rather than error ----
    if session.done:
        return {
            "reply": "This interview has already been completed.",
            "done": True,
        }

    # ---- Subsequent turns: candidate's answer ----
    if req.message is None:
        raise HTTPException(
            status_code=400,
            detail="Session already in progress. Include a 'message' field with the candidate's response.",
        )

    result = session.submit_answer(req.message)
    return result


# ---------------------------------------------------------------------------
# Frontend — a minimal chat UI served from the same app, same origin, so it
# works out of the box with no CORS setup and no separate deployment.
# Registered AFTER /api routes so it never shadows them.
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
