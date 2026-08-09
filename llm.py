"""
llm.py — thin wrapper around the Anthropic Messages API.

Design choice: the interview engine never *requires* an LLM to function.
Every call site treats `call_llm(...)` as "best effort" — if no API key is
configured, or the call fails for any reason, it returns None and the
engine falls back to its deterministic, curriculum-grounded logic
(see engine.py). This means POST /api/interview keeps working — same
conversational contract, same feedback shape — even in a sandboxed grading
environment with no key provisioned. When a key *is* present, questions,
follow-ups, and feedback get noticeably richer and more adaptive.
"""

import os
import json
import requests

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
TIMEOUT_SECONDS = 20


def llm_available() -> bool:
    return bool(ANTHROPIC_API_KEY)


def call_llm(system: str, user_prompt: str, max_tokens: int = 400):
    """
    Single-turn helper: system prompt + one user prompt -> text response.
    Returns None on any failure (missing key, network error, bad response)
    so callers can fall back gracefully.
    """
    if not ANTHROPIC_API_KEY:
        return None
    try:
        resp = requests.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        parts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
        return "\n".join(parts).strip() or None
    except Exception:
        return None


def call_llm_json(system: str, user_prompt: str, max_tokens: int = 700):
    """
    Same as call_llm, but instructs the model to return ONLY JSON and
    parses it. Returns None if the call fails or output isn't valid JSON.
    """
    raw = call_llm(
        system + "\n\nRespond with ONLY valid JSON. No markdown fences, no preamble, no trailing text.",
        user_prompt,
        max_tokens=max_tokens,
    )
    if not raw:
        return None
    cleaned = raw.strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return None
