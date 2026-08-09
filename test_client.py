"""
test_client.py — drives a full interview conversation against a running
server to demonstrate the endpoint satisfies the spec: >=8 questions,
>=4 distinct curriculum days, adaptive follow-ups, and a structured
feedback payload at the end.

Usage:
    python3 main.py               # in one terminal
    python3 test_client.py        # in another (defaults to CAND-006)
    python3 test_client.py CAND-003 http://localhost:8000
"""
import json
import sys
import uuid
import requests

BASE = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
CAND_ID = sys.argv[1] if len(sys.argv) > 1 else "CAND-006"

candidates = json.load(open("data/candidates.json"))["candidates"]
candidate = next(c for c in candidates if c["member"]["id"] == CAND_ID)

session_id = str(uuid.uuid4())

# A mix of answer qualities to exercise both branches of the engine —
# short/hedged answers should trigger follow-ups, detailed grounded
# answers should move the interview on to the next topic.
CANNED_ANSWERS = [
    "Not sure, I don't really remember the details of that one.",
    "I used ChromaDB locally for fast iteration during development, then compared it against Pinecone for a managed, scalable option once we needed multi-region availability and didn't want to run our own infra.",
    "Yeah I did that.",
    "We built a query router that checked whether the question needed structured lookup from SQLite or semantic search from the vector store, then merged and deduplicated results by source before ranking them.",
    "Maybe, I think so, not 100% sure honestly.",
    "I used LangChain agents with a ReAct loop so the model could decide which tool to call based on the reasoning trace, and I logged every tool call for debugging.",
    "I skipped that day so I don't have hands-on experience with it.",
    "For the capstone I wired together the retrieval engine, the agent orchestration layer, and MCP tool calls behind a FastAPI backend, with a React frontend consuming a streaming endpoint.",
    "Not really, didn't get to that part.",
    "We containerized both services with Docker and used health checks plus environment-based config for the Kubernetes deployment.",
    "Honestly I'm blanking on this one, we didn't spend much time here.",
    "I set up specialized agents per domain and a router agent that classified the incoming query before delegating to the right specialist, then merged their outputs into one response.",
    "We used Pydantic models to validate every tool call's structured output before it reached the user, and logged the raw and validated versions for auditing.",
]

print(f"=== Interviewing {candidate['member']['name']} ({CAND_ID}) ===\n")

resp = requests.post(f"{BASE}/api/interview", json={"sessionId": session_id, "candidate": candidate})
resp.raise_for_status()
data = resp.json()
print("INTERVIEWER:", data["reply"], "\n")

turn = 0
question_count = 1
while not data.get("done") and turn < len(CANNED_ANSWERS):
    answer = CANNED_ANSWERS[turn]
    print("CANDIDATE:", answer, "\n")
    resp = requests.post(f"{BASE}/api/interview", json={"sessionId": session_id, "message": answer})
    resp.raise_for_status()
    data = resp.json()
    print("INTERVIEWER:", data["reply"], "\n")
    question_count += 1
    turn += 1

print("=" * 60)
print("done:", data.get("done"))
print("total interviewer turns (incl. opening):", question_count)
if "feedback" in data:
    print("\nFEEDBACK:")
    print(json.dumps(data["feedback"], indent=2))
