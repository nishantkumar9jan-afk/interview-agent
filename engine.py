"""
engine.py — the brain of the AI Interview Agent.

Design summary
---------------
1. CandidateProfile turns the raw candidate mission history into a ranked
   interview plan: days the candidate visibly struggled with (many attempts,
   or explicitly failed) are prioritized, one easy "warm-up" day opens the
   interview, and a couple of first-try days are mixed in later as a sanity
   check against the weak-day findings. This is the "adaptive" part of the
   brief — the interview a struggling candidate gets is a different shape
   than the one a strong candidate gets, even though both pull from the
   same curriculum.

2. Every question is grounded in that day's actual objectives/tools from
   curriculum.json — never generic ("tell me about AI") questions.

3. After every answer, a lightweight signal analyzer (word count, hedging
   language, keyword grounding against the day's tools/objectives) scores
   the response. That score decides whether to ask a follow-up (dig into a
   vague or thin answer) or move to the next topic — and it doubles as the
   evidence base for the final structured feedback, so feedback isn't a
   black box even when no LLM key is configured.

4. An LLM (Anthropic Claude) is used opportunistically to make question
   phrasing, follow-ups, and the feedback narrative sharper and more
   conversational. It is never load-bearing: every path has a deterministic
   fallback so the endpoint is fully functional with zero external calls.
"""

import json
import re
import threading
import uuid
from pathlib import Path

from llm import call_llm, call_llm_json, llm_available

DATA_DIR = Path(__file__).parent / "data"

MIN_QUESTIONS = 8
MIN_DAYS = 4
MAX_QUESTIONS = 12          # hard cap so a struggling candidate's interview still ends
MAX_FOLLOWUPS_PER_DAY = 1   # keeps total question count bounded and predictable

HEDGE_PHRASES = [
    "not sure", "i don't know", "i dont know", "no idea", "not familiar",
    "never used", "skip this", "pass on this", "not confident", "i think maybe",
    "i'm not certain", "im not certain", "can't remember", "cant remember",
]

# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------
class Curriculum:
    def __init__(self, path=DATA_DIR / "curriculum.json"):
        raw = json.loads(Path(path).read_text())
        self.cohort = raw.get("cohort", "")
        self.days = {d["day"]: d for d in raw["days"]}

    def get(self, day_number: int):
        return self.days.get(day_number)

    def all_day_numbers(self):
        return sorted(self.days.keys())


CURRICULUM = Curriculum()

# ---------------------------------------------------------------------------
# Candidate analysis
# ---------------------------------------------------------------------------
class CandidateProfile:
    """
    Wraps one candidate record (member + missions + signals, matching the
    shape of an entry in candidates.json) and derives an interview plan.
    """

    def __init__(self, candidate: dict):
        # Be lenient about shape: accept {member, missions, signals} (the
        # documented candidate.json schema) but don't hard-fail on minor
        # variations — a missing "signals" block, for instance, shouldn't
        # break session start.
        self.member = candidate.get("member", candidate)
        self.missions = candidate.get("missions", [])
        self.signals = candidate.get("signals", {})

        self.name = self.member.get("name", "Candidate")
        self.job_role = self.member.get("jobRole", "the role")
        self.years_experience = self.member.get("yearsExperience", 0)

    def _mission_severity(self, m: dict) -> int:
        """Higher = more evidence the candidate struggled with this day."""
        if m.get("skipped"):
            return 100  # untested entirely — worth probing
        if m.get("passed") is False:
            return 90 + m.get("attempts", 0)
        attempts = m.get("attempts", 1)
        if attempts >= 4:
            return 50 + attempts
        return 0  # comfortably passed, not a priority target

    def build_interview_plan(self, min_days: int = MIN_DAYS) -> list[int]:
        """
        Returns an ordered list of curriculum day numbers to interview on.
        Order = [1 warm-up strong day, ...weak days by severity desc,
                 ...1-2 strong days mixed in near the end as a cross-check].
        Always returns at least `min_days` distinct valid curriculum days,
        backfilling from core curriculum days if the candidate's own
        mission history is too sparse (e.g. an early-exit candidate).
        """
        scored = []
        strong = []
        for m in self.missions:
            day_num = m.get("day")
            if CURRICULUM.get(day_num) is None:
                continue
            sev = self._mission_severity(m)
            if sev > 0:
                scored.append((sev, day_num))
            elif m.get("passed"):
                strong.append(day_num)

        scored.sort(key=lambda x: -x[0])
        weak_days = [d for _, d in scored]

                plan: list[int] = []

        # Warm-up: earliest strong day (usually Day 1) sets a comfortable tone.
        if strong:
            warmup = min(strong)
            plan.append(warmup)
            strong.remove(warmup)

        for d in weak_days:
            if d not in plan:
                plan.append(d)

        # Mix in up to 2 strong days later as a cross-check against the
        # weak-day findings.
        for d in sorted(strong)[:2]:
            if d not in plan:
                plan.append(d)

        # Backfill from core, broadly-applicable curriculum days.
        core_backfill = [7, 10, 12, 16, 22, 31, 8, 21, 27]

        for d in core_backfill:
            if len(plan) >= MIN_QUESTIONS:
                break
            if d not in plan and CURRICULUM.get(d):
                plan.append(d)

        return plan[:MIN_QUESTIONS]   # cap plan length; follow-ups will fill out the rest

    def signal_for_day(self, day_number: int) -> dict:
        for m in self.missions:
            if m.get("day") == day_number:
                return m
        return {}


# ---------------------------------------------------------------------------
# Response scoring (used for follow-up decisions AND feedback evidence)
# ---------------------------------------------------------------------------
def score_response(answer: str, day: dict) -> dict:
    text = (answer or "").strip()
    lower = text.lower()
    words = re.findall(r"[a-zA-Z']+", lower)
    word_count = len(words)

    hedge = any(phrase in lower for phrase in HEDGE_PHRASES)

    vocab = set()
    for t in day.get("tools", []):
        vocab.update(re.findall(r"[a-zA-Z']+", t.lower()))
    for o in day.get("objectives", []):
        vocab.update(re.findall(r"[a-zA-Z']+", o.lower()))
    vocab = {w for w in vocab if len(w) > 3}  # drop short/common tokens
    overlap = len(set(words) & vocab)

    length_score = min(word_count / 40, 1.0)          # 40+ words = full credit
    keyword_score = min(overlap / 3, 1.0)              # 3+ grounded terms = full credit
    hedge_penalty = 0.4 if hedge else 0.0

    confidence = max(0.0, min(1.0, 0.5 * length_score + 0.5 * keyword_score - hedge_penalty))

    return {
        "word_count": word_count,
        "hedge": hedge,
        "keyword_overlap": overlap,
        "confidence": round(confidence, 2),
        "thin": word_count < 15 or hedge,
    }


# ---------------------------------------------------------------------------
# Question / follow-up generation
# ---------------------------------------------------------------------------
_TEMPLATES_BY_TYPE = {
    "SETUP": "Tell me about your experience with this: \"{obj}\" — what tripped you up the first time, and where did {tools} fit into that workflow?",
    "BUILD": "Walk me through this part of your work: \"{obj}\" — what did you use {tools} for, and what would you do differently if you rebuilt it today?",
    "LEARN": "How would you explain this to a teammate who's never seen {tools}: \"{obj}\"? Where do people usually get it wrong?",
    "AI_CORE": "Let's dig into the fundamentals behind this: \"{obj}\" — how does {tools} actually make that possible? Walk me through the mechanics.",
    "SHIP_IT": "This looks like a 'ship it' milestone: \"{obj}\", using {tools}. What broke first under real conditions, and how did you catch it?",
    "OPTIMIZE": "You worked on this: \"{obj}\". What was your baseline before optimizing, what did {tools} let you measure, and what was the actual improvement?",
    "CAPSTONE": "For your capstone, how did {tools} come together to let you accomplish this: \"{obj}\"? What's the one decision you'd defend hardest in a design review?",
}


def _pick_objective(day: dict, avoid: set) -> str:
    objs = [o for o in day.get("objectives", []) if o not in avoid]
    return (objs or day.get("objectives", ["this topic"]))[0]


def generate_question(day: dict, candidate: CandidateProfile, is_warmup: bool, asked_objs: set) -> str:
    obj = _pick_objective(day, asked_objs)
    tools = ", ".join(day.get("tools", [])[:3]) or "the tools from that day"
    day_type = day.get("type", "BUILD")
    signal = candidate.signal_for_day(day["day"])

    if llm_available():
        struggle_note = ""
        if signal.get("skipped"):
            struggle_note = "The candidate SKIPPED this day entirely in the curriculum — treat this as untested ground, ask an accessible but real question."
        elif signal.get("passed") is False:
            struggle_note = f"The candidate FAILED this mission after {signal.get('attempts', '?')} attempts — probe for genuine understanding, don't let a rehearsed answer pass unchallenged."
        elif signal.get("attempts", 1) >= 4:
            struggle_note = f"The candidate needed {signal.get('attempts')} attempts to pass this mission — there may be a shaky foundation here worth probing."
        elif signal.get("attempts", 1) == 1:
            struggle_note = "The candidate passed this on the first attempt — this can be a slightly harder, more probing question."

        prompt = f"""You are conducting a technical interview for a {candidate.job_role} candidate ({candidate.years_experience} yrs experience) named {candidate.name}.

Curriculum day {day['day']}: "{day['title']}" ({day_type})
Objectives: {day.get('objectives')}
Tools/concepts: {day.get('tools')}
{struggle_note}

Write ONE open-ended, conversational interview question (1-3 sentences) that tests real understanding of this day's material — not a yes/no question, not asking them to write code, but something that reveals whether they actually understand the concepts and tradeoffs. {"This is the opening question of the interview, so start with a brief, warm one-line welcome before it." if is_warmup else "Do not include any greeting — just the question."}
Return ONLY the question text (and welcome line if applicable), nothing else."""
        result = call_llm(
            "You are a sharp, friendly, no-nonsense senior technical interviewer.",
            prompt,
            max_tokens=200,
        )
        if result:
            return result.strip()

    # ---- deterministic fallback ----
    template = _TEMPLATES_BY_TYPE.get(day_type, _TEMPLATES_BY_TYPE["BUILD"])
    q = template.format(obj=obj, tools=tools)
    if is_warmup:
        return f"Hi {candidate.name.split()[0]}, thanks for making time today. Let's start easy. {q}"
    return q


def generate_followup(day: dict, question: str, answer: str, score: dict) -> str | None:
    """Returns a follow-up question string, or None to move on."""
    if not score["thin"]:
        return None

    if llm_available():
        prompt = f"""Interview question asked: "{question}"
Candidate's answer: "{answer}"
Curriculum context — day {day['day']} "{day['title']}", objectives: {day.get('objectives')}, tools: {day.get('tools')}

The answer seems thin, vague, or hedged. Write ONE short, specific follow-up question (1-2 sentences) that gives the candidate a concrete chance to demonstrate real understanding — reference a specific tool or objective from the context above. Return ONLY the follow-up question text."""
        result = call_llm("You are a sharp, friendly senior technical interviewer.", prompt, max_tokens=150)
        if result:
            return result.strip()

    # deterministic fallback: probe a specific tool they haven't mentioned
    tools = day.get("tools", [])
    tool = tools[0] if tools else "the approach"
    return f"Let's go one level deeper — specifically, how did you use {tool} here, and what would break if you removed it?"


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------
def generate_feedback(candidate: CandidateProfile, transcript: list[dict], day_scores: dict) -> dict:
    """
    day_scores: {day_number: {"day_title":..., "confidences":[...], "thin_count":int}}
    """
    ranked = sorted(
        day_scores.items(),
        key=lambda kv: (sum(kv[1]["confidences"]) / max(len(kv[1]["confidences"]), 1)),
        reverse=True,
    )
    strengths_days = [d for d, s in ranked if (sum(s["confidences"]) / max(len(s["confidences"]), 1)) >= 0.55][:3]
    gap_days = [d for d, s in ranked if (sum(s["confidences"]) / max(len(s["confidences"]), 1)) < 0.4][:3]

    if llm_available():
        prompt = f"""Here is a full technical interview transcript for {candidate.name}, a {candidate.job_role} candidate ({candidate.years_experience} yrs experience).

Transcript:
{json.dumps(transcript, indent=2)}

Per-topic confidence scores (0-1, higher = stronger answer), derived from response length, terminology grounding, and hedging language:
{json.dumps({str(k): round(sum(v['confidences'])/max(len(v['confidences']),1), 2) for k, v in day_scores.items()}, indent=2)}

Write structured interview feedback as JSON with exactly these keys:
- "summary": 2-4 sentence overall assessment, direct and specific (not generic praise).
- "strengths": array of 2-4 concise, specific bullet points citing actual curriculum topics discussed.
- "gaps": array of 2-4 concise, specific bullet points on weaker areas, citing actual curriculum topics.
- "next": array of 2-4 concrete, actionable next steps (what to study/practice/revisit) tied to the gaps.
Keep every array item under 20 words."""
        result = call_llm_json("You are a senior technical interviewer writing hiring feedback.", prompt, max_tokens=700)
        if result and all(k in result for k in ("summary", "strengths", "gaps", "next")):
            return result

    # ---- deterministic fallback ----
    def title_for(d):
        day = CURRICULUM.get(d)
        return day["title"] if day else f"Day {d}"

    total_answers = sum(len(s["confidences"]) for s in day_scores.values())
    avg_conf = sum(sum(s["confidences"]) for s in day_scores.values()) / max(total_answers, 1)

    summary = (
        f"{candidate.name} covered {len(day_scores)} curriculum topics across the interview with an average "
        f"response confidence of {round(avg_conf * 100)}%. "
        + ("Responses were generally well-grounded and specific." if avg_conf >= 0.55
           else "Several responses were thin or hedged, suggesting gaps worth probing further in a follow-up round.")
    )

    strengths = [f"Solid, well-grounded answers on \"{title_for(d)}\"." for d in strengths_days] or \
                ["Completed the full interview without disengaging from any topic."]
    gaps = [f"Answers on \"{title_for(d)}\" were thin, vague, or hedged." for d in gap_days] or \
           ["No major gaps stood out; consider a deeper technical round to differentiate further."]
    next_steps = [f"Revisit the objectives for \"{title_for(d)}\" and practice explaining the reasoning, not just the steps." for d in gap_days] or \
                 ["Proceed to a hands-on technical assessment to validate depth beyond conversation."]

    return {"summary": summary, "strengths": strengths, "gaps": gaps, "next": next_steps}


# ---------------------------------------------------------------------------
# Session state machine
# ---------------------------------------------------------------------------
class InterviewSession:
    def __init__(self, session_id: str, candidate: dict):
        self.session_id = session_id
        self.candidate = CandidateProfile(candidate)
        self.plan = self.candidate.build_interview_plan()
        self.plan_index = 0
        self.followups_used = 0
        self.asked_objectives_by_day: dict[int, set] = {}
        self.current_day = None
        self.current_question = None
        self.questions_asked = 0
        self.distinct_days: set[int] = set()
        self.transcript: list[dict] = []
        self.day_scores: dict[int, dict] = {}
        self.done = False

    def _advance_to_next_day(self) -> bool:
        """Returns False if the plan is exhausted."""
        if self.plan_index >= len(self.plan):
            return False
        self.current_day = self.plan[self.plan_index]
        self.plan_index += 1
        self.followups_used = 0
        self.asked_objectives_by_day.setdefault(self.current_day, set())
        return True

    def start(self) -> str:
        ok = self._advance_to_next_day()
        if not ok:
            self.done = True
            return "We don't have enough curriculum data to run this interview."
        day = CURRICULUM.get(self.current_day)
        question = generate_question(day, self.candidate, is_warmup=True, asked_objs=set())
        self.current_question = question
        self.questions_asked += 1
        self.distinct_days.add(self.current_day)
        self.transcript.append({"role": "interviewer", "day": self.current_day, "content": question})
        return question

    def _should_end(self) -> bool:
        if self.questions_asked >= MAX_QUESTIONS:
            return True
        plan_exhausted = self.plan_index >= len(self.plan) and self.followups_used >= MAX_FOLLOWUPS_PER_DAY
        met_minimums = self.questions_asked >= MIN_QUESTIONS and len(self.distinct_days) >= MIN_DAYS
        return met_minimums and plan_exhausted

    def submit_answer(self, message: str) -> dict:
        day = CURRICULUM.get(self.current_day)
        score = score_response(message, day)
        self.transcript.append({"role": "candidate", "day": self.current_day, "content": message})

        bucket = self.day_scores.setdefault(
            self.current_day, {"day_title": day["title"], "confidences": [], "thin_count": 0}
        )
        bucket["confidences"].append(score["confidence"])
        if score["thin"]:
            bucket["thin_count"] += 1

        self.asked_objectives_by_day[self.current_day].add(
            _pick_objective(day, set())
        )

        # Decide: follow-up on this day, or move to next day / end.
        followup = None
        if self.followups_used < MAX_FOLLOWUPS_PER_DAY:
            followup = generate_followup(day, self.current_question, message, score)

        if followup and not self._should_end():
            self.followups_used += 1
            self.current_question = followup
            self.questions_asked += 1
            self.transcript.append({"role": "interviewer", "day": self.current_day, "content": followup})
            return {"reply": followup, "done": False}

        if self._should_end():
            self.done = True
            feedback = generate_feedback(self.candidate, self.transcript, self.day_scores)
            return {"reply": "Interview completed. Thanks for your time!", "done": True, "feedback": feedback}

                # move to next day
        advanced = self._advance_to_next_day()

        if not advanced:
            if self.questions_asked < MIN_QUESTIONS:
                return {
                    "reply": "Please continue the interview.",
                    "done": False
                }

            self.done = True
            feedback = generate_feedback(
                self.candidate,
                self.transcript,
                self.day_scores
            )

            return {
                "reply": "Interview completed. Thanks for your time!",
                "done": True,
                "feedback": feedback
            }

        day = CURRICULUM.get(self.current_day)
        question = generate_question(
            day,
            self.candidate,
            is_warmup=False,
            asked_objs=self.asked_objectives_by_day.get(
                self.current_day, set()
            ),
        )
        self.current_question = question
        self.questions_asked += 1
        self.distinct_days.add(self.current_day)
        self.transcript.append({
            "role": "interviewer",
            "day": self.current_day,
            "content": question
        })
        return {"reply": question, "done": False}

    

        

# ---------------------------------------------------------------------------
# In-memory session store (no persistence required per spec)
# ---------------------------------------------------------------------------
class SessionStore:
    def __init__(self):
        self._sessions: dict[str, InterviewSession] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str, candidate: dict) -> InterviewSession:
        with self._lock:
            session = InterviewSession(session_id, candidate)
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> InterviewSession | None:
        with self._lock:
            return self._sessions.get(session_id)


SESSIONS = SessionStore()
