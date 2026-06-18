"""Score how well-written a prompt is — concisely, without lecturing.

Design intent: give a number, a one-line verdict, the prompt's strengths, and AT
MOST 3 high-impact suggestions (phrased as "consider", not "you must"). If the
prompt is already strong, say so and stop. The goal is a quick signal, not a
prompt-engineering tutorial.

`assess()` is a deterministic heuristic (offline, testable). `assess_llm()` asks
Claude for a short critique when ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# dimension -> (weight, gentle suggestion when it's missing)
_DIMENSIONS = {
    "clear task": (25, "Lead with a clear action verb so the ask is unambiguous."),
    "context/role": (20, "Add a line of context or a role (e.g. 'You are a...') to frame it."),
    "specific": (20, "Replace vague references ('it', 'this') with the specific subject."),
    "constraints": (15, "State any constraints - length, tone, or what to avoid."),
    "output format": (12, "Say what the output should look like (e.g. JSON, a bulleted list)."),
    "example": (8, "A short example of a good answer can sharpen results."),
}


# dimensions for AGENT INSTRUCTIONS (a system prompt / agent config), with the
# gentle suggestion shown when each is missing.
_INSTRUCTION_DIMENSIONS = {
    "role / purpose": (15, "State who the agent is and its goal (e.g. 'You are a ... that ...')."),
    "tools / actions": (20, "List the tools/actions it may use, and when to use each."),
    "permissions / boundaries": (20, "Define what it must NOT do and any authorization limits."),
    "safety / refusal rules": (15, "Say when to refuse - unauthorized, unsafe, or out-of-scope requests."),
    "data sources": (10, "Name the source(s) of truth and require it to cite them."),
    "output format": (10, "Specify the response format or structure."),
    "uncertainty handling": (10, "Tell it to ask or say 'I don't know' instead of guessing."),
}


@dataclass
class PromptScore:
    score: int
    level: str
    summary: str
    strengths: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    present: dict[str, bool] = field(default_factory=dict)
    example: str = ""   # a concrete improved shape, built around the user's ask


def _checks(prompt: str) -> dict[str, bool]:
    p = prompt.strip()
    low = p.lower()
    n = len(re.findall(r"\w+", low))
    vague = len(re.findall(r"\b(it|this|that|thing|stuff|something|etc)\b", low))
    return {
        "clear task": bool(re.search(
            r"\b(write|create|generate|list|summari|explain|translate|classif|extract|"
            r"review|analy(?:z|s)e|compare|draft|fix|implement|design|answer|describe|"
            r"convert|rewrite|evaluate|test|build|find|identify|plan|recommend)\w*\b", low)
        ) or p.endswith("?"),
        "context/role": bool(re.search(
            r"\b(you are|act as|context|background|given|as an?|the goal|we are|i am|i'm)\b", low)
        ) or n >= 25,
        "specific": n >= 12 and vague <= 2,
        "constraints": bool(re.search(
            r"\b(must|should|only|do not|don'?t|avoid|limit|at most|no more than|within|"
            r"in \d+ (words|sentences|bullets|lines)|tone|formal|concise|professional)\b", low)),
        "output format": bool(re.search(
            r"\b(json|yaml|table|markdown|bullet|numbered list|list|format|schema|csv|"
            r"step by step|steps|headings?)\b", low)),
        "example": bool(re.search(r"(e\.?g\.?\b|for example|for instance|example\s*:|sample)", low)),
    }


def _level(score: int) -> tuple[str, str]:
    if score >= 85:
        return "Strong", "This is a strong, well-scoped prompt."
    if score >= 65:
        return "Good", "Good prompt — a couple of quick wins below."
    if score >= 45:
        return "Fair", "Workable, but a few additions would help a lot."
    return "Needs work", "This prompt is likely to give inconsistent results."


def _example_shape(prompt: str, present: dict[str, bool]) -> str:
    """Build a concrete 'good' shape around the user's actual ask.

    Fills only the parts that are missing, so it reads as a template to complete
    — not a lecture. Angle-bracket slots are for the user to fill in.
    """
    task = prompt.strip().rstrip(".")
    lines = []
    if not present["context/role"]:
        lines.append("You are an expert assistant for this task.")
    lines.append(f"Task: {task}.")
    if not present["constraints"]:
        lines.append("Constraints: be accurate and concise; if unsure, say so.")
    if not present["output format"]:
        lines.append("Output: return a clear, structured response (e.g. short bullets or JSON).")
    if not present["example"]:
        lines.append("Example of a good answer: <add one short example>.")
    return "\n".join(lines)


def assess(prompt: str) -> PromptScore:
    if not prompt.strip():
        return PromptScore(0, "Empty", "No prompt provided.", suggestions=["Enter a prompt to score it."])
    present = _checks(prompt)
    score = sum(_DIMENSIONS[d][0] for d, ok in present.items() if ok)
    level, summary = _level(score)
    strengths = [d for d, ok in present.items() if ok]
    # only the highest-impact gaps, capped at 3, and never when already strong
    missing = sorted((d for d, ok in present.items() if not ok),
                     key=lambda d: _DIMENSIONS[d][0], reverse=True)
    suggestions = [] if score >= 85 else [_DIMENSIONS[d][1] for d in missing[:3]]
    example = "" if score >= 85 else _example_shape(prompt, present)
    return PromptScore(score=score, level=level, summary=summary, strengths=strengths,
                       suggestions=suggestions, present=present, example=example)


def _instruction_checks(text: str) -> dict[str, bool]:
    low = text.lower()

    def has(*w: str) -> bool:
        return any(x in low for x in w)

    return {
        "role / purpose": has("you are", "your role", "act as", "purpose", "responsible for", "assistant that", "agent that"),
        "tools / actions": has("tool", "function", "action", "api", "call", "search", "create", "update", "retrieve", "use the"),
        "permissions / boundaries": has("must not", "do not", "never", "not allowed", "only", "authorized", "permission", "scope", "restricted", "limit"),
        "safety / refusal rules": has("refuse", "decline", "reject", "unsafe", "confirm before", "do not reveal", "not share", "without authorization", "out of scope"),
        "data sources": has("source", "knowledge base", "database", "jira", "confluence", "snowflake", "retrieve", "ground", "cite", "documentation", "system of record"),
        "output format": has("format", "respond with", "json", "structure", "bullet", "steps", "template", "schema", "reply in"),
        "uncertainty handling": has("if unsure", "if you don't know", "i don't know", "ask", "clarif", "unknown", "escalate", "not sure", "insufficient", "no data"),
    }


def _instruction_example(present: dict[str, bool]) -> str:
    slots = {
        "role / purpose": "You are <role> whose goal is <goal>.",
        "tools / actions": "Tools you may use: <list>. Use <tool> when <condition>.",
        "permissions / boundaries": "You must not <forbidden actions>; act only within <scope/authorization>.",
        "safety / refusal rules": "Refuse requests that are unauthorized, unsafe, or out of scope.",
        "data sources": "Use <source of truth> as the only data source, and cite it.",
        "output format": "Respond in <format - e.g. concise bullets or JSON>.",
        "uncertainty handling": "If data is missing or the request is ambiguous, ask or say you don't know - never guess.",
    }
    return "\n".join(text for dim, text in slots.items() if not present[dim])


def assess_instructions(text: str) -> PromptScore:
    """Score an agent's instructions/configuration. Heuristic, offline."""
    if not text.strip():
        return PromptScore(0, "Empty", "No instructions provided.",
                           suggestions=["Paste the agent's instructions to score them."])
    present = _instruction_checks(text)
    score = sum(_INSTRUCTION_DIMENSIONS[d][0] for d, ok in present.items() if ok)
    level, summary = _level(score)
    strengths = [d for d, ok in present.items() if ok]
    missing = sorted((d for d, ok in present.items() if not ok),
                     key=lambda d: _INSTRUCTION_DIMENSIONS[d][0], reverse=True)
    suggestions = [] if score >= 85 else [_INSTRUCTION_DIMENSIONS[d][1] for d in missing[:3]]
    example = "" if score >= 85 else _instruction_example(present)
    return PromptScore(score=score, level=level, summary=summary, strengths=strengths,
                       suggestions=suggestions, present=present, example=example)


def assess_llm(prompt: str) -> PromptScore:
    """Short, non-preachy critique from Claude. Needs ANTHROPIC_API_KEY."""
    import json
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required for assess_llm")
    import anthropic

    client = anthropic.Anthropic()
    system = (
        "You rate how well-written a prompt is. Be concise and encouraging, not "
        "preachy. Reply with ONLY JSON: {\"score\": 0-100, \"summary\": \"<one "
        "sentence>\", \"suggestions\": [\"<at most 3, phrased as 'consider…'>\"], "
        "\"improved\": \"<a rewritten version of THIS prompt that keeps the user's "
        "intent but applies best practices>\"}. If the prompt is already strong, "
        "return an empty suggestions list and set improved to an empty string."
    )
    resp = client.messages.create(
        model=os.environ.get("PRS_JUDGE_MODEL", "claude-opus-4-8"),
        max_tokens=600, thinking={"type": "adaptive"}, system=system,
        messages=[{"role": "user", "content": f"PROMPT:\n{prompt}"}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    data = json.loads(text)
    score = int(data["score"])
    level, _ = _level(score)
    return PromptScore(score=score, level=level, summary=str(data.get("summary", "")),
                       suggestions=list(data.get("suggestions", []))[:3],
                       example=str(data.get("improved", "")))
