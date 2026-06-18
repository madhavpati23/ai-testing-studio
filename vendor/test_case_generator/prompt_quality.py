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
    "clear task": (18, "Lead with a clear action verb so the ask is unambiguous."),
    "role / domain": (15, "Name the role/domain to assume (e.g. 'Act as a project manager')."),
    "audience & purpose": (15, "Say who it's for and why (the audience and the goal)."),
    "key details": (15, "Include the specific points to cover; avoid vague references."),
    "constraints (tone & length)": (15, "State the tone, the length, and what to avoid."),
    "output format": (14, "Specify the output format (email, bullet points, report, JSON...)."),
    "example / style": (8, "Give a short example or a style sample to match."),
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
            r"convert|rewrite|evaluate|test|build|find|identify|plan|recommend|help)\w*\b", low)
        ) or p.endswith("?"),
        "role / domain": bool(re.search(
            r"\b(you are|act as|as an?|role|expert in|specialist|persona|responsible for)\b", low)),
        "audience & purpose": bool(re.search(
            r"\b(audience|for an?|for the|reader|manager|client|team|customer|user|stakeholder|"
            r"recipient|purpose|so that|goal|in order to|to help|intended|status update|escalation)\b", low)),
        "key details": (n >= 12 and vague <= 2) or bool(re.search(
            r"\b(include|details?|points?|cover|specifically|key|the following)\b", low)),
        "constraints (tone & length)": bool(re.search(
            r"\b(must|should|only|do not|don'?t|avoid|limit|at most|no more than|within|"
            r"in \d+ (words|sentences|bullets|lines|paragraphs)|tone|formal|concise|professional|"
            r"polite|assertive|brief|detailed|length|accurate|complete)\b", low)),
        "output format": bool(re.search(
            r"\b(json|yaml|table|markdown|bullet|numbered|list|format|schema|csv|step|heading|"
            r"email|report|script|release notes|slide|memo|paragraph)\b", low)),
        "example / style": bool(re.search(
            r"(e\.?g\.?\b|for example|for instance|\bexample\b|\bsample\b|such as|match this style|like the following)", low)),
    }


def _level(score: int) -> tuple[str, str]:
    if score >= 85:
        return "Strong", "This is a strong, well-scoped prompt."
    if score >= 65:
        return "Good", "Good prompt — a couple of quick wins below."
    if score >= 45:
        return "Fair", "Workable, but a few additions would help a lot."
    return "Needs work", "This prompt is likely to give inconsistent results."


# task keywords -> a sensible role for the rewrite
_ROLE_HINTS = [
    (("project", "manager", "scrum", "delivery", "sprint", "stakeholder"), "an experienced project manager"),
    (("iam", "identity", "access management", "okta", "active directory", "provisioning"), "an IAM specialist"),
    (("instructor", "teach", "lesson", "course", "training", "tutorial", "student", "learn", "document", "guide"),
     "an experienced instructor and technical writer"),
    (("code", "function", "program", "bug", "refactor", "api", "python", "javascript", "sql", "implement"),
     "a senior software engineer"),
    (("test", "qa", "quality", "validation"), "a senior QA engineer"),
    (("email", "message", "reply", "customer", "support"), "a professional communications assistant"),
    (("summar", "tl;dr", "digest"), "an expert summarizer"),
    (("data", "analy", "metric", "chart", "report"), "a data analyst"),
    (("legal", "contract", "policy", "compliance"), "a careful legal assistant"),
    (("market", "copy", "ad", "campaign", "blog"), "a skilled marketing copywriter"),
]


def _infer_role(low: str) -> str:
    for keys, role in _ROLE_HINTS:
        if any(k in low for k in keys):
            return role
    return "an expert assistant"


def _clean_task(prompt: str) -> str:
    t = " ".join(prompt.strip().split())
    t = re.sub(r"\bi\b", "I", t)        # fix a lone lowercase 'i'
    t = t.rstrip(". ")
    return (t[0].upper() + t[1:]) if t else "Complete the request"


def _rewrite(prompt: str, present: dict[str, bool]) -> str:
    """Produce a concrete, ready-to-use rewrite of the prompt (no placeholders).

    Keeps the user's task, adds a sensible role and the missing best-practice
    elements as real instructions. A Claude-backed rewrite (assess_llm) tailors
    it further to the exact content.
    """
    low = prompt.lower()
    lines: list[str] = []
    if not present["role / domain"]:
        lines.append(f"You are {_infer_role(low)}.")
    lines.append(f"{_clean_task(prompt)}.")
    if not present["audience & purpose"]:
        lines.append("- Audience & purpose: write it for a professional audience and make the goal explicit.")
    if not present["key details"]:
        lines.append("- Cover the key points the request needs, and state any assumptions you make.")
    if not present["constraints (tone & length)"]:
        lines.append("- Use a professional, concise tone; keep it as brief as the task allows; be accurate and complete.")
    if not present["output format"]:
        lines.append("- Format it appropriately (e.g. email, bullet points, or a short report).")
    if not present["example / style"]:
        lines.append("- Include a brief example to illustrate where it helps.")
    lines.append("- Provide the final, ready-to-use answer; if anything is unclear, ask instead of guessing.")
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
    example = "" if score >= 85 else _rewrite(prompt, present)
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
