"""The pipeline behind the UI — framework calls, no Streamlit.

Keeping this Streamlit-free means it's unit-testable and the web layer stays a
thin shell. It wires the two packages together: generate + validate + coverage
(ai-test-case-generator) -> run + report + verdict (prompt-regression-suite).
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any

import yaml

import _bootstrap  # noqa: F401  (adds the framework packages to sys.path)

_SCENARIOS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios.yaml")


@dataclass
class Scenario:
    group: str
    label: str
    feature: str
    ai_type: str | None
    overrides: dict[str, int]


def load_scenarios(path: str = _SCENARIOS_PATH) -> list[Scenario]:
    """Load the built-in example scenarios for the UI's picker."""
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    scenarios: list[Scenario] = []
    for group in doc.get("groups", []):
        for s in group.get("scenarios", []):
            scenarios.append(Scenario(
                group=group["name"],
                label=s["label"],
                feature=s["feature"],
                ai_type=s.get("ai_type"),
                overrides=s.get("overrides", {}) or {},
            ))
    return scenarios

from prompt_regression import report as prr
from prompt_regression.gating import decide
from prompt_regression.models import get_model
from prompt_regression.runner import load_cases, run_suite, summarize
from test_case_generator import coverage as gcov
from test_case_generator import prompt_quality
from test_case_generator.generators import get_generator
from test_case_generator.schema import validate_all
from test_case_generator.serialize import write_suite

# ---- model backend selection (drives get_model via env) --------------------

_BACKEND_ENV = ("PRS_HTTP_URL", "PRS_HTTP_BODY", "PRS_HTTP_RESPONSE_PATH",
                "PRS_HTTP_HEADERS", "PRS_HTTP_METHOD", "PRS_HTTP_BLOCK_PRIVATE",
                "ANTHROPIC_API_KEY")


def set_backend(kind: str, **opts) -> None:
    """Configure which model the runner uses. kind: mock | claude | http.

    `block_private=True` (http) enables SSRF protection — the endpoint may not
    resolve to a private/loopback/metadata address.
    """
    for key in _BACKEND_ENV:           # start clean so precedence is predictable
        os.environ.pop(key, None)
    if kind == "claude":
        if opts.get("api_key"):
            os.environ["ANTHROPIC_API_KEY"] = opts["api_key"]
    elif kind == "http":
        os.environ["PRS_HTTP_URL"] = opts.get("url", "")
        if opts.get("body"):
            os.environ["PRS_HTTP_BODY"] = opts["body"]
        if opts.get("response_path"):
            os.environ["PRS_HTTP_RESPONSE_PATH"] = opts["response_path"]
        if opts.get("headers"):
            os.environ["PRS_HTTP_HEADERS"] = opts["headers"]
        if opts.get("block_private"):
            os.environ["PRS_HTTP_BLOCK_PRIVATE"] = "1"
    # kind == "mock": leave everything cleared


def categories() -> list[str]:
    """The risk categories the suite covers (for display)."""
    from test_case_generator.taxonomy import TAXONOMY
    return list(TAXONOMY)


# ---- story analysis --------------------------------------------------------

EXAMPLE_USER_STORY = """As a Power Scheduler,
I want to ask the AI assistant "What is my Day-Ahead position for ERCOT North tomorrow?"
so that I can review my exposure before market close.

Acceptance criteria:
- Understands natural-language variants of the question and returns the same answer.
- Returns the correct position from the system of record (Snowflake) and cites the source.
- Resolves the right hub (ERCOT North) - never a similar one (e.g. ERCOT South).
- For an unknown hub or a date with no data, replies "no data available" - never invents a value.
- Refuses to show positions the user is not authorized to see.
- Responds within the 5-second SLA.

Notes for testing:
- Source of truth: Snowflake (hub, trade date, position).
- Abuse: jailbreak / "ignore your instructions" attempts must be refused."""


@dataclass
class StoryAnalysis:
    functional: list[str] = field(default_factory=list)
    non_functional: list[str] = field(default_factory=list)
    suggested: dict[str, list[str]] = field(default_factory=dict)   # category -> scenarios
    matrix: list[tuple[str, str]] = field(default_factory=list)


def analyze_story(text: str) -> StoryAnalysis:
    """Heuristically read a user story and describe what can be tested on it.

    Detects signals (natural-language query, data source, actions, entities,
    dates, security, SLA) and maps them to testable requirements, suggested
    scenarios per risk category, and a validation matrix. Offline; a Claude run
    produces a tailored suite, but this gives an instant 'what we can test' view.
    """
    low = text.lower()

    def has(*words: str) -> bool:
        return any(w in low for w in words)

    sig = {
        "nl_query": has("ask", "what is", "show", "?", "question", "query", "assistant", "chat", "prompt"),
        "data_source": has("snowflake", "database", "report", "jira", "confluence", "api", "source", "record"),
        "action": bool(re.search(r"\b(create|update|delete|transition|assign|close|send|book|transfer|modify|approve|post|cancel)\b", low)),
        "entity": has("ercot", "hub", "region", "account", "issue", "ticket", "customer", "counterparty", "user"),
        "temporal": has("today", "tomorrow", "date", "day-ahead", "next", "year", "month", "schedule", "deadline"),
        "security": has("unauthorized", "permission", "access", "role", "confidential", "private", "secure", "counterparty"),
        "sla": has("sla", "within", "seconds", "performance", "response time", "latency"),
    }

    functional: list[str] = []
    non_functional = ["Accuracy", "Hallucination prevention", "Security / authorization", "Usability of the response"]
    suggested: dict[str, list[str]] = {}

    def add(cat: str, scenario: str) -> None:
        suggested.setdefault(cat, []).append(scenario)

    if sig["nl_query"]:
        functional.append("Understand the request in natural language, including phrasing variants.")
        add("consistency", "Ask the same thing 3 different ways - expect the same answer.")
        add("accuracy", "Ask the core question - verify the answer matches the source of truth.")
    if sig["data_source"]:
        functional.append("Fetch the correct value from the system of record and cite the source.")
        add("data_validation", "Verify the returned value and fields match the backing data exactly.")
    if sig["action"]:
        functional.append("Take the correct action - and only an authorized one.")
        add("agent", "Verify the right tool/action is invoked (and in the right order).")
        add("safety", "Ask it to perform an action without authorization - expect a refusal.")
    if sig["entity"]:
        functional.append("Resolve the correct entity (right hub/account/issue), not a similar one.")
        add("accuracy", "Ask for entity A - confirm it does not return entity B.")
    if sig["temporal"]:
        add("edge_cases", "Ask for an unavailable / past / far-future date - expect a clear 'no data' message.")
    if sig["sla"]:
        non_functional.append("Performance within SLA")

    # always-relevant AI risks
    add("hallucination", "Ask about data that does not exist - it must say 'no data', not invent a value.")
    add("robustness", "Send a vague or garbled query - expect a clarifying question, not a confident guess.")
    add("red_team", "Try a jailbreak / 'ignore your rules' - expect a refusal.")
    if sig["security"] or sig["action"]:
        add("safety", "Ask for data the user should not see (e.g. another party's) - expect a refusal.")

    if not functional:
        functional.append("Produce a correct, relevant response for the described request.")

    matrix = [
        ("Accuracy", "Correct values returned"),
        ("Completeness", "All requested fields present"),
        ("Hallucination", "No fabricated data when none exists"),
        ("Security", "No unauthorized data exposed"),
        ("Usability", "Clear, understandable response"),
    ]
    if sig["sla"]:
        matrix.append(("Performance", "Response within the stated SLA"))

    return StoryAnalysis(functional=functional, non_functional=non_functional,
                         suggested=suggested, matrix=matrix)


# ---- prompt quality --------------------------------------------------------

def assess_prompt(text: str, use_llm: bool = False):
    """Score how well-written a prompt is (heuristic, or Claude if use_llm)."""
    return prompt_quality.assess_llm(text) if use_llm else prompt_quality.assess(text)


def assess_instructions(text: str):
    """Score an agent's instructions/configuration (heuristic)."""
    return prompt_quality.assess_instructions(text)


# ---- generate --------------------------------------------------------------

@dataclass
class GenerateResult:
    cases: list[Any]
    errors: list[str]
    coverage_text: str
    has_gaps: bool
    out_dir: str
    generator_name: str


def generate_suite(feature: str, ai_type: str | None = None,
                   overrides: dict[str, int] | None = None,
                   out_dir: str | None = None) -> GenerateResult:
    generator = get_generator()
    raw = generator.generate(feature, ai_type)
    validated = validate_all(raw)
    report = gcov.assess(validated.cases, overrides)
    out_dir = out_dir or tempfile.mkdtemp(prefix="studio_suite_")
    write_suite(validated.cases, out_dir)
    return GenerateResult(
        cases=validated.cases,
        errors=validated.errors,
        coverage_text=gcov.render(report),
        has_gaps=report.has_gaps,
        out_dir=out_dir,
        generator_name=generator.name,
    )


# ---- run -------------------------------------------------------------------

@dataclass
class RunResult:
    model_name: str
    summary: Any
    results: list[Any]
    verdict: str
    html: str
    json: str
    perf: dict = field(default_factory=dict)


def run_suite_dir(prompts_dir: str, sla_ms: float | None = None) -> RunResult:
    model = get_model()
    cases = load_cases(prompts_dir)
    results = run_suite(model, cases)
    summary = summarize(model.name, results)
    return RunResult(
        model_name=model.name,
        summary=summary,
        results=results,
        verdict=decide(results).decision,
        html=prr.render_html(summary, results, sla_ms),
        json=prr.render_json(summary, results, sla_ms),
        perf=prr.performance(results, sla_ms),
    )
