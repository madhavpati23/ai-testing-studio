"""The pipeline behind the UI — framework calls, no Streamlit.

Keeping this Streamlit-free means it's unit-testable and the web layer stays a
thin shell. It wires the two packages together: generate + validate + coverage
(ai-test-case-generator) -> run + report + verdict (prompt-regression-suite).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
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


# ---- prompt quality --------------------------------------------------------

def assess_prompt(text: str, use_llm: bool = False):
    """Score how well-written a prompt is (heuristic, or Claude if use_llm)."""
    return prompt_quality.assess_llm(text) if use_llm else prompt_quality.assess(text)


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


def run_suite_dir(prompts_dir: str) -> RunResult:
    model = get_model()
    cases = load_cases(prompts_dir)
    results = run_suite(model, cases)
    summary = summarize(model.name, results)
    return RunResult(
        model_name=model.name,
        summary=summary,
        results=results,
        verdict=decide(results).decision,
        html=prr.render_html(summary, results),
        json=prr.render_json(summary, results),
    )
