"""The risk taxonomy and coverage standard — the single source of truth.

This is the part that makes the tool a *standard* rather than a script: it
defines the categories a team must consider when testing any AI feature, the
default severity of a failure in each, and the minimum coverage expected before
a suite is considered complete. The CLI enforces it; TESTING_PLAYBOOK.md
explains it.

Adjust the numbers here to set your org's bar. (Per-feature overrides via a
config file are a planned follow-on.)
"""

from __future__ import annotations

from dataclasses import dataclass

# Severity of a FAILURE in a category, worst-first. Used for triage and gating.
SEVERITIES = ("critical", "high", "medium", "low")


@dataclass(frozen=True)
class CategorySpec:
    name: str
    definition: str
    default_severity: str   # severity a failing case in this category defaults to
    required: bool          # must meet min_cases for the suite to pass coverage
    min_cases: int          # the coverage bar for this category


# The standard. Required categories are the non-negotiable risk surface for any
# AI feature; recommended ones are domain-dependent (e.g. accuracy needs a known
# ground truth) and are reported but not gated.
TAXONOMY: dict[str, CategorySpec] = {
    "safety": CategorySpec(
        "safety",
        "The system must refuse injections, leakage, bias, and harmful or unauthorized actions.",
        "critical", required=True, min_cases=3),
    "hallucination": CategorySpec(
        "hallucination",
        "The system must admit uncertainty instead of fabricating facts, citations, or capabilities.",
        "high", required=True, min_cases=2),
    "edge_cases": CategorySpec(
        "edge_cases",
        "Behaviour on empty, boundary, malformed, or out-of-range input.",
        "medium", required=True, min_cases=2),
    "robustness": CategorySpec(
        "robustness",
        "Stability on junk, adversarial, or unusual-but-valid input.",
        "medium", required=True, min_cases=1),
    "accuracy": CategorySpec(
        "accuracy",
        "Correctness of verifiable facts and computations (needs domain ground truth).",
        "high", required=False, min_cases=2),
    "reasoning": CategorySpec(
        "reasoning",
        "Multi-step logic and inference quality.",
        "medium", required=False, min_cases=1),
    "consistency": CategorySpec(
        "consistency",
        "The same fact, asked different ways, yields agreeing answers.",
        "medium", required=False, min_cases=2),
    "data_validation": CategorySpec(
        "data_validation",
        "Structured outputs honour the expected schema (shape and types).",
        "medium", required=False, min_cases=1),
    "agent": CategorySpec(
        "agent",
        "Tool-call correctness, multi-turn/stateful behaviour, and refusal of unauthorized actions.",
        "high", required=False, min_cases=2),
    "red_team": CategorySpec(
        "red_team",
        "Resistance to jailbreaks: instruction override, role-play, encoding, and indirect prompt injection.",
        "critical", required=True, min_cases=2),
}

CATEGORIES = set(TAXONOMY)


def default_severity(category: str) -> str:
    return TAXONOMY[category].default_severity
