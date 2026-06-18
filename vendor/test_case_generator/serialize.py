"""Write validated cases as prompt-regression-suite YAML.

The suite loads one category per file (a top-level `category:` plus a `cases:`
list). We group the generated cases by category and write one file each, so the
output drops directly into a `prompts/` directory.
"""

from __future__ import annotations

import os
from collections import defaultdict

import yaml

from .schema import Case


def _case_to_dict(case: Case) -> dict:
    # severity/status/reviewer are metadata for triage and governance; the
    # runner ignores keys it doesn't recognise.
    d = {
        "id": case.id,
        "severity": case.severity,
        "status": case.status,
        "prompt": case.prompt,
        "validator": case.validator,
        "args": case.args,
    }
    if case.reviewer:
        d["reviewer"] = case.reviewer
    return d


def write_suite(cases: list[Case], out_dir: str) -> list[str]:
    """Write one YAML file per category into out_dir. Returns the paths written."""
    os.makedirs(out_dir, exist_ok=True)
    by_cat: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        by_cat[case.category].append(case)

    written: list[str] = []
    for category, items in sorted(by_cat.items()):
        path = os.path.join(out_dir, f"{category}.yaml")
        doc = {"category": category, "cases": [_case_to_dict(c) for c in items]}
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True, width=100)
        written.append(path)
    return written
