"""Baselines turn a one-off test run into regression testing.

Save the per-case pass/fail of a known-good run as a baseline. On a later run
(new model version, edited prompt, tweaked system instructions), diff against
the baseline to surface exactly which cases changed -- the regressions you care
about, separated from cases that were already failing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .runner import Result


def to_baseline(results: list[Result]) -> dict[str, bool]:
    """Reduce a run to the part that should be stable: case id -> passed."""
    return {r.case.id: r.passed for r in results}


def save(results: list[Result], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_baseline(results), fh, indent=2, sort_keys=True)
        fh.write("\n")


def load(path: str) -> dict[str, bool]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class Diff:
    regressed: list[str]   # passed in baseline, fails now  <- the alarm
    fixed: list[str]       # failed in baseline, passes now
    added: list[str]       # new case, not in baseline
    removed: list[str]     # in baseline, no longer present

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressed)


def diff(baseline: dict[str, bool], results: list[Result]) -> Diff:
    current = to_baseline(results)
    regressed, fixed, added = [], [], []
    for cid, now_pass in current.items():
        if cid not in baseline:
            added.append(cid)
        elif baseline[cid] and not now_pass:
            regressed.append(cid)
        elif not baseline[cid] and now_pass:
            fixed.append(cid)
    removed = [cid for cid in baseline if cid not in current]
    return Diff(
        regressed=sorted(regressed),
        fixed=sorted(fixed),
        added=sorted(added),
        removed=sorted(removed),
    )
