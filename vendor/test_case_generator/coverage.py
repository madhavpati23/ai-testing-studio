"""Measure a suite against the coverage standard in taxonomy.py.

Turns "did we test enough?" from a judgement call into a policy: required
categories must meet their minimum case count, or the suite has a gap. This is
what lets a manager gate a release on coverage rather than vibes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Case
from .taxonomy import TAXONOMY


@dataclass
class CategoryCoverage:
    name: str
    count: int
    min_cases: int
    required: bool

    @property
    def ok(self) -> bool:
        return self.count >= self.min_cases

    @property
    def is_gap(self) -> bool:
        return self.required and not self.ok


@dataclass
class CoverageReport:
    rows: list[CategoryCoverage]
    total_cases: int

    @property
    def gaps(self) -> list[CategoryCoverage]:
        return [r for r in self.rows if r.is_gap]

    @property
    def has_gaps(self) -> bool:
        return bool(self.gaps)


def effective_standard(overrides: dict[str, int] | None = None) -> dict[str, tuple[bool, int]]:
    """Merge per-feature overrides onto the org standard.

    Any category named in `overrides` becomes REQUIRED at the given minimum —
    that's how a high-risk feature raises the bar above the default.
    """
    std = {name: (spec.required, spec.min_cases) for name, spec in TAXONOMY.items()}
    for category, minimum in (overrides or {}).items():
        std[category] = (True, minimum)
    return std


def assess(cases: list[Case], overrides: dict[str, int] | None = None) -> CoverageReport:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1
    std = effective_standard(overrides)
    rows = [
        CategoryCoverage(name=name, count=counts.get(name, 0),
                         min_cases=minimum, required=required)
        for name, (required, minimum) in std.items()
    ]
    return CoverageReport(rows=rows, total_cases=len(cases))


def render(report: CoverageReport) -> str:
    line = "-" * 60
    out = [line, "  COVERAGE vs STANDARD", line]
    for r in sorted(report.rows, key=lambda x: (not x.required, x.name)):
        tag = "REQUIRED" if r.required else "optional"
        if r.is_gap:
            mark = "GAP "
        elif r.ok:
            mark = "ok  "
        else:
            mark = "--  "  # optional, below target — informational only
        out.append(f"  [{mark}] {r.name:<16} {r.count}/{r.min_cases:<3} ({tag})")
    out.append(line)
    if report.has_gaps:
        names = ", ".join(r.name for r in report.gaps)
        out.append(f"  RESULT: BELOW STANDARD -- required gaps in: {names}")
    else:
        out.append("  RESULT: meets the required coverage standard.")
    out.append(line)
    return "\n".join(out)
