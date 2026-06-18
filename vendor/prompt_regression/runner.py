"""Load test cases, run them against a model, and judge each answer."""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field
from typing import Any

import yaml

from .models import Model
from .validators import judge


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    prompt: str
    validator: str
    args: dict[str, Any]
    severity: str = "medium"   # carried from the generator; drives the gating verdict
    turns: tuple[str, ...] | None = None   # set for multi-turn (agent) cases


@dataclass
class Result:
    case: Case
    answer: str
    passed: bool          # cleared the pass-rate threshold across `runs`
    detail: str
    runs: int = 1
    passes: int = 1
    flaky: bool = False   # passed some runs but not all — unstable behaviour
    latency_ms: float = 0.0   # mean response time across runs


def load_cases(prompts_dir: str) -> list[Case]:
    """Read every *.yaml file in prompts_dir into a flat list of cases."""
    cases: list[Case] = []
    seen: set[str] = set()
    for path in sorted(glob.glob(os.path.join(prompts_dir, "*.yaml"))):
        with open(path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        category = doc.get("category", os.path.splitext(os.path.basename(path))[0])
        for raw in doc.get("cases", []):
            cid = raw["id"]
            if cid in seen:
                raise ValueError(f"duplicate case id: {cid}")
            seen.add(cid)
            turns_raw = raw.get("turns")
            prompt = raw.get("prompt")
            if turns_raw:
                turns = tuple(turns_raw)
                prompt = prompt or " | ".join(turns_raw)   # display string
            else:
                turns = None
                if not prompt:
                    raise ValueError(f"case {cid} needs a 'prompt' or 'turns'")
            cases.append(Case(
                id=cid,
                category=category,
                prompt=prompt,
                validator=raw["validator"],
                args=raw.get("args", {}),
                severity=raw.get("severity", "medium"),
                turns=turns,
            ))
    if not cases:
        raise ValueError(f"no test cases found in {prompts_dir}")
    return cases


def answer_for(model: Model, case: Case) -> str:
    """Get the model's answer for a case (multi-turn if the case has turns)."""
    if case.turns:
        converse = getattr(model, "converse", None)
        if callable(converse):
            return converse(list(case.turns))
        # adapter has no conversational mode: send the transcript as one prompt
        return model.ask("\n".join(case.turns))
    return model.ask(case.prompt)


def run_suite(model: Model, cases: list[Case], repeat: int = 1,
              pass_threshold: float = 1.0) -> list[Result]:
    """Run each case `repeat` times; it passes if the pass rate >= threshold.

    LLMs are non-deterministic, so a single green run is weak evidence. Running
    N times and gating on a threshold turns a lucky pass into a real signal and
    surfaces flaky cases (those that pass only sometimes).
    """
    results: list[Result] = []
    for case in cases:
        outcomes = []  # (passed, answer, detail) per run
        latencies = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            answer = answer_for(model, case)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            ok, detail = judge(answer, case.validator, case.args)
            outcomes.append((ok, answer, detail))
        passes = sum(1 for ok, _, _ in outcomes if ok)
        passed = (passes / repeat) >= pass_threshold
        flaky = 0 < passes < repeat
        # show a failing run if there is one (most informative), else the last
        rep = next((o for o in outcomes if not o[0]), outcomes[-1])
        results.append(Result(case=case, answer=rep[1], passed=passed, detail=rep[2],
                              runs=repeat, passes=passes, flaky=flaky,
                              latency_ms=sum(latencies) / len(latencies)))
    return results


@dataclass
class Summary:
    model: str
    total: int
    passed: int
    by_category: dict[str, tuple[int, int]] = field(default_factory=dict)  # cat -> (pass, total)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total * 100 if self.total else 0.0


def summarize(model_name: str, results: list[Result]) -> Summary:
    by_cat: dict[str, list[int]] = {}
    passed = 0
    for r in results:
        bucket = by_cat.setdefault(r.case.category, [0, 0])
        bucket[1] += 1
        if r.passed:
            bucket[0] += 1
            passed += 1
    return Summary(
        model=model_name,
        total=len(results),
        passed=passed,
        by_category={k: (v[0], v[1]) for k, v in by_cat.items()},
    )
