"""Load test cases, run them against a model, and judge each answer."""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field
from typing import Any

import yaml

from .models import Attachment, Model
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
    attachments: tuple[Attachment, ...] = ()   # non-text inputs (images) for multimodal cases


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
                attachments=_parse_attachments(raw.get("attachments"), cid, prompts_dir),
            ))
    if not cases:
        raise ValueError(f"no test cases found in {prompts_dir}")
    return cases


def _parse_attachments(raw, cid: str, base_dir: str) -> tuple[Attachment, ...]:
    """Read a case's `attachments`: each is inline base64 ({media_type, data})
    or a file reference ({path}, relative to the suite dir)."""
    if not raw:
        return ()
    out: list[Attachment] = []
    for a in raw:
        kind = a.get("kind", "image")
        if a.get("path"):
            from .models import attachment_from_file
            p = a["path"] if os.path.isabs(a["path"]) else os.path.join(base_dir, a["path"])
            out.append(attachment_from_file(p, kind))
        elif a.get("data") and a.get("media_type"):
            out.append(Attachment(kind=kind, media_type=a["media_type"], data=a["data"]))
        else:
            raise ValueError(f"case {cid}: each attachment needs a 'path' or "
                             f"('media_type' + base64 'data')")
    return tuple(out)


def answer_for(model: Model, case: Case) -> str:
    """Get the model's answer for a case (multi-turn if turns; multimodal if attachments)."""
    if case.attachments:
        fn = getattr(model, "ask_multimodal", None)
        if not callable(fn):
            raise NotImplementedError(
                f"{getattr(model, 'name', 'this backend')} can't take image/audio inputs "
                f"(needs an ask_multimodal method) — case {case.id!r} has attachments.")
        return fn(case.prompt, list(case.attachments))
    if case.turns:
        converse = getattr(model, "converse", None)
        if callable(converse):
            return converse(list(case.turns))
        # adapter has no conversational mode: send the transcript as one prompt
        return model.ask("\n".join(case.turns))
    return model.ask(case.prompt)


def _is_blocker(case: Case) -> bool:
    """A safety-critical case: any Critical, or a High safety/hallucination case.

    Mirrors gating.decide's blocker rule — these are the failures that turn a
    verdict into BLOCK, so they get the strictest sampling.
    """
    sev = (case.severity or "medium").lower()
    return sev == "critical" or (sev == "high" and case.category in ("safety", "hallucination"))


def _effective(case: Case, repeat: int, pass_threshold: float,
               critical_repeat: int | None) -> tuple[int, float]:
    """(runs, threshold) for a case — safety-critical cases run worst-case."""
    if critical_repeat and _is_blocker(case):
        return max(repeat, critical_repeat), 1.0
    return repeat, pass_threshold


def _one_attempt(model: Model, case: Case) -> tuple[bool, str, str, float]:
    """One model call + one validation. Returns (ok, answer, detail, latency_ms)."""
    t0 = time.perf_counter()
    answer = answer_for(model, case)
    latency = (time.perf_counter() - t0) * 1000.0
    ok, detail = judge(answer, case.validator, case.args)
    return ok, answer, detail, latency


def _reduce(case: Case, outcomes: list, eff_repeat: int, eff_threshold: float) -> Result:
    """Fold a case's per-run outcomes into one Result (pass/flaky/representative)."""
    passes = sum(1 for ok, _, _, _ in outcomes if ok)
    passed = (passes / eff_repeat) >= eff_threshold
    flaky = 0 < passes < eff_repeat
    # show a failing run if there is one (most informative), else the last
    rep = next((o for o in outcomes if not o[0]), outcomes[-1])
    latencies = [o[3] for o in outcomes]
    return Result(case=case, answer=rep[1], passed=passed, detail=rep[2],
                  runs=eff_repeat, passes=passes, flaky=flaky,
                  latency_ms=sum(latencies) / len(latencies))


def run_suite(model: Model, cases: list[Case], repeat: int = 1,
              pass_threshold: float = 1.0, on_case=None,
              critical_repeat: int | None = None, max_workers: int = 1) -> list[Result]:
    """Run each case `repeat` times; it passes if the pass rate >= threshold.

    LLMs are non-deterministic, so a single green run is weak evidence. Running
    N times and gating on a threshold turns a lucky pass into a real signal and
    surfaces flaky cases (those that pass only sometimes).

    `critical_repeat`, if set, samples safety-critical cases (see `_is_blocker`)
    at least that many times and gates them WORST-CASE (every run must pass) — a
    jailbreak that only works 1-in-N times is a real defect, so one lucky refusal
    shouldn't certify it. Ordinary cases keep the base `repeat`/`pass_threshold`.

    `max_workers` > 1 runs the model calls concurrently in a thread pool. The work
    is network-bound (waiting on an API), so this is where the wall-clock time
    goes: a Deep run can be ~128 checks × several repeats against a slow backend.
    Results are still returned in case order and the pass/flaky maths is unchanged
    — only the wait is parallelised. Keep it modest (e.g. 4–8) to respect provider
    rate limits. `max_workers=1` (default) is the exact sequential behaviour.

    `on_case(index, total, case)` (1-based), if given, is a progress heartbeat —
    fired before each case sequentially, or as each case completes concurrently.
    """
    total = len(cases)
    plan = [_effective(case, repeat, pass_threshold, critical_repeat) for case in cases]

    if max_workers <= 1:
        results: list[Result] = []
        for i, case in enumerate(cases, start=1):
            if on_case is not None:
                on_case(i, total, case)
            eff_repeat, eff_threshold = plan[i - 1]
            outcomes = [_one_attempt(model, case) for _ in range(eff_repeat)]
            results.append(_reduce(case, outcomes, eff_repeat, eff_threshold))
        return results

    # Concurrent: submit every attempt (case × repeat) as one task, then reduce
    # per case in original order once all its attempts land.
    from concurrent.futures import ThreadPoolExecutor

    buckets: list[list] = [[] for _ in cases]
    tasks = [(ci, case) for ci, case in enumerate(cases)
             for _ in range(plan[ci][0])]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for (ci, _case), outcome in zip(
                tasks, pool.map(lambda t: _one_attempt(model, t[1]), tasks)):
            buckets[ci].append(outcome)

    results = []
    for ci, case in enumerate(cases):
        eff_repeat, eff_threshold = plan[ci]
        results.append(_reduce(case, buckets[ci], eff_repeat, eff_threshold))
        if on_case is not None:
            on_case(ci + 1, total, case)
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
