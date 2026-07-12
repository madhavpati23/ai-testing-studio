"""Tests for concurrent suite execution (max_workers)."""

from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
from prompt_regression.runner import Case, run_suite


class _SlowModel:
    """Each ask() sleeps, and records peak concurrency, so we can prove overlap."""
    name = "slow"

    def __init__(self, delay=0.05):
        self.delay = delay
        self._lock = threading.Lock()
        self._active = 0
        self.peak = 0

    def ask(self, prompt):
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
        time.sleep(self.delay)
        with self._lock:
            self._active -= 1
        return "ok"          # matches the 'contains: ok' validator below


def _cases(n):
    return [Case(id=f"c{i}", category="accuracy", prompt=f"q{i}",
                 validator="contains", args={"value": "ok"}, severity="low")
            for i in range(n)]


def test_concurrent_matches_sequential_results():
    cases = core.build_certification("quick")
    seq = core.run_selected(cases, model=core.make_model("mock"), max_workers=1)
    par = core.run_selected(cases, model=core.make_model("mock"), max_workers=8)
    # identical order and pass/fail — concurrency changes only the wait
    assert [r.case.id for r in seq.results] == [r.case.id for r in par.results]
    assert [r.passed for r in seq.results] == [r.passed for r in par.results]
    assert seq.verdict == par.verdict


def test_workers_actually_overlap():
    model = _SlowModel(delay=0.05)
    run_suite(model, _cases(8), max_workers=4)
    assert model.peak >= 2          # genuinely ran calls in parallel


def test_concurrency_is_faster_for_network_bound_work():
    seq_model = _SlowModel(delay=0.05)
    t0 = time.perf_counter()
    run_suite(seq_model, _cases(8), max_workers=1)
    seq_t = time.perf_counter() - t0

    par_model = _SlowModel(delay=0.05)
    t0 = time.perf_counter()
    run_suite(par_model, _cases(8), max_workers=8)
    par_t = time.perf_counter() - t0
    assert par_t < seq_t * 0.7      # a clear speedup, allowing for overhead


def test_critical_repeat_worst_case_still_holds_under_concurrency():
    # a case that refuses only on its first answer must still FAIL worst-case
    class Flaky:
        name = "flaky"
        def __init__(self):
            self.lock = threading.Lock()
            self.n = 0
        def ask(self, prompt):
            with self.lock:
                self.n += 1
                first = self.n == 1
            return "ok" if first else "no"
    case = Case(id="crit", category="safety", prompt="p",
                validator="contains", args={"value": "ok"}, severity="critical")
    res = run_suite(Flaky(), [case], repeat=1, critical_repeat=4, max_workers=4)[0]
    assert res.runs == 4 and res.passes == 1 and res.passed is False and res.flaky is True
