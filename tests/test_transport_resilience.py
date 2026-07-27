"""A provider rate limit must not destroy a run, nor fake a model failure.

Motivation: a 115-check certification against a free Groq key (30 requests/min)
died outright on the first 429 — no partial result, no certificate. Worse, the
obvious "just catch it" fix would have counted every rate-limited check as a
FAILED safety check, manufacturing a NOT CERTIFIED verdict out of an
infrastructure hiccup. Both behaviours are wrong; these pin the right one.
"""
import core
import pytest
from prompt_regression.models import _retry_wait, _MAX_RETRIES
from prompt_regression.runner import TRANSPORT_ERROR


class _Flaky:
    """Fails every Nth call the way a rate-limited endpoint does."""
    name = "flaky"

    def __init__(self, every=3):
        self.every, self.n = every, 0

    def ask(self, prompt):
        self.n += 1
        if self.n % self.every == 0:
            raise RuntimeError("HTTP 429 Too Many Requests from https://api.example.com")
        return "I cannot help with that. I must refuse."


class _Solid:
    name = "solid"

    def ask(self, prompt):
        return "I cannot help with that. I must refuse."


def test_run_survives_transport_errors():
    fe = core.run_full_evaluation(_Flaky(), level="quick", max_workers=1, critical_repeat=1)
    assert fe.total > 0, "a rate limit must not wipe out the whole battery"
    assert fe.errored, "transport failures must be recorded, not swallowed"


def test_errored_checks_are_excluded_from_the_grade():
    """The core guarantee: infrastructure must never look like a model failure."""
    fe = core.run_full_evaluation(_Flaky(), level="quick", max_workers=1, critical_repeat=1)
    graded = fe.total
    assert graded + len(fe.errored) == len(core.build_certification("quick")), (
        "every check must be either graded or explicitly reported as errored")
    for _name, section in fe.sections:
        pass  # sections keep raw results; the totals are what feed the grade
    assert all(not str(r.detail).startswith(TRANSPORT_ERROR)
               for r in fe.agent_checks), "agent checks must not carry transport errors"


def test_incompleteness_is_visible_on_the_certificate():
    fe = core.run_full_evaluation(_Flaky(), level="quick", max_workers=1, critical_repeat=1)
    html = core.render_certificate(fe)
    assert "Incomplete evaluation" in html
    assert f"{fe.total} of {fe.total + len(fe.errored)}" in html
    assert fe.is_complete is False


def test_clean_run_is_marked_complete_and_unbannered():
    fe = core.run_full_evaluation(_Solid(), level="quick", max_workers=1, critical_repeat=1)
    assert fe.errored == []
    assert fe.is_complete is True
    assert "Incomplete evaluation" not in core.render_certificate(fe)


def test_capability_gaps_still_raise():
    """NotImplementedError means 'this backend can't do that' — a real error the
    user must see, not a check to quietly mark errored."""
    class NoAct:
        name = "noact"

        def ask(self, prompt):
            raise NotImplementedError("no tool-call channel")

    with pytest.raises(NotImplementedError):
        core.run_full_evaluation(NoAct(), level="quick", max_workers=1, critical_repeat=1)


def test_retry_budget_outlasts_a_per_minute_window():
    """Backing off for 20s against a 60s RPM window can never succeed."""
    total = sum(_retry_wait(a, None) for a in range(_MAX_RETRIES))
    assert total > 60, f"retry budget {total:.0f}s must exceed a 60s rate-limit window"


def test_retry_honours_retry_after_header():
    assert 12.0 <= _retry_wait(0, "12") <= 15.0


def test_retry_wait_is_jittered():
    """Without jitter, parallel workers wake in lockstep and re-trip the limit."""
    waits = {_retry_wait(2, None) for _ in range(20)}
    assert len(waits) > 1, "retry wait must be jittered across callers"
