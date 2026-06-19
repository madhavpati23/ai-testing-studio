"""Tests for the features added after the original generate/run flow:
session-scoped models (BYOK), golden-set parsing, the certification battery,
and the practice question bank.
"""

import os

import core


# ---- make_model: session-scoped, never touches the process env --------------

def test_make_model_mock():
    m = core.make_model("mock")
    assert m.name == "mock-helpbot-v2"


def test_make_model_http_keeps_env_clean():
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("PRS_HTTP_URL", None)
    m = core.make_model("http", {
        "url": "https://api.example.com/v1/chat/completions",
        "headers": '{"Authorization": "Bearer secret-token"}',
        "response_path": "choices.0.message.content",
    })
    assert m.name.startswith("http:")
    assert m.headers.get("Authorization") == "Bearer secret-token"
    # the key/url must NOT leak into the shared process environment
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "PRS_HTTP_URL" not in os.environ


def test_make_model_http_blocks_private_by_default():
    # safe default: no explicit block_private -> SSRF guard is on
    m = core.make_model("http", {"url": "https://api.example.com/x"})
    assert m.block_private is True


def test_generate_suite_with_explicit_mock_kind():
    gen = core.generate_suite("password reset", kind="mock", opts={})
    assert gen.generator_name == "mock"
    assert len(gen.cases) >= 8


# ---- golden set --------------------------------------------------------------

def test_golden_template_parses_clean():
    cases, errors = core.build_golden(core.GOLDEN_TEMPLATE)
    assert errors == []
    assert len(cases) == 5
    assert all(c.validator in ("contains", "not_contains", "regex", "equals_number")
               for c in cases)


def test_golden_reports_bad_rows():
    bad = "prompt,expected,validator\nHi there,,contains\nWhat is 2+2?,4,bogus_validator\n"
    cases, errors = core.build_golden(bad)
    assert cases == []
    assert len(errors) == 2  # missing expected + unsupported validator


def test_golden_requires_header():
    cases, errors = core.build_golden("just,some,data\n1,2,3\n")
    assert cases == []
    assert errors


# ---- certification battery ---------------------------------------------------

def test_certification_builds_and_is_valid():
    cases = core.build_certification()
    assert len(cases) >= 20
    cats = {c.category for c in cases}
    # covers the major risk dimensions
    for must in ("safety", "red_team", "hallucination", "accuracy"):
        assert must in cats
    assert core.certification_dimensions() >= 8


# ---- practice bank -----------------------------------------------------------

def test_practice_bank_size_and_mapping():
    assert core.question_bank_size() >= 500
    ids = {e.id for e in core.practice_exercises()}
    # every question maps to a known exercise
    for ex_id, _probe in core.question_bank():
        assert ex_id in ids


def test_practice_expected_verdict():
    assert core.expected_verdict("direct-injection") == "It FAILED"
    assert core.expected_verdict("consistency") == "It PASSED"


def test_practice_difficulty_known():
    for e in core.practice_exercises():
        assert core.difficulty(e.id) in core.DIFFICULTIES


# ---- judge calibration -------------------------------------------------------

def test_calibration_template_parses():
    rows, errors = core.parse_calibration_csv(core.CALIBRATION_TEMPLATE)
    assert errors == []
    assert len(rows) == 6
    assert all(isinstance(h, bool) for _c, _a, h in rows)


def test_calibration_requires_columns():
    rows, errors = core.parse_calibration_csv("a,b\n1,2\n")
    assert rows == []
    assert errors


def test_calibrate_judge_agreement():
    rows = [("c1", "a1", True), ("c2", "a2", False), ("c3", "a3", True)]
    # a judge that always says pass agrees on the two True rows -> 2/3
    res = core.calibrate_judge(rows, lambda a, c: (True, "always"))
    assert res.total == 3
    assert res.agree == 2
    assert round(res.agreement) == 67
    assert res.verdict == "DO NOT TRUST"


def test_calibrate_judge_perfect():
    rows = [("c", "a", True), ("c", "a", True)]
    res = core.calibrate_judge(rows, lambda a, c: (True, "ok"))
    assert res.agreement == 100.0
    assert res.verdict == "TRUSTWORTHY"


# ---- multi-turn conversation -------------------------------------------------

def test_conversation_memory_against_mock():
    m = core.make_model("mock")
    run = core.run_conversation(
        ["My name is Sam.", "What is my name?"],
        validator="contains", expected="Sam", model=m)
    assert run.results[0].passed
    assert "Sam" in run.results[0].answer


def test_conversation_requires_turns():
    import pytest
    with pytest.raises(ValueError):
        core.run_conversation(["   ", ""], validator="contains", expected="x",
                              model=core.make_model("mock"))
