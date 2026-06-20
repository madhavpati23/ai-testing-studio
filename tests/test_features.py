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


def test_certification_levels():
    quick = core.build_certification("quick")
    standard = core.build_certification("standard")
    assert len(quick) == 22
    assert len(standard) > len(quick)        # extended battery adds probes
    assert len(core.build_certification("thorough")) == len(standard)


def test_stress_cases_sample_and_validate():
    cases = core.build_stress_cases(80)
    assert len(cases) == 80
    assert all(c.validator == "regex" for c in cases)
    # full coverage: every skill maps to a validator
    assert set(core._SKILL_TEST) == {e.id for e in core.practice_exercises()}


def test_deep_full_evaluation_adds_stress():
    m = core.make_model("mock")
    fe = core.run_full_evaluation(m, level="deep", stress_n=40)
    names = [n for n, _ in fe.sections]
    assert "Randomized stress battery" in names
    assert fe.total >= 48 + 40


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


# ---- RAG grounding -----------------------------------------------------------

class _FakeModel:
    name = "fake"

    def __init__(self, answer):
        self._answer = answer

    def ask(self, _prompt):
        return self._answer


def test_grounding_faithful_and_correct():
    r = core.run_grounding(
        "The sky is blue.", "What colour is the sky?",
        model=_FakeModel("Blue."), grounding_judge=lambda c, a: (True, "supported"),
        expected="blue")
    assert r.verdict == "GROUNDED"
    assert r.expected_ok is True


def test_grounding_detects_hallucination():
    r = core.run_grounding(
        "The sky is blue.", "How far is the sun?",
        model=_FakeModel("About 150 million km."),
        grounding_judge=lambda c, a: (False, "not in context"))
    assert r.verdict == "NOT GROUNDED"
    assert r.grounded is False


def test_grounding_faithful_but_wrong():
    r = core.run_grounding(
        "The sky is blue.", "colour?",
        model=_FakeModel("I don't have that information."),
        grounding_judge=lambda c, a: (True, "ok"), expected="blue")
    assert r.verdict == "GROUNDED BUT WRONG"
    assert r.expected_ok is False


# ---- full evaluation (combined scorecard) ------------------------------------

def test_full_evaluation_combines_dimensions():
    m = core.make_model("mock")
    fe = core.run_full_evaluation(m)
    assert fe.total >= 20                       # certification probes
    assert fe.verdict in ("SHIP", "NEEDS SIGN-OFF", "BLOCK")
    assert fe.by_category                        # pooled per-dimension
    assert len(fe.sections) == 1                 # certification only


def test_full_evaluation_includes_golden():
    m = core.make_model("mock")
    gcases, _ = core.build_golden(core.GOLDEN_TEMPLATE)
    fe = core.run_full_evaluation(m, golden_cases=gcases)
    assert len(fe.sections) == 2                 # certification + golden
    assert fe.total >= 25


# ---- certificate -------------------------------------------------------------

def test_certification_grade_mapping():
    assert core.certification_grade(98, "SHIP") == ("A", "CERTIFIED")
    assert core.certification_grade(90, "NEEDS SIGN-OFF") == ("B", "CONDITIONALLY CERTIFIED")
    # a BLOCK can never earn an A/B no matter the score
    assert core.certification_grade(99, "BLOCK")[0] == "C"
    assert core.certification_grade(99, "BLOCK")[1] == "NOT CERTIFIED"
    assert core.certification_grade(30, "BLOCK") == ("F", "NOT CERTIFIED")


def test_render_certificate_is_html():
    fe = core.run_full_evaluation(core.make_model("mock"))
    html = core.render_certificate(fe)
    assert html.lower().startswith("<!doctype")
    assert "Certificate of AI Evaluation" in html
    assert fe.model_name in html
