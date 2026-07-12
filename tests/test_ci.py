"""Tests for the headless CI certification (studio_ci)."""

from __future__ import annotations

import json
import os
import sys
import xml.dom.minidom as minidom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
import studio_ci


def _fe(level: str = "quick"):
    model = core.make_model("mock")
    return core.run_full_evaluation(model, level=level)


def test_report_is_json_serializable_and_complete():
    fe = _fe()
    grade, status = core.certification_grade(fe.pass_rate, fe.verdict)
    report = studio_ci.to_report(fe, grade, status)
    # round-trips through JSON without error
    reloaded = json.loads(json.dumps(report))
    assert reloaded["total"] == fe.total
    assert len(reloaded["checks"]) == fe.total          # one entry per pooled check
    assert reloaded["grade"] == grade and reloaded["verdict"] == fe.verdict
    assert set(reloaded["by_category"]) == set(fe.by_category)


def test_junit_is_well_formed_xml():
    fe = _fe()
    grade, status = core.certification_grade(fe.pass_rate, fe.verdict)
    xml = studio_ci.to_junit(fe, studio_ci.to_report(fe, grade, status))
    dom = minidom.parseString(xml)                       # raises if malformed
    suite = dom.getElementsByTagName("testsuite")[0]
    assert int(suite.getAttribute("tests")) == fe.total
    cases = dom.getElementsByTagName("testcase")
    assert len(cases) == fe.total


def test_decide_exit_policies():
    fe = _fe()  # mock scores poorly -> verdict BLOCK
    assert fe.verdict == "BLOCK"
    # block bar: a BLOCK verdict fails the pipeline
    assert studio_ci.decide_exit(fe, "F", "block", None)[0] == 1
    # never bar: always passes regardless of verdict
    assert studio_ci.decide_exit(fe, "F", "never", None)[0] == 0
    # min-grade gate is independent of the verdict bar
    assert studio_ci.decide_exit(fe, "F", "never", "B")[0] == 1
    assert studio_ci.decide_exit(fe, "A", "never", "B")[0] == 0


def test_run_end_to_end_writes_reports(tmp_path):
    jp, xp = tmp_path / "r.json", tmp_path / "r.xml"
    args = studio_ci.build_parser().parse_args(
        ["--backend", "mock", "--level", "quick", "--fail-on", "block",
         "--json", str(jp), "--junit", str(xp), "--quiet"])
    import io
    code = studio_ci.run(args, out=io.StringIO())
    assert code == 1                                     # mock BLOCKs -> pipeline fails
    assert json.loads(jp.read_text())["verdict"] == "BLOCK"
    minidom.parseString(xp.read_text())                  # valid XML on disk


def test_save_persists_to_history(tmp_path):
    import history
    db = str(tmp_path / "h.db")
    args = studio_ci.build_parser().parse_args(
        ["--backend", "mock", "--level", "quick", "--fail-on", "never",
         "--save", "--db", db, "--label", "sha-123", "--quiet"])
    import io
    assert studio_ci.run(args, out=io.StringIO()) == 0
    rows = history.list_runs(path=db)
    assert len(rows) == 1 and rows[0].label == "sha-123"


def test_multimodal_requires_vision_backend():
    args = studio_ci.build_parser().parse_args(
        ["--backend", "mock", "--level", "quick", "--multimodal", "--quiet"])
    import io
    # the mock backend can't take images -> a clean usage error, not a crash
    assert studio_ci.run(args, out=io.StringIO()) == 2


def test_http_backend_requires_url():
    args = studio_ci.build_parser().parse_args(["--backend", "http"])
    import io
    assert studio_ci.run(args, out=io.StringIO()) == 2   # usage error, not a crash
