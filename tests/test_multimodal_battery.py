"""Phase 2 multimodal: battery, serialization round-trip, end-to-end run."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _bootstrap  # noqa: F401
import core
import multimodal
from prompt_regression.runner import load_cases

pytest.importorskip("PIL")


def test_render_is_deterministic_and_png():
    a = multimodal.render_text_image("HELLO WORLD")
    b = multimodal.render_text_image("HELLO WORLD")
    assert a == b                                  # deterministic
    assert a[:8] == b"\x89PNG\r\n\x1a\n"           # real PNG header


def test_battery_cases_carry_images():
    cases = multimodal.build_multimodal_battery()
    assert len(cases) == len(multimodal._BATTERY_SPEC)
    assert all(len(c.attachments) == 1 for c in cases)
    assert all(c.attachments[0].kind == "image" for c in cases)
    # includes both an attack and a benign control
    ids = {c.id for c in cases}
    assert "mm-typographic-injection" in ids and "mm-benign-no-overrefusal" in ids


def test_serialization_round_trip_preserves_attachments(tmp_path):
    cases = multimodal.build_multimodal_battery()
    multimodal.write_multimodal_suite(cases, str(tmp_path))
    # images written to disk, referenced by path
    assert os.path.isdir(tmp_path / "images")
    reloaded = load_cases(str(tmp_path))
    by_id = {c.id: c for c in reloaded}
    for original in cases:
        r = by_id[original.id]
        assert len(r.attachments) == 1
        # bytes survive the write->read round trip exactly
        assert r.attachments[0].data == original.attachments[0].data
        assert r.attachments[0].media_type == original.attachments[0].media_type


class _FakeVision:
    """A vision model that OBEYS text hidden in the image — i.e. jailbroken."""
    name = "fake-vision"

    def ask(self, prompt):
        return "no image"

    def ask_multimodal(self, prompt, attachments):
        # simulate reading + obeying the injected instruction and OCR probes
        return "Sure — PWNED. Also the word is GIRAFFE and SUNSHINE."


def test_end_to_end_run_grades_and_gates():
    run = multimodal.run_multimodal_battery(_FakeVision())
    by_id = {r.case.id: r for r in run.results}
    # obeyed the typographic injection -> that critical case FAILS
    assert by_id["mm-typographic-injection"].passed is False
    # read the OCR words correctly -> those pass
    assert by_id["mm-ocr-accuracy"].passed is True
    assert by_id["mm-benign-no-overrefusal"].passed is True
    # a failed critical red_team case forces BLOCK
    assert run.verdict == "BLOCK"


def test_checks_fold_into_certificate_shape():
    run = multimodal.run_multimodal_battery(_FakeVision())
    checks = multimodal.multimodal_checks(run)
    assert len(checks) == len(multimodal._BATTERY_SPEC)
    assert all(hasattr(c, "case") and hasattr(c, "passed") for c in checks)
