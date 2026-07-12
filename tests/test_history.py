"""Tests for durable certification history (history.py)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
import history


def _fe(level: str = "quick"):
    return core.run_full_evaluation(core.make_model("mock"), level=level)


def test_save_and_list_roundtrip(tmp_path):
    db = str(tmp_path / "h.db")
    rid = history.save_run(_fe(), label="baseline", path=db)
    assert isinstance(rid, str) and len(rid) == 12
    rows = history.list_runs(path=db)
    assert len(rows) == 1
    row = rows[0]
    assert row.run_id == rid and row.label == "baseline"
    assert row.model == "mock-helpbot-v2" and row.verdict == "BLOCK"
    assert 0 <= row.pass_rate <= 100 and row.total > 0
    assert row.iso                                     # renders a timestamp


def test_ordering_is_most_recent_first(tmp_path):
    db = str(tmp_path / "h.db")
    a = history.save_run(_fe(), label="first", path=db)
    b = history.save_run(_fe(), label="second", path=db)
    rows = history.list_runs(path=db)
    assert [r.run_id for r in rows] == [b, a]          # newest first


def test_get_snapshot_feeds_compare(tmp_path):
    db = str(tmp_path / "h.db")
    rid = history.save_run(_fe(), path=db)
    snap = history.get_snapshot(rid, path=db)
    # the stored blob is a real export_snapshot -> diffs against itself cleanly
    diff = core.compare_snapshots(snap, snap)
    assert not diff.has_regressions and not diff.newly_passed


def test_regression_since_previous(tmp_path):
    db = str(tmp_path / "h.db")
    assert history.regression_since_previous("mock-helpbot-v2", path=db) is None
    history.save_run(_fe(), path=db)
    assert history.regression_since_previous("mock-helpbot-v2", path=db) is None  # only one
    history.save_run(_fe(), path=db)
    diff = history.regression_since_previous("mock-helpbot-v2", path=db)
    assert diff is not None
    # identical mock runs -> deterministic, so nothing should flip
    assert not diff.has_regressions and not diff.newly_passed


def test_delete_and_clear(tmp_path):
    db = str(tmp_path / "h.db")
    rid = history.save_run(_fe(), path=db)
    assert history.delete_run(rid, path=db) is True
    assert history.delete_run(rid, path=db) is False   # already gone
    history.save_run(_fe(), path=db)
    history.save_run(_fe(), path=db)
    assert history.clear(path=db) == 2
    assert history.list_runs(path=db) == []


def test_disabled_on_public_deploy(monkeypatch):
    monkeypatch.setenv("PRS_STUDIO_PUBLIC", "1")
    assert history.is_enabled() is False
    monkeypatch.delenv("PRS_STUDIO_PUBLIC", raising=False)
    assert history.is_enabled() is True


def test_filter_by_model(tmp_path):
    db = str(tmp_path / "h.db")
    history.save_run(_fe(), path=db)
    assert len(history.list_runs(model="mock-helpbot-v2", path=db)) == 1
    assert history.list_runs(model="nonexistent", path=db) == []
