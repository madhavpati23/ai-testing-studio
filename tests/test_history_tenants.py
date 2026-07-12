"""Multi-tenant history (gap #5, Phase 1): tenant scoping, migration, Postgres."""

from __future__ import annotations

import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
import history


def _fe():
    return core.run_full_evaluation(core.make_model("mock"), level="quick")


# ---- tenant isolation (SQLite, runs everywhere) ---------------------------

def test_tenants_are_isolated(tmp_path):
    db = str(tmp_path / "h.db")
    a = history.save_run(_fe(), label="A", path=db, tenant_id="acme", user_id="u1")
    history.save_run(_fe(), label="B", path=db, tenant_id="globex", user_id="u2")

    acme = history.list_runs(path=db, tenant_id="acme")
    assert [r.run_id for r in acme] == [a]
    assert acme[0].tenant_id == "acme" and acme[0].user_id == "u1"
    # the default tenant sees neither
    assert history.list_runs(path=db, tenant_id=history.DEFAULT_TENANT) == []


def test_cross_tenant_read_and_delete_are_blocked(tmp_path):
    db = str(tmp_path / "h.db")
    rid = history.save_run(_fe(), path=db, tenant_id="acme")
    # another tenant can't fetch or delete acme's run
    assert history.get_snapshot(rid, path=db, tenant_id="globex") is None
    assert history.delete_run(rid, path=db, tenant_id="globex") is False
    # acme still can
    assert history.get_snapshot(rid, path=db, tenant_id="acme") is not None
    assert history.delete_run(rid, path=db, tenant_id="acme") is True


def test_clear_and_regression_are_tenant_scoped(tmp_path):
    db = str(tmp_path / "h.db")
    history.save_run(_fe(), path=db, tenant_id="acme")
    history.save_run(_fe(), path=db, tenant_id="acme")
    history.save_run(_fe(), path=db, tenant_id="globex")
    # regression only considers this tenant's runs
    assert history.regression_since_previous("mock-helpbot-v2", path=db, tenant_id="globex") is None
    assert history.regression_since_previous("mock-helpbot-v2", path=db, tenant_id="acme") is not None
    # clear removes only the named tenant
    assert history.clear(path=db, tenant_id="acme") == 2
    assert len(history.list_runs(path=db, tenant_id="globex")) == 1


# ---- migration: an old (pre-tenant) DB keeps working ----------------------

def test_migration_adds_tenant_columns_to_old_db(tmp_path):
    db = str(tmp_path / "old.db")
    # simulate the previous schema (no tenant_id/user_id) with one legacy row
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE runs (
        run_id TEXT PRIMARY KEY, ts REAL NOT NULL, label TEXT NOT NULL DEFAULT '',
        model TEXT NOT NULL, level TEXT NOT NULL, grade TEXT NOT NULL, status TEXT NOT NULL,
        verdict TEXT NOT NULL, pass_rate REAL NOT NULL, passed INTEGER NOT NULL,
        total INTEGER NOT NULL, runs_per_check INTEGER NOT NULL, snapshot_json TEXT NOT NULL)""")
    conn.execute("INSERT INTO runs VALUES ('old123', 1.0, 'legacy', 'mock-helpbot-v2', "
                 "'quick', 'F', 'NOT CERTIFIED', 'BLOCK', 10.0, 1, 10, 1, '{}')")
    conn.commit()
    conn.close()

    # opening through history migrates it; the legacy row lands in the default tenant
    rows = history.list_runs(path=db, tenant_id=history.DEFAULT_TENANT)
    assert [r.run_id for r in rows] == ["old123"]
    assert rows[0].tenant_id == history.DEFAULT_TENANT and rows[0].user_id == history.DEFAULT_USER
    # and new writes still work on the migrated DB
    history.save_run(_fe(), path=db, tenant_id="acme")
    assert len(history.list_runs(path=db, tenant_id="acme")) == 1


# ---- is_enabled semantics --------------------------------------------------

def test_postgres_dsn_enables_history_even_when_public(monkeypatch):
    monkeypatch.setenv("PRS_STUDIO_PUBLIC", "1")
    assert history.is_enabled() is False                 # local sqlite stays off
    monkeypatch.setenv("PRS_STUDIO_DB_URL", "postgresql://x/y")
    assert history.is_enabled() is True                  # a real shared store is on


# ---- key-gated real Postgres integration ----------------------------------

@pytest.mark.skipif(not os.environ.get("PRS_TEST_PG_DSN"),
                    reason="no PRS_TEST_PG_DSN — real Postgres integration test skipped")
def test_real_postgres_tenant_isolation(monkeypatch):
    pytest.importorskip("psycopg")
    monkeypatch.setenv("PRS_STUDIO_DB_URL", os.environ["PRS_TEST_PG_DSN"])
    tag = "acme-" + os.urandom(4).hex()
    other = "globex-" + os.urandom(4).hex()
    try:
        rid = history.save_run(_fe(), tenant_id=tag)
        assert [r.run_id for r in history.list_runs(tenant_id=tag)] == [rid]
        assert history.list_runs(tenant_id=other) == []
    finally:
        history.clear(tenant_id=tag)
