"""Durable certification history — a track record across sessions.

Single-user by default: an opt-in local SQLite file (a laptop, a CI runner) that
remembers past certifications so "grade over time" and "did this regress?"
survive a page refresh. Each row stores the exact `core.export_snapshot` JSON, so
any two runs feed straight into `core.compare_snapshots` — history and the
regression diff are the same data, not two systems.

Multi-tenant when you need it (gap #5, Phase 1): set PRS_STUDIO_DB_URL to a
Postgres DSN and the same API writes to a shared server-side database instead,
with every row scoped by `tenant_id` (the isolation boundary) and `user_id` (for
attribution). Reads/writes/deletes filter by tenant, so one tenant never sees
another's runs. (True database row-level security, SSO, and roles are later
phases; this is the storage + data-model swap the rest builds on.)

Safety: local SQLite persistence stays disabled on a public deploy
(PRS_STUDIO_PUBLIC=1) so it can't silently record a shared instance — but a
deliberately configured Postgres backend is enabled, because that IS the
multi-tenant store.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from dataclasses import dataclass

import core

_DEFAULT_DB = os.path.join(os.path.expanduser("~"), ".ai_testing_studio", "history.db")

# The default tenant/user for single-user local mode — existing callers that pass
# no tenant behave exactly as before, all under one "local" tenant.
DEFAULT_TENANT = "local"
DEFAULT_USER = "local"


def _dsn() -> str | None:
    """A Postgres DSN if configured — switches the backend to server-side."""
    return os.environ.get("PRS_STUDIO_DB_URL") or None


def is_enabled() -> bool:
    """History on? A configured Postgres store is always on (it's the shared DB);
    local SQLite stays off on a public deploy so it can't record other users."""
    if _dsn():
        return True
    return os.environ.get("PRS_STUDIO_PUBLIC", "") != "1"


def db_path() -> str:
    """The SQLite file to use — PRS_STUDIO_DB overrides the per-user default."""
    return os.environ.get("PRS_STUDIO_DB") or _DEFAULT_DB


@dataclass
class RunRow:
    run_id: str
    ts: float           # unix seconds
    tenant_id: str
    user_id: str
    label: str
    model: str
    level: str
    grade: str
    status: str
    verdict: str
    pass_rate: float
    passed: int
    total: int
    runs_per_check: int

    @property
    def iso(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.ts))


_COLS = ("run_id", "ts", "tenant_id", "user_id", "label", "model", "level", "grade",
         "status", "verdict", "pass_rate", "passed", "total", "runs_per_check")


# ---- backend abstraction (SQLite | Postgres) --------------------------------

def _is_sqlite(conn) -> bool:
    return isinstance(conn, sqlite3.Connection)


def _exec(conn, sql: str, params=()):
    """Run SQL written with `?` placeholders, translating for Postgres (`%s`).

    The dialect is read off the actual connection object, so passing an explicit
    sqlite connection while a Postgres DSN is set can't mismatch placeholders.
    """
    if not _is_sqlite(conn):
        sql = sql.replace("?", "%s")
    return conn.execute(sql, params)


def _schema_stmts(sqlite_dialect: bool) -> list[str]:
    real = "REAL" if sqlite_dialect else "DOUBLE PRECISION"
    return [
        f"""CREATE TABLE IF NOT EXISTS runs (
            run_id         TEXT PRIMARY KEY,
            ts             {real} NOT NULL,
            tenant_id      TEXT NOT NULL DEFAULT '{DEFAULT_TENANT}',
            user_id        TEXT NOT NULL DEFAULT '{DEFAULT_USER}',
            label          TEXT NOT NULL DEFAULT '',
            model          TEXT NOT NULL,
            level          TEXT NOT NULL,
            grade          TEXT NOT NULL,
            status         TEXT NOT NULL,
            verdict        TEXT NOT NULL,
            pass_rate      {real} NOT NULL,
            passed         INTEGER NOT NULL,
            total          INTEGER NOT NULL,
            runs_per_check INTEGER NOT NULL,
            snapshot_json  TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS idx_runs_tenant_model_ts ON runs (tenant_id, model, ts)",
    ]


def _existing_columns(conn) -> set[str]:
    if _is_sqlite(conn):
        return {r[1] for r in conn.execute("PRAGMA table_info(runs)").fetchall()}
    rows = conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'runs'").fetchall()
    return {r["column_name"] for r in rows}


def _migrate(conn) -> None:
    """Add tenant_id/user_id to a pre-multi-tenant table so old local DBs keep working."""
    cols = _existing_columns(conn)
    for name, default in (("tenant_id", DEFAULT_TENANT), ("user_id", DEFAULT_USER)):
        if name not in cols:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {name} TEXT NOT NULL DEFAULT '{default}'")


def connect(path: str | None = None):
    """Open (and initialize) the history store. Postgres if PRS_STUDIO_DB_URL is
    set, else a local SQLite file. Caller closes it."""
    dsn = _dsn()
    if dsn:
        import psycopg
        from psycopg.rows import dict_row
        conn = psycopg.connect(dsn, row_factory=dict_row)
        _init_schema(conn, sqlite_dialect=False)
        return conn
    path = path or db_path()
    if path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _init_schema(conn, sqlite_dialect=True)
    return conn


def _init_schema(conn, sqlite_dialect: bool) -> None:
    """Create the table, migrate columns on a legacy DB, THEN create the index.

    Order matters: the index references tenant_id, so on an old (pre-tenant) table
    the column must be added before the index is built."""
    table_stmt, index_stmt = _schema_stmts(sqlite_dialect)
    conn.execute(table_stmt)
    _migrate(conn)
    conn.execute(index_stmt)
    conn.commit()


def _row(r) -> RunRow:
    return RunRow(**{k: r[k] for k in _COLS})


# ---- API (tenant-scoped; defaults preserve single-user behaviour) -----------

def save_run(fe: "core.FullEvalResult", label: str = "", conn=None, path: str | None = None,
             tenant_id: str = DEFAULT_TENANT, user_id: str = DEFAULT_USER) -> str:
    """Persist a certification and return its run_id.

    Stores summary columns for fast listing/charting plus the full
    `export_snapshot` JSON for later regression diffing, scoped to `tenant_id`.
    """
    grade, status = core.certification_grade(fe.pass_rate, fe.verdict)
    run_id = uuid.uuid4().hex[:12]
    snapshot = core.export_snapshot(fe)
    own = conn is None
    conn = conn or connect(path)
    try:
        _exec(conn,
              "INSERT INTO runs (run_id, ts, tenant_id, user_id, label, model, level, grade, "
              "status, verdict, pass_rate, passed, total, runs_per_check, snapshot_json) "
              "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (run_id, time.time(), tenant_id, user_id, label, fe.model_name, fe.level, grade,
               status, fe.verdict, float(fe.pass_rate), int(fe.passed), int(fe.total),
               int(fe.runs), snapshot))
        conn.commit()
    finally:
        if own:
            conn.close()
    return run_id


def list_runs(limit: int = 50, model: str | None = None, conn=None, path: str | None = None,
              tenant_id: str = DEFAULT_TENANT) -> list[RunRow]:
    """Most-recent-first summaries for one tenant (no snapshot payload)."""
    own = conn is None
    conn = conn or connect(path)
    try:
        sql = f"SELECT {', '.join(_COLS)} FROM runs WHERE tenant_id = ?"
        params: list = [tenant_id]
        if model:
            sql += " AND model = ?"
            params.append(model)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return [_row(r) for r in _exec(conn, sql, params).fetchall()]
    finally:
        if own:
            conn.close()


def get_snapshot(run_id: str, conn=None, path: str | None = None,
                 tenant_id: str = DEFAULT_TENANT) -> str | None:
    """The stored export_snapshot JSON for a run in this tenant, for diffing."""
    own = conn is None
    conn = conn or connect(path)
    try:
        r = _exec(conn, "SELECT snapshot_json FROM runs WHERE run_id = ? AND tenant_id = ?",
                  (run_id, tenant_id)).fetchone()
        return r["snapshot_json"] if r else None
    finally:
        if own:
            conn.close()


def delete_run(run_id: str, conn=None, path: str | None = None,
               tenant_id: str = DEFAULT_TENANT) -> bool:
    own = conn is None
    conn = conn or connect(path)
    try:
        cur = _exec(conn, "DELETE FROM runs WHERE run_id = ? AND tenant_id = ?",
                    (run_id, tenant_id))
        conn.commit()
        return cur.rowcount > 0
    finally:
        if own:
            conn.close()


def clear(conn=None, path: str | None = None, tenant_id: str = DEFAULT_TENANT) -> int:
    """Delete every run for one tenant. Returns how many were removed."""
    own = conn is None
    conn = conn or connect(path)
    try:
        cur = _exec(conn, "DELETE FROM runs WHERE tenant_id = ?", (tenant_id,))
        conn.commit()
        return cur.rowcount
    finally:
        if own:
            conn.close()


def regression_since_previous(model: str, conn=None, path: str | None = None,
                              tenant_id: str = DEFAULT_TENANT) -> "core.RegressionDiff | None":
    """Diff a model's two most recent runs in this tenant. None if fewer than two.

    Reuses core.compare_snapshots, so the history view and the manual snapshot
    diff report regressions identically.
    """
    own = conn is None
    conn = conn or connect(path)
    try:
        rows = _exec(conn,
                     "SELECT snapshot_json FROM runs WHERE tenant_id = ? AND model = ? "
                     "ORDER BY ts DESC LIMIT 2", (tenant_id, model)).fetchall()
        if len(rows) < 2:
            return None
        after, before = rows[0]["snapshot_json"], rows[1]["snapshot_json"]
        return core.compare_snapshots(before, after)
    finally:
        if own:
            conn.close()
