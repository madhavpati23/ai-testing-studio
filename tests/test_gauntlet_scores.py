"""Sir Leaks-a-Lot progress persistence + leaderboard (history.py)."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import history


def test_save_get_upsert(tmp_path):
    db = str(tmp_path / "h.db")
    history.save_gauntlet("neo", solved=2, attempts=9, best_level=2, path=db)
    row = history.get_gauntlet("neo", path=db)
    assert row.solved == 2 and row.attempts == 9 and row.best_level == 2
    # upsert: same player updates in place (no duplicate)
    history.save_gauntlet("neo", solved=4, attempts=15, best_level=4, path=db)
    assert history.get_gauntlet("neo", path=db).solved == 4
    assert len(history.gauntlet_leaderboard(path=db)) == 1


def test_leaderboard_ranks_by_solved_then_attempts(tmp_path):
    db = str(tmp_path / "h.db")
    history.save_gauntlet("trinity", solved=7, attempts=40, best_level=7, path=db)
    history.save_gauntlet("neo",     solved=7, attempts=22, best_level=7, path=db)  # fewer attempts
    history.save_gauntlet("cypher",  solved=3, attempts=5,  best_level=3, path=db)
    board = history.gauntlet_leaderboard(path=db)
    assert [r.player for r in board] == ["neo", "trinity", "cypher"]


def test_gauntlet_scores_are_tenant_scoped(tmp_path):
    db = str(tmp_path / "h.db")
    history.save_gauntlet("neo", 5, 10, 5, path=db, tenant_id="acme")
    history.save_gauntlet("neo", 1, 2, 1, path=db, tenant_id="globex")
    assert history.get_gauntlet("neo", path=db, tenant_id="acme").solved == 5
    assert len(history.gauntlet_leaderboard(path=db, tenant_id="acme")) == 1
    assert history.get_gauntlet("neo", path=db, tenant_id="globex").solved == 1
