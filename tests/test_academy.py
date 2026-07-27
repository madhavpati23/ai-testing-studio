"""Red-Team Academy curriculum + progress logic."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import academy as A
import history


def test_curriculum_shape():
    assert len(A.MODULES) >= 6
    ids = [l.id for l in A.all_lessons()]
    assert len(ids) == len(set(ids)) == A.total_lessons()          # unique lesson ids
    for l in A.all_lessons():
        assert l.concept and l.examples and l.task and l.quiz_q
        assert 0 <= l.quiz_answer < len(l.quiz_options)            # answer in range


def test_progress_and_graduation():
    assert A.progress_pct(set()) == 0.0
    assert not A.is_graduate(set())
    everything = {l.id for l in A.all_lessons()}
    assert A.progress_pct(everything) == 100.0
    assert A.is_graduate(everything)
    assert len(A.modules_completed(everything)) == len(A.MODULES)


def test_module_completion_requires_all_its_lessons():
    m = A.MODULES[0]
    partial = {m.lessons[0].id}
    assert not A.module_done(m, partial) if len(m.lessons) > 1 else A.module_done(m, partial)
    full = {l.id for l in m.lessons}
    assert A.module_done(m, full)


def test_resume_bullets_grow_with_progress():
    assert "in progress" in " ".join(A.resume_bullets(set())).lower()
    grad = A.resume_bullets({l.id for l in A.all_lessons()})
    assert any("red-teaming program" in b.lower() for b in grad)


def test_certificate_html_renders():
    html = A.certificate_html("Ada Lovelace", {l.id for l in A.all_lessons()})
    assert "Ada Lovelace" in html and "PRACTITIONER" in html and "<table" in html


def test_kv_store_roundtrip(tmp_path):
    db = str(tmp_path / "h.db")
    assert history.get_kv("neo", "academy", path=db) is None
    history.save_kv("neo", "academy", '["m1l1","m1l2"]', path=db)
    assert history.get_kv("neo", "academy", path=db) == '["m1l1","m1l2"]'
    history.save_kv("neo", "academy", '["m1l1"]', path=db)                  # upsert
    assert history.get_kv("neo", "academy", path=db) == '["m1l1"]'
    # tenant isolation
    assert history.get_kv("neo", "academy", path=db, tenant_id="other") is None
