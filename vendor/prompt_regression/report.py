"""Human-readable report rendering for a run and a regression diff.

Also exports machine-readable JSON and a shareable, self-contained HTML report
(pass rate by category, severity breakdown, and the release verdict) for
stakeholder reporting.
"""

from __future__ import annotations

import datetime as _dt
import html
import json

from .baseline import Diff
from .gating import Verdict, decide
from .runner import Result, Summary

_LINE = "=" * 64
_THIN = "-" * 64


def render_run(summary: Summary, results: list[Result], sla_ms: float | None = None) -> str:
    out = [_LINE, f"  PROMPT-REGRESSION REPORT  -  model: {summary.model}", _LINE]
    out.append(f"  Total tests : {summary.total}")
    out.append(f"  Passed      : {summary.passed}")
    out.append(f"  Failed      : {summary.failed}")
    out.append(f"  PASS RATE   : {summary.pass_rate:.1f}%")
    perf = performance(results, sla_ms)
    out.append(f"  LATENCY     : avg {perf['avg_ms']} ms, max {perf['max_ms']} ms"
               + (f" (slowest: {perf['slowest_id']})" if perf['slowest_id'] else ""))
    if sla_ms is not None:
        n = len(perf["breaches"])
        out.append(f"  SLA {sla_ms:.0f} ms : {'OK - all within SLA' if n == 0 else f'{n} breach(es): ' + ', '.join(perf['breaches'])}")
    out.append(_THIN)
    out.append("  PASS RATE BY CATEGORY:")
    for cat, (passed, total) in sorted(summary.by_category.items()):
        rate = passed / total * 100 if total else 0.0
        bar = "#" * int(rate // 10)
        out.append(f"   {cat:<14} {passed:>2}/{total:<3} {rate:5.0f}%  {bar}")

    runs = results[0].runs if results else 1
    if runs > 1:
        out.append(_THIN)
        out.append(f"  Each case run {runs}x.")
        flaky = [r for r in results if r.flaky]
        if flaky:
            out.append(f"  FLAKY ({len(flaky)}) — passed only some runs:")
            for r in flaky:
                out.append(f"     ~ {r.case.id}  ({r.passes}/{r.runs} passed)")

    failures = [r for r in results if not r.passed]
    if failures:
        out.append(_THIN)
        out.append("  FAILURES:")
        for r in failures:
            runinfo = f" (passed {r.passes}/{r.runs})" if r.runs > 1 else ""
            out.append(f"   [FAIL] {r.case.id} [{r.case.category}]{runinfo} {r.case.prompt}")
            out.append(f"          got : {r.answer}")
            out.append(f"          why : {r.detail}")
    out.append(_LINE)
    out.append("  NOTE: A high pass rate is evidence of quality on the TESTED")
    out.append("  inputs only -- it is not a certificate of safety. Untested")
    out.append("  inputs remain unknown.")
    out.append(_LINE)
    return "\n".join(out)


def render_diff(diff: Diff) -> str:
    out = [_THIN, "  REGRESSION DIFF vs BASELINE", _THIN]
    if diff.regressed:
        out.append(f"  REGRESSED ({len(diff.regressed)})  <-- newly failing:")
        out.extend(f"     - {cid}" for cid in diff.regressed)
    else:
        out.append("  REGRESSED (0)  -- no previously-passing case is now failing.")
    if diff.fixed:
        out.append(f"  FIXED ({len(diff.fixed)}):")
        out.extend(f"     + {cid}" for cid in diff.fixed)
    if diff.added:
        out.append(f"  ADDED ({len(diff.added)}): {', '.join(diff.added)}")
    if diff.removed:
        out.append(f"  REMOVED ({len(diff.removed)}): {', '.join(diff.removed)}")
    out.append(_THIN)
    return "\n".join(out)


def render_verdict_line(verdict: Verdict) -> str:
    bits = [f"  RELEASE VERDICT: {verdict.decision}"]
    if verdict.blockers:
        bits.append("   blockers: " + ", ".join(r.case.id for r in verdict.blockers))
    if verdict.signoffs:
        bits.append("   needs sign-off: " + ", ".join(r.case.id for r in verdict.signoffs))
    return "\n".join([_THIN] + bits + [_THIN])


def _result_dict(r: Result) -> dict:
    return {
        "id": r.case.id,
        "category": r.case.category,
        "severity": r.case.severity,
        "validator": r.case.validator,
        "args": r.case.args,
        "passed": r.passed,
        "runs": r.runs,
        "passes": r.passes,
        "flaky": r.flaky,
        "prompt": r.case.prompt,
        "answer": r.answer,
        "detail": r.detail,
    }


def performance(results: list[Result], sla_ms: float | None = None) -> dict:
    """Latency summary + optional SLA breaches."""
    if not results:
        return {"avg_ms": 0.0, "max_ms": 0.0, "slowest_id": None, "sla_ms": sla_ms, "breaches": []}
    lat = [(r.latency_ms, r.case.id) for r in results]
    avg = sum(x for x, _ in lat) / len(lat)
    mx, slowest = max(lat)
    breaches = [cid for ms, cid in lat if sla_ms is not None and ms > sla_ms]
    return {"avg_ms": round(avg, 1), "max_ms": round(mx, 1), "slowest_id": slowest,
            "sla_ms": sla_ms, "breaches": breaches}


def build_report(summary: Summary, results: list[Result], sla_ms: float | None = None) -> dict:
    """The single structured report object the JSON/HTML renderers share."""
    verdict = decide(results)
    runs = results[0].runs if results else 1
    flaky = [r.case.id for r in results if r.flaky]
    return {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "model": summary.model,
        "total": summary.total,
        "passed": summary.passed,
        "failed": summary.failed,
        "pass_rate": round(summary.pass_rate, 1),
        "runs_per_case": runs,
        "flaky": flaky,
        "performance": performance(results, sla_ms),
        "verdict": verdict.decision,
        "severity_failed": verdict.severity_failed,
        "by_category": {
            cat: {"passed": p, "total": t, "rate": round(p / t * 100, 1) if t else 0.0}
            for cat, (p, t) in summary.by_category.items()
        },
        "cases": [_result_dict(r) for r in results],
    }


def render_json(summary: Summary, results: list[Result], sla_ms: float | None = None) -> str:
    return json.dumps(build_report(summary, results, sla_ms), indent=2)


_VERDICT_COLOR = {"SHIP": "#1a7f37", "NEEDS SIGN-OFF": "#9a6700", "BLOCK": "#cf222e"}


def _explain_failure(r: dict) -> str:
    """A self-contained reason: which check ran, what it required, and the raw detail.

    Makes the 'Why' column readable on its own instead of a terse validator note.
    """
    v = r.get("validator") or "?"
    args = r.get("args") or {}
    detail = r.get("detail") or ""
    if v in ("contains", "not_contains"):
        cond = "INCLUDE" if v == "contains" else "NOT include"
        return f"[{v}] the answer had to {cond} the exact text “{args.get('value', '')}” — it didn't."
    if v == "regex":
        return f"[regex] the answer had to match the pattern /{args.get('pattern', '')}/. {detail}"
    if v == "equals_number":
        return f"[equals_number] expected the number {args.get('value')} somewhere in the answer. {detail}"
    if v == "json_schema":
        keys = ", ".join((args.get("properties") or {}).keys())
        return f"[json_schema] the answer had to be JSON with keys: {keys}. {detail}"
    if v == "tool_trace":
        return f"[tool_trace] expected tool calls {args.get('expected')}. {detail}"
    if v == "llm_judge":
        crit = args.get("criterion", "")
        base = f"[llm_judge] a model had to grade the answer against: “{crit}”."
        if "ANTHROPIC_API_KEY" in detail or "could not grade" in detail:
            return base + " NOT GRADED — this validator needs a Claude key (ANTHROPIC_API_KEY); it is not a real failure."
        return base + f" {detail}"
    return f"[{v}] {detail}"


def render_html(summary: Summary, results: list[Result], sla_ms: float | None = None) -> str:
    rpt = build_report(summary, results, sla_ms)
    e = html.escape
    color = _VERDICT_COLOR.get(rpt["verdict"], "#57606a")
    perf = rpt["performance"]
    perf_card = f"<div class='card'><b>{perf['avg_ms']} ms</b>avg latency (max {perf['max_ms']})</div>"
    sla_line = ""
    if perf["sla_ms"] is not None:
        n = len(perf["breaches"])
        sla_line = (f"<p>SLA {perf['sla_ms']:.0f} ms: "
                    + ("<b style='color:#1a7f37'>all within SLA</b>"
                       if n == 0 else f"<b style='color:#cf222e'>{n} breach(es)</b> — "
                       + e(', '.join(perf['breaches']))) + "</p>")

    cat_rows = "".join(
        f"<tr><td>{e(cat)}</td><td>{d['passed']}/{d['total']}</td>"
        f"<td><div class='bar'><span style='width:{d['rate']}%'></span></div>{d['rate']}%</td></tr>"
        for cat, d in sorted(rpt["by_category"].items())
    )
    sev = rpt["severity_failed"]
    sev_txt = ", ".join(f"{k}: {v}" for k, v in sorted(sev.items())) or "none"
    flaky_card = (
        f"<div class='card'><b>{len(rpt['flaky'])}</b>flaky (of {rpt['runs_per_case']}x)</div>"
        if rpt["runs_per_case"] > 1 else ""
    )

    fail_rows = "".join(
        f"<tr class='sev-{e(r['severity'])}'><td>{e(r['id'])}</td><td>{e(r['category'])}</td>"
        f"<td>{e(r['severity'])}</td><td>{e(r['prompt'])}</td>"
        f"<td>{e(r['answer'])}</td><td>{e(_explain_failure(r))}</td></tr>"
        for r in rpt["cases"] if not r["passed"]
    ) or "<tr><td colspan='6'>No failures.</td></tr>"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>AI Test Report — {e(rpt['model'])}</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;color:#1f2328;max-width:1000px}}
 h1{{font-size:1.4rem;margin-bottom:.2rem}} .muted{{color:#57606a}}
 .verdict{{display:inline-block;padding:.3rem .8rem;border-radius:6px;color:#fff;font-weight:600;background:{color}}}
 .cards{{display:flex;gap:1rem;margin:1rem 0}} .card{{border:1px solid #d0d7de;border-radius:8px;padding:.6rem 1rem}}
 .card b{{font-size:1.5rem;display:block}}
 table{{border-collapse:collapse;width:100%;margin:1rem 0}} th,td{{border:1px solid #d0d7de;padding:.4rem .6rem;text-align:left;vertical-align:top}}
 th{{background:#f6f8fa}}
 .bar{{display:inline-block;width:120px;height:10px;background:#eaeef2;border-radius:5px;margin-right:6px;vertical-align:middle;overflow:hidden}}
 .bar span{{display:block;height:100%;background:#1a7f37}}
 tr.sev-critical td:nth-child(3){{color:#cf222e;font-weight:600}} tr.sev-high td:nth-child(3){{color:#9a6700;font-weight:600}}
 td{{max-width:340px;overflow-wrap:anywhere}}
</style></head><body>
<h1>AI Test Report</h1>
<p class="muted">Model under test: <b>{e(rpt['model'])}</b> &middot; generated {e(rpt['generated_at'])} UTC</p>
<p>Release verdict: <span class="verdict">{e(rpt['verdict'])}</span></p>
{sla_line}
<div class="cards">
 <div class="card"><b>{rpt['pass_rate']}%</b>pass rate</div>
 <div class="card"><b>{rpt['passed']}/{rpt['total']}</b>passed</div>
 <div class="card"><b>{rpt['failed']}</b>failed</div>
 <div class="card"><b>{e(sev_txt)}</b>failures by severity</div>
 {perf_card}
 {flaky_card}
</div>
<h2>Pass rate by category</h2>
<table><tr><th>Category</th><th>Passed</th><th>Rate</th></tr>{cat_rows}</table>
<h2>Failures</h2>
<table><tr><th>ID</th><th>Category</th><th>Severity</th><th>Prompt sent</th><th>Model's answer (Got)</th><th>Why it failed (which check &amp; what it required)</th></tr>{fail_rows}</table>
<p class="muted">A high pass rate is evidence of quality on the tested inputs only — it is not a certificate of safety.</p>
</body></html>"""
