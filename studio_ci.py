"""Headless certification for CI/CD — the Studio's grade as a pipeline gate.

The Studio UI answers "is this AI ready?" for a human. This runs the *same*
engine (core.run_full_evaluation) with no browser, writes machine-readable
reports (JSON + JUnit XML), and sets a non-zero exit code when the AI fails the
policy — so a certificate can gate a pull request the way any test suite does.

Nothing new is graded here: it reuses make_model / make_judge / run_full_evaluation
/ certification_grade, so a CI result matches what the UI would show for the same
config. Safety cases are judge-graded (unless --no-judge or a mock backend) and
safety-critical cases run worst-case, exactly as in the app.

Examples:
  # Offline smoke test (no keys, no network) — regex-graded, proves the wiring.
  python studio_ci.py --backend mock --level quick

  # Certify a deployed HTTP agent; fail the pipeline on a BLOCK verdict.
  python studio_ci.py --backend http --url "$AGENT_URL" \
      --response-path output --level standard --fail-on block \
      --json report.json --junit report.junit.xml

  # Gate on a minimum letter grade AND run the user's ground truth.
  python studio_ci.py --backend claude --min-grade B --golden truth.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from xml.sax.saxutils import escape, quoteattr

import _bootstrap  # noqa: F401  (adds the framework packages to sys.path)
import core

# --fail-on policies, worst→lenient. A run "fails" (exit 1) when its verdict is
# at or above the chosen bar, or (for "any") when any single check failed.
_FAIL_ON = ("block", "signoff", "any", "never")
_GRADE_RANK = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}


def build_backend_opts(args: argparse.Namespace) -> dict:
    """Assemble the make_model/make_judge opts dict from parsed CLI args."""
    opts: dict = {}
    if args.api_key:
        opts["api_key"] = args.api_key
    if args.url:
        opts["url"] = args.url
    if args.headers:
        opts["headers"] = args.headers        # JSON string, parsed downstream
    if args.body:
        opts["body"] = args.body
    if args.response_path:
        opts["response_path"] = args.response_path
    if args.method:
        opts["method"] = args.method
    # SSRF protection on by default for HTTP backends; --allow-private opts out.
    opts["block_private"] = not args.allow_private
    return opts


def _pooled_results(fe: "core.FullEvalResult") -> list:
    """Every graded check behind the certificate, flattened (battery + agent)."""
    pooled = []
    for _name, rr in fe.sections:
        pooled += list(rr.results)
    pooled += list(fe.agent_checks)
    return pooled


def to_report(fe: "core.FullEvalResult", grade: str, status: str) -> dict:
    """A JSON-serializable summary of a full evaluation."""
    pooled = _pooled_results(fe)
    return {
        "model": fe.model_name,
        "level": fe.level,
        "grade": grade,
        "status": status,
        "verdict": fe.verdict,
        "pass_rate": round(fe.pass_rate, 2),
        "passed": fe.passed,
        "total": fe.total,
        "runs_per_check": fe.runs,
        "by_category": {k: {"passed": v[0], "total": v[1]}
                        for k, v in fe.by_category.items()},
        "sections": [name for name, _ in fe.sections],
        "checks": [{
            "id": r.case.id,
            "category": r.case.category,
            "severity": r.case.severity,
            "passed": bool(r.passed),
            "flaky": bool(getattr(r, "flaky", False)),
            "runs": getattr(r, "runs", 1),
            "passes": getattr(r, "passes", 1),
            "detail": r.detail or "",
        } for r in pooled],
    }


def to_junit(fe: "core.FullEvalResult", report: dict) -> str:
    """Render the pooled checks as a JUnit XML suite (no external dependency).

    One <testcase> per check; a failed check carries a <failure> whose message is
    the validator's detail. Most CI systems (GitHub Actions, GitLab, Jenkins)
    render this natively, so a failed safety case shows up as a named test.
    """
    pooled = _pooled_results(fe)
    failures = sum(1 for r in pooled if not r.passed)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append(
        f'<testsuite name="ai-studio-certification" '
        f'tests="{len(pooled)}" failures="{failures}" errors="0" '
        f'timestamp={quoteattr(str(report["model"]))}>')
    for r in pooled:
        name = quoteattr(r.case.id)
        classname = quoteattr(f"{r.case.category}.{r.case.severity}")
        lines.append(f'  <testcase name={name} classname={classname}>')
        if not r.passed:
            msg = quoteattr((r.detail or "check failed")[:200])
            lines.append(f'    <failure message={msg}>{escape(r.detail or "")}</failure>')
        lines.append('  </testcase>')
    lines.append('</testsuite>')
    return "\n".join(lines)


def decide_exit(fe: "core.FullEvalResult", grade: str, fail_on: str,
                min_grade: str | None) -> tuple[int, str]:
    """Return (exit_code, reason). 0 = pass the pipeline, 1 = fail it."""
    if min_grade and _GRADE_RANK[grade] > _GRADE_RANK[min_grade.upper()]:
        return 1, f"grade {grade} is below the required minimum {min_grade.upper()}"
    if fail_on == "block" and fe.verdict == "BLOCK":
        return 1, "verdict is BLOCK (a critical/high-safety check failed)"
    if fail_on == "signoff" and fe.verdict in ("BLOCK", "NEEDS SIGN-OFF"):
        return 1, f"verdict is {fe.verdict}"
    if fail_on == "any" and fe.passed < fe.total:
        return 1, f"{fe.total - fe.passed} of {fe.total} checks failed"
    return 0, "policy satisfied"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="studio_ci",
        description="Run the AI Testing Studio certification headlessly and gate a pipeline on it.")
    b = p.add_argument_group("backend (the AI under test)")
    b.add_argument("--backend", choices=list(core._BACKEND_KIND.values()) if hasattr(core, "_BACKEND_KIND")
                   else ["mock", "claude", "http", "http_agent"], default="mock",
                   help="mock | claude | http | http_agent (default: mock)")
    b.add_argument("--url", help="endpoint URL for http / http_agent backends")
    b.add_argument("--api-key", help="API key (else read from the environment, e.g. ANTHROPIC_API_KEY)")
    b.add_argument("--headers", help="extra HTTP headers as a JSON object string")
    b.add_argument("--body", help="HTTP body template (use {PROMPT} as the placeholder)")
    b.add_argument("--response-path", help="dotted path to the answer in the JSON response (http)")
    b.add_argument("--method", default="POST", help="HTTP method (default: POST)")
    b.add_argument("--allow-private", action="store_true",
                   help="disable SSRF protection (allow private/loopback endpoints)")

    e = p.add_argument_group("evaluation")
    e.add_argument("--level", choices=["quick", "standard", "thorough", "deep"], default="standard")
    e.add_argument("--repeat", type=int, default=1, help="runs per check (default: 1)")
    e.add_argument("--critical-repeat", type=int, default=3,
                   help="worst-case runs for safety-critical checks (default: 3)")
    e.add_argument("--stress", type=int, default=0, help="add N randomized stress probes")
    e.add_argument("--workers", type=int, default=1,
                   help="concurrent model calls (default: 1). Speeds up network-bound "
                        "backends; keep modest to respect provider rate limits.")
    e.add_argument("--golden", help="ground-truth file (.csv/.xlsx/.pdf) to fold into the grade")
    e.add_argument("--multimodal", action="store_true",
                   help="also run the image red-team battery (needs a vision backend, e.g. claude)")
    e.add_argument("--system-prompt", help="system prompt the AI runs under")
    e.add_argument("--system-prompt-file", help="read the system prompt from a file")
    e.add_argument("--no-judge", action="store_true",
                   help="skip the LLM judge (safety cases fall back to regex — weaker)")

    g = p.add_argument_group("gate & output")
    g.add_argument("--fail-on", choices=_FAIL_ON, default="signoff",
                   help="exit non-zero when the verdict reaches this bar (default: signoff)")
    g.add_argument("--min-grade", choices=["A", "B", "C", "D", "F"],
                   help="also fail if the letter grade is below this")
    g.add_argument("--json", dest="json_path", help="write the JSON report here")
    g.add_argument("--junit", dest="junit_path", help="write the JUnit XML report here")
    g.add_argument("--save", action="store_true",
                   help="persist this run to the local history store (for grade-over-time)")
    g.add_argument("--db", help="history SQLite path (default: $PRS_STUDIO_DB or ~/.ai_testing_studio)")
    g.add_argument("--label", default="", help="a label for this run in history (e.g. a git SHA)")
    g.add_argument("--tenant", default="local",
                   help="tenant to scope this run under (multi-tenant Postgres history)")
    g.add_argument("--user", default="local", help="user id to attribute this run to")
    g.add_argument("--quiet", action="store_true", help="print only the one-line verdict")
    return p


def _resolve_system_prompt(args: argparse.Namespace) -> str | None:
    if args.system_prompt_file:
        with open(args.system_prompt_file, encoding="utf-8") as fh:
            return fh.read().strip() or None
    return (args.system_prompt or "").strip() or None


def run(args: argparse.Namespace, out=sys.stdout) -> int:
    kind = args.backend
    opts = build_backend_opts(args)
    system_prompt = _resolve_system_prompt(args)

    if kind in ("http", "http_agent") and not opts.get("url"):
        print("error: --url is required for the http / http_agent backends", file=sys.stderr)
        return 2

    golden_cases = None
    if args.golden:
        with open(args.golden, "rb") as fh:
            golden_cases, gerrors = core.build_golden_from_file(fh.read(), args.golden)
        for err in gerrors:
            print(f"golden: {err}", file=sys.stderr)

    model = core.make_model(kind, opts, system_prompt=system_prompt)
    judge = None
    if not args.no_judge and kind != "mock":
        try:
            judge = core.make_judge(kind, opts, system_prompt=system_prompt)
        except Exception as exc:                       # judge is best-effort
            print(f"warning: could not build judge ({exc}); safety cases use regex fallback",
                  file=sys.stderr)

    agent_checks = None
    if args.multimodal:
        if not hasattr(model, "ask_multimodal"):
            print("error: --multimodal needs a vision backend (e.g. --backend claude); "
                  f"{kind!r} can't take image inputs", file=sys.stderr)
            return 2
        import multimodal
        mm = multimodal.run_multimodal_battery(model, judge=judge)
        agent_checks = multimodal.multimodal_checks(mm)

    fe = core.run_full_evaluation(
        model, golden_cases=golden_cases or None, judge=judge,
        level=args.level, repeat=args.repeat, stress_n=args.stress,
        agent_checks=agent_checks,
        critical_repeat=args.critical_repeat, max_workers=args.workers)

    grade, status = core.certification_grade(fe.pass_rate, fe.verdict)
    report = to_report(fe, grade, status)

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    if args.junit_path:
        with open(args.junit_path, "w", encoding="utf-8") as fh:
            fh.write(to_junit(fe, report))
    if args.save:
        import history
        run_id = history.save_run(fe, label=args.label, path=args.db,
                                  tenant_id=args.tenant, user_id=args.user)
        if not args.quiet:
            print(f"  Saved to history as {run_id}"
                  + (f" (label: {args.label})" if args.label else ""), file=out)

    exit_code, reason = decide_exit(fe, grade, args.fail_on, args.min_grade)

    if args.quiet:
        print(f"{status} — Grade {grade} — {fe.verdict} "
              f"({fe.passed}/{fe.total}) — {'PASS' if exit_code == 0 else 'FAIL'}", file=out)
    else:
        print(f"\n  AI Testing Studio — certification ({fe.level})", file=out)
        print(f"  Model:   {fe.model_name}", file=out)
        print(f"  Grade:   {grade}   Status: {status}", file=out)
        print(f"  Verdict: {fe.verdict}   Score: {fe.pass_rate:.0f}% ({fe.passed}/{fe.total})", file=out)
        if fe.by_category:
            print("  By category:", file=out)
            for cat, (pw, tot) in sorted(fe.by_category.items()):
                flag = "" if pw == tot else "  <-- failures"
                print(f"    {cat:<20} {pw}/{tot}{flag}", file=out)
        fails = [r for r in _pooled_results(fe) if not r.passed]
        if fails:
            print("  Failed checks:", file=out)
            for r in fails[:20]:
                print(f"    [{r.case.severity}] {r.case.id}: {r.detail}", file=out)
            if len(fails) > 20:
                print(f"    ... and {len(fails) - 20} more", file=out)
        gate = "PASS" if exit_code == 0 else "FAIL"
        print(f"\n  Pipeline gate: {gate} ({reason})\n", file=out)

    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"certification errored: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
