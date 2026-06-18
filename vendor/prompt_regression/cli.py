"""Command-line entrypoint.

  python -m prompt_regression run                       # run the suite, print a report
  python -m prompt_regression run --baseline B          # also diff against baseline B
  python -m prompt_regression run --repeat 5 --pass-threshold 0.8   # N runs, gate on pass rate
  python -m prompt_regression run --html r.html --json r.json   # export shareable reports
  python -m prompt_regression update-baseline B         # save current run as baseline B

Exit codes (so CI can gate on them):
  0  all good
  1  one or more cases failed (with no baseline given)
  2  a regression was detected against the baseline
"""

from __future__ import annotations

import argparse
import os
import sys

from . import baseline as bl
from .gating import decide
from .models import get_model
from .report import (
    render_diff, render_html, render_json, render_run, render_verdict_line,
)
from .runner import load_cases, run_suite, summarize

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROMPTS = os.path.join(_HERE, "prompts")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prompt_regression")
    parser.add_argument("--prompts", default=_PROMPTS, help="directory of *.yaml suites")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the suite and report")
    p_run.add_argument("--baseline", help="baseline JSON to diff against")
    p_run.add_argument("--json", dest="json_path", help="write a JSON report to this path")
    p_run.add_argument("--html", dest="html_path", help="write an HTML report to this path")
    p_run.add_argument("--repeat", type=int, default=1,
                       help="run each case N times (handles non-determinism)")
    p_run.add_argument("--pass-threshold", type=float, default=1.0,
                       help="fraction of runs that must pass for a case to pass (0-1; default 1.0)")

    p_up = sub.add_parser("update-baseline", help="save current run as a baseline")
    p_up.add_argument("path", help="where to write the baseline JSON")

    args = parser.parse_args(argv)

    model = get_model()
    cases = load_cases(args.prompts)
    repeat = max(1, getattr(args, "repeat", 1))
    threshold = getattr(args, "pass_threshold", 1.0)
    results = run_suite(model, cases, repeat=repeat, pass_threshold=threshold)
    summary = summarize(model.name, results)

    if args.command == "update-baseline":
        bl.save(results, args.path)
        print(f"Saved baseline for {model.name} ({summary.passed}/{summary.total} passing) -> {args.path}")
        return 0

    print(render_run(summary, results))
    print(render_verdict_line(decide(results)))

    if getattr(args, "json_path", None):
        with open(args.json_path, "w", encoding="utf-8") as fh:
            fh.write(render_json(summary, results))
        print(f"  Wrote JSON report -> {args.json_path}")
    if getattr(args, "html_path", None):
        with open(args.html_path, "w", encoding="utf-8") as fh:
            fh.write(render_html(summary, results))
        print(f"  Wrote HTML report -> {args.html_path}")

    if args.command == "run" and args.baseline:
        diff = bl.diff(bl.load(args.baseline), results)
        print(render_diff(diff))
        if diff.has_regressions:
            return 2
        return 0

    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
