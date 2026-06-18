"""Command-line entrypoint.

  python -m test_case_generator generate --feature "password reset email" --out-dir prompts
  python -m test_case_generator generate --spec feature.txt --out-dir prompts [--strict]
  python -m test_case_generator generate --config suite.yaml [--strict]
  python -m test_case_generator coverage --prompts prompts [--config suite.yaml] [--approved-only] [--strict]
  python -m test_case_generator review --prompts prompts [--strict]

`generate` creates cases (mock by default, Claude if ANTHROPIC_API_KEY is set),
validates each against the schema, writes the survivors as prompt-regression-suite
YAML, and reports coverage against the standard in taxonomy.py.

`coverage` assesses an existing suite directory against the same standard.

Exit codes (so CI can gate on them):
  0  ok
  1  nothing valid to write / bad input
  2  --strict was set and the suite is below the required coverage standard
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import yaml

from . import coverage as cov
from . import prompt_quality
from .config import load_config
from .generators import get_generator
from .schema import Case, validate_all
from .serialize import write_suite


def _load_cases(prompts_dir: str) -> list[Case]:
    raw: list[dict] = []
    for path in sorted(glob.glob(os.path.join(prompts_dir, "*.yaml"))):
        doc = yaml.safe_load(open(path, encoding="utf-8")) or {}
        category = doc.get("category", os.path.splitext(os.path.basename(path))[0])
        for case in doc.get("cases", []):
            case.setdefault("category", category)
            raw.append(case)
    return validate_all(raw).cases


def _cmd_generate(args) -> int:
    feature = args.feature
    ai_type = None
    out_dir = args.out_dir
    overrides: dict[str, int] = {}

    if args.config:
        cfg = load_config(args.config)
        feature, ai_type, overrides = cfg.feature, cfg.ai_type, cfg.coverage
        # an explicit --out-dir wins over the config's default
        out_dir = args.out_dir if args.out_dir != "prompts" else cfg.out_dir
    elif args.spec:
        feature = open(args.spec, encoding="utf-8").read().strip()
    if not feature:
        print("error: empty feature description", file=sys.stderr)
        return 1

    generator = get_generator()
    raw = generator.generate(feature, ai_type)
    result = validate_all(raw)

    print(f"Generator   : {generator.name}")
    print(f"Generated   : {len(raw)} raw case(s)")
    print(f"Valid       : {len(result.cases)}")
    if result.errors:
        print(f"Dropped     : {len(result.errors)}")
        for err in result.errors:
            print(f"   - {err}")
    if not result.cases:
        print("No valid cases to write.", file=sys.stderr)
        return 1

    paths = write_suite(result.cases, out_dir)
    print(f"Wrote {len(result.cases)} case(s) across {len(paths)} file(s) to {out_dir}")

    report = cov.assess(result.cases, overrides)
    print(cov.render(report))
    if args.strict and report.has_gaps:
        return 2
    return 0


def _cmd_review(args) -> int:
    cases = _load_cases(args.prompts)
    if not cases:
        print(f"No valid cases found in {args.prompts}", file=sys.stderr)
        return 1
    by_status: dict[str, list] = {}
    for c in cases:
        by_status.setdefault(c.status, []).append(c)

    print(f"Review status for {args.prompts} ({len(cases)} case(s)):")
    for status in ("approved", "reviewed", "draft"):
        print(f"  {status:<9}: {len(by_status.get(status, []))}")

    unapproved = [c for c in cases if c.status != "approved"]
    if unapproved:
        print(f"  Unapproved ({len(unapproved)}) -- not release-ready:")
        for c in unapproved:
            print(f"    [{c.status:<8}] {c.id}  ({c.severity} {c.category})")
    else:
        print("  All cases approved.")

    if args.strict and unapproved:
        return 2
    return 0


def _cmd_assess_prompt(args) -> int:
    text = args.text if args.text else open(args.file, encoding="utf-8").read()
    score = prompt_quality.assess_llm(text) if args.llm else prompt_quality.assess(text)
    print(f"Prompt score : {score.score}/100  ({score.level})")
    print(f"  {score.summary}")
    if score.strengths:
        print("  Strengths: " + ", ".join(score.strengths))
    if score.suggestions:
        print("  Consider:")
        for s in score.suggestions:
            print(f"   - {s}")
    return 0


def _cmd_coverage(args) -> int:
    cases = _load_cases(args.prompts)
    if not cases:
        print(f"No valid cases found in {args.prompts}", file=sys.stderr)
        return 1
    approved_only = getattr(args, "approved_only", False)
    if approved_only:
        cases = [c for c in cases if c.status == "approved"]
        print("(release view: counting approved cases only)")
    overrides = load_config(args.config).coverage if args.config else {}
    report = cov.assess(cases, overrides)
    print(f"Assessed {report.total_cases} case(s) in {args.prompts}")
    print(cov.render(report))
    if args.strict and report.has_gaps:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="test_case_generator")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a test suite for a feature")
    src = gen.add_mutually_exclusive_group(required=True)
    src.add_argument("--feature", help="feature/requirement described inline")
    src.add_argument("--spec", help="path to a file describing the feature")
    src.add_argument("--config", help="path to a suite.yaml (feature + coverage overrides)")
    gen.add_argument("--out-dir", default="prompts", help="where to write the YAML suite")
    gen.add_argument("--strict", action="store_true", help="exit 2 if below the coverage standard")
    gen.set_defaults(func=_cmd_generate)

    cvg = sub.add_parser("coverage", help="assess an existing suite against the standard")
    cvg.add_argument("--prompts", default="prompts", help="suite directory of *.yaml")
    cvg.add_argument("--config", help="suite.yaml whose coverage overrides to apply")
    cvg.add_argument("--approved-only", action="store_true",
                     help="count only approved cases (release-baseline view)")
    cvg.add_argument("--strict", action="store_true", help="exit 2 if below the coverage standard")
    cvg.set_defaults(func=_cmd_coverage)

    rev = sub.add_parser("review", help="show the review/approval status of a suite")
    rev.add_argument("--prompts", default="prompts", help="suite directory of *.yaml")
    rev.add_argument("--strict", action="store_true", help="exit 2 if any case is unapproved")
    rev.set_defaults(func=_cmd_review)

    ap = sub.add_parser("assess-prompt", help="score how well-written a prompt is")
    apsrc = ap.add_mutually_exclusive_group(required=True)
    apsrc.add_argument("--text", help="the prompt text to score")
    apsrc.add_argument("--file", help="path to a file containing the prompt")
    ap.add_argument("--llm", action="store_true", help="use Claude for the critique (needs API key)")
    ap.set_defaults(func=_cmd_assess_prompt)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
