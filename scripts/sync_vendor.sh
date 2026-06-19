#!/usr/bin/env bash
# Re-sync the vendored framework copies from the sibling source repos.
# Run from the ai-testing-studio repo root. CI (vendor-drift job) enforces sync.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
cd "$here"

rm -rf vendor/test_case_generator vendor/prompt_regression
cp -r ../ai-test-case-generator/src/test_case_generator vendor/test_case_generator
cp -r ../prompt-regression-suite/src/prompt_regression  vendor/prompt_regression
find vendor -name __pycache__ -type d -prune -exec rm -rf {} +

echo "Re-synced vendor/ from ../ai-test-case-generator and ../prompt-regression-suite."
