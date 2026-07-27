# Vendored packages — do not edit here

`test_case_generator/` and `prompt_regression/` are **bundled snapshots** of the
two framework repos, copied in so this app is self-contained and can deploy from
a private repo with no external git dependencies.

Edit the source in the upstream repos, not here. To re-sync after changes there:

```bash
rm -rf vendor/test_case_generator vendor/prompt_regression
cp -r ../ai-test-case-generator/src/test_case_generator vendor/test_case_generator
cp -r ../prompt-regression-suite/src/prompt_regression  vendor/prompt_regression
find vendor -name __pycache__ -type d -prune -exec rm -rf {} +
```

## ⚠️ The snapshots have drifted AHEAD of upstream — diff before you sync

The instruction above is currently unsafe to run blind. Fixes have landed
directly in `vendor/` over time, so a blanket re-copy would silently *delete*
working code. Known as of 2026-07-27:

- `prompt_regression/models.py` — vendor has multimodal support (`Attachment`,
  `ask_multimodal`) that upstream does not.
- `prompt_regression/runner.py` — vendor has the concurrent runner
  (`max_workers`, `_one_attempt`, transport-error isolation); upstream is still
  the older sequential version.

So: **diff first, re-sync per file, and port vendor-only changes upstream
before copying anything down.**

```bash
diff -ru ../prompt-regression-suite/src/prompt_regression vendor/prompt_regression
diff -ru ../ai-test-case-generator/src/test_case_generator vendor/test_case_generator
```

`taxonomy.py` is back in sync (2026-07-27) and is the one file that must stay
that way: a category missing from the taxonomy causes every case in it to be
**silently dropped** from the battery, which is how seven risk dimensions once
stopped being tested without a single failing test. `tests/test_battery_integrity.py`
now guards this.
