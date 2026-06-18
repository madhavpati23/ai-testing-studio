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
