# AI Testing Studio

A browser UI that makes the whole AI-testing toolchain usable by anyone — no
terminal required. Describe a feature → generate a risk-based test suite → run it
against a model → get a report with a **ship / no-ship verdict**.

It's a thin [Streamlit](https://streamlit.io) shell over two packages:

- [**ai-test-case-generator**](https://github.com/madhavpati23/ai-test-case-generator) — designs the suite + enforces a coverage standard
- [**prompt-regression-suite**](https://github.com/madhavpati23/prompt-regression-suite) — runs it (mock / Claude / any endpoint) + reports a verdict

```
Describe feature ──▶ Generate suite ──▶ Run ──▶ Report + verdict
   (+ AI type,        (coverage check)   (mock /   (HTML, JSON,
    overrides)                            Claude /   downloadable)
                                          endpoint)
```

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL it prints (usually http://localhost:8501).

> Locally, the two framework packages are picked up automatically from the
> sibling repos via [`_bootstrap.py`](_bootstrap.py) — no install needed if
> `ai-test-case-generator/` and `prompt-regression-suite/` sit next to this repo.
> For a standalone/cloud install, `requirements.txt` pulls them from GitHub.

## What you can do

1. **Pick the model under test** (sidebar): offline **Mock**, the **Claude API**
   (paste a key), or **any HTTP endpoint** (URL + body template + response path).
2. **Describe the feature** and pick an **AI type** (chatbot / rag / classifier /
   summarizer / agent). Optionally raise the **coverage bar** for this feature.
3. **Generate** — see the cases (id, category, severity, validator) and a coverage
   check against the standard.
4. **Run** — execute the suite and view the **report** inline, with the verdict
   (**SHIP / NEEDS SIGN-OFF / BLOCK**). Download the HTML/JSON report and the YAML suite.

## Architecture

- [`core.py`](core.py) — the Streamlit-free pipeline (generate → validate →
  coverage → run → report → verdict). Unit-tested in [`tests/`](tests/).
- [`app.py`](app.py) — the UI; a thin layer that calls `core`.

Keeping the logic in `core.py` means the web layer carries no business logic and
the pipeline is testable without a browser.

## Deploy (free)

Push to GitHub and deploy on [Streamlit Community Cloud](https://share.streamlit.io):
point it at `app.py`. `requirements.txt` installs the framework packages from
GitHub, so the deployed app is self-contained.

## License

MIT — see [LICENSE](LICENSE).
