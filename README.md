# AI Testing Studio

A browser UI that makes the whole AI-testing toolchain usable by anyone — no
terminal required. Describe a feature → generate a risk-based test suite → run it
against a model → get a report with a **ship / no-ship verdict**.

▶️ **Live demo** (offline mock, no signup): https://ai-testing-studio-jsrj4bqyatgfc7jzz8qzgz.streamlit.app/

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

The UI is organised into three working tabs (plus a "How it works" reference):

**🧪 Test a feature** — the main flow:
1. **Pick the model under test** (sidebar): offline **Mock**, the **Claude API**
   (paste a key), or **any HTTP endpoint** (OpenAI-compatible preset built in).
2. **Describe the feature** — a short phrase or a **full user story with
   acceptance criteria** — pick an **AI type**, and **declare what the AI can do**
   (takes actions? returns JSON? stateful?) so only fitting cases are generated.
   Optionally raise the **coverage bar**.
3. **Generate** a **starter scaffold** of cases (id, category, severity,
   validator) + a coverage check against the standard. The offline mock fills
   generic risk-category templates — a starting point a human (or the Claude
   backend) tailors, not a finished suite.
4. **Run** (with an optional **SLA in ms**) — view the **report** inline with
   metric tiles (pass rate, verdict, avg latency) and the verdict
   (**SHIP / NEEDS SIGN-OFF / BLOCK**). Download the HTML/JSON report and the YAML suite.

**✍️ Prompt & instructions** — paste a **prompt** or an **agent's instructions**
and get a quality score, a few concise pointers, and a concrete **suggested
rewrite** (task-type aware; Claude does a fully tailored rewrite when available).

**📄 Example audit** — a real adversarial audit run with this methodology: 13
sharp probes against a live model, judged with explicit pass criteria, with a
documented defect and a ship / no-ship verdict.

**ℹ️ How it works** — the flow, the risk categories, and the verdict legend.

## Architecture

- [`core.py`](core.py) — the Streamlit-free pipeline (generate → validate →
  coverage → run → report → verdict). Unit-tested in [`tests/`](tests/).
- [`app.py`](app.py) — the UI; a thin layer that calls `core`.

Keeping the logic in `core.py` means the web layer carries no business logic and
the pipeline is testable without a browser.

## Deploy (free)

Push to GitHub and deploy on [Streamlit Community Cloud](https://share.streamlit.io):
point it at `app.py`. The framework packages are **bundled in `vendor/`**, so the
app is fully self-contained — **it deploys from a private repo with no external
dependencies, and the live app URL is still public.**

**For a public instance, set `PRS_STUDIO_PUBLIC=1`** (in the app's *Advanced
settings → Secrets/env*). This is important: the app serves all sessions from
one process, so a public instance must not accept secrets or arbitrary URLs.
With it set, the Studio restricts to the **offline mock** — no API-key field, no
outbound requests. Visitors get the full generate → run → report demo safely;
anyone who wants to test a real model clones the repo and runs it locally.

## Security

- **Public mode** (`PRS_STUDIO_PUBLIC=1`) disables the Claude/HTTP backends so
  the shared instance handles no secrets and makes no outbound calls.
- The HTTP adapter only allows `http`/`https` (no `file://` etc.) and, with
  `PRS_HTTP_BLOCK_PRIVATE=1`, refuses private/loopback/metadata addresses (SSRF)
  and won't follow redirects — see prompt-regression-suite.
- All YAML is parsed with `safe_load`; there is no `eval`/`exec`/`subprocess`.

## License

MIT — see [LICENSE](LICENSE).
