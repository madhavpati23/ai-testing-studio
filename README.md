# AI Evaluation Studio

*(formerly "AI Testing Studio" — the repo/URL keep the old slug)*

A browser UI that makes the whole AI-testing toolchain usable by anyone — no
terminal required. Describe a feature → generate a risk-based test suite → run it
against a model → get a report with a **ship / no-ship verdict**.

▶️ **Live demo** (offline Demo bot, no signup): https://ai-testing-studio-jsrj4bqyatgfc7jzz8qzgz.streamlit.app/

> **A user story is optional.** AI testing needs two things — an *oracle* (what a
> correct answer is) and the *right inputs* (including adversarial ones). A user
> story is just one handy source of the oracle when the AI implements a defined
> feature. Plenty of the most valuable testing — **behavioural model audits,
> red-teaming, bias/safety** — has no story at all (see the **Example audit**).

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

One idea — **give an AI a verdict you can defend** — organised as a journey, not a
pile of peer tabs. First, in the sidebar, pick the **model under test**: the offline
**Demo bot** (a planted-bug dummy, free, no key) or a real model — **Claude** or
**Groq (free)** / any HTTP endpoint — with **your own key** (kept in your session,
never written to the server). Then:

**👋 Start here** — the one idea (the *three roles*: model under test · designer /
your ground truth · judge) and a "pick your path" map.

**🎯 Evaluate** — the core. One question, **four ways to answer it**:
- **🏁 Full evaluation** — the integrated run: certification **plus your golden set**
  against one model, pooled into **one** cross-dimension scorecard and verdict. This is
  where the tabs work together — "is this model good?" answered in a single click.
- **Against your ground truth** — upload an `input → expected` CSV; the verdict is
  judged against truth *you* defined (the most trustworthy run).
- **Across risk dimensions** — a fixed **deploy-readiness certification** (~22 probes:
  injection, hallucination, bias, PII, …) with a per-dimension scorecard, the failing
  probes, and a SHIP / NEEDS-SIGN-OFF / BLOCK verdict.
- **From a feature description** — generate a draft suite (a real model designs
  feature-specific cases; the Demo bot fills generic scaffolds), review, and run it
  with optional **runs-per-case** (non-determinism / flaky detection) and an **SLA**.

> **The tabs work together.** Calibrate a judge once (⚖️ Judge) and it's reused for
> every `llm_judge` grading in Evaluate and Multi-turn — *calibrate once, trusted
> everywhere* — and all tabs run against the one **model under test** you pick in the
> sidebar.

**🔁 Behaviors** — specialised agent checks: **Multi-turn** (memory, context & scope
across a conversation) and **RAG grounding** (is the answer faithful to a provided
source, or hallucinated beyond it — *grounded / grounded-but-wrong / not grounded*).

**⚖️ Judge** — calibrate an **LLM-as-judge** against your own human labels
(`criterion, answer, human_pass`) and see how often it agrees with you (**agreement %**
→ *trustworthy / use with caution / do not trust*). Backend-agnostic, so open-ended
quality can be graded without a Claude key; it also grades `llm_judge` cases in a run.

**✍️ Prompt scorer** — a utility: paste a **prompt** or **agent instructions** and get
a quality score, pointers, and a concrete **suggested rewrite**.

**🎓 Practice** — learn AI testing hands-on: a randomised bank of **500+ probes
across 19 skills** (injection, hallucination, bias, PII, over-refusal, …). Fire a
probe at the bot, record your verdict, then **reveal** what an expert looks for.
Auto-scored against the Demo bot (which has planted bugs, so failures are real to
catch with no key); filter by skill or difficulty.

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

**Bring-your-own-key is safe on a shared instance:** a user's key is kept only in
their **Streamlit session** and built into the model object **per request** — it is
never written to the server's environment, stored, or logged (see
[`core.make_model`](core.py)). So visitors can test real models with their own key
without a local install. Setting **`PRS_STUDIO_PUBLIC=1`** is still recommended on a
public deploy (it's a no-op for keys now, but documents intent).

## Security

- **Session-scoped keys** — API keys are passed per request via `core.make_model`,
  never placed in the process environment, so one user's key can't bleed into
  another's session.
- **SSRF guard on by default** — the HTTP backend refuses private / loopback /
  metadata addresses and won't follow redirects. A user must explicitly tick
  *"Allow private / localhost addresses"* to reach a trusted local endpoint
  (e.g. Ollama). Only `http`/`https` schemes are allowed (no `file://`).
- **Rate-limit resilient** — transient `429`/`503` are retried with backoff
  (honouring `Retry-After`).
- All YAML is parsed with `safe_load`; there is no `eval`/`exec`/`subprocess`.

## License

MIT — see [LICENSE](LICENSE).
