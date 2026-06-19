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

The UI is organised into eight working tabs (plus a "How it works" reference):

**🧪 Test a feature** — the main flow:
1. **Pick the model under test** (sidebar): the offline **Demo bot** (a dummy with
   planted bugs, for free demos + Practice), the **Claude API** (paste a key), or
   **any HTTP endpoint** (**Groq** free + OpenAI-compatible presets built in).
2. **Describe the feature** — a short phrase or a full user story. *(Advanced
   options, hidden by default: declare what the AI can do so only fitting cases
   are generated, and raise the coverage bar.)*
3. **Generate** a **starter scaffold** of cases (id, category, severity,
   validator) + a coverage check against the standard. The offline Demo bot fills
   generic risk-category templates; **select a real backend (Groq/Claude) to have
   the model design feature-specific cases** instead of the generic scaffold.
4. **Run** (optional **runs-per-case** for non-determinism + **SLA in ms**) — view
   the **report** inline with metric tiles (pass rate, verdict, avg latency,
   flaky cases) and the verdict (**SHIP / NEEDS SIGN-OFF / BLOCK**). Download the
   HTML/JSON report and the YAML suite.
5. **🛡️ Deploy-readiness certification** — run a fixed, comprehensive battery
   (~22 probes across every risk dimension) against the chosen bot, with a
   per-dimension scorecard, the failing probes (and the bot's replies), and a
   certification verdict. *Risk-based, not absolute — a strong general bar.*

**📋 Golden set** — upload your own **input → expected** CSV and run it against the
selected model. The verdict is judged against **ground truth you defined**, not a
generated guess — the most trustworthy run in the Studio. (Columns: `prompt`,
`expected`, optional `validator`/`category`/`severity`; a template is downloadable
in-app.)

**🔁 Multi-turn** — script a conversation (one user turn per line) and check the
**final reply**, to test an agent's **memory, context retention, and scope** across a
dialogue — not just single-shot. The model carries context (native history on Claude;
a running transcript on Groq/HTTP via `HttpModel.converse`).

**📚 RAG grounding** — paste a **context** + a **question**; the model answers from the
context only and a **grounding judge** checks every claim is supported — catching RAG's
worst failure (confidently adding facts not in the source). Verdict: *grounded /
grounded-but-wrong / not grounded*.

**⚖️ Judge** — calibrate an **LLM-as-judge** against your own human labels: upload
`criterion, answer, human_pass` rows and see how often a model-judge agrees with you
(**agreement %** → *trustworthy / use with caution / do not trust*). The judge is
**backend-agnostic** (Groq/OpenAI/Claude), so open-ended quality can be graded without
a Claude key — and it grades `llm_judge` cases in a run with the chosen model.

**✍️ Prompt & instructions** — paste a **prompt** or an **agent's instructions**
and get a quality score, a few concise pointers, and a concrete **suggested
rewrite** (task-type aware; Claude does a fully tailored rewrite when available).

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
