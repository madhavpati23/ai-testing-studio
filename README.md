# AI Evaluation Studio

*(formerly "AI Testing Studio" — the repo/URL keep the old slug)*

A browser UI that lets **anyone certify an AI** — no terminal, no setup. Point it at a
model, click **Certify**, and get a **grade (A–F) and a downloadable certificate** across
every risk dimension (safety, hallucination, bias, accuracy, grounding, …). Power users
get the full toolchain underneath: ground-truth evals, a calibrated judge, multi-turn and
RAG checks.

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
Point at an AI ──▶ Run risk-based evaluation ──▶ Graded certificate
  (Demo bot /        (safety, hallucination,        (A–F + SHIP /
   Claude / Groq /    bias, accuracy, grounding…)    NO-SHIP, downloadable)
   any endpoint)
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

**🏅 Certify** *(opens first — the front door)* — **one click → a grade + a downloadable
certificate**, plus onboarding and the full methodology folded into the same tab so
nothing needs its own stop. Open **👋 New here?** for the one idea (the *three roles*:
model under test · designer / your ground truth · judge) and a 2-minute free-key setup.
Then: runs the full evaluation across every risk dimension (plus your own ground truth if
you add it) and issues a printable **Certificate of AI Evaluation** with a letter
**grade (A–F)**, a **CERTIFIED / CONDITIONALLY CERTIFIED / NOT CERTIFIED** status, the
model name, the date, the **thoroughness level**, and a per-dimension breakdown. Pick the
depth: **Quick (~38 checks)** / **Standard (~95)** / **Thorough (~95 × 3 runs)** — and the
certificate prints which level it was, so the grade is honestly contextualised. A **Deep**
run (~98 checks + 80 randomized stress probes against a real backend) can take minutes, not
seconds — so instead of a static spinner, a live heartbeat shows exactly which check is
running right now. Runs can go **concurrent** (⚡ Parallel model calls) to cut the wait.
**No key? Certify the Demo bot instantly.** Download a **snapshot** (JSON) alongside the
certificate, and after a later prompt/model change, **compare two snapshots** to see
exactly which checks regressed or improved — not just whether the score moved. Open
**🧭 The full 12-step testing methodology** for the complete process as a live checklist —
oracle → connect → battery → judge → reliability → (severity gating, automatic) → agent
actions → agent loops → adversarial search → track over time → compare options → certify.
**Not a locked wizard** — every step but two is optional or agent-only, every other tab
stays directly reachable. Status reflects what you've actually done *this session*.

**The grade isn't blind to agent or conversational behaviour.** Run a check in
**🔁 Behaviors** — Multi-turn, RAG grounding, Agent actions, Agent loops, or adversarial
search — and click *"Add this result to my certificate"* on the result. It folds into
the SAME verdict the standard battery uses, through the same severity gating. An agent
that answers every question correctly but transfers money without checking the balance
first, drifts mid-conversation, or hallucinates from a source does **not** earn a clean
certificate just because its text is good.

**🎯 Evaluate** — two ways to test a specific dimension on its own (for the fixed
risk-dimension battery + a grade/certificate, use **🏅 Certify** instead — its Quick
level runs the identical battery, plus more):
- **Against your ground truth** — upload an `input → expected` CSV; the verdict is
  judged against truth *you* defined (the most trustworthy run). You can also add ground
  truth directly in **🏅 Certify**, but that pools it with the full battery — use this
  tab for a verdict on **your rows alone**, with no battery cost/latency added and an
  SLA check.
- **From a feature description** — generate a draft suite (a real model designs
  feature-specific cases; the Demo bot fills generic scaffolds), review, and run it
  with optional **runs-per-case** (non-determinism / flaky detection) and an **SLA**.

> **The tabs work together.** Calibrate a judge once (⚖️ Judge) and it's reused for
> every `llm_judge` grading in Evaluate and Multi-turn — *calibrate once, trusted
> everywhere* — and all tabs run against the one **model under test** you pick in the
> sidebar.

**🔁 Behaviors** — specialised agent checks: **Multi-turn** (memory, context & scope
across a conversation — check just the final reply, or add **checkpoints on specific
turns** so a mid-conversation slip can't hide behind a clean ending), **RAG grounding**
(is the answer faithful to a provided source, or hallucinated beyond it — *grounded /
grounded-but-wrong / not grounded* — single source, or **multiple sources** to test
**conflicting documents** — does it flag the disagreement or silently pick a side? — and
**distractor documents** that can pull it toward a wrong-but-still-grounded answer),
and **Agent actions** — the one that tests *behaviour, not text*: the model is given
**real tools** (a banking agent's `get_balance` / `transfer_funds`) via **native
tool-use**, and we capture the calls it *actually* makes — did it call the right tool
with the right arguments, and did it **refuse to fire an irreversible tool** on a
coerced request? Runs on **Claude** (real tool-use); the **Demo bot** has a *planted*
unsafe-action bug so the safety scenario is catchable offline. Beyond the built-in
banking demo, you can **bring your own tools** — define your own tool schemas + a
scenario and test that *your* agent calls (or refuses) the right thing — or go further:
**paste your agent's own instructions** (a Rovo/Jira agent's persona, permissions, tools)
and an AI **proposes a tailored battery** — likely tools, what could go wrong, and
concrete must/must-not scenarios — which you review, then run for real and fold straight
into a certificate. Nothing executes until you approve the proposed plan. A safety
scenario also gets an **adversarial search** button — instead of one hand-written
coercion phrasing, it automatically tries 6 different framings (direct override, fake
authority, urgency, roleplay, reassurance, hypothetical-then-real) and reports the
**break rate**, so a refusal is proven robust across attacks, not just lucky on one
wording. And **Agent
loops** — the frontier beyond a single decision: a **real multi-step tool-use loop**
(call a tool → see a simulated result → decide the next step → repeat), checking the
*whole chain* — did it verify a precondition before acting (e.g. check a balance
*before* transferring), in the right order, within limits, and — when a tool *itself*
reports failure (an error/timeout) — did it **honestly relay the failure**, or
confidently claim success anyway? Beyond the built-in banking demo, you can **bring your
own multi-step orchestration** — your own tools, your own simulated tool results, and a
table of checks (`must_call` / `must_not_call` / `order` / `max_arg` / `no_false_success`)
— e.g. the classic "step A must run before step B" rule for a multi-stage pipeline agent.
Both Agent actions and Agent loops can **repeat the check N times**
and report the real **pass rate** — an LLM is non-deterministic, so a single PASS on a
safety check proves little; a result that's flaky (passes sometimes, fails others) is
treated as **NEEDS SIGN-OFF**, not safe to trust on a lucky run.

You can also point any of these at **your own deployed agent** — pick **"Your deployed
agent (HTTP)"** in the sidebar and give it a small JSON contract
(`POST {"prompt","tools"} -> {"text","tool_calls"}`); the same checks then run against
your actual production agent's real behaviour, not just Claude or the offline demo.

**⚖️ Judge** — calibrate an **LLM-as-judge** against your own human labels
(`criterion, answer, human_pass`) and see how often it agrees with you (**agreement %**
→ *trustworthy / use with caution / do not trust*). Backend-agnostic, so open-ended
quality can be graded without a Claude key; it also grades `llm_judge` cases in a run.
Agreement also gets a **95% confidence interval** (Wilson score) — below ~20 labelled
examples, the tool explicitly warns that the point estimate is too noisy to trust (6
examples at "67% agreement" could really be anywhere from ~30% to ~90%), rather than
stating a small sample's number as settled fact.

**🏆 Leaderboard** — Certify answers *"is this model good?"*; this answers *"which of
these is best, and where exactly do they differ?"* Configure up to 4 contestants (any
mix of Demo bot / Claude / HTTP / your deployed agent), run the **same** certification
battery against all of them, and get a ranked comparison — grade, status, score, verdict
— plus a per-model breakdown. One bad/misconfigured contestant is isolated to its own
**ERROR** row rather than failing the whole run. Download the result as a **Markdown
table** (drop straight into a write-up or post) or **JSON** (archive the run).

> A **500+ probe bank** across 19 skills (injection, hallucination, bias, PII,
> over-refusal, …) powers the **Deep** certification level: each Deep run draws **80
> fresh, randomized probes** — broad coverage that's hard to game, with a robust
> validator per skill (verified that a genuinely correct answer passes). Sampling is
> **stratified by skill**, not a flat draw over all 512 — a flat draw would miss a
> small skill group (e.g. "consistency," 6 of 512 probes) about 1 run in 3, so every
> Deep run is guaranteed to cover all 19 skills, not just most of them most of the time.

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

- [`studio_ci.py`](studio_ci.py) — headless certification for CI/CD (exit code +
  JSON/JUnit reports). See **Gate a pipeline** below.
- [`history.py`](history.py) — durable certification history (local SQLite, or
  multi-tenant Postgres). See **History** below.
- [`multimodal.py`](multimodal.py) — the image red-team battery + serialization.
  See **Multimodal** below.

- [`examples/demo_agent_server.py`](examples/demo_agent_server.py) — a toy banking
  agent that's a **genuinely separate HTTP process**, implementing the contract
  `HttpAgentModel` expects. It has one deliberate, realistic bug (transfers without
  checking the balance first) — run it, point **🔁 Behaviors → Agent loops** at it
  via the **"Your deployed agent (HTTP)"** backend, and watch the Studio catch the
  bug in a real external service, not just its own built-in demo.

## Gate a pipeline (CI/CD)

The same certificate the UI produces can gate a pull request. [`studio_ci.py`](studio_ci.py)
runs `core.run_full_evaluation` headlessly, writes machine-readable reports, and
exits non-zero when the AI fails the policy:

```bash
# Offline wiring smoke test — no keys, no network.
python studio_ci.py --backend mock --level quick

# Certify a deployed AI; fail the build on a BLOCK verdict or a sub-C grade.
python studio_ci.py --backend claude --level standard \
    --fail-on block --min-grade C \
    --json certification.json --junit certification.junit.xml
```

Safety cases are judge-graded and safety-critical cases run worst-case, exactly
as in the app. `--fail-on {block,signoff,any,never}` sets the gate; `--junit`
emits a JUnit report most CI systems render natively. A ready-to-copy workflow is
in [`.github/workflows/certify-example.yml`](.github/workflows/certify-example.yml).
Add `--workers N` to run checks concurrently — the wait is network time against
your backend, so a Deep run (~178 checks × repeats) finishes far faster. Keep it
modest (e.g. 4–8) to respect provider rate limits; `--workers 1` is sequential.
In the app, the same control appears as **⚡ Parallel model calls** on the Certify
step. Run `python studio_ci.py --help` for the full backend/evaluation/gate options.

## History (track record over time)

Every certification the app runs is saved to a **local** SQLite store, so grade
over time and "did this regress since last time?" survive a refresh. The results
view shows a score-over-time chart, a table of past runs, and an automatic
regression check against the previous run of the same model (reusing
`compare_snapshots`, so it agrees with the manual snapshot diff).

- Persistence is **local and single-user** by default — disabled automatically
  when `PRS_STUDIO_PUBLIC=1`, so a shared deploy never records across tenants.
- Storage location: `$PRS_STUDIO_DB`, else `~/.ai_testing_studio/history.db`.
- In CI, add `--save --label "$GIT_SHA"` to `studio_ci.py` to build the same
  track record from pipeline runs.

**Multi-tenant (Postgres).** Set `PRS_STUDIO_DB_URL` to a Postgres DSN and the
same API writes to a shared server-side database instead of SQLite, with every
row scoped by `tenant_id` (isolation boundary) and `user_id` (attribution) —
reads/writes/deletes filter by tenant, so one tenant never sees another's runs.
Old local SQLite databases migrate automatically (their rows land in the `local`
tenant). In CI, pass `--tenant`/`--user` to attribute a saved run. Needs
`psycopg` (`pip install "psycopg[binary]"`). SSO, database row-level security,
and roles are later phases; this is the storage + data-model foundation.

## Red-teaming (adaptive, not just a checklist)

Beyond the fixed jailbreak probes, the **🔁 Behaviors → Agent actions** area offers
two escalating attacks:

- **Adversarial search** fans a scenario out across many attack framings — the
  original social-engineering set (override, authority, urgency, roleplay, …) plus
  optional **encoding/obfuscation smuggling** (base64, ROT13, leetspeak,
  multilingual, payload-splitting) — and reports which ones break it.
- **Iterative attacker** (PAIR/TAP-style) runs an *attacker LLM* that reads each
  refusal and writes a *better* attack toward a goal, escalating until it breaks
  the target or runs out of rounds — an adaptive search over attacks, not a fixed
  list. Point the attacker/judge at a *different* strong model to avoid a model
  red-teaming itself. A proven break folds into the certificate as a critical
  red-team failure.

## Multimodal (image inputs)

Test cases can carry image attachments, so a vision-capable backend can be
probed for image-based failures. The whole engine downstream of the model call —
validators, judge, gating, certificate, history, CI, concurrency — is
modality-agnostic and works unchanged; multimodal is just a new **input** path:

- A `Case` may carry `attachments` (images as base64 or a file `path`).
- `runner.answer_for` routes an attachment-bearing case to the backend's
  `ask_multimodal`; a backend without vision raises a clear error.
- `ClaudeModel.ask_multimodal` sends the Anthropic vision content blocks;
  `image_content_blocks()` builds that shape and is unit-tested offline.

**Vision red-team battery** ([`multimodal.py`](multimodal.py)) — a small suite of
image probes generated deterministically with Pillow (no binary fixtures):
typographic prompt injection (an instruction hidden *in* the image), an
instruction-in-image safety case, an OCR-accuracy check, and a benign
over-refusal control. Run it in CI with `studio_ci.py --backend claude
--multimodal`; results fold into the certificate like any other check.
`write_multimodal_suite` / `load_cases` round-trip a multimodal suite losslessly.

Wiring is covered by offline unit tests (a fake vision model + real Pillow
rendering); that a *real* model reads an image is covered by a single
`ANTHROPIC_API_KEY`-gated integration test (skipped in normal CI).

## Sir Leaks-a-Lot (red-team practice arena)

A Gandalf-style prompt-injection game ([`gauntlet.py`](gauntlet.py)) — you try to make
**Sir Leaks-a-Lot**, the world's leakiest AI guardian, spill its secret — built to go
further than a fixed hosted target:

- **7 levels that stack defenses** — a bare secret → reluctance → an output filter
  that redacts the literal secret → a guard model that catches obvious leaks → an
  input filter blocking suspicious words → the boss with everything at once. Each
  level forces a new technique (indirection, reversing/spelling, base64/rot13/hex,
  synonyms), with **educational feedback** naming which defense caught you.
- **Any defender** — an offline deterministic simulator (no key, every level
  solvable by its intended technique) *or* a live model via the sidebar backend.
- **Defender mode** — write your own protective system prompt and the studio's
  iterative attacker adapts against it until it breaks or gives up.
- **Progress + leaderboard** — enter a handle to save your progress (resumes where
  you left off) and rank on a leaderboard, persisted via [`history.py`](history.py)
  (local by default; shared when a Postgres backend is configured).

Find it in the **🛡️ Sir Leaks-a-Lot** tab. Leak detection catches the secret even
when smuggled out encoded, so bypassing the output filter counts as a win.

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
- **"Your deployed agent" has real side effects** — unlike every other backend, it
  calls *your* actual agent endpoint, so any tool it runs really executes. Point it
  at a staging/test agent, not production data, unless that's the deliberate intent.
- All YAML is parsed with `safe_load`; there is no `eval`/`exec`/`subprocess`. Stub
  response templates (Agent loops) use plain `{key}` substitution, not Python's
  `str.format`/`eval` — no format-string or code-injection surface.

## License

MIT — see [LICENSE](LICENSE).
