"""AI Testing Studio — a browser UI for the generate -> run -> report toolchain.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import core

# In a shared public deployment, one process serves all sessions — so we must
# NOT accept secrets (ANTHROPIC_API_KEY) or arbitrary URLs (SSRF). Set
# PRS_STUDIO_PUBLIC=1 on the public instance: it restricts to the offline mock.
PUBLIC = str(os.environ.get("PRS_STUDIO_PUBLIC", "")).strip().lower() in ("1", "true", "yes", "on")


def _secret(name: str) -> str | None:
    """Read an API key from Streamlit Secrets, if configured. Safe if absent.

    Lets a *private* deployment hold the key server-side (Settings -> Secrets) so
    no key is ever typed into the UI. Never use this on a public instance.
    """
    try:
        val = st.secrets.get(name)
    except Exception:
        return None
    return str(val) if val else None

# HTTP-backend presets so common targets are one click (no typing).
_HTTP_PRESETS = {
    "Custom": None,
    "Groq (free, OpenAI-compatible)": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "body": '{"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": {PROMPT}}]}',
        "response_path": "choices.0.message.content",
        "headers": '{"Authorization": "Bearer gsk_..."}',
    },
    "OpenAI-compatible": {
        "url": "https://api.openai.com/v1/chat/completions",
        "body": '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": {PROMPT}}]}',
        "response_path": "choices.0.message.content",
        "headers": '{"Authorization": "Bearer sk-..."}',
    },
}
_BACKEND_KIND = {"Demo bot (offline)": "mock", "Claude API": "claude", "HTTP endpoint": "http"}
AI_TYPES = ["(none)", "chatbot", "rag", "classifier", "summarizer", "agent"]

st.set_page_config(page_title="AI Testing Studio", page_icon="🧪", layout="wide")

st.markdown(
    """
    <style>
      .hero {background:linear-gradient(135deg,#0f172a 0%,#134e4a 100%);
             color:#e2e8f0;padding:1.4rem 1.8rem;border-radius:16px;margin-bottom:.8rem;
             box-shadow:0 8px 24px rgba(2,6,23,.18);}
      .hero h1 {font-family:ui-monospace,"JetBrains Mono",Menlo,Consolas,monospace;
                font-size:1.9rem;margin:0;letter-spacing:-.5px;color:#f8fafc;}
      .hero h1 .accent{color:#34d399;}
      .hero p {margin:.35rem 0 0;color:#cbd5e1;font-size:1rem;}
      .chip{display:inline-block;background:#ecfdf5;color:#065f46;border:1px solid #a7f3d0;
            border-radius:999px;padding:2px 11px;margin:3px;font-size:12px;font-weight:600;}
      .pq-callout{background:#ecfdf5;border:1px solid #6ee7b7;border-left:4px solid #10b981;
                  border-radius:10px;padding:.7rem 1rem;margin:.2rem 0 .6rem;color:#065f46;}
      .pq-callout b{color:#047857;}
      .pq-badge{display:inline-block;background:#10b981;color:#fff;border-radius:999px;
                padding:1px 9px;font-size:11px;font-weight:700;margin-right:6px;vertical-align:middle;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>🧪 AI Testing <span class="accent">Studio</span></h1>'
    '<p>Describe an AI feature → generate a risk-based test suite → run it → get a '
    'report with a ship / no-ship verdict.</p></div>',
    unsafe_allow_html=True,
)

# ---- form state (shared across tabs) --------------------------------------
for _key, _default in {"feature_input": "", "aitype_input": "(none)", "pq_text": "",
                       "ov_safety": 0, "ov_accuracy": 0, "ov_agent": 0}.items():
    st.session_state.setdefault(_key, _default)


def _clear_pq():
    st.session_state["pq_text"] = ""


def _clear_feature():
    st.session_state["feature_input"] = ""
    st.session_state["aitype_input"] = "(none)"
    st.session_state["ex_choice"] = "(custom)"
    st.session_state["ov_safety"] = 0
    st.session_state["ov_accuracy"] = 0
    st.session_state["ov_agent"] = 0
    for k in ("_applied_example", "gen", "run"):
        st.session_state.pop(k, None)


# ---- sidebar: which model to test against ---------------------------------
with st.sidebar:
    st.header("Model under test")
    backends = ["Demo bot (offline)"] if PUBLIC else ["Demo bot (offline)", "Claude API", "HTTP endpoint"]
    backend = st.radio("Backend", backends,
                       help="The Demo bot is a built-in offline dummy with planted bugs — for "
                            "free demos and Practice, no key. Claude/HTTP test a real model.")
    if PUBLIC:
        st.caption("This is a public demo — it runs the offline **Demo bot** only (a dummy with "
                   "planted bugs, for trying the pipeline and Practice). Clone the repo to test "
                   "the Claude API or your own endpoint.")
    backend_opts: dict[str, str] = {}
    if backend == "Claude API":
        _sk = _secret("ANTHROPIC_API_KEY")
        if _sk:
            backend_opts["api_key"] = _sk
            st.caption("🔐 Using **ANTHROPIC_API_KEY** from Secrets.")
        else:
            backend_opts["api_key"] = st.text_input("ANTHROPIC_API_KEY", type="password")
    elif backend == "HTTP endpoint":
        for _k, _d in {"http_url": "", "http_body": '{"prompt": {PROMPT}}',
                       "http_response_path": "output", "http_headers": ""}.items():
            st.session_state.setdefault(_k, _d)

        def _apply_http_preset():
            p = _HTTP_PRESETS.get(st.session_state.get("http_preset"))
            if p:
                st.session_state["http_url"] = p["url"]
                st.session_state["http_body"] = p["body"]
                st.session_state["http_response_path"] = p["response_path"]
                st.session_state["http_headers"] = p["headers"]

        st.selectbox("Preset", list(_HTTP_PRESETS), key="http_preset", on_change=_apply_http_preset)
        backend_opts["url"] = st.text_input("Endpoint URL", key="http_url",
                                            placeholder="https://api.example.com/chat")
        backend_opts["body"] = st.text_input("Body template", key="http_body",
                                             help="The token {PROMPT} is replaced with the JSON-encoded prompt.")
        backend_opts["response_path"] = st.text_input("Response path", key="http_response_path",
                                                      help='Dotted path to the answer, e.g. choices.0.message.content')

        # Pull the bearer key from Secrets when one matches the chosen preset, so
        # no key is typed into the UI (use this only on a *private* deployment).
        _preset = st.session_state.get("http_preset", "")
        _secret_name = ("GROQ_API_KEY" if _preset.startswith("Groq")
                        else "OPENAI_API_KEY" if _preset.startswith("OpenAI") else None)
        _hk = _secret(_secret_name) if _secret_name else None
        if _hk:
            backend_opts["headers"] = json.dumps({"Authorization": f"Bearer {_hk}"})
            st.caption(f"🔐 Using **{_secret_name}** from Secrets for the Authorization header.")
        else:
            backend_opts["headers"] = st.text_input("Headers (JSON)", key="http_headers",
                                                    placeholder='{"Authorization": "Bearer ..."}')
        if _preset.startswith("Groq") and not _hk:
            st.caption("Free key: sign up at console.groq.com → API Keys → create one "
                       "(starts `gsk_`), and paste it into the Authorization header above. "
                       "Run this **locally** — don't paste keys into the public app.")

    st.divider()
    st.markdown(
        "**How the verdict works**\n\n"
        "- **BLOCK** — any Critical fails, or a High safety/hallucination fails\n"
        "- **NEEDS SIGN-OFF** — any other High fails\n"
        "- **SHIP** — no Critical/High failures"
    )

tab_test, tab_prompt, tab_practice, tab_audit, tab_help = st.tabs(
    ["🧪 Test a feature", "✍️ Prompt & instructions", "🎓 Practice",
     "📄 Example audit", "ℹ️ How it works"]
)
_AUDIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "reports", "claude-audit-2026-06-18.md")

# ============================================================================
# TAB 1 — Test a feature (the main flow)
# ============================================================================
with tab_test:
    st.subheader("1 · Describe the feature")
    scenarios = core.load_scenarios()
    by_label = {f"{s.group} — {s.label}": s for s in scenarios}
    choice = st.selectbox("Start from an example (optional)", ["(custom)"] + list(by_label), key="ex_choice")
    if choice != "(custom)" and st.session_state.get("_applied_example") != choice:
        s = by_label[choice]
        st.session_state["feature_input"] = s.feature
        st.session_state["aitype_input"] = s.ai_type or "(none)"
        st.session_state["ov_safety"] = int(s.overrides.get("safety", 0))
        st.session_state["ov_accuracy"] = int(s.overrides.get("accuracy", 0))
        st.session_state["ov_agent"] = int(s.overrides.get("agent", 0))
        st.session_state["_applied_example"] = choice

    col1, col2 = st.columns([3, 1])
    with col1:
        feature = st.text_area(
            "Feature or input to test", key="feature_input", height=140,
            placeholder=(
                "Describe what's under test — a short phrase, a feature, or any input.\n\n"
                "e.g. \"password reset via email\"\n"
                "or  \"banking agent that transfers funds between a user's accounts\"\n\n"
                "If you have acceptance criteria, paste them too — they become the oracle."
            ),
        )
    with col2:
        ai_type = st.selectbox("AI type", AI_TYPES, key="aitype_input")
        st.caption("Tip: the more precisely you state what a correct answer is, the "
                   "sharper the generated cases (especially with the Claude backend).")

    # Power-user knobs, hidden by default so the main flow stays simple:
    # capability gating (which scaffold cases apply) + coverage overrides.
    _CAP_OPTS = {
        "Takes actions (create / update / delete)": "acts",
        "Returns structured data (JSON / API)": "structured",
        "Stateful service (has status / on-off)": "stateful",
    }
    _flag_to_label = {v: k for k, v in _CAP_OPTS.items()}
    _derived = {"agent": ["acts"], "classifier": ["structured"]}.get(ai_type, [])
    if st.session_state.get("_caps_for_aitype") != ai_type:
        st.session_state["caps_select"] = [_flag_to_label[f] for f in _derived]
        st.session_state["_caps_for_aitype"] = ai_type

    with st.expander("⚙️ Advanced options (optional)"):
        cap_labels = st.multiselect(
            "What can this AI do?  (only fitting tests are generated)",
            list(_CAP_OPTS), key="caps_select",
            help="Leave empty for a read-only / text AI (e.g. a Q&A or document agent). "
                 "Tick what applies and the generator skips cases that don't fit.",
        )
        st.caption("Coverage overrides — each category set here becomes REQUIRED at the given minimum.")
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            v_safety = st.number_input("safety min", min_value=0, max_value=20, step=1, key="ov_safety")
        with oc2:
            v_accuracy = st.number_input("accuracy min", min_value=0, max_value=20, step=1, key="ov_accuracy")
        with oc3:
            v_agent = st.number_input("agent min", min_value=0, max_value=20, step=1, key="ov_agent")
    capabilities = [_CAP_OPTS[lbl] for lbl in cap_labels]
    overrides = {k: int(v) for k, v in
                 {"safety": v_safety, "accuracy": v_accuracy, "agent": v_agent}.items() if v}

    gc1, gc2, _ = st.columns([1, 1, 4])
    if gc1.button("⚙️ Generate test suite", type="primary", disabled=not feature):
        core.set_backend(_BACKEND_KIND[backend], **backend_opts)
        with st.spinner("Generating cases…"):
            try:
                st.session_state["gen"] = core.generate_suite(
                    feature, None if ai_type == "(none)" else ai_type,
                    overrides or None, capabilities=capabilities)
            except Exception as exc:
                st.session_state.pop("gen", None)
                st.error(f"Could not generate the suite: {exc}")
    gc2.button("Clear", on_click=_clear_feature, disabled=not feature, key="clear_feature")

    gen = st.session_state.get("gen")
    if gen:
        _is_mock_gen = gen.generator_name == "mock"
        if _is_mock_gen:
            st.success(f"Generated a **starter scaffold** of {len(gen.cases)} case(s) "
                       f"with the offline **Demo bot** generator.")
        else:
            st.success(f"Designed **{len(gen.cases)} tailored case(s)** for your feature "
                       f"using **`{gen.generator_name}`**.")
        if gen.errors:
            st.warning(f"Dropped {len(gen.errors)} invalid case(s): " + "; ".join(gen.errors))
        if _is_mock_gen:
            st.caption("This is a **generic scaffold, not a finished suite** — the offline Demo bot "
                       "fills the same risk-category templates regardless of feature. **Select a "
                       "real backend (e.g. Groq) and re-generate** to have the model design "
                       "feature-specific cases. Review each case and untick any that don't apply.")
        else:
            st.caption("These cases were **designed for your feature** by the selected model. "
                       "Still review them — especially the *expected* answers in each validator — "
                       "before trusting them as a baseline. Untick any that don't fit.")
        _df = pd.DataFrame(
            [{"keep": True, "id": c.id, "category": c.category, "severity": c.severity,
              "validator": c.validator, "prompt": c.prompt} for c in gen.cases]
        )
        _edited = st.data_editor(
            _df, hide_index=True, use_container_width=True,
            disabled=["id", "category", "severity", "validator", "prompt"],
            column_config={
                "keep": st.column_config.CheckboxColumn("keep", help="Untick to exclude this case"),
                "prompt": st.column_config.TextColumn("prompt", width="large"),
            },
            key="case_editor",
        )
        _kept_ids = set(_edited[_edited["keep"]]["id"])
        kept_cases = [c for c in gen.cases if c.id in _kept_ids]
        st.caption(f"**{len(kept_cases)} of {len(gen.cases)}** cases selected.")

        with st.expander("🔍 View full prompts (word-wrapped)", expanded=False):
            st.caption("The table above truncates long prompts; here they are in full.")
            for c in gen.cases:
                st.markdown(f"**`{c.id}`**  ·  `{c.category}` / {c.severity} / `{c.validator}`")
                st.markdown(f"> {c.prompt}")
                st.divider()
        (st.error if gen.has_gaps else st.info)(
            ("⚠️ Below coverage standard\n\n" if gen.has_gaps else "✅ Coverage\n\n")
            + "```\n" + gen.coverage_text + "\n```"
        )

        st.subheader("2 · Run the suite")
        rcol1, rcol2, rcol3 = st.columns([2, 1, 1])
        do_run = rcol1.button(f"▶️ Run {len(kept_cases)} selected case(s)",
                              type="primary", disabled=not kept_cases)
        repeat_in = rcol2.number_input(
            "Runs per case", min_value=1, max_value=20, value=1, step=1,
            help="The model is non-deterministic — run each case N times and measure a "
                 "pass rate. A case that passes only sometimes is flagged FLAKY. "
                 "Use 3–5 when testing a real model (Claude/HTTP).")
        sla_in = rcol3.number_input("SLA (ms, optional)", min_value=0, max_value=120000,
                                    value=0, step=100,
                                    help="Flag cases whose response time exceeds this. 0 = off.")
        if repeat_in > 1:
            thr_pct = st.slider(
                "Pass threshold — a case passes only if it succeeds in at least this "
                "share of its runs", min_value=50, max_value=100, value=100, step=10,
                format="%d%%")
        else:
            thr_pct = 100
        if do_run:
            core.set_backend(_BACKEND_KIND[backend], **backend_opts)
            with st.spinner(f"Running {len(kept_cases)} case(s)"
                            + (f" × {repeat_in} runs…" if repeat_in > 1 else "…")):
                try:
                    st.session_state["run"] = core.run_selected(
                        kept_cases, sla_ms=sla_in or None,
                        repeat=int(repeat_in), pass_threshold=thr_pct / 100)
                except Exception as exc:
                    st.session_state.pop("run", None)
                    st.error(f"The run failed against **{backend}**: {exc}\n\n"
                             "Check the endpoint URL, body template, and response path "
                             "in the sidebar — or switch to the offline Demo bot backend.")

        run = st.session_state.get("run")
        if run:
            st.subheader("3 · Report")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Pass rate", f"{run.summary.pass_rate:.0f}%")
            m2.metric("Passed", f"{run.summary.passed}/{run.summary.total}")
            m3.metric("Failed", run.summary.failed)
            m4.metric("Verdict", run.verdict)
            _perf = run.perf or {}
            _breaches = _perf.get("breaches", [])
            m5.metric("Avg latency", f"{_perf.get('avg_ms', 0)} ms",
                      delta=(f"{len(_breaches)} over SLA" if _breaches else None),
                      delta_color="inverse")
            _flaky = [r for r in run.results if getattr(r, "flaky", False)]
            _runs_per = max((getattr(r, "runs", 1) for r in run.results), default=1)
            if _runs_per > 1:
                msg = (f"Ran each case **{_runs_per}×**. "
                       + (f"⚠️ **{len(_flaky)} flaky** case(s) — passed some runs, not all: "
                          + ", ".join(f"`{r.case.id}`" for r in _flaky)
                          if _flaky else "✅ No flaky cases — behaviour was stable across runs."))
                (st.warning if _flaky else st.success)(msg)
            verdict_style = {"SHIP": "success", "NEEDS SIGN-OFF": "warning", "BLOCK": "error"}
            getattr(st, verdict_style.get(run.verdict, "info"))(
                f"Release verdict: **{run.verdict}**  ·  model: `{run.model_name}`")
            if str(run.model_name).startswith("mock"):
                st.warning(
                    "**Demo bot — this verdict is illustrative, not a real evaluation.** "
                    "The offline demo bot returns canned, feature-independent answers, "
                    "so the result is roughly the same whatever feature you type (only the "
                    "capability checkboxes change the case count). It demonstrates the "
                    "pipeline end-to-end with no API key. **For a real evaluation that "
                    "actually depends on your feature, run the Claude or HTTP backend** "
                    "(locally — see the README).")
            components.html(run.html, height=620, scrolling=True)

            rc1, rc2, rc3 = st.columns(3)
            rc1.download_button("⬇️ HTML report", run.html, "report.html", "text/html")
            rc2.download_button("⬇️ JSON report", run.json, "report.json", "application/json")
            bundle = "\n".join(
                f"# === {os.path.basename(p)} ===\n" + open(p, encoding="utf-8").read()
                for p in sorted(glob.glob(os.path.join(gen.out_dir, "*.yaml")))
            )
            rc3.download_button("⬇️ Generated suite (YAML)", bundle, "suite.yaml", "text/yaml")

    # ---- deploy-readiness certification battery -----------------------------
    st.divider()
    st.subheader("🛡️ Deploy-readiness certification")
    st.caption(
        f"A fixed, comprehensive battery — **{len(core.CERTIFICATION_CASES)} probes** across "
        f"**{core.certification_dimensions()}** risk dimensions (safety, security/red-team, "
        "hallucination, accuracy, reasoning, consistency, robustness, bias, format, and "
        "refusal-calibration). Validators check for the *correct* behaviour, so a strong model "
        "passes and a weak one fails. Run it against a real bot (Groq/Claude) to certify it for "
        "deploy. *Certification is risk-based, not absolute — this is a strong general bar, not a "
        "guarantee.*")

    # Drop a stale result if the backend changed since the last certification run.
    if st.session_state.get("_cert_backend") not in (None, backend):
        st.session_state.pop("cert_run", None)

    cc1, cc2, cc3 = st.columns([1.4, 1, 1])
    do_cert = cc1.button("🛡️ Run certification battery", type="primary", key="run_cert")
    cert_repeat = cc2.number_input("Runs per case", min_value=1, max_value=10, value=1, step=1,
                                   key="cert_repeat",
                                   help="Run each probe N times and measure a pass rate "
                                        "(non-determinism). 3–5 for a real model.")
    if st.session_state.get("cert_run") is not None:
        cc3.button("Clear result", key="clear_cert",
                   on_click=lambda: st.session_state.pop("cert_run", None))
    if do_cert:
        core.set_backend(_BACKEND_KIND[backend], **backend_opts)
        st.session_state["_cert_backend"] = backend
        with st.spinner("Running the certification battery…"):
            try:
                _cases = core.build_certification()
                st.session_state["cert_run"] = core.run_selected(
                    _cases, repeat=int(cert_repeat))
            except Exception as exc:
                st.session_state.pop("cert_run", None)
                st.error(f"Certification run failed against **{backend}**: {exc}")

    cert = st.session_state.get("cert_run")
    if cert:
        st.caption("Showing your most recent certification run (it stays until you re-run, "
                   "clear it, or switch backend).")
        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.metric("Score", f"{cert.summary.pass_rate:.0f}%")
        cm2.metric("Passed", f"{cert.summary.passed}/{cert.summary.total}")
        cm3.metric("Failed", cert.summary.failed)
        cm4.metric("Verdict", cert.verdict)
        _cv_style = {"SHIP": "success", "NEEDS SIGN-OFF": "warning", "BLOCK": "error"}
        getattr(st, _cv_style.get(cert.verdict, "info"))(
            f"Certification verdict: **{cert.verdict}**  ·  model: `{cert.model_name}`  "
            + ("— ready to deploy on this bar." if cert.verdict == "SHIP"
               else "— do not deploy until the failures below are resolved."))
        if str(cert.model_name).startswith("mock"):
            st.warning("**Demo bot** — it has planted bugs, so it deliberately fails many probes "
                       "(it is *not* deploy-ready, by design). Run Groq/Claude to certify a real bot.")
        # per-dimension scorecard
        _by_cat: dict[str, list[int]] = {}
        for r in cert.results:
            row = _by_cat.setdefault(r.case.category, [0, 0])
            row[1] += 1
            if r.passed:
                row[0] += 1
        _score_rows = {
            "risk dimension": [], "passed": [], "status": []}
        for c, (p, t) in sorted(_by_cat.items()):
            _score_rows["risk dimension"].append(c)
            _score_rows["passed"].append(f"{p}/{t}")
            _score_rows["status"].append("✅ pass" if p == t else ("⚠️ partial" if p else "❌ fail"))
        st.markdown("**Scorecard by risk dimension**")
        st.table(_score_rows)
        _fails = [r for r in cert.results if not r.passed]
        if _fails:
            with st.expander(f"❌ {len(_fails)} failing probe(s) — what to fix", expanded=False):
                for r in _fails:
                    st.markdown(f"**`{r.case.id}`** (`{r.case.category}` / {r.case.severity})")
                    st.markdown(f"> Prompt: {r.case.prompt}")
                    st.caption(f"Bot replied: {r.answer[:300]}")
                    st.divider()
        cdl1, cdl2 = st.columns(2)
        cdl1.download_button("⬇️ Certification HTML", cert.html, "certification.html", "text/html")
        cdl2.download_button("⬇️ Certification JSON", cert.json, "certification.json", "application/json")

# ============================================================================
# TAB 2 — Prompt & instructions scorer
# ============================================================================
with tab_prompt:
    st.markdown(
        '<div class="pq-callout"><span class="pq-badge">TOOL</span>'
        '<b style="font-size:1.1rem;">✍️ Score a prompt or agent instructions</b><br>'
        'Get a quality score, a few quick pointers, and an example of how it could look '
        '— no lecturing.</div>',
        unsafe_allow_html=True,
    )
    pq_mode = st.radio("What are you scoring?", ["Prompt", "Agent instructions"],
                       horizontal=True, key="pq_mode")
    is_instr = pq_mode == "Agent instructions"
    st.caption(("Paste an agent's instructions/config to check it defines role, tools, "
                "permissions, refusal rules, and data sources.") if is_instr else
               ("Paste a prompt to get a score, a couple of quick pointers, and an "
                "example of how it could look."))
    pq_text = st.text_area("Instructions to score" if is_instr else "Prompt to score",
                           height=130, key="pq_text",
                           placeholder=("Paste the agent's instructions…" if is_instr
                                        else "Paste the prompt you wrote…"))
    pq_llm = (not is_instr) and (not PUBLIC) and st.checkbox(
        "Use Claude for the critique (needs the Claude backend + key)")

    bc1, bc2, _ = st.columns([1, 1, 4])
    do_score = bc1.button("Score this", type="primary", disabled=not pq_text.strip())
    bc2.button("Clear", on_click=_clear_pq, disabled=not pq_text)

    if do_score:
        if pq_llm:
            core.set_backend("claude", api_key=backend_opts.get("api_key", ""))
        try:
            score = (core.assess_instructions(pq_text) if is_instr
                     else core.assess_prompt(pq_text, use_llm=pq_llm))
        except Exception as exc:
            st.error(f"Could not score with Claude: {exc}")
        else:
            tone = "success" if score.score >= 85 else "info" if score.score >= 65 else "warning"
            getattr(st, tone)(f"**{score.score}/100 — {score.level}.** {score.summary}")
            if score.strengths:
                st.caption("Strengths: " + ", ".join(score.strengths))
            for s in score.suggestions:
                st.markdown(f"- {s}")
            if score.example:
                if is_instr:
                    heading, blurb = "🧩 Suggested instructions template", \
                        "Fill the slots and use this as the agent's instructions."
                elif pq_llm:
                    heading, blurb = "✍️ Rewritten prompt", \
                        "A stronger version of your prompt — copy and use it."
                else:
                    heading, blurb = "✍️ Suggested rewrite", \
                        ("A stronger, ready-to-use rewrite of your prompt. For one tailored to "
                         "your exact content, use the Claude backend.")
                with st.container(border=True):
                    st.markdown(f"#### {heading}")
                    st.caption(blurb)
                    st.code(score.example, language="text")

# ============================================================================
# TAB 3 — Practice (guided, hands-on AI-testing drills)
# ============================================================================
with tab_practice:
    st.markdown(
        '<div class="pq-callout"><span class="pq-badge">LEARN</span>'
        '<b style="font-size:1.1rem;">🎓 Practice testing an AI</b><br>'
        'Pick a drill, fire the probe at the bot under test, judge the answer yourself '
        '— then reveal what an expert tester looks for.</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Tip: these probes are tuned so the **offline Demo bot** already fails them — so you "
        "can practice catching real bugs with no key. Then switch to **Groq (free)** or "
        "**Claude** in the sidebar and run the same probes against a real bot, where a "
        "strong model should pass. (Real backends: run locally.)")

    st.caption(f"Drawn at random from **{core.question_bank_size()}** practice questions "
               f"across **{len(core.practice_exercises())}** skills — no two sessions are the same.")

    st.session_state.setdefault("practice_score", {"correct": 0, "total": 0})
    _practice_is_mock = _BACKEND_KIND[backend] == "mock"
    if _practice_is_mock:
        # The Demo bot has a known answer key, so we can auto-grade and keep score.
        sccol1, sccol2 = st.columns([3, 1])
        if sccol2.button("Reset score", key="practice_reset_score"):
            st.session_state["practice_score"] = {"correct": 0, "total": 0}
            for _k in [k for k in st.session_state if str(k).startswith("practice_scored_")]:
                del st.session_state[_k]
            st.rerun()
        _sc = st.session_state["practice_score"]
        _pct = round(100 * _sc["correct"] / _sc["total"]) if _sc["total"] else 0
        sccol1.metric("🎯 Your score this session",
                      f"{_sc['correct']} / {_sc['total']} correct",
                      f"{_pct}%" if _sc["total"] else None)
        sccol1.caption("Auto-graded because you're testing the **Demo bot** (it has a known answer key).")
    else:
        # A real bot has no answer key — scoring is off; the learner self-assesses.
        st.info(f"🎯 **Self-assessed mode** — you're testing a real bot (`{backend}`), which has no "
                "fixed answer key, so automatic scoring is off. Judge each answer yourself, then "
                "**reveal** the expert analysis to check your call. Switch to the **Demo bot** backend "
                "(sidebar) to get an auto-graded score.")

    # Optional focus: narrow the pool to chosen skills / difficulties.
    with st.expander("🎚️ Focus your session (optional) — pick skills or difficulty"):
        _title_to_id = {f"{e.title}  ·  {core.difficulty(e.id)}": e.id
                        for e in core.practice_exercises()}
        _sel_titles = st.multiselect("Skills (leave empty for all 19)", list(_title_to_id),
                                     key="practice_filter_skills")
        _sel_diff = st.multiselect("Difficulty (leave empty for all)", core.DIFFICULTIES,
                                   key="practice_filter_diff")
        st.caption("Filters apply to the **next** question.")
    _skill_ids = [_title_to_id[t] for t in _sel_titles] or None
    _diffs = _sel_diff or None

    # Draw the first question, or a fresh one on demand.
    if "practice_q" not in st.session_state:
        _ex0, _p0 = core.random_question(skills=_skill_ids, difficulties=_diffs)
        st.session_state["practice_q"] = {"ex_id": _ex0.id, "probe": _p0, "n": 0}
    if st.button("🎲 New random question", key="practice_next"):
        _curr = st.session_state["practice_q"]["probe"]
        _exn, _pn = core.random_question(avoid=_curr, skills=_skill_ids, difficulties=_diffs)
        st.session_state["practice_q"] = {
            "ex_id": _exn.id, "probe": _pn, "n": st.session_state["practice_q"]["n"] + 1}

    q = st.session_state["practice_q"]
    ex = core.exercise_by_id(q["ex_id"])
    n = q["n"]

    st.markdown(f"**Skill:** {ex.skill}  ·  category `{ex.category}`  ·  "
                f"difficulty **{core.difficulty(ex.id)}**")
    st.info(f"**Your task:** {ex.brief}")

    probe_key = f"practice_probe_{n}"
    st.session_state.setdefault(probe_key, q["probe"])
    probe = st.text_area("Probe to send (edit it — crafting the probe is half the skill)",
                         key=probe_key, height=90)

    send = st.button("▶️ Send to the bot", type="primary",
                     key=f"practice_send_{n}", disabled=not probe.strip())

    if send:
        core.set_backend(_BACKEND_KIND[backend], **backend_opts)
        with st.spinner("Asking the bot…"):
            try:
                model_name, answer = core.ask_once(probe)
                st.session_state[f"practice_ans_{n}"] = (model_name, answer)
                st.session_state.pop(f"practice_reveal_{n}", None)  # fresh answer -> re-judge
            except Exception as exc:
                st.session_state.pop(f"practice_ans_{n}", None)
                st.error(f"The call failed against **{backend}**: {exc}")

    got = st.session_state.get(f"practice_ans_{n}")
    if not got:
        st.caption("Send the probe to see the bot's answer, then record your verdict to "
                   "reveal the expert analysis.")
    else:
        model_name, answer = got
        with st.container(border=True):
            st.markdown(f"**The bot replied**  ·  `{model_name}`")
            st.markdown(f"> {answer}")

        vcol1, vcol2, _ = st.columns([2.4, 1, 2])
        verdict = vcol1.radio("Your verdict — did it pass or fail this probe?",
                              ["It PASSED", "It FAILED", "Not sure"],
                              key=f"practice_verdict_{n}", horizontal=True)
        _is_mock = _BACKEND_KIND[backend] == "mock"
        if vcol2.button("OK — reveal", type="primary", key=f"practice_ok_{n}"):
            st.session_state[f"practice_reveal_{n}"] = True
            if _is_mock and not st.session_state.get(f"practice_scored_{n}"):
                _exp = core.expected_verdict(ex.id)
                st.session_state["practice_score"]["total"] += 1
                if verdict == _exp:
                    st.session_state["practice_score"]["correct"] += 1
                st.session_state[f"practice_scored_{n}"] = True

        if st.session_state.get(f"practice_reveal_{n}"):
            if _is_mock:
                _exp = core.expected_verdict(ex.id)
                if verdict == _exp:
                    st.success(f"✅ Correct — the right call here is **{_exp}**.")
                else:
                    st.error(f"❌ Not quite — the right call is **{_exp}**. Here's why:")
            else:
                st.info("No auto-grade against a real bot (there's no answer key) — use the "
                        "analysis below to check your own judgement.")
            with st.container(border=True):
                st.markdown("#### 🔎 Expert analysis")
                st.caption(f"Your call: **{verdict}**")
                st.markdown(f"**What to inspect:** {ex.look_for}")
                st.markdown(f"**Pass / fail rule:** {ex.pass_criterion}")
                st.markdown(f"**Why it matters:** {ex.why}")
                st.markdown(f"**Common rookie mistake:** {ex.pitfall}")


# ============================================================================
# TAB 4 — Example audit (a real report produced with this methodology)
# ============================================================================
with tab_audit:
    st.caption("A real adversarial audit run with this methodology — 13 sharp probes "
               "against a live model, judged with explicit pass criteria.")
    try:
        st.markdown(open(_AUDIT_PATH, encoding="utf-8").read())
    except OSError:
        st.info("Audit report file not found.")


# ============================================================================
# TAB 5 — How it works
# ============================================================================
with tab_help:
    st.subheader("How it works")

    st.markdown("#### What AI testing actually needs")
    st.markdown(
        "Testing an AI comes down to two things — **not** a user story:\n\n"
        "1. **An oracle** — a clear statement of what a *correct* answer is, so you "
        "can judge pass/fail.\n"
        "2. **The right inputs** — including the nasty ones: edge cases, ambiguous "
        "phrasing, and adversarial probes (injection, jailbreak, hallucination bait)."
    )
    st.info(
        "**Is a user story required? No.** If the AI implements a *defined feature* "
        "(e.g. invoice approval), a story's acceptance criteria are a handy source of "
        "the oracle. But much of the most valuable AI testing — **auditing a model's "
        "behaviour, red-teaming, bias/safety** — has no story at all. The **Example "
        "audit** tab is exactly that: a story-free behavioural audit that found a real "
        "defect."
    )

    st.markdown("#### The flow in this tool")
    st.markdown(
        "1. **Describe** what's under test — a short phrase, a feature, or (optionally) "
        "a full user story.\n"
        "2. **Generate** a risk-based starter scaffold and check it against the "
        "coverage standard.\n"
        "3. **Run** it against the offline Demo bot, the Claude API, or any HTTP endpoint.\n"
        "4. Get a **report** with pass-rate, severity, latency, and a ship / no-ship "
        "**verdict**."
    )
    st.caption("The generated cases are a scaffold — a human (or the Claude backend) "
               "defines the real oracle. Defining 'correct' is the test design; this "
               "tool runs, gates, and reports it.")

    st.markdown("#### What's in each tab")
    st.markdown(
        "- **🧪 Test a feature** — generate + run a suite, plus a fixed **deploy-readiness "
        "certification** battery across every risk dimension with a per-dimension scorecard.\n"
        "- **✍️ Prompt & instructions** — score and rewrite a prompt or an agent's instructions.\n"
        "- **🎓 Practice** — learn by doing: 500+ probes across 19 skills; fire one, judge it, "
        "then reveal what an expert looks for (auto-scored against the Demo bot).\n"
        "- **📄 Example audit** — a real adversarial audit that found a documented defect."
    )
    st.markdown("#### Risk categories covered")
    st.markdown('<div>' + "".join(f'<span class="chip">{c}</span>' for c in core.categories()) + '</div>',
                unsafe_allow_html=True)
    st.markdown("#### The release verdict")
    st.markdown(
        "- **BLOCK** — any Critical case fails, or a High `safety`/`hallucination` case fails\n"
        "- **NEEDS SIGN-OFF** — any other High case fails\n"
        "- **SHIP** — no Critical/High failures remain"
    )
    st.caption("Performance/SLA is reported alongside, but does not change the verdict — speed and "
               "correctness are kept as separate signals.")
