"""AI Evaluation Studio — a browser UI for the generate -> run -> report toolchain.

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

# Keys are session-scoped (built per request via core.make_model, never written
# to the process env), so a shared instance can offer bring-your-own-key safely.
# Set PRS_STUDIO_PUBLIC=1 on a public instance to force the SSRF guard on the HTTP
# backend (refuse private/loopback/metadata URLs).
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


def _active_judge(kind: str, opts: dict):
    """Prefer a judge you calibrated this session; otherwise build a fresh one.

    This is what makes the tabs work together: calibrate a judge once (Judge tab)
    and every llm_judge grading in Evaluate / Multi-turn uses *that* validated
    judge. Returns (judge_callable, badge_text).
    """
    cj = st.session_state.get("calibrated_judge")
    if cj and cj.get("fn") is not None:
        return cj["fn"], (f"your **calibrated** judge `{cj.get('model_name', '?')}` "
                          f"({cj['agreement']:.0f}% human agreement → {cj['verdict']})")
    return core.make_judge(kind, opts), ("an **uncalibrated** judge — calibrate one in the "
                                         "⚖️ Judge tab to validate it first")


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

st.set_page_config(page_title="AI Evaluation Studio", page_icon="🧪", layout="wide")

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
    '<div class="hero"><h1>🧪 AI Evaluation <span class="accent">Studio</span></h1>'
    '<p>Describe an AI feature → generate a risk-based test suite → run it → get a '
    'report with a ship / no-ship verdict.</p></div>',
    unsafe_allow_html=True,
)

# ---- form state (shared across tabs) --------------------------------------
for _key, _default in {"feature_input": "", "aitype_input": "(none)",
                       "ov_safety": 0, "ov_accuracy": 0, "ov_agent": 0}.items():
    st.session_state.setdefault(_key, _default)


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
    backends = ["Demo bot (offline)", "Claude API", "HTTP endpoint"]
    backend = st.radio("Backend", backends,
                       help="The Demo bot is a built-in offline dummy with planted bugs — for "
                            "free demos, no key. Claude/HTTP test a real model with "
                            "your own key (used only for your session).")

    # Results belong to the backend they were produced against. When the user
    # switches backends, clear every cached run so a stale verdict from the old
    # model can't sit there looking current — and tell them why it vanished.
    _RESULT_KEYS = ("gen", "run", "cert_run", "certify", "golden_run",
                    "convo_run", "rag_run", "calib", "calibrated_judge")
    if st.session_state.get("_last_backend", backend) != backend:
        for _k in _RESULT_KEYS:
            st.session_state.pop(_k, None)
        st.warning(f"Switched to **{backend}** — cleared earlier results. "
                   "Re-run against the new backend.")
    st.session_state["_last_backend"] = backend

    backend_opts: dict[str, str] = {}
    if backend != "Demo bot (offline)":
        st.caption("🔑 **Bring your own key.** It's kept only in *your* browser session and sent "
                   "directly to the provider per request — never written to the server's "
                   "environment, never stored or logged.")
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
                       "(starts `gsk_`) and paste it into the Authorization header above.")

        # SSRF guard ON by default; private/loopback/metadata addresses are refused
        # unless the user explicitly allows them (only for a trusted local endpoint).
        _allow_local = st.checkbox(
            "Allow private / localhost addresses (e.g. a local Ollama server)",
            value=False,
            help="Off by default for safety — the app refuses internal/metadata IPs so it "
                 "can't be used to reach private infrastructure. Tick only for an endpoint "
                 "you run and trust on your own machine/network.")
        backend_opts["block_private"] = not _allow_local

    st.divider()
    st.markdown(
        "**How the verdict works**\n\n"
        "- **BLOCK** — any Critical fails, or a High safety/hallucination fails\n"
        "- **NEEDS SIGN-OFF** — any other High fails\n"
        "- **SHIP** — no Critical/High failures"
    )

_AUDIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "reports", "claude-audit-2026-06-18.md")

# ============================================================================
# Flow functions — each renders one evaluation flow. They read module globals
# (backend, backend_opts, …) set by the sidebar. The tab "spine" at the bottom
# dispatches to them, so the UI reads as a journey, not a pile of peer tabs.
# ============================================================================

# ---- Evaluate · from a feature description (generate a draft suite + run) ----
def _flow_feature():
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
        with st.spinner("Generating cases…"):
            try:
                st.session_state["gen"] = core.generate_suite(
                    feature, None if ai_type == "(none)" else ai_type,
                    overrides or None, capabilities=capabilities,
                    kind=_BACKEND_KIND[backend], opts=backend_opts)
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
            with st.spinner(f"Running {len(kept_cases)} case(s)"
                            + (f" × {repeat_in} runs…" if repeat_in > 1 else "…")):
                try:
                    _judge, _judge_badge = ((None, None) if _BACKEND_KIND[backend] == "mock"
                                            else _active_judge(_BACKEND_KIND[backend], backend_opts))
                    st.session_state["run_judge_badge"] = _judge_badge
                    st.session_state["run"] = core.run_selected(
                        kept_cases, sla_ms=sla_in or None,
                        repeat=int(repeat_in), pass_threshold=thr_pct / 100,
                        model=core.make_model(_BACKEND_KIND[backend], backend_opts),
                        judge=_judge)
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
            _jbadge = st.session_state.get("run_judge_badge")
            if _jbadge and any(r.case.validator == "llm_judge" for r in run.results):
                st.caption(f"⚖️ Open-ended (`llm_judge`) cases were graded by {_jbadge}.")
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


# ---- Evaluate · across risk dimensions (deploy-readiness certification) ------
def _flow_certification():
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
        st.session_state["_cert_backend"] = backend
        with st.spinner("Running the certification battery…"):
            try:
                _cases = core.build_certification()
                st.session_state["cert_run"] = core.run_selected(
                    _cases, repeat=int(cert_repeat),
                    model=core.make_model(_BACKEND_KIND[backend], backend_opts))
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

# ---- Certify (the common-man front door: one click -> a certificate) --------
def _flow_certify():
    st.subheader("🏅 Certify an AI")
    st.markdown("Run a full evaluation across every risk dimension and get a **shareable "
                "certificate** — in one click.")
    _kind = _BACKEND_KIND[backend]
    if _kind == "mock":
        st.info("You're set to the **Demo bot** — click below to certify it instantly, no key "
                "needed. (It has planted bugs on purpose, so expect a low grade.) **To certify a "
                "*real* AI**, pick **Claude** or **Groq (free)** in the sidebar — see "
                "*👋 Start here* for the 3-step key setup.")
    else:
        st.caption(f"Certifying **{backend}** — your key stays in your session, never stored.")

    with st.expander("➕ Add your own ground truth (optional)"):
        st.caption("Upload a CSV of `prompt, expected` answers you trust; they're folded into the "
                   "certificate. Leave empty to certify on the standard battery alone.")
        up = st.file_uploader("Golden set CSV", type=["csv"], key="certify_golden")
    gcases = []
    if up is not None:
        try:
            gcases, gerr = core.build_golden(up.getvalue().decode("utf-8", errors="replace"))
            if gerr:
                st.warning("Some rows skipped:\n\n- " + "\n- ".join(gerr))
            if gcases:
                st.caption(f"Added **{len(gcases)}** of your own check(s).")
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")

    _THOROUGH = {
        "Quick — ~22 checks, 1 run (fast smoke test)": ("quick", 1, 0),
        "Standard — ~48 checks, 1 run (recommended)": ("standard", 1, 0),
        "Thorough — ~48 checks, 3 runs each (most rigorous)": ("thorough", 3, 0),
        "Deep — ~48 + 80 randomized stress probes (hardest to game)": ("deep", 1, 80),
    }
    tc1, tc2 = st.columns([2, 1])
    thoroughness = tc1.selectbox("Thoroughness", list(_THOROUGH), index=1, key="certify_level")
    _level, _runs, _stress = _THOROUGH[thoroughness]
    tc2.caption("More checks + more runs = a more defensible grade, but more API calls "
                "(mind free-tier rate limits). **Deep** draws 80 fresh probes from a 500+ "
                "bank, so no two Deep runs are identical.")

    if st.button("🏅 Certify this AI", type="primary", key="run_certify"):
        _cj, _cb = ((None, None) if _kind == "mock" else _active_judge(_kind, backend_opts))
        st.session_state["certify_badge"] = _cb
        with st.spinner(f"Running the {_level} evaluation across every dimension…"):
            try:
                st.session_state["certify"] = core.run_full_evaluation(
                    core.make_model(_kind, backend_opts),
                    golden_cases=gcases or None, judge=_cj,
                    level=_level, repeat=_runs, stress_n=_stress)
            except Exception as exc:
                st.session_state.pop("certify", None)
                st.error(f"Certification failed against **{backend}**: {exc}")

    fe = st.session_state.get("certify")
    if fe:
        letter, status = core.certification_grade(fe.pass_rate, fe.verdict)
        gm1, gm2, gm3 = st.columns(3)
        gm1.metric("Grade", letter)
        gm2.metric("Status", status)
        gm3.metric("Score", f"{fe.pass_rate:.0f}%")
        _sv = {"CERTIFIED": "success", "CONDITIONALLY CERTIFIED": "warning", "NOT CERTIFIED": "error"}
        getattr(st, _sv.get(status, "info"))(
            f"**{status} — Grade {letter}** · {fe.passed}/{fe.total} checks passed · "
            f"model `{fe.model_name}`")
        _cb2 = st.session_state.get("certify_badge")
        if _cb2:
            st.caption(f"⚖️ Open-ended cases graded by {_cb2}.")

        cert_html = core.render_certificate(fe)
        st.download_button("⬇️ Download the certificate", cert_html,
                           "ai-evaluation-certificate.html", "text/html", type="primary")
        st.markdown("**Your certificate**")
        components.html(cert_html, height=560, scrolling=True)

        with st.expander("See the full breakdown (which checks, and what failed)"):
            _rows = {"risk dimension": [], "passed": [], "result": []}
            for c, (p, t) in sorted(fe.by_category.items()):
                _rows["risk dimension"].append(c)
                _rows["passed"].append(f"{p}/{t}")
                _rows["result"].append("✅ pass" if p == t else ("⚠️ partial" if p else "❌ fail"))
            st.table(_rows)
            for _name, _run in fe.sections:
                st.markdown(f"**{_name}** — {_run.summary.passed}/{_run.summary.total} · {_run.verdict}")
                components.html(_run.html, height=360, scrolling=True)


# ---- Evaluate · against your ground truth (golden set) ----------------------
def _flow_golden():
    st.markdown(
        '<div class="pq-callout"><span class="pq-badge">TRUTH</span>'
        '<b style="font-size:1.1rem;">📋 Test against your own ground truth</b><br>'
        'Upload a CSV of <b>input → expected</b> pairs you trust, and run them against the '
        'selected model. The verdict is judged against <b>truth you defined</b>, not a '
        'generated guess — this is the most trustworthy run in the Studio.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "**CSV columns:** `prompt`, `expected` (required); `validator`, `category`, "
        "`severity` (optional).\n"
        "- `validator` (default **contains**): `contains` · `not_contains` · `regex` · "
        "`equals_number`\n"
        "- `expected` is the substring / regex / number the answer must satisfy."
    )
    st.download_button("⬇️ Download a CSV template", core.GOLDEN_TEMPLATE,
                       "golden-set-template.csv", "text/csv")

    up = st.file_uploader("Upload your golden-set CSV", type=["csv"], key="golden_csv")
    gcases, gerrors = [], []
    if up is not None:
        try:
            _text = up.getvalue().decode("utf-8", errors="replace")
            gcases, gerrors = core.build_golden(_text)
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
    if gerrors:
        st.warning("Skipped some rows:\n\n- " + "\n- ".join(gerrors))
    if gcases:
        st.success(f"Loaded **{len(gcases)}** ground-truth case(s).")
        st.dataframe(
            pd.DataFrame([{"id": c.id, "category": c.category, "severity": c.severity,
                           "prompt": c.prompt, "validator": c.validator, "expected": c.args}
                          for c in gcases]),
            hide_index=True, use_container_width=True)

        grc1, grc2, grc3 = st.columns([1.6, 1, 1])
        do_golden = grc1.button("▶️ Run golden set", type="primary", key="run_golden")
        g_repeat = grc2.number_input("Runs per case", min_value=1, max_value=10, value=1, step=1,
                                     key="golden_repeat",
                                     help="Run each case N times (non-determinism). 3–5 for a real model.")
        g_sla = grc3.number_input("SLA (ms, optional)", min_value=0, max_value=120000,
                                  value=0, step=100, key="golden_sla")
        if do_golden:
            with st.spinner(f"Running {len(gcases)} case(s) against {backend}…"):
                try:
                    st.session_state["golden_run"] = core.run_selected(
                        gcases, sla_ms=g_sla or None, repeat=int(g_repeat),
                        model=core.make_model(_BACKEND_KIND[backend], backend_opts))
                except Exception as exc:
                    st.session_state.pop("golden_run", None)
                    st.error(f"Run failed against **{backend}**: {exc}")

        grun = st.session_state.get("golden_run")
        if grun:
            st.subheader("Result")
            gm1, gm2, gm3, gm4 = st.columns(4)
            gm1.metric("Score", f"{grun.summary.pass_rate:.0f}%")
            gm2.metric("Passed", f"{grun.summary.passed}/{grun.summary.total}")
            gm3.metric("Failed", grun.summary.failed)
            gm4.metric("Verdict", grun.verdict)
            _gv = {"SHIP": "success", "NEEDS SIGN-OFF": "warning", "BLOCK": "error"}
            getattr(st, _gv.get(grun.verdict, "info"))(
                f"Verdict against your ground truth: **{grun.verdict}**  ·  model: `{grun.model_name}`")
            components.html(grun.html, height=560, scrolling=True)
            gd1, gd2 = st.columns(2)
            gd1.download_button("⬇️ HTML report", grun.html, "golden-report.html", "text/html")
            gd2.download_button("⬇️ JSON report", grun.json, "golden-report.json", "application/json")
    elif up is None:
        st.caption("No file yet — download the template, fill in your own prompts and expected "
                   "answers, and upload it. Tip: run it against **Groq/Claude**, not the Demo bot.")

# ---- Behaviours · multi-turn conversation -----------------------------------
def _flow_multiturn():
    st.markdown(
        '<div class="pq-callout"><span class="pq-badge">AGENT</span>'
        '<b style="font-size:1.1rem;">🔁 Test across a conversation</b><br>'
        'Single-turn tests miss what agents get wrong: <b>memory, context retention, staying in '
        'scope over a dialogue</b>. Script several user turns; the model carries context, and the '
        'check runs on the <b>final reply</b>.</div>',
        unsafe_allow_html=True,
    )
    st.caption("The model keeps context across turns (true multi-turn on Claude; a running "
               "transcript on Groq/HTTP). Classic test: state a fact, then ask for it back.")

    # What the check is looking for, up front, as a legend.
    ml1, ml2 = st.columns(2)
    ml1.success("**PASS**  \nThe final reply still honours the check — context held.")
    ml2.error("**FAIL**  \nThe model forgot, drifted, or broke scope by the last turn.")

    with st.container(border=True):
        st.markdown("##### 📥 The conversation")
        _convo_default = "My name is Sam and my account ID is 4471.\nWhat is my account ID?"
        convo = st.text_area("Conversation — one user turn per line", value=_convo_default,
                             height=130, key="convo_turns")

        ac1, ac2 = st.columns([1, 2])
        convo_validator = ac1.selectbox(
            "Check the final reply with",
            ["contains", "not_contains", "regex", "equals_number", "llm_judge"], key="convo_validator")
        convo_expected = ac2.text_input(
            "Expected (substring / regex / number / judge criterion)", value="4471",
            key="convo_expected")
        st.caption("💡 The check runs on the **last line's reply**. Add turns to probe memory "
                   "(state a fact early, ask it back later) or scope (try to pull it off-task).")

    if st.button("▶️ Run conversation", type="primary", key="run_convo",
                 disabled=not convo.strip() or not convo_expected.strip()):
        _turns = [ln for ln in convo.splitlines() if ln.strip()]
        with st.spinner(f"Running {len(_turns)} turn(s) against {backend}…"):
            try:
                _kind = _BACKEND_KIND[backend]
                _cjudge, _cbadge = (None, None)
                if convo_validator == "llm_judge" and _kind != "mock":
                    _cjudge, _cbadge = _active_judge(_kind, backend_opts)
                st.session_state["convo_judge_badge"] = _cbadge
                st.session_state["convo_run"] = core.run_conversation(
                    _turns, validator=convo_validator, expected=convo_expected,
                    model=core.make_model(_kind, backend_opts), judge=_cjudge)
            except Exception as exc:
                st.session_state.pop("convo_run", None)
                st.error(f"Conversation run failed against **{backend}**: {exc}")

    crun = st.session_state.get("convo_run")
    if crun:
        res = crun.results[0]
        (st.success if res.passed else st.error)(
            f"{'✅ PASS' if res.passed else '❌ FAIL'} · verdict **{crun.verdict}** · "
            f"model `{crun.model_name}`")
        with st.container(border=True):
            st.markdown("**Final reply**")
            st.markdown(f"> {res.answer}")
            if not res.passed and res.detail:
                st.caption(f"Why: {res.detail}")
        _cb = st.session_state.get("convo_judge_badge")
        if _cb:
            st.caption(f"⚖️ Graded by {_cb}.")

# ---- Behaviours · RAG grounding ---------------------------------------------
def _flow_rag():
    st.markdown(
        '<div class="pq-callout"><span class="pq-badge">RAG</span>'
        '<b style="font-size:1.1rem;">📚 Grounding / faithfulness check</b><br>'
        'A retrieval system\'s worst failure is <b>confidently adding facts that aren\'t in the '
        'retrieved source</b>. Paste the context, ask a question — the model answers from the '
        'context only, and a grounding judge checks every claim is actually supported.</div>',
        unsafe_allow_html=True,
    )
    _rag_kind = _BACKEND_KIND[backend]
    if _rag_kind == "mock":
        st.warning("Pick a **real backend** (Claude / Groq / OpenAI) — grounding needs a model to "
                   "answer and a model to grade faithfulness. The Demo bot can't.")

    # What the three possible verdicts mean, up front, as a legend.
    lc1, lc2, lc3 = st.columns(3)
    lc1.success("**GROUNDED**  \nEvery claim is supported by the context.")
    lc2.warning("**GROUNDED BUT WRONG**  \nFaithful, but missed the expected answer.")
    lc3.error("**NOT GROUNDED**  \nAdded or contradicted facts — a hallucination.")

    with st.container(border=True):
        st.markdown("##### 📥 The retrieval")
        rag_context = st.text_area(
            "Context — the retrieved source the answer must stick to", height=160, key="rag_context",
            value="Acme Cloud's Pro plan costs $49/month and includes 2 TB of storage and email "
                  "support. The Free plan includes 10 GB of storage and community support only.")
        rrc1, rrc2 = st.columns([2, 1])
        rag_question = rrc1.text_input("Question", value="How much does the Pro plan cost and what "
                                       "support does it include?", key="rag_question")
        rag_expected = rrc2.text_input("Expected (optional substring)", value="$49", key="rag_expected")
        st.caption("💡 To see a hallucination caught, ask something the context can't answer — "
                   "e.g. *“What's the Enterprise plan price?”* — and watch for **NOT GROUNDED**.")

    if _rag_kind == "mock":
        st.caption("⚪ Disabled — connect a real backend (Claude / Groq / OpenAI) in the sidebar to enable.")
    if st.button("📚 Run grounding check", type="primary", key="run_rag",
                 disabled=_rag_kind == "mock" or not rag_context.strip() or not rag_question.strip()):
        with st.spinner(f"Answering from context + grading faithfulness with {backend}…"):
            try:
                st.session_state["rag_run"] = core.run_grounding(
                    rag_context, rag_question,
                    model=core.make_model(_rag_kind, backend_opts),
                    grounding_judge=core.make_grounding_judge(_rag_kind, backend_opts),
                    expected=rag_expected.strip() or None)
            except Exception as exc:
                st.session_state.pop("rag_run", None)
                st.error(f"Grounding check failed against **{backend}**: {exc}")

    rag = st.session_state.get("rag_run")
    if rag:
        _rv = {"GROUNDED": "success", "GROUNDED BUT WRONG": "warning", "NOT GROUNDED": "error"}
        getattr(st, _rv.get(rag.verdict, "info"))(
            f"**{rag.verdict}** · model `{rag.model_name}`")
        with st.container(border=True):
            st.markdown("**The model's answer**")
            st.markdown(f"> {rag.answer}")
            st.caption(f"Faithfulness judge: {rag.reason}")
            if rag.expected is not None:
                st.caption(f"Expected substring “{rag.expected}”: "
                           + ("✅ found" if rag.expected_ok else "❌ not found"))

# ---- Judge calibration ------------------------------------------------------
def _flow_judge():
    st.markdown(
        '<div class="pq-callout"><span class="pq-badge">JUDGE</span>'
        '<b style="font-size:1.1rem;">⚖️ Calibrate an LLM judge</b><br>'
        'For open-ended quality (faithfulness, a refusal that <i>actually</i> refuses) a keyword '
        'check is too brittle — you grade with a model. But a judge is only trustworthy if it '
        '<b>agrees with humans</b>. Upload labelled examples and measure that agreement before you '
        'rely on it.</div>',
        unsafe_allow_html=True,
    )
    _judge_kind = _BACKEND_KIND[backend]
    if _judge_kind == "mock":
        st.warning("Pick a **real backend** (Claude / Groq / OpenAI) in the sidebar — the Demo bot "
                   "can't grade. Tip: use a *strong* model as the judge, ideally **different** from "
                   "the model you're testing (self-grading is biased).")
    else:
        st.caption(f"Judge model: **`{backend}`**. Use a strong model, ideally different from the "
                   "one under test (self-grading is biased).")

    # What the agreement score earns the judge, up front, as a legend.
    jl1, jl2, jl3 = st.columns(3)
    jl1.success("**TRUSTWORTHY**  \nHigh agreement — safe to grade with.")
    jl2.warning("**USE WITH CAUTION**  \nDecent, but check the disagreements.")
    jl3.error("**DO NOT TRUST**  \nToo far from you — tighten or change judge.")

    with st.container(border=True):
        st.markdown("##### 📥 Your labelled examples")
        st.markdown(
            "**CSV columns:** `criterion`, `answer`, `human_pass` (true/false) — your human "
            "judgement of whether each answer satisfies the criterion."
        )
        st.download_button("⬇️ Download a calibration template", core.CALIBRATION_TEMPLATE,
                           "judge-calibration-template.csv", "text/csv")
        cup = st.file_uploader("Upload your labelled calibration CSV", type=["csv"], key="calib_csv")
    crows, cerrors = [], []
    if cup is not None:
        try:
            crows, cerrors = core.parse_calibration_csv(cup.getvalue().decode("utf-8", errors="replace"))
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
    if cerrors:
        st.warning("Skipped some rows:\n\n- " + "\n- ".join(cerrors))
    if crows:
        st.success(f"Loaded **{len(crows)}** labelled example(s).")
        if _judge_kind == "mock":
            st.caption("⚪ Disabled — connect a real backend (Claude / Groq / OpenAI) in the sidebar to enable.")
        if st.button("⚖️ Calibrate the judge", type="primary", key="run_calib",
                     disabled=_judge_kind == "mock"):
            with st.spinner(f"Grading {len(crows)} example(s) with {backend}…"):
                try:
                    _jfn = core.make_judge(_judge_kind, backend_opts)
                    cal = core.calibrate_judge(crows, _jfn)
                    st.session_state["calib"] = cal
                    # Store the judge so every run can reuse it (the tabs work together).
                    st.session_state["calibrated_judge"] = {
                        "fn": _jfn, "agreement": cal.agreement, "verdict": cal.verdict,
                        "model_name": getattr(_jfn, "model_name", backend)}
                except Exception as exc:
                    st.session_state.pop("calib", None)
                    st.session_state.pop("calibrated_judge", None)
                    st.error(f"Calibration failed against **{backend}**: {exc}")

        cal = st.session_state.get("calib")
        if cal:
            jm1, jm2 = st.columns(2)
            jm1.metric("Agreement with humans", f"{cal.agreement:.0f}%", f"{cal.agree}/{cal.total}")
            jm2.metric("Judge verdict", cal.verdict)
            _jv = {"TRUSTWORTHY": "success", "USE WITH CAUTION": "warning", "DO NOT TRUST": "error"}
            getattr(st, _jv.get(cal.verdict, "info"))(
                f"This judge agreed with your labels **{cal.agreement:.0f}%** of the time → "
                f"**{cal.verdict}**. " + ("Safe to use for grading." if cal.verdict == "TRUSTWORTHY"
                else "Disagreements below — tighten your criteria or pick a stronger judge."))
            st.caption("✅ This calibrated judge is now used for **llm_judge** grading in Evaluate "
                       "and Multi-turn — calibrate once, trusted everywhere.")
            st.markdown("**Where the judge landed (❌ = disagreed with your label)**")
            st.dataframe(
                pd.DataFrame([{
                    "match": "✅" if m else "❌",
                    "human": "pass" if h else "fail",
                    "judge": ("pass" if j else "fail") if j is not None else "error",
                    "criterion": crit, "answer": ans, "judge reason": reason,
                } for (crit, ans, h, j, reason, m) in cal.rows]),
                hide_index=True, use_container_width=True)
    elif cup is None:
        st.caption("No file yet — download the template, label each row with your own pass/fail "
                   "judgement, and upload it to measure how well the judge matches you.")


# ---- Example audit ----------------------------------------------------------
def _flow_audit():
    st.caption("A real adversarial audit run with this methodology — 13 sharp probes "
               "against a live model, judged with explicit pass criteria.")
    try:
        st.markdown(open(_AUDIT_PATH, encoding="utf-8").read())
    except OSError:
        st.info("Audit report file not found.")


# ---- How it works -----------------------------------------------------------
def _flow_help():
    st.subheader("How it works — from an AI to a certificate")
    st.markdown("The whole journey, end to end. The short version: **point the Studio at an AI → "
                "click Certify → it runs a battery of checks, judges each answer, and issues a "
                "graded certificate.**")

    st.markdown("#### Step 1 — Have an AI you can test")
    st.markdown(
        "You can certify any AI the Studio can *reach programmatically*:\n"
        "- **The built-in Demo bot** — no key, certifies instantly (it has planted bugs, so it "
        "scores low on purpose — good for trying the flow).\n"
        "- **The Claude API** — paste an `ANTHROPIC_API_KEY`.\n"
        "- **Any OpenAI-compatible / HTTP endpoint** — Groq (free), OpenAI, Together, a local "
        "Ollama, or your own model server.\n\n"
        "⚠️ A **web-only chatbot with no API** (e.g. a consumer chat page) can't be automated "
        "here — test it by hand using the method in the **📄 Example audit** tab."
    )

    st.markdown("#### Step 2 — Connect it (sidebar)")
    st.markdown(
        "In **Model under test**, pick the backend and paste your key if it's a real model. "
        "**Your key stays in your browser session** — it's sent to the provider per request and "
        "never written to the server, stored, or logged. No key? Leave it on the **Demo bot**."
    )
    st.caption("Free path: console.groq.com → create a key (gsk_…) → sidebar → HTTP endpoint → "
               "Groq preset → paste it. See 👋 Start here for the 2-minute version.")

    st.markdown("#### Step 3 — Certify (one click)")
    st.markdown(
        "Open **🏅 Certify**, choose a thoroughness (Quick ~22 / Standard ~48 / Thorough ~48×3 / "
        "**Deep** ~48 + 80 randomized stress probes), and click **Certify this AI**. Under the hood it:\n"
        "1. **Builds the battery** — fixed probes across every risk dimension below.\n"
        "2. **Sends each probe to your AI** and collects the answer (with retry on rate limits).\n"
        "3. **Judges every answer** — a validator per probe (refusal regex, no-leak check, exact "
        "number, valid JSON…), and your **calibrated judge** for open-ended ones.\n"
        "4. **Adds your golden set** if you uploaded one (your own input → expected truth).\n"
        "5. **Pools the results** → a per-dimension scorecard and a severity-gated **verdict**.\n"
        "6. **Grades** it (A–F + status) and **renders a downloadable certificate**."
    )

    st.markdown("#### Step 4 — Read & share the certificate")
    st.markdown(
        "You get a **letter grade (A–F)**, a **CERTIFIED / CONDITIONALLY CERTIFIED / NOT "
        "CERTIFIED** status, the score, the model name, the **thoroughness level**, and a "
        "per-dimension breakdown — **downloadable as a printable certificate.**"
    )

    st.markdown("#### Step 5 — Go deeper for a *real* verdict (optional)")
    st.markdown(
        "The default certificate is a strong general bar. To make it trustworthy for *your* use "
        "case:\n"
        "- **📋 Add your ground truth** (a golden-set CSV) — domain-specific truth, folded into "
        "the certificate. *Biggest upgrade.*\n"
        "- **⚖️ Calibrate the judge** — prove it agrees with humans, then it's reused everywhere.\n"
        "- **🔁 Behaviors** — for agents: multi-turn memory and RAG grounding.\n"
        "- **Thorough** level — re-tests each probe several times for consistency, not luck."
    )

    st.markdown("#### What the grade means")
    st.markdown(
        "- **CERTIFIED (A/B/C)** — no Critical or High safety/hallucination check failed.\n"
        "- **CONDITIONALLY CERTIFIED** — only lower-severity issues; fix-then-ship.\n"
        "- **NOT CERTIFIED** — a Critical or High safety/hallucination check failed; don't deploy "
        "as-is (a blocker caps the grade at C no matter the score)."
    )
    st.caption("It's a **risk-based assessment at the chosen depth, not an absolute guarantee** — "
               "honest by design. Speed/latency is reported alongside but never changes the grade.")

    st.markdown("#### Risk dimensions covered")
    st.markdown('<div>' + "".join(f'<span class="chip">{c}</span>' for c in core.categories()) + '</div>',
                unsafe_allow_html=True)


# ---- Start here -------------------------------------------------------------
def _flow_start_here():
    st.subheader("👋 Welcome — give any AI a verdict you can defend")
    st.markdown(
        "Put a model or agent **under test** and get a **SHIP / NO-SHIP verdict** across the "
        "dimensions that matter — judged against *truth*, not vibes."
    )
    st.info(
        "**The one idea — three roles.** It's easy to mix these up, so here they are once:\n\n"
        "1. **The model under test** — the AI you're judging (pick it in the sidebar).\n"
        "2. **The designer / your ground truth** — where the test cases *come from* (you upload "
        "them, or a model drafts them).\n"
        "3. **The judge** — for open-ended quality, a model grades the answer — and you "
        "*calibrate* that judge against your own labels before trusting it."
    )
    st.markdown("#### Pick your path")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown("**🏅 Just certify an AI**")
        st.caption("Go to **🏅 Certify** → click **Certify this AI**. With the **Demo bot** it "
                   "works instantly, no key. You get a **grade + a downloadable certificate**.")
    with pc2:
        st.markdown("**🔑 Certify a real AI**")
        st.caption("In the sidebar pick **Groq (free)** or **Claude**, paste your key (3 steps "
                   "below; it stays in your session), then **Certify**.")
    with pc3:
        st.markdown("**🔬 Go deep**")
        st.caption("Add your **ground truth** (📋 in Evaluate), **calibrate a judge** (⚖️), or pick "
                   "**Deep** in Certify to draw 80 fresh probes from a 500+ bank — hardest to game.")
    st.markdown("#### Get a free Groq key (≈2 min)")
    st.markdown(
        "1. Go to **console.groq.com** → sign in → **API Keys** → **Create**.\n"
        "2. Copy the key (starts `gsk_`).\n"
        "3. Sidebar → **HTTP endpoint** → preset **Groq** → paste it in the Authorization header."
    )
    st.caption("Then open **🏅 Certify** and click the button. The **ℹ️ How it works** tab "
               "explains the method behind the grade.")


# ============================================================================
# The tab spine — a journey, dispatching to the flow functions above.
# ============================================================================
(tab_certify, tab_start, tab_eval, tab_behav, tab_judge,
 tab_audit, tab_help) = st.tabs(
    ["🏅 Certify", "👋 Start here", "🎯 Evaluate", "🔁 Behaviors", "⚖️ Judge",
     "📄 Example audit", "ℹ️ How it works"]
)

with tab_certify:
    _flow_certify()

with tab_start:
    _flow_start_here()

with tab_eval:
    st.markdown("**Put an AI under test and get a verdict.** Choose how you want to judge it:")
    st.caption("For a one-click grade + certificate, use the **🏅 Certify** tab. These modes are "
               "for testing a specific dimension on its own.")
    eval_mode = st.radio(
        "How do you want to evaluate?",
        ["📋 Against your ground truth — upload input → expected (most trustworthy)",
         "🛡️ Across risk dimensions — a fixed deploy-readiness certification",
         "🧪 From a feature description — generate a draft suite, then run it"],
        key="eval_mode")
    st.divider()
    if eval_mode.startswith("📋"):
        _flow_golden()
    elif eval_mode.startswith("🛡️"):
        _flow_certification()
    else:
        _flow_feature()

with tab_behav:
    st.markdown("**Specialised checks for agent behaviour** — beyond a single question.")
    beh_mode = st.radio(
        "Which behaviour?",
        ["🔁 Multi-turn — memory, context & scope across a conversation",
         "📚 RAG grounding — is the answer faithful to a provided source?"],
        key="beh_mode")
    st.divider()
    if beh_mode.startswith("🔁"):
        _flow_multiturn()
    else:
        _flow_rag()

with tab_judge:
    _flow_judge()
with tab_audit:
    _flow_audit()
with tab_help:
    _flow_help()
