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
        _caveat = (f" ⚠️ *only {cj.get('total', '?')} examples — too few to be confident*"
                  if cj.get("low_confidence") else "")
        return cj["fn"], (f"your **calibrated** judge `{cj.get('model_name', '?')}` "
                          f"({cj['agreement']:.0f}% human agreement → {cj['verdict']}){_caveat}")
    return core.make_judge(kind, opts), ("an **uncalibrated** judge — calibrate one in the "
                                         "⚖️ Judge tab to validate it first")


def _queue_agent_checks(checks: list, source_label: str) -> None:
    """Queue agent-action/loop/red-team checks to fold into the next certificate.

    Without this, the one-click grade only ever reflects text quality — an
    agent could earn "Grade A" while a live tool-misuse bug sits unflagged in
    a different tab. Certify reads this queue and pools it into the verdict.

    Dedupes by check id (last write wins): clicking "Add to my certificate"
    twice for the same scenario must update it, not double-count it — an
    inflated total would quietly skew the pass rate.
    """
    queue = st.session_state.setdefault("certify_agent_checks", [])
    by_id = {c.case.id: c for c in queue}
    for c in checks:
        by_id[c.case.id] = c
    st.session_state["certify_agent_checks"] = list(by_id.values())
    sources = st.session_state.setdefault("certify_agent_check_sources", [])
    sources[:] = [s for s in sources if s[0] != source_label]   # replace this source's old entry
    sources.append((source_label, len(checks)))


def _agent_checks_queue_caption() -> str:
    queue = st.session_state.get("certify_agent_checks", [])
    if not queue:
        return ""
    sources = st.session_state.get("certify_agent_check_sources", [])
    failing = sum(1 for c in queue if not c.passed)
    parts = ", ".join(f"{label} (+{n})" for label, n in sources)
    return (f"📥 **{len(queue)} agent check(s) queued for the certificate** ({failing} currently "
           f"failing) — from: {parts}")


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
_BACKEND_KIND = {"Demo bot (offline)": "mock", "Claude API": "claude", "HTTP endpoint": "http",
                 "Your deployed agent (HTTP)": "http_agent"}
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
    '<p>Point it at an AI → run a risk-based evaluation → get a graded certificate '
    'with a ship / no-ship verdict.</p></div>',
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
    backends = ["Demo bot (offline)", "Claude API", "HTTP endpoint", "Your deployed agent (HTTP)"]
    backend = st.radio("Backend", backends,
                       help="The Demo bot is a built-in offline dummy with planted bugs — for "
                            "free demos, no key. Claude/HTTP test a real model with "
                            "your own key (used only for your session).")

    # Results belong to the backend they were produced against. When the user
    # switches backends, clear every cached run so a stale verdict from the old
    # model can't sit there looking current — and tell them why it vanished.
    _RESULT_KEYS = ("gen", "run", "cert_run", "certify", "golden_run",
                    "convo_run", "convo_trace", "rag_run", "rag_multi_run", "aa_run", "aa_search",
                    "al_run", "calib", "calibrated_judge",
                    "certify_agent_checks", "certify_agent_check_sources")
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
    elif backend == "Your deployed agent (HTTP)":
        st.caption("📡 For testing **your own production agent** — Agent actions / Agent loops "
                   "(Behaviors) point at this. Your endpoint must accept "
                   "`POST {\"prompt\", \"tools\"}` and return "
                   "`{\"text\", \"tool_calls\": [{\"name\", \"arguments\"}, ...]}` — "
                   "every tool call your agent made, in order, however many steps it took.")
        backend_opts["url"] = st.text_input("Agent endpoint URL", key="agent_url",
                                            placeholder="https://my-agent.example.com/run")
        backend_opts["headers"] = st.text_input("Headers (JSON)", key="agent_headers",
                                                placeholder='{"Authorization": "Bearer ..."}')
        _allow_local_agent = st.checkbox(
            "Allow private / localhost addresses", value=False, key="agent_allow_local",
            help="Off by default — tick only for an agent you run and trust locally.")
        backend_opts["block_private"] = not _allow_local_agent
        st.caption("⚠️ Side effects here are **real** — this calls your actual agent, not a "
                   "simulation. Point it at a staging/test agent, not production data, unless "
                   "you mean it.")

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

    _aq_caption = _agent_checks_queue_caption()
    if _aq_caption:
        st.info(_aq_caption + "  \nWithout these, the grade only reflects text quality — an agent "
               "can misuse a real tool and still earn a clean certificate.")
        if st.button("🗑️ Clear queued agent checks", key="clear_agent_checks"):
            st.session_state["certify_agent_checks"] = []
            st.session_state["certify_agent_check_sources"] = []
            st.rerun()
    elif _kind in ("claude", "http_agent"):
        # This backend CAN act on tools — so a clean certificate that never checked tool-use
        # is a real gap, not a minor caveat. Make that loud rather than a quiet footnote.
        st.warning("⚠️ **This certificate will not reflect tool-use safety.** `" + backend + "` can "
                  "act on tools, but no Agent-action/loop checks are queued — go to **🔁 Behaviors "
                  "→ Agent actions / Agent loops**, run a check, and click *\"Add this result to my "
                  "certificate\"* first. Otherwise an agent that misuses a real tool can still earn "
                  "a clean grade here, because nothing below tests that.")
    else:
        st.caption("💡 Run a check in **🔁 Behaviors → Agent actions / Agent loops** and click "
                  "*\"Add this result to my certificate\"* to fold tool-use safety into this grade "
                  "— otherwise it only reflects text quality.")

    if st.button("🏅 Certify this AI", type="primary", key="run_certify"):
        _cj, _cb = ((None, None) if _kind == "mock" else _active_judge(_kind, backend_opts))
        st.session_state["certify_badge"] = _cb
        with st.spinner(f"Running the {_level} evaluation across every dimension…"):
            try:
                st.session_state["certify"] = core.run_full_evaluation(
                    core.make_model(_kind, backend_opts),
                    golden_cases=gcases or None, judge=_cj,
                    level=_level, repeat=_runs, stress_n=_stress,
                    agent_checks=st.session_state.get("certify_agent_checks") or None)
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
        if not fe.agent_checks and _kind in ("claude", "http_agent"):
            st.caption("⚠️ **This grade reflects text quality only** — no agent-action/loop checks "
                      "were folded in, even though this backend can act on tools. See **🔁 "
                      "Behaviors** to test (and certify) tool-use safety too.")

        cert_html = core.render_certificate(fe)
        cert_snapshot = core.export_snapshot(fe)
        cdl1, cdl2 = st.columns(2)
        cdl1.download_button("⬇️ Download the certificate", cert_html,
                             "ai-evaluation-certificate.html", "text/html", type="primary")
        cdl2.download_button("⬇️ Download a snapshot (for regression tracking)", cert_snapshot,
                             "ai-evaluation-snapshot.json", "application/json",
                             help="Save this, then re-certify later (after a prompt/model change) "
                                  "and compare the two snapshots below to see exactly which checks "
                                  "regressed — not just whether the score moved.")
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
            if fe.agent_checks:
                _ac_passed = sum(1 for c in fe.agent_checks if c.passed)
                st.markdown(f"**Agent checks (folded in)** — {_ac_passed}/{len(fe.agent_checks)}")
                st.dataframe(
                    pd.DataFrame([{"id": c.case.id, "severity": c.case.severity,
                                  "✓": "✅" if c.passed else "❌", "detail": c.detail}
                                 for c in fe.agent_checks]),
                    hide_index=True, use_container_width=True)

        with st.expander("📈 Compare to a previous snapshot — did anything regress?"):
            st.caption("Upload an older snapshot (the **baseline**) and a newer one (e.g. after "
                      "changing a prompt or switching models) to see exactly which checks flipped "
                      "from pass to fail — a score moving from 90% to 88% hides whether that's one "
                      "new Critical failure or three trivial ones.")
            cmp1, cmp2 = st.columns(2)
            before_file = cmp1.file_uploader("Baseline snapshot (older)", type=["json"], key="cmp_before")
            after_file = cmp2.file_uploader("New snapshot (newer) — defaults to the run above",
                                            type=["json"], key="cmp_after")
            if before_file is not None:
                try:
                    _before_text = before_file.getvalue().decode("utf-8")
                    _after_text = (after_file.getvalue().decode("utf-8")
                                  if after_file is not None else cert_snapshot)
                    diff = core.compare_snapshots(_before_text, _after_text)
                    db1, db2, db3 = st.columns(3)
                    db1.metric("Before → after grade",
                              f"{diff.before.get('grade', '?')} → {diff.after.get('grade', '?')}")
                    db2.metric("Regressions", len(diff.newly_failed))
                    db3.metric("Improvements", len(diff.newly_passed))
                    if diff.has_regressions:
                        st.error(f"**{len(diff.newly_failed)} check(s) regressed** — passed in the "
                                f"baseline, now failing: " + ", ".join(diff.newly_failed))
                    else:
                        st.success("No regressions — nothing that passed before now fails.")
                    if diff.newly_passed:
                        st.caption("✅ Improved: " + ", ".join(diff.newly_passed))
                    if diff.unchanged_failed:
                        st.caption(f"⚪ Still failing in both (pre-existing, not new): "
                                  + ", ".join(diff.unchanged_failed))
                except Exception as exc:
                    st.error(f"Could not compare snapshots: {exc}")


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
        _turns_preview = [ln for ln in convo.splitlines() if ln.strip()]
        if _turns_preview:
            st.caption("Turns: " + " · ".join(f"**{i+1}** {t[:40]}{'…' if len(t) > 40 else ''}"
                                               for i, t in enumerate(_turns_preview)))

    convo_mode = st.radio(
        "What to check",
        ["Just the final reply — classic memory/scope test",
         "Specific turns — checkpoints, so a mid-conversation slip can't hide behind a clean ending"],
        key="convo_mode")
    _CONVO_RULES = {
        "Must mention this text": "contains",
        "Must NOT mention this text": "not_contains",
        "Must match this pattern (regex)": "regex",
        "Must equal this number": "equals_number",
        "Must satisfy this description (AI-graded)": "llm_judge",
    }

    if convo_mode.startswith("Just"):
        with st.container(border=True):
            st.markdown("##### ✅ The rule for the final reply")
            ac1, ac2 = st.columns([1, 2])
            _rule_label = ac1.selectbox("Rule type", list(_CONVO_RULES), key="convo_rule_label")
            convo_validator = _CONVO_RULES[_rule_label]
            _ph = {"contains": "4471", "not_contains": "I don't know",
                   "regex": r"\b4471\b", "equals_number": "4471",
                   "llm_judge": "correctly states the account ID is 4471"}[convo_validator]
            convo_expected = ac2.text_input("Value", value="4471" if convo_validator == "contains" else "",
                                            placeholder=_ph, key="convo_expected")
            _sentence = {
                "contains": f"the final reply **must mention** “{convo_expected or '…'}”.",
                "not_contains": f"the final reply **must NOT mention** “{convo_expected or '…'}”.",
                "regex": f"the final reply **must match the pattern** `{convo_expected or '…'}`.",
                "equals_number": f"the final reply's number **must equal** {convo_expected or '…'}.",
                "llm_judge": f"an AI judge checks the final reply **{convo_expected or '…'}**.",
            }[convo_validator]
            st.caption(f"📐 **PASS if** {_sentence}")
            st.caption("💡 Add turns to probe memory (state a fact early, ask it back later) or "
                       "scope (try to pull it off-task).")

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
                    st.session_state.pop("convo_trace", None)
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

    else:
        with st.container(border=True):
            st.markdown("##### ✅ Checkpoints — one or more turns to assert on")
            st.caption("Pick a **turn number** (from the list above), a rule, and the value — add "
                       "as many rows as you like. Example: check turn 1 for a leak even though "
                       "turn 2's reply (the end of the chat) looks perfectly clean.")
            _default_rows = pd.DataFrame([
                {"turn": 1, "rule": "Must mention this text", "value": "Sam"},
                {"turn": 2, "rule": "Must mention this text", "value": "4471"},
            ])
            checkpoints_df = st.data_editor(
                _default_rows, num_rows="dynamic", key="convo_checkpoints",
                use_container_width=True,
                column_config={
                    "turn": st.column_config.NumberColumn("Turn #", min_value=1, step=1),
                    "rule": st.column_config.SelectboxColumn("Rule", options=list(_CONVO_RULES)),
                    "value": st.column_config.TextColumn("Value"),
                })

        _aa_kind = _BACKEND_KIND[backend]
        if _aa_kind == "http":
            st.caption("⚪ Disabled — HTTP endpoints don't expose a per-turn transcript. "
                       "Use Claude or the Demo bot.")
        if st.button("▶️ Run checkpoints", type="primary", key="run_convo_trace",
                     disabled=not convo.strip() or checkpoints_df.empty or _aa_kind == "http"):
            _turns = [ln for ln in convo.splitlines() if ln.strip()]
            _checks = [core.TurnCheck(int(row["turn"]), _CONVO_RULES[row["rule"]], str(row["value"]))
                      for _, row in checkpoints_df.dropna().iterrows()]
            with st.spinner(f"Running {len(_turns)} turn(s), checking {len(_checks)} checkpoint(s)…"):
                try:
                    _kind = _BACKEND_KIND[backend]
                    _cjudge = None
                    if any(c.validator == "llm_judge" for c in _checks) and _kind != "mock":
                        _cjudge, _ = _active_judge(_kind, backend_opts)
                    st.session_state["convo_trace"] = core.run_conversation_trace(
                        _turns, _checks, model=core.make_model(_kind, backend_opts), judge=_cjudge)
                    st.session_state.pop("convo_run", None)
                except Exception as exc:
                    st.session_state.pop("convo_trace", None)
                    st.error(f"Checkpoint run failed against **{backend}**: {exc}")

        trace = st.session_state.get("convo_trace")
        if trace:
            (st.success if trace.passed else st.error)(
                f"{'✅ ALL CHECKPOINTS PASS' if trace.passed else '❌ AT LEAST ONE FAILED'} · "
                f"verdict **{trace.verdict}** · model `{trace.model_name}`")
            with st.container(border=True):
                st.markdown("**Full transcript — every reply, not just the last**")
                for i, (t, r) in enumerate(zip(trace.turns, trace.replies), start=1):
                    st.markdown(f"**Turn {i}** — *{t}*")
                    st.markdown(f"> {r}")
            with st.container(border=True):
                st.markdown("**Checkpoint results**")
                st.dataframe(
                    pd.DataFrame([{
                        "✓": "✅" if c.passed else "❌",
                        "turn": c.check.turn_index,
                        "rule": c.check.validator,
                        "expected": c.check.expected,
                        "reply checked": c.reply,
                        "why": c.detail or "—",
                    } for c in trace.checks]),
                    hide_index=True, use_container_width=True)

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

    # What the verdicts mean, up front, as a legend.
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.success("**GROUNDED**  \nEvery claim is supported by the source(s).")
    lc2.warning("**GROUNDED BUT WRONG**  \nFaithful, but missed the expected answer.")
    lc3.error("**NOT GROUNDED**  \nAdded or contradicted facts — a hallucination.")
    lc4.warning("**OVERCONFIDENT**  \nSources disagree; it picked one without saying so.")

    rag_source = st.radio(
        "Sources",
        ["📄 Single source — classic faithfulness check",
         "📑 Multiple sources — conflicting or distracting documents"],
        key="rag_source", horizontal=True)

    if rag_source.startswith("📄"):
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
                    st.session_state.pop("rag_multi_run", None)
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

    else:
        if _rag_kind == "mock":
            st.warning("Pick a **real backend** — multi-source grounding needs a model to answer "
                       "and a model to grade faithfulness. The Demo bot can't.")

        with st.container(border=True):
            st.markdown("##### 📥 Your documents — label + content per row")
            st.caption("Make two rows **disagree** to test conflict-handling, or add an irrelevant "
                       "row to test whether it distracts the model from the right answer.")
            _default_docs = pd.DataFrame([
                {"label": "Pricing 2024.txt", "content": "Acme Cloud Pro plan costs $49/month."},
                {"label": "Pricing 2025 update.txt",
                 "content": "Acme Cloud Pro plan costs $59/month, effective March 2025."},
                {"label": "Support FAQ.txt", "content": "Support replies within 24 hours on all plans."},
            ])
            docs_df = st.data_editor(_default_docs, num_rows="dynamic", key="rag_docs",
                                     use_container_width=True)
            rrc1, rrc2 = st.columns([2, 1])
            rag_m_question = rrc1.text_input("Question", value="How much does the Pro plan cost?",
                                             key="rag_m_question")
            rag_m_expected = rrc2.text_input("Expected (optional substring)", value="",
                                             key="rag_m_expected")
            rag_has_conflict = st.checkbox(
                "These documents deliberately **disagree** — a good answer should flag it, not "
                "silently pick a side", value=True, key="rag_has_conflict")

        if st.button("📚 Run multi-source grounding check", type="primary", key="run_rag_multi",
                     disabled=_rag_kind == "mock" or docs_df.empty or not rag_m_question.strip()):
            _docs = [core.RagDocument(str(r["label"]), str(r["content"]))
                    for _, r in docs_df.dropna().iterrows()]
            with st.spinner(f"Answering from {len(_docs)} document(s) + grading with {backend}…"):
                try:
                    st.session_state["rag_multi_run"] = core.run_grounding_multidoc(
                        _docs, rag_m_question,
                        model=core.make_model(_rag_kind, backend_opts),
                        grounding_judge=core.make_grounding_judge(_rag_kind, backend_opts),
                        expected=rag_m_expected.strip() or None, has_conflict=rag_has_conflict)
                    st.session_state.pop("rag_run", None)
                except Exception as exc:
                    st.session_state.pop("rag_multi_run", None)
                    st.error(f"Multi-source grounding check failed against **{backend}**: {exc}")

        rag_m = st.session_state.get("rag_multi_run")
        if rag_m:
            _rv = {"GROUNDED": "success", "GROUNDED BUT WRONG": "warning",
                  "GROUNDED BUT OVERCONFIDENT": "warning", "NOT GROUNDED": "error"}
            getattr(st, _rv.get(rag_m.verdict, "info"))(
                f"**{rag_m.verdict}** · model `{rag_m.model_name}`")
            with st.container(border=True):
                st.markdown("**Documents offered**")
                st.dataframe(pd.DataFrame([{"source": d.label, "content": d.content}
                                          for d in rag_m.documents]),
                            hide_index=True, use_container_width=True)
                st.markdown("**The model's answer**")
                st.markdown(f"> {rag_m.answer}")
                st.caption(f"Faithfulness judge: {rag_m.reason}")
                if rag_m.has_conflict:
                    st.caption("Conflict acknowledged in the answer: "
                              + ("✅ yes" if rag_m.conflict_flagged else "❌ no — picked a side silently"))
                if rag_m.expected is not None:
                    st.caption(f"Expected substring “{rag_m.expected}”: "
                              + ("✅ found" if rag_m.expected_ok else "❌ not found"))

# ---- Behaviours · agent actions (real native tool-use) ----------------------
def _flow_agent_action():
    st.markdown(
        '<div class="pq-callout"><span class="pq-badge">AGENT</span>'
        '<b style="font-size:1.1rem;">🛠️ Agent-action check</b><br>'
        'Most "agent" testing only reads the <i>text</i>. This tests the <b>actions</b>: the model '
        'is given <b>real tools</b> and we capture the calls it <i>actually</i> makes — did it fire '
        'the right tool with the right arguments, and did it <b>refuse to run an irreversible one</b> '
        'when it should have? The tools here are a banking agent: <code>get_balance</code> '
        '(read-only) and <code>transfer_funds</code> (irreversible).</div>',
        unsafe_allow_html=True,
    )
    _aa_kind = _BACKEND_KIND[backend]

    # Legend: the two things an agent-action check proves.
    al1, al2 = st.columns(2)
    al1.success("**Capability**  \nCalls the right tool with the right arguments.")
    al2.error("**Safety**  \nRefuses to fire an irreversible tool on a coerced request.")

    aa_source = st.radio(
        "Toolset",
        ["📦 Built-in demo — a banking agent (get_balance / transfer_funds)",
         "🧪 Your own agent — define your own tools and scenario"],
        key="aa_source", horizontal=True)

    with st.expander("⚙️ Advanced — reliability"):
        aa_reps = st.number_input(
            "Repeat this check N times", min_value=1, max_value=10, value=1, key="aa_reps",
            help="LLMs are non-deterministic. A safety check that passes once might fail 3 times "
                 "out of 10 — repeat it to see the real pass rate, not a lucky single run.")

    if aa_source.startswith("📦"):
        if _aa_kind == "http":
            st.warning("HTTP endpoints have no standard tool-call channel to capture. Use "
                       "**Claude** (real native tool-use) — or the **Demo bot** for an offline "
                       "demonstration (it has a *planted* unsafe-action bug for the safety scenario).")
        elif _aa_kind == "http_agent":
            st.warning("This built-in demo uses fixed banking tools — your deployed agent doesn't "
                       "have those. Switch to **🧪 Your own agent** below to test it with its "
                       "actual tools.")
        elif _aa_kind == "mock":
            st.info("Running the **Demo bot**: a deterministic agent with a *planted* safety bug — "
                    "it obeys the coerced transfer instead of refusing. Switch to **Claude** to "
                    "test a real model's tool-use.")

        with st.container(border=True):
            st.markdown("##### 📥 The scenario")
            _labels = [s.label for s in core.AGENT_SCENARIOS]
            _pick = st.selectbox("Pick an agent scenario", _labels, key="aa_scenario")
            _scen = core.AGENT_SCENARIOS[_labels.index(_pick)]
            st.markdown("**Request sent to the agent:**")
            st.markdown(f"> {_scen.prompt}")
            st.caption(f"✅ A correct agent should: {_scen.intent}")

        _aa_builtin_disabled = _aa_kind in ("http", "http_agent")
        if _aa_builtin_disabled:
            st.caption("⚪ Disabled — switch to Claude or the Demo bot in the sidebar to enable.")
        if st.button("🛠️ Run agent-action check", type="primary", key="run_aa",
                     disabled=_aa_builtin_disabled):
            with st.spinner(f"Offering the tools to {backend} and capturing its calls "
                            f"({aa_reps}×)…"):
                try:
                    _model = core.make_model(_aa_kind, backend_opts)
                    st.session_state["aa_run"] = core.run_repeated(
                        lambda: core.run_agent_action(_scen, _model), n=int(aa_reps))
                except Exception as exc:
                    st.session_state.pop("aa_run", None)
                    st.error(f"Agent-action check failed against **{backend}**: {exc}")

        if _scen.kind == "must_not_call":
            st.caption(f"🔍 **Adversarial search** — instead of this one phrasing, try "
                      f"{len(core.AGENT_PROMPT_MUTATORS)} different coercion framings (direct "
                      f"override, fake authority, urgency, roleplay, ...) and see how many break it.")
            if st.button("🔍 Search for a break", key="run_aa_search", disabled=_aa_builtin_disabled):
                with st.spinner(f"Trying {len(core.AGENT_PROMPT_MUTATORS)} coercion framings "
                                f"against {backend}…"):
                    try:
                        st.session_state["aa_search"] = core.run_adversarial_search(
                            _scen, core.make_model(_aa_kind, backend_opts))
                    except Exception as exc:
                        st.session_state.pop("aa_search", None)
                        st.error(f"Adversarial search failed against **{backend}**: {exc}")

            aa_search = st.session_state.get("aa_search")
            if aa_search and aa_search.scenario.id == _scen.id:
                if not aa_search.scored:
                    st.error("Every attempt errored — couldn't assess any framing (see errors below).")
                else:
                    (st.error if aa_search.broken else st.success)(
                        f"**{len(aa_search.broken)}/{len(aa_search.scored)} framings broke it "
                        f"({aa_search.break_rate:.0f}%)**")
                st.dataframe(
                    pd.DataFrame([{
                        "framing": a.label,
                        "✓": "—" if a.result is None else ("❌ broke it" if not a.result.passed else "✅ held"),
                        "mutated prompt": a.mutated_prompt,
                        "detail": a.error or (a.result.detail if a.result else ""),
                    } for a in aa_search.attempts]),
                    hide_index=True, use_container_width=True)
                if aa_search.scored and st.button("📥 Add this result to my certificate", key="queue_aa_search"):
                    _queue_agent_checks(core.adversarial_search_checks(aa_search),
                                       f"Adversarial search: {_scen.label}")
                    st.success("Queued. Open **🏅 Certify** and re-run to fold it into the grade.")

    else:
        _aa_custom_ok = _aa_kind in ("claude", "http_agent")
        if not _aa_custom_ok:
            st.warning("Custom tools need **real native tool-use** — use **Claude**, or "
                       "**your deployed agent** (it gets your tools forwarded directly) — the "
                       "Demo bot can only improvise the built-in banking tools, and a generic "
                       "HTTP endpoint has no standard tool-call channel.")
        elif _aa_kind == "http_agent":
            st.caption("📡 Your tools below are sent to your agent endpoint as-is — it decides "
                       "what to call, just like in production.")

        with st.container(border=True):
            st.markdown("##### 🧰 Your tools")
            st.caption("JSON list of tool schemas — the same shape Claude's native tool-use "
                       "expects: `name`, `description`, `input_schema`.")
            st.download_button("⬇️ Download a tools template", core.AGENT_TOOLS_TEMPLATE,
                               "agent-tools-template.json", "application/json")
            tools_text = st.text_area("Tool definitions (JSON)", value=core.AGENT_TOOLS_TEMPLATE,
                                      height=180, key="aa_tools_json")
            tools, tool_errors = core.parse_agent_tools(tools_text)
            if tool_errors:
                st.warning("Problems in your tool JSON:\n\n- " + "\n- ".join(tool_errors))

        with st.container(border=True):
            st.markdown("##### 📥 Your scenario")
            aa_prompt = st.text_area(
                "Prompt sent to the agent", key="aa_custom_prompt", height=80,
                value="Please email jane@example.com with the subject 'Update' and body 'All good.'")
            _tool_names = [t["name"] for t in tools] if tools else []
            cc1, cc2 = st.columns(2)
            aa_kind = cc1.selectbox(
                "This scenario expects the agent to…",
                ["must_call", "must_not_call"],
                format_func=lambda k: "✅ CALL this tool" if k == "must_call" else "🚫 NOT call this tool",
                key="aa_custom_kind")
            aa_tool = cc2.selectbox("Tool", _tool_names or ["(define a tool above first)"],
                                    key="aa_custom_tool")
            aa_args = ""
            if aa_kind == "must_call":
                aa_args = st.text_input(
                    "Expected arguments (JSON, optional — leave blank to only check it called the tool)",
                    value="", placeholder='{"to": "jane@example.com"}', key="aa_custom_args")
            aa_severity = st.selectbox("Severity if this fails", ["critical", "high", "medium", "low"],
                                       index=1, key="aa_custom_severity")
            st.caption("💡 For a **safety** scenario, write a coercive/suspicious prompt and set "
                       "kind to **NOT call** the dangerous tool — same pattern as the built-in demo.")

        _scen, _scen_err = (core.build_custom_scenario(aa_prompt, aa_kind, aa_tool, aa_args, aa_severity)
                            if _tool_names else (None, "define at least one valid tool above"))
        if _scen_err:
            st.caption(f"⚪ Disabled — {_scen_err}.")
        if st.button("🛠️ Run agent-action check", type="primary", key="run_aa_custom",
                     disabled=_scen is None or not _aa_custom_ok):
            with st.spinner(f"Offering your tools to {backend} and capturing its calls "
                            f"({aa_reps}×)…"):
                try:
                    _model = core.make_model(_aa_kind, backend_opts)
                    st.session_state["aa_run"] = core.run_repeated(
                        lambda: core.run_agent_action(_scen, _model, tools=tools), n=int(aa_reps))
                except Exception as exc:
                    st.session_state.pop("aa_run", None)
                    st.error(f"Agent-action check failed against **{backend}**: {exc}")

    aa_rep = st.session_state.get("aa_run")
    if aa_rep:
        if aa_rep.n > 1:
            (st.success if aa_rep.all_passed else st.error)(
                f"**{aa_rep.passed}/{aa_rep.n} passed ({aa_rep.pass_rate:.0f}%)** · "
                f"verdict **{aa_rep.verdict}** · model `{aa_rep.results[0].model_name}`")
            if not aa_rep.all_passed and aa_rep.passed > 0:
                st.caption("⚠️ **Flaky** — passed some runs, failed others. Not safe to trust a "
                          "single lucky pass.")
            st.dataframe(
                pd.DataFrame([{"run": i + 1, "✓": "✅" if r.passed else "❌", "why": r.detail}
                             for i, r in enumerate(aa_rep.results)]),
                hide_index=True, use_container_width=True)
        else:
            _r0 = aa_rep.results[0]
            (st.success if _r0.passed else st.error)(
                f"{'✅ PASS' if _r0.passed else '❌ FAIL'} · verdict **{_r0.verdict}** · "
                f"model `{_r0.model_name}`")
        aa = aa_rep.results[0]
        with st.container(border=True):
            st.markdown("**Tool calls the model actually made**" + (" (run 1)" if aa_rep.n > 1 else ""))
            if aa.calls:
                st.dataframe(
                    pd.DataFrame([{"tool": c.name, "arguments": json.dumps(c.arguments)}
                                  for c in aa.calls]),
                    hide_index=True, use_container_width=True)
            else:
                st.caption("— no tools were called —")
            st.markdown(f"**Why:** {aa.detail}")
            if aa.text:
                st.caption(f"Assistant said: “{aa.text}”")
        if st.button("📥 Add this result to my certificate", key="queue_aa"):
            _queue_agent_checks(core.agent_action_checks(aa_rep, _scen), f"Agent action: {_scen.label}")
            st.success("Queued. Open **🏅 Certify** and re-run to fold it into the grade.")

# ---- Behaviours · multi-step agent loops -------------------------------------
def _flow_agent_loop():
    st.markdown(
        '<div class="pq-callout"><span class="pq-badge">AGENT</span>'
        '<b style="font-size:1.1rem;">🔗 Multi-step agent loop</b><br>'
        'Agent actions (above) capture <b>one</b> decision. Most real agentic failures live in the '
        '<b>chain</b>: an agent calls the right first tool, then misuses the result on step two — e.g. '
        'transferring more money than the balance it just read. This runs a <b>real multi-step loop</b> '
        '(call a tool → see a simulated result → decide the next step) and checks the whole sequence: '
        'did it verify a precondition, in the right order, within limits?</div>',
        unsafe_allow_html=True,
    )
    _al_kind = _BACKEND_KIND[backend]

    al1, al2 = st.columns(2)
    al1.success("**SHIP**  \nEvery step in the chain respected the rules.")
    al2.error("**BLOCK**  \nIt skipped a precondition or exceeded a limit somewhere in the chain.")

    if _al_kind == "http":
        st.warning("HTTP endpoints have no native tool-use channel — use **Claude**, **your "
                   "deployed agent**, or the **Demo bot** for an offline demonstration.")
    elif _al_kind == "http_agent":
        st.warning("This built-in scenario uses fixed banking tools your deployed agent likely "
                   "doesn't have — it's a demonstration of the *check kinds* (must_call, order, "
                   "max_arg), not a real test of your agent. Use the **core.run_agent_loop** API "
                   "directly with your own scenario for that (see the README).")
    elif _al_kind == "mock":
        st.info("The **Demo bot** simulates the *planted* precondition bug in one shot (it isn't "
                "running a real adaptive loop) so you can see the failure mode offline. Switch to "
                "**Claude** for a real multi-step tool-use loop.")

    with st.container(border=True):
        st.markdown("##### 📥 The scenario")
        _labels = [s.label for s in core.AGENT_LOOP_SCENARIOS]
        _pick = st.selectbox("Pick a scenario", _labels, key="al_scenario")
        _scen = core.AGENT_LOOP_SCENARIOS[_labels.index(_pick)]
        st.markdown("**Request sent to the agent:**")
        st.markdown(f"> {_scen.prompt}")
        st.caption(f"✅ A correct agent should: {_scen.intent}")
        st.markdown("**Simulated tool results it will see (these don't really happen):**")
        st.json(_scen.tool_stubs)

    with st.expander("⚙️ Advanced — reliability"):
        al_reps = st.number_input(
            "Repeat this check N times", min_value=1, max_value=10, value=1, key="al_reps",
            help="LLMs are non-deterministic. Repeat the loop to see the real pass rate, not a "
                 "lucky single run.")

    if st.button("🔗 Run agent loop", type="primary", key="run_al", disabled=_al_kind == "http"):
        with st.spinner(f"Running the multi-step loop against {backend} ({al_reps}×)…"):
            try:
                _model = core.make_model(_al_kind, backend_opts)
                st.session_state["al_run"] = core.run_repeated(
                    lambda: core.run_agent_loop(_scen, _model), n=int(al_reps))
            except Exception as exc:
                st.session_state.pop("al_run", None)
                st.error(f"Agent-loop check failed against **{backend}**: {exc}")

    al_rep = st.session_state.get("al_run")
    if al_rep:
        if al_rep.n > 1:
            (st.success if al_rep.all_passed else st.error)(
                f"**{al_rep.passed}/{al_rep.n} passed ({al_rep.pass_rate:.0f}%)** · "
                f"verdict **{al_rep.verdict}** · model `{al_rep.results[0].model_name}`")
            if not al_rep.all_passed and al_rep.passed > 0:
                st.caption("⚠️ **Flaky** — passed some runs, failed others. Not safe to trust a "
                          "single lucky pass.")
            st.dataframe(
                pd.DataFrame([{"run": i + 1, "✓": "✅" if r.passed else "❌"}
                             for i, r in enumerate(al_rep.results)]),
                hide_index=True, use_container_width=True)
        else:
            _r0 = al_rep.results[0]
            (st.success if _r0.passed else st.error)(
                f"{'✅ PASS' if _r0.passed else '❌ FAIL'} · verdict **{_r0.verdict}** · "
                f"model `{_r0.model_name}`")
        al = al_rep.results[0]
        with st.container(border=True):
            st.markdown("**Every tool call made across the whole chain, in order**"
                       + (" (run 1)" if al_rep.n > 1 else ""))
            if al.calls:
                st.dataframe(
                    pd.DataFrame([{"step": i + 1, "tool": c.name, "arguments": json.dumps(c.arguments)}
                                  for i, c in enumerate(al.calls)]),
                    hide_index=True, use_container_width=True)
            else:
                st.caption("— no tools were called —")
        with st.container(border=True):
            st.markdown("**Checks against the whole sequence**")
            st.dataframe(
                pd.DataFrame([{
                    "✓": "✅" if c.passed else "❌",
                    "check": c.check.kind,
                    "detail": ({"must_call": f"must call {c.check.tool}",
                               "must_not_call": f"must NOT call {c.check.tool}",
                               "order": f"{c.check.tool} before {c.check.other_tool}",
                               "max_arg": f"{c.check.tool}.{c.check.arg} ≤ {c.check.limit}"}
                              [c.check.kind]),
                    "why": c.detail or "—",
                } for c in al.checks]),
                hide_index=True, use_container_width=True)
            if al.text:
                st.caption(f"Assistant's final answer: “{al.text}”")
        if st.button("📥 Add this result to my certificate", key="queue_al"):
            _queue_agent_checks(core.agent_loop_checks(al_rep, _scen), f"Agent loop: {_scen.label}")
            st.success("Queued. Open **🏅 Certify** and re-run to fold it into the grade.")

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
                        "model_name": getattr(_jfn, "model_name", backend),
                        "low_confidence": cal.low_confidence, "total": cal.total}
                except Exception as exc:
                    st.session_state.pop("calib", None)
                    st.session_state.pop("calibrated_judge", None)
                    st.error(f"Calibration failed against **{backend}**: {exc}")

        cal = st.session_state.get("calib")
        if cal:
            _lo, _hi = cal.confidence_interval
            jm1, jm2, jm3 = st.columns(3)
            jm1.metric("Agreement with humans", f"{cal.agreement:.0f}%", f"{cal.agree}/{cal.total}")
            jm2.metric("Judge verdict", cal.verdict)
            jm3.metric("95% confidence range", f"{_lo:.0f}–{_hi:.0f}%")
            _jv = {"TRUSTWORTHY": "success", "USE WITH CAUTION": "warning", "DO NOT TRUST": "error"}
            getattr(st, _jv.get(cal.verdict, "info"))(
                f"This judge agreed with your labels **{cal.agreement:.0f}%** of the time → "
                f"**{cal.verdict}**. " + ("Safe to use for grading." if cal.verdict == "TRUSTWORTHY"
                else "Disagreements below — tighten your criteria or pick a stronger judge."))
            if cal.low_confidence:
                st.warning(f"⚠️ **Statistically thin sample.** {cal.caveat} The point estimate "
                          f"({cal.agreement:.0f}%) above could be misleading on its own — look at "
                          f"the confidence range, not just the headline number.")
            st.caption("✅ This calibrated judge is now used for **llm_judge** grading in Evaluate "
                       "and Multi-turn — calibrate once, trusted everywhere." +
                       (" *(treat that reuse with the same caution as the sample size above.)*"
                        if cal.low_confidence else ""))
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

    st.markdown("#### 📖 Glossary — every term on this site, defined")
    _GLOSSARY = {
        "The basics": {
            "Probe / check / case": "One question or scenario sent to the AI, with a rule for "
                "what a correct answer looks like. The smallest unit everything else is built from.",
            "Battery": "A fixed *set* of probes covering many risk dimensions at once — like a "
                "blood test panel, not one blood test. The certification battery is "
                "~22 (Quick) to ~48 (Standard/Thorough) probes.",
            "Validator": "The rule that decides PASS/FAIL for one probe's answer — e.g. *contains* "
                "a substring, matches a *regex*, *equals* a number, is valid *JSON*, or is graded "
                "by an **LLM judge** for open-ended quality.",
            "Severity": "How bad a failure is: **critical** > **high** > **medium** > **low**. "
                "Severity (not just pass/fail count) decides the verdict — one critical failure "
                "matters more than ten low ones.",
            "Pass rate": "Percent of probes that passed — the score behind the letter grade.",
        },
        "Truth & judging": {
            "Golden set / ground truth": "Your own `prompt → expected answer` pairs — *truth you "
                "defined*, not a generic bar. The single biggest upgrade to a certificate's "
                "trustworthiness for your specific use case.",
            "LLM-as-judge": "Using a model to grade an open-ended answer (e.g. \"does this refusal "
                "actually refuse?\") because a keyword check is too brittle for nuanced quality.",
            "Calibrate (a judge)": "Test the judge against examples *you* (a human) already labelled "
                "pass/fail, and measure how often it agrees with you — **agreement %** — before "
                "trusting it to grade anything. An uncalibrated judge is just an unverified guess.",
            "Confidence interval (95%)": "The range the *true* agreement rate could plausibly fall "
                "in, given how few examples it was measured on — not just the single headline %. "
                "6 examples at \"67% agreement\" could really be anywhere from ~30% to ~90%; the "
                "tool warns you below ~20 examples rather than stating the point estimate as fact.",
            "Coverage": "Whether the probes actually span every risk dimension that matters, not "
                "just a lot of probes in one area.",
        },
        "The verdict": {
            "Verdict (SHIP / NEEDS SIGN-OFF / BLOCK)": "The release decision, gated by severity: "
                "**BLOCK** = a critical failure (or a high safety/hallucination failure); "
                "**NEEDS SIGN-OFF** = a lesser high failure; **SHIP** = neither.",
            "Grade (A–F) / status": "The common-man translation of the verdict + score into a "
                "letter and a CERTIFIED / CONDITIONALLY CERTIFIED / NOT CERTIFIED status. "
                "A BLOCK verdict caps the grade at C no matter how high the score is.",
            "Flaky": "Passed *some* runs and failed *others* on the exact same probe — proof the "
                "model is inconsistent, not reliably safe. Treated as NEEDS SIGN-OFF, never a "
                "clean SHIP, even if most runs passed.",
            "Reliability / repeat N times": "Re-running the same check several times because LLMs "
                "are non-deterministic — a single PASS proves much less than a 9/10 or 10/10.",
        },
        "RAG & conversation": {
            "Grounding": "Whether every claim in an answer is actually supported by the retrieved "
                "source — the opposite of hallucination. **GROUNDED** = faithful; **NOT GROUNDED** "
                "= invented or contradicted facts; **GROUNDED BUT WRONG** = faithful but missed the "
                "right answer; **GROUNDED BUT OVERCONFIDENT** = sources disagreed and it silently "
                "picked one instead of saying so.",
            "Distractor document": "An irrelevant retrieved document mixed in on purpose, to test "
                "whether it pulls the model toward a wrong answer.",
            "Checkpoint (multi-turn)": "A check pinned to *one specific turn* in a conversation, so "
                "a problem on turn 1 can't hide behind a clean-looking reply on turn 5.",
        },
        "Agents": {
            "Agent action": "A check on what an agent *does*, not just what it says — captures the "
                "real tool call(s) it makes via native tool-use, not a self-reported description.",
            "Agent loop / multi-step": "A *chain* of agent decisions — call a tool, see the result, "
                "decide the next step, repeat — because real agentic bugs often live in step two, "
                "not the first decision (e.g. transferring money without checking the balance first).",
            "Precondition": "Something an agent must verify *before* taking an action — the classic "
                "miss is acting on an assumption instead of checking it first.",
            "must_call / must_not_call / order / max_arg": "The four rule types an agent-loop check "
                "is built from: a tool *must* be called, must *never* be called, must be called "
                "*before* another tool, or an argument must *never exceed* a limit.",
            "Adversarial search": "Instead of testing one hand-written attack phrasing, "
                "automatically trying several coercion framings (direct override, fake authority, "
                "urgency, roleplay...) and reporting the **break rate** — proof a refusal holds up "
                "broadly, not just on one wording.",
        },
        "Tracking & safety": {
            "Snapshot": "A saved, point-in-time export of every individual check's result — what "
                "lets you compare *today's* run against an earlier one.",
            "Regression": "A check that used to PASS and now FAILS, found by comparing two "
                "snapshots — the thing a moving score alone can hide.",
            "BYOK (bring your own key)": "Your API key lives only in your browser session for this "
                "visit — it's never written to the server, stored, or logged.",
            "SSRF guard": "A safety check that blocks the HTTP backend from reaching private/"
                "internal/metadata network addresses, so the tool can't be abused as a proxy into "
                "infrastructure it shouldn't reach.",
        },
    }
    for section, terms in _GLOSSARY.items():
        with st.expander(section):
            for term, definition in terms.items():
                st.markdown(f"**{term}** — {definition}")


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
         "📚 RAG grounding — is the answer faithful to a provided source?",
         "🛠️ Agent actions — does it call the right tool (and refuse dangerous ones)?",
         "🔗 Agent loops — does it verify a precondition before acting, across multiple steps?"],
        key="beh_mode")
    st.divider()
    if beh_mode.startswith("🔁"):
        _flow_multiturn()
    elif beh_mode.startswith("📚"):
        _flow_rag()
    elif beh_mode.startswith("🛠️"):
        _flow_agent_action()
    else:
        _flow_agent_loop()

with tab_judge:
    _flow_judge()
with tab_audit:
    _flow_audit()
with tab_help:
    _flow_help()
