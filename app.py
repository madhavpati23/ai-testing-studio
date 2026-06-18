"""AI Testing Studio — a browser UI for the generate -> run -> report toolchain.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import glob
import os

import streamlit as st
import streamlit.components.v1 as components

import core

# In a shared public deployment, one process serves all sessions — so we must
# NOT accept secrets (ANTHROPIC_API_KEY) or arbitrary URLs (SSRF). Set
# PRS_STUDIO_PUBLIC=1 on the public instance: it restricts to the offline mock.
PUBLIC = str(os.environ.get("PRS_STUDIO_PUBLIC", "")).strip().lower() in ("1", "true", "yes", "on")

# HTTP-backend presets so common targets are one click (no typing).
_HTTP_PRESETS = {
    "Custom": None,
    "Rovo proxy (localhost:8000)": {
        "url": "http://localhost:8000/ask",
        "body": '{"prompt": {PROMPT}}',
        "response_path": "output",
        "headers": "",
    },
    "OpenAI-compatible": {
        "url": "https://api.openai.com/v1/chat/completions",
        "body": '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": {PROMPT}}]}',
        "response_path": "choices.0.message.content",
        "headers": '{"Authorization": "Bearer sk-..."}',
    },
}
_BACKEND_KIND = {"Mock (offline)": "mock", "Claude API": "claude", "HTTP endpoint": "http"}
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


def _load_example_story():
    st.session_state["feature_input"] = core.EXAMPLE_USER_STORY
    st.session_state["ex_choice"] = "(custom)"
    st.session_state.pop("_applied_example", None)


def _clear_feature():
    st.session_state["feature_input"] = ""
    st.session_state["aitype_input"] = "(none)"
    st.session_state["ex_choice"] = "(custom)"
    st.session_state["ov_safety"] = 0
    st.session_state["ov_accuracy"] = 0
    st.session_state["ov_agent"] = 0
    for k in ("_applied_example", "gen", "run", "story_analysis"):
        st.session_state.pop(k, None)


# ---- sidebar: which model to test against ---------------------------------
with st.sidebar:
    st.header("Model under test")
    backends = ["Mock (offline)"] if PUBLIC else ["Mock (offline)", "Claude API", "HTTP endpoint"]
    backend = st.radio("Backend", backends,
                       help="Mock runs offline with no key. Claude/HTTP test a real model.")
    if PUBLIC:
        st.caption("This is a public demo — it runs the offline mock only. "
                   "Clone the repo to test the Claude API or your own endpoint.")
    backend_opts: dict[str, str] = {}
    if backend == "Claude API":
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
        backend_opts["headers"] = st.text_input("Headers (JSON)", key="http_headers",
                                                placeholder='{"Authorization": "Bearer ..."}')
        if st.session_state.get("http_preset", "").startswith("Rovo"):
            st.caption("Start the rovo-test-proxy first: `uvicorn app:app --port 8000`. "
                       "If you set PROXY_TOKEN, add it as a header: {\"X-Proxy-Token\": \"…\"}.")

    st.divider()
    st.markdown(
        "**How the verdict works**\n\n"
        "- **BLOCK** — any Critical fails, or a High safety/hallucination fails\n"
        "- **NEEDS SIGN-OFF** — any other High fails\n"
        "- **SHIP** — no Critical/High failures"
    )

tab_test, tab_prompt, tab_story, tab_help = st.tabs(
    ["🧪 Test a feature", "✍️ Prompt & instructions", "🧭 Story analysis", "ℹ️ How it works"]
)

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
            "Feature or user story", key="feature_input", height=140,
            placeholder=(
                "A short phrase — or paste a full user story with acceptance criteria, e.g.\n\n"
                "As a user, I want to reset my password via email.\n"
                "Acceptance criteria:\n"
                "- A reset link is emailed and expires in 30 minutes\n"
                "- The link works only once"
            ),
        )
    with col2:
        ai_type = st.selectbox("AI type", AI_TYPES, key="aitype_input")
        st.caption("Tip: with the Claude backend, a full user story yields a test per acceptance criterion. "
                   "See the **Story analysis** tab for what we'd test.")

    with st.expander("Coverage overrides (optional) — raise the bar for this feature"):
        st.caption("Each category set here becomes REQUIRED at the given minimum.")
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            v_safety = st.number_input("safety min", min_value=0, max_value=20, step=1, key="ov_safety")
        with oc2:
            v_accuracy = st.number_input("accuracy min", min_value=0, max_value=20, step=1, key="ov_accuracy")
        with oc3:
            v_agent = st.number_input("agent min", min_value=0, max_value=20, step=1, key="ov_agent")
        overrides = {k: int(v) for k, v in
                     {"safety": v_safety, "accuracy": v_accuracy, "agent": v_agent}.items() if v}

    gc1, gc2, _ = st.columns([1, 1, 4])
    if gc1.button("⚙️ Generate test suite", type="primary", disabled=not feature):
        core.set_backend(_BACKEND_KIND[backend], **backend_opts)
        with st.spinner("Generating cases…"):
            st.session_state["gen"] = core.generate_suite(
                feature, None if ai_type == "(none)" else ai_type, overrides or None)
    gc2.button("Clear", on_click=_clear_feature, disabled=not feature, key="clear_feature")

    gen = st.session_state.get("gen")
    if gen:
        st.success(f"Generated {len(gen.cases)} case(s) with **{gen.generator_name}**.")
        if gen.errors:
            st.warning(f"Dropped {len(gen.errors)} invalid case(s): " + "; ".join(gen.errors))
        st.dataframe(
            [{"id": c.id, "category": c.category, "severity": c.severity,
              "validator": c.validator, "prompt": c.prompt} for c in gen.cases],
            use_container_width=True, hide_index=True,
        )
        (st.error if gen.has_gaps else st.info)(
            ("⚠️ Below coverage standard\n\n" if gen.has_gaps else "✅ Coverage\n\n")
            + "```\n" + gen.coverage_text + "\n```"
        )

        st.subheader("2 · Run the suite")
        rcol1, rcol2 = st.columns([2, 1])
        do_run = rcol1.button("▶️ Run against the selected model", type="primary")
        sla_in = rcol2.number_input("SLA (ms, optional)", min_value=0, max_value=120000,
                                    value=0, step=100,
                                    help="Flag cases whose response time exceeds this. 0 = off.")
        if do_run:
            core.set_backend(_BACKEND_KIND[backend], **backend_opts)
            with st.spinner("Running…"):
                st.session_state["run"] = core.run_suite_dir(gen.out_dir, sla_ms=sla_in or None)

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
            verdict_style = {"SHIP": "success", "NEEDS SIGN-OFF": "warning", "BLOCK": "error"}
            getattr(st, verdict_style.get(run.verdict, "info"))(
                f"Release verdict: **{run.verdict}**  ·  model: `{run.model_name}`")
            components.html(run.html, height=620, scrolling=True)

            rc1, rc2, rc3 = st.columns(3)
            rc1.download_button("⬇️ HTML report", run.html, "report.html", "text/html")
            rc2.download_button("⬇️ JSON report", run.json, "report.json", "application/json")
            bundle = "\n".join(
                f"# === {os.path.basename(p)} ===\n" + open(p, encoding="utf-8").read()
                for p in sorted(glob.glob(os.path.join(gen.out_dir, "*.yaml")))
            )
            rc3.download_button("⬇️ Generated suite (YAML)", bundle, "suite.yaml", "text/yaml")

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
# TAB 3 — Story analysis
# ============================================================================
with tab_story:
    st.subheader("What can we test on this story?")
    st.caption("Reads the feature / user story from the **Test a feature** tab.")
    feat = st.session_state.get("feature_input", "").strip()

    c1, c2, _ = st.columns([1.4, 1.4, 3])
    if c1.button("🧭 Analyse this story", disabled=not feat):
        st.session_state["story_analysis"] = core.analyze_story(feat)
    c2.button("📋 Load the example story", on_click=_load_example_story)

    if not feat:
        st.info("Enter a feature in **Test a feature**, or click **Load the example story** above.")

    a = st.session_state.get("story_analysis")
    if a and feat:
        with st.container(border=True):
            st.markdown("**Testable requirements — functional**")
            st.markdown("\n".join(f"- {x}" for x in a.functional))
            st.markdown("**Non-functional:** " + ", ".join(a.non_functional))
            st.markdown("**Suggested tests, by risk category**")
            for cat, items in a.suggested.items():
                st.markdown(f"- **`{cat}`** — " + "; ".join(items))
            st.markdown("**Validation matrix**")
            st.table({"Area": [m[0] for m in a.matrix], "What to verify": [m[1] for m in a.matrix]})
        st.caption("Heuristic preview. **Generate** (Test a feature tab) turns this into a runnable "
                   "suite — tailored to your acceptance criteria with the Claude backend.")

    st.divider()
    st.markdown("#### 📋 How to write a user story for AI testing")
    st.caption("A clear AI-testing story: role, want, so-that, explicit acceptance criteria, and "
               "testing notes (source of truth + abuse cases).")
    st.code(core.EXAMPLE_USER_STORY, language="text")

# ============================================================================
# TAB 4 — How it works
# ============================================================================
with tab_help:
    st.subheader("How it works")
    st.markdown(
        "1. **Describe** a feature or paste a full user story.\n"
        "2. **Generate** a risk-based suite and check it against the coverage standard.\n"
        "3. **Run** it against the offline mock, the Claude API, or any HTTP endpoint.\n"
        "4. Get a **report** with pass-rate, severity, latency, and a ship / no-ship **verdict**."
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
