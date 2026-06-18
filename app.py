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

st.set_page_config(page_title="AI Testing Studio", page_icon="🧪", layout="wide")

st.markdown(
    """
    <style>
      /* hero */
      .hero {background:linear-gradient(135deg,#0f172a 0%,#134e4a 100%);
             color:#e2e8f0;padding:1.5rem 1.8rem;border-radius:16px;margin-bottom:1rem;
             box-shadow:0 8px 24px rgba(2,6,23,.18);}
      .hero h1 {font-family:ui-monospace,"JetBrains Mono",Menlo,Consolas,monospace;
                font-size:2rem;margin:0;letter-spacing:-.5px;color:#f8fafc;}
      .hero h1 .accent{color:#34d399;}
      .hero p {margin:.4rem 0 0;color:#cbd5e1;font-size:1.02rem;}
      /* step strip */
      .steps{display:flex;gap:.6rem;margin:.2rem 0 1rem;flex-wrap:wrap;}
      .step{flex:1;min-width:180px;background:#f8fafc;border:1px solid #e2e8f0;
            border-left:4px solid #10b981;border-radius:10px;padding:.6rem .9rem;}
      .step b{display:block;color:#0f172a;font-size:.95rem;}
      .step span{color:#64748b;font-size:.85rem;}
      /* category chips */
      .chips{margin:.2rem 0 .4rem;}
      .chip{display:inline-block;background:#ecfdf5;color:#065f46;border:1px solid #a7f3d0;
            border-radius:999px;padding:2px 11px;margin:3px;font-size:12px;font-weight:600;}
      .section-num{color:#10b981;font-weight:700;}
      /* prompt-check callout */
      .pq-callout{background:#ecfdf5;border:1px solid #6ee7b7;border-left:4px solid #10b981;
                  border-radius:10px;padding:.7rem 1rem;margin:.2rem 0 .4rem;color:#065f46;}
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

st.markdown(
    '<div class="steps">'
    '<div class="step"><b>1 · Describe</b><span>A feature or full user story</span></div>'
    '<div class="step"><b>2 · Generate</b><span>Risk-based cases + coverage check</span></div>'
    '<div class="step"><b>3 · Run</b><span>Mock, Claude, or any endpoint</span></div>'
    '<div class="step"><b>4 · Verdict</b><span>Ship / sign-off / block + report</span></div>'
    '</div>',
    unsafe_allow_html=True,
)

_chips = "".join(f'<span class="chip">{c}</span>' for c in core.categories())
st.markdown(f'<div class="chips">Covers: {_chips}</div>', unsafe_allow_html=True)

# ---- sidebar: which model to test against ---------------------------------
with st.sidebar:
    st.header("Model under test")
    backends = ["Mock (offline)"] if PUBLIC else ["Mock (offline)", "Claude API", "HTTP endpoint"]
    backend = st.radio(
        "Backend", backends,
        help="Mock runs offline with no key. Claude/HTTP test a real model.",
    )
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

_BACKEND_KIND = {"Mock (offline)": "mock", "Claude API": "claude", "HTTP endpoint": "http"}

# ---- optional: quick prompt quality check ---------------------------------
st.session_state.setdefault("pq_text", "")

st.markdown(
    '<div class="pq-callout"><span class="pq-badge">NEW</span>'
    '<b style="font-size:1.1rem;">✍️ Score your prompt instantly</b><br>'
    'Paste any prompt to get a quality score, a few quick pointers, and an example '
    'of how it could look — open the panel below to try it.</div>',
    unsafe_allow_html=True,
)

with st.expander("🔎  Open the prompt quality check", expanded=False):
    st.caption("Paste a prompt to get a score, a couple of quick pointers, and an "
               "example of how it could look — no lecturing.")
    pq_text = st.text_area("Prompt to score", height=110, key="pq_text",
                           placeholder="Paste the prompt you wrote…")
    pq_llm = (not PUBLIC) and st.checkbox("Use Claude for the critique (needs the Claude backend + key)")

    def _clear_pq():
        st.session_state["pq_text"] = ""

    bc1, bc2, _ = st.columns([1, 1, 4])
    do_score = bc1.button("Score this prompt", type="primary", disabled=not pq_text.strip())
    bc2.button("Clear", on_click=_clear_pq, disabled=not pq_text)

    if do_score:
        if pq_llm:
            core.set_backend("claude", api_key=backend_opts.get("api_key", ""))
        try:
            score = core.assess_prompt(pq_text, use_llm=pq_llm)
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
                st.markdown("**Example of how it could look:**")
                st.code(score.example, language="text")

# ---- step 1: describe + generate ------------------------------------------
AI_TYPES = ["(none)", "chatbot", "rag", "classifier", "summarizer", "agent"]

# initialise form state once so the example picker can pre-fill it
for key, default in {"feature_input": "", "aitype_input": "(none)",
                     "ov_safety": 0, "ov_accuracy": 0, "ov_agent": 0}.items():
    st.session_state.setdefault(key, default)

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
        "Feature or user story",
        key="feature_input",
        height=140,
        placeholder=(
            "A short phrase — or paste a full user story with acceptance criteria, e.g.\n\n"
            "As a user, I want to reset my password via email.\n"
            "Acceptance criteria:\n"
            "- A reset link is emailed and expires in 30 minutes\n"
            "- The link works only once\n"
            "- Unknown emails get a neutral 'if an account exists…' response"
        ),
    )
with col2:
    ai_type = st.selectbox("AI type", AI_TYPES, key="aitype_input")
    st.caption("Tip: with the Claude backend, a full user story yields a test per acceptance criterion.")

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

def _clear_feature():
    st.session_state["feature_input"] = ""
    st.session_state["aitype_input"] = "(none)"
    st.session_state["ex_choice"] = "(custom)"
    st.session_state["ov_safety"] = 0
    st.session_state["ov_accuracy"] = 0
    st.session_state["ov_agent"] = 0
    for k in ("_applied_example", "gen", "run"):
        st.session_state.pop(k, None)


gc1, gc2, _ = st.columns([1, 1, 4])
if gc1.button("⚙️ Generate test suite", type="primary", disabled=not feature):
    core.set_backend(_BACKEND_KIND[backend], **backend_opts)
    with st.spinner("Generating cases…"):
        gen = core.generate_suite(
            feature,
            None if ai_type == "(none)" else ai_type,
            overrides or None,
        )
    st.session_state["gen"] = gen
gc2.button("Clear", on_click=_clear_feature, disabled=not feature, key="clear_feature")

# ---- step 1 results --------------------------------------------------------
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

    # ---- step 2: run -------------------------------------------------------
    st.subheader("2 · Run the suite")
    if st.button("▶️ Run against the selected model", type="primary"):
        core.set_backend(_BACKEND_KIND[backend], **backend_opts)
        with st.spinner("Running…"):
            run = core.run_suite_dir(gen.out_dir)
        st.session_state["run"] = run

    run = st.session_state.get("run")
    if run:
        st.subheader("3 · Report")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pass rate", f"{run.summary.pass_rate:.0f}%")
        m2.metric("Passed", f"{run.summary.passed}/{run.summary.total}")
        m3.metric("Failed", run.summary.failed)
        m4.metric("Verdict", run.verdict)
        verdict_style = {"SHIP": "success", "NEEDS SIGN-OFF": "warning", "BLOCK": "error"}
        getattr(st, verdict_style.get(run.verdict, "info"))(
            f"Release verdict: **{run.verdict}**  ·  model: `{run.model_name}`"
        )
        components.html(run.html, height=620, scrolling=True)

        rc1, rc2, rc3 = st.columns(3)
        rc1.download_button("⬇️ HTML report", run.html, "report.html", "text/html")
        rc2.download_button("⬇️ JSON report", run.json, "report.json", "application/json")
        # bundle the generated YAML for download
        bundle = "\n".join(
            f"# === {os.path.basename(p)} ===\n" + open(p, encoding="utf-8").read()
            for p in sorted(glob.glob(os.path.join(gen.out_dir, "*.yaml")))
        )
        rc3.download_button("⬇️ Generated suite (YAML)", bundle, "suite.yaml", "text/yaml")
