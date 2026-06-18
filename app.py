"""AI Testing Studio — a browser UI for the generate -> run -> report toolchain.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

import core

st.set_page_config(page_title="AI Testing Studio", page_icon="🧪", layout="wide")

st.title("🧪 AI Testing Studio")
st.caption(
    "Describe an AI feature → generate a risk-based test suite → run it → get a "
    "report with a ship / no-ship verdict. Powered by ai-test-case-generator + "
    "prompt-regression-suite."
)

# ---- sidebar: which model to test against ---------------------------------
with st.sidebar:
    st.header("Model under test")
    backend = st.radio(
        "Backend", ["Mock (offline)", "Claude API", "HTTP endpoint"],
        help="Mock runs offline with no key. Claude/HTTP test a real model.",
    )
    backend_opts: dict[str, str] = {}
    if backend == "Claude API":
        backend_opts["api_key"] = st.text_input("ANTHROPIC_API_KEY", type="password")
    elif backend == "HTTP endpoint":
        backend_opts["url"] = st.text_input("Endpoint URL", placeholder="https://api.example.com/chat")
        backend_opts["body"] = st.text_input("Body template", value='{"prompt": {PROMPT}}',
                                             help="The token {PROMPT} is replaced with the JSON-encoded prompt.")
        backend_opts["response_path"] = st.text_input("Response path", value="output",
                                                      help='Dotted path to the answer, e.g. choices.0.message.content')
        backend_opts["headers"] = st.text_input("Headers (JSON)", value="",
                                                placeholder='{"Authorization": "Bearer ..."}')

    st.divider()
    st.markdown(
        "**How the verdict works**\n\n"
        "- **BLOCK** — any Critical fails, or a High safety/hallucination fails\n"
        "- **NEEDS SIGN-OFF** — any other High fails\n"
        "- **SHIP** — no Critical/High failures"
    )

_BACKEND_KIND = {"Mock (offline)": "mock", "Claude API": "claude", "HTTP endpoint": "http"}

# ---- step 1: describe + generate ------------------------------------------
AI_TYPES = ["(none)", "chatbot", "rag", "classifier", "summarizer", "agent"]

# initialise form state once so the example picker can pre-fill it
for key, default in {"feature_input": "", "aitype_input": "(none)",
                     "ov_safety": 0, "ov_accuracy": 0, "ov_agent": 0}.items():
    st.session_state.setdefault(key, default)

st.subheader("1 · Describe the feature")

scenarios = core.load_scenarios()
by_label = {f"{s.group} — {s.label}": s for s in scenarios}
choice = st.selectbox("Start from an example (optional)", ["(custom)"] + list(by_label))
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

if st.button("⚙️ Generate test suite", type="primary", disabled=not feature):
    core.set_backend(_BACKEND_KIND[backend], **backend_opts)
    with st.spinner("Generating cases…"):
        gen = core.generate_suite(
            feature,
            None if ai_type == "(none)" else ai_type,
            overrides or None,
        )
    st.session_state["gen"] = gen

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
        verdict_style = {"SHIP": "success", "NEEDS SIGN-OFF": "warning", "BLOCK": "error"}
        getattr(st, verdict_style.get(run.verdict, "info"))(
            f"Release verdict: **{run.verdict}**  ·  model: `{run.model_name}`  ·  "
            f"pass rate {run.summary.pass_rate:.1f}% ({run.summary.passed}/{run.summary.total})"
        )
        components.html(run.html, height=620, scrolling=True)

        rc1, rc2, rc3 = st.columns(3)
        rc1.download_button("⬇️ HTML report", run.html, "report.html", "text/html")
        rc2.download_button("⬇️ JSON report", run.json, "report.json", "application/json")
        # bundle the generated YAML for download
        import glob
        import os
        bundle = "\n".join(
            f"# === {os.path.basename(p)} ===\n" + open(p, encoding="utf-8").read()
            for p in sorted(glob.glob(os.path.join(gen.out_dir, "*.yaml")))
        )
        rc3.download_button("⬇️ Generated suite (YAML)", bundle, "suite.yaml", "text/yaml")
