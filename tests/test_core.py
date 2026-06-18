import os

import core


def test_full_flow_with_mock():
    core.set_backend("mock")
    gen = core.generate_suite("password reset email", "chatbot")
    assert len(gen.cases) >= 8
    assert os.path.isdir(gen.out_dir)

    run = core.run_suite_dir(gen.out_dir)
    assert run.summary.total == len(gen.cases)
    assert run.verdict in ("SHIP", "NEEDS SIGN-OFF", "BLOCK")
    assert "<!doctype html>" in run.html.lower()
    assert run.model_name == "mock-helpbot-v2"


def test_agent_type_adds_agent_cases():
    core.set_backend("mock")
    gen = core.generate_suite("customer support agent", "agent")
    assert any(c.category == "agent" for c in gen.cases)


def test_coverage_override_creates_gap():
    core.set_backend("mock")
    gen = core.generate_suite("password reset", "chatbot", overrides={"accuracy": 3})
    assert gen.has_gaps  # mock emits no accuracy cases, so requiring it gaps


def test_set_backend_manages_env():
    core.set_backend("http", url="https://api.example.com/chat", response_path="output")
    assert os.environ["PRS_HTTP_URL"] == "https://api.example.com/chat"
    core.set_backend("mock")
    assert "PRS_HTTP_URL" not in os.environ
    assert "ANTHROPIC_API_KEY" not in os.environ
