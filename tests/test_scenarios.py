import core
from test_case_generator.config import AI_TYPES
from test_case_generator.taxonomy import CATEGORIES


def test_scenarios_load_and_are_grouped():
    scenarios = core.load_scenarios()
    assert len(scenarios) >= 10
    assert len({s.group for s in scenarios}) >= 3        # multiple groups


def test_every_scenario_is_valid():
    for s in core.load_scenarios():
        assert s.feature.strip()
        assert s.ai_type is None or s.ai_type in AI_TYPES
        for cat, minimum in s.overrides.items():
            assert cat in CATEGORIES
            assert isinstance(minimum, int) and minimum > 0


def test_a_scenario_runs_end_to_end():
    core.set_backend("mock")
    s = next(x for x in core.load_scenarios() if x.ai_type == "agent")
    gen = core.generate_suite(s.feature, s.ai_type, s.overrides or None)
    assert any(c.category == "agent" for c in gen.cases)
    run = core.run_suite_dir(gen.out_dir)
    assert run.verdict in ("SHIP", "NEEDS SIGN-OFF", "BLOCK")
