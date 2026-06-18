import core


def test_analyze_detects_query_data_and_hallucination():
    story = ('As a Power Scheduler, I want to ask the AI "What is my Day-Ahead position '
             'for ERCOT North tomorrow?" so I can review exposure. Source: Snowflake. '
             'Respond within the 5-second SLA.')
    a = core.analyze_story(story)
    assert a.functional                                   # some functional reqs found
    cats = a.suggested
    assert "hallucination" in cats                        # always-on AI risk
    assert "accuracy" in cats and "data_validation" in cats
    assert "Performance within SLA" in a.non_functional   # SLA detected
    assert any(area == "Performance" for area, _ in a.matrix)


def test_analyze_detects_agent_actions_and_safety():
    a = core.analyze_story("Jira agent that can transition and update issues on request.")
    assert "agent" in a.suggested
    assert "safety" in a.suggested                        # actions -> authorization tests


def test_analyze_always_includes_core_ai_risks():
    a = core.analyze_story("a simple assistant")
    for must in ("hallucination", "robustness", "red_team"):
        assert must in a.suggested


def test_example_story_is_present_and_structured():
    assert "Acceptance criteria:" in core.EXAMPLE_USER_STORY
    assert core.EXAMPLE_USER_STORY.lower().startswith("as a")
