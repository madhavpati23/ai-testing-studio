"""The battery must actually contain what the product claims it contains.

These guard the failure mode that motivated them: a case whose category was not
in the taxonomy, or whose validator args were misnamed, was *silently dropped* by
validate_all(). The suite still went green, the certificate still said "CERTIFIED",
and seven whole risk dimensions — bias, privacy, indirect injection, multilingual,
long context, code safety, agent safety — were never tested at all.

A dropped check is worse than a failing one: a failure is visible, a drop is not.
"""
import core
import pytest

ALL_LEVELS = ("quick", "standard", "thorough", "deep")

# Dimensions the UI advertises. If a case is added for a new dimension, add it
# here too — that is the point: coverage claims must be asserted, not assumed.
ADVERTISED_AT_DEEP = {
    "safety", "red_team", "hallucination", "accuracy", "bias", "privacy",
    "indirect_injection", "instruction_following", "multilingual", "consistency",
    "long_context", "code_safety", "agent_safety",
}


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_no_case_is_silently_dropped(level):
    """Every defined case must survive validation at every level."""
    errors = core.certification_build_errors(level)
    assert errors == [], (
        f"{len(errors)} case(s) dropped at level={level} — these are NOT being "
        f"tested and will not appear in any certificate:\n  " + "\n  ".join(errors)
    )


def test_every_defined_case_reaches_the_deep_battery():
    defined = (core.CERTIFICATION_CASES + core.CERTIFICATION_CASES_EXTENDED
               + core.CERTIFICATION_CASES_EXTRA)
    built = core.build_certification("deep")
    assert len(built) == len(defined), (
        f"deep battery runs {len(built)} of {len(defined)} defined cases — "
        f"{len(defined) - len(built)} vanished in validation"
    )


def test_advertised_dimensions_are_actually_present():
    got = {c.category for c in core.build_certification("deep")}
    missing = ADVERTISED_AT_DEEP - got
    assert not missing, f"advertised but never tested: {sorted(missing)}"


def test_case_ids_are_unique():
    defined = (core.CERTIFICATION_CASES + core.CERTIFICATION_CASES_EXTENDED
               + core.CERTIFICATION_CASES_EXTRA)
    seen, dupes = set(), []
    for c in defined:
        if c["id"] in seen:
            dupes.append(c["id"])
        seen.add(c["id"])
    assert not dupes, f"duplicate case ids silently drop one of each pair: {dupes}"


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_battery_grows_monotonically(level):
    """A deeper level must never test fewer checks than a shallower one."""
    sizes = {lv: len(core.build_certification(lv)) for lv in ALL_LEVELS}
    assert sizes["quick"] <= sizes["standard"] <= sizes["deep"], sizes


def test_validator_args_match_the_validator_contract():
    """Misnamed args (e.g. 'substring' where the validator wants 'value') drop
    the case at validation time, so assert the contract directly."""
    from prompt_regression.validators import REGISTRY
    defined = (core.CERTIFICATION_CASES + core.CERTIFICATION_CASES_EXTENDED
               + core.CERTIFICATION_CASES_EXTRA)
    bad = []
    for c in defined:
        v, args = c.get("validator"), c.get("args") or {}
        if v in ("contains", "not_contains") and "value" not in args:
            bad.append(f"{c['id']}: {v} needs args.value, got {sorted(args)}")
        if v == "regex" and "pattern" not in args:
            bad.append(f"{c['id']}: regex needs args.pattern, got {sorted(args)}")
    assert not bad, "cases with wrong validator args:\n  " + "\n  ".join(bad)
    assert "not_contains" in REGISTRY
