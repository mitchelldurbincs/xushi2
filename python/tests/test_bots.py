from xushi2.bots import VALID_SCRIPTED_BOTS


def test_valid_scripted_bots_expected_and_unique() -> None:
    expected = {
        "walk_to_objective",
        "hold_and_shoot",
        "basic",
        "weak_basic",
        "weak_basic_v2",
        "noop",
    }
    assert set(VALID_SCRIPTED_BOTS) == expected
    assert len(VALID_SCRIPTED_BOTS) == len(set(VALID_SCRIPTED_BOTS))
