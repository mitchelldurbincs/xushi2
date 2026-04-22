from __future__ import annotations

from eval.eval_memory_toy import AblationResult, _check_gate, ablation_modes_differ


def test_ablation_modes_mutate_hidden_state_differently():
    assert ablation_modes_differ(num_episodes=10, seed=0)


def test_check_gate_passes_for_expected_ranges():
    ok, failures = _check_gate(
        AblationResult(mode="normal", mean=-0.05, ci95=0.01, n=500),
        AblationResult(mode="zero_every_tick", mean=-1.0, ci95=0.02, n=500),
        AblationResult(mode="random_every_tick", mean=-1.1, ci95=0.03, n=500),
    )
    assert ok
    assert failures == []


def test_check_gate_fails_on_normal_bound():
    ok, failures = _check_gate(
        AblationResult(mode="normal", mean=-0.30, ci95=0.01, n=500),
        AblationResult(mode="zero_every_tick", mean=-1.0, ci95=0.02, n=500),
        AblationResult(mode="random_every_tick", mean=-1.1, ci95=0.03, n=500),
    )
    assert not ok
    assert any("normal_mean" in f for f in failures)
