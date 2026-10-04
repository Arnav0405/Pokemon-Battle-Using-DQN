"""Offline tests for the curve/metric code (no Showdown server, no training).

The convergence rule is the part worth pinning down: it decides what the report claims, and it is
easy to make it either trivially true or trivially false by picking a bad window. These tests fix
the intended semantics -- a flat run converges, a run that is still climbing at the end does not --
and pin the noise floor to the formula, so a future "tune the tolerance" edit cannot silently turn
the test into a tautology.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from lab_ca import plots as plots_module
from lab_ca.plots import (
    CONVERGENCE_SIGMA_MULTIPLE,
    CONVERGENCE_TOLERANCE_FLOOR,
    CONVERGENCE_WINDOW,
    EVEN_MATCH_RATE,
    _binomial_half_width,
    binomial_sigma,
    block_means,
    comparison_seaborn,
    compute_metrics,
    load_episodes,
    load_frame,
    main,
    moving_average,
)


def _rows(wins, rewards=None, steps=None, outcome=None):
    """Minimal episodes.jsonl rows: only the fields the metrics actually read."""
    rewards = rewards if rewards is not None else [30.0 if win else -30.0 for win in wins]
    steps = steps if steps is not None else [25] * len(wins)
    return [
        {
            "episode": index + 1,
            "win": int(win),
            "cumulative_reward": float(rewards[index]),
            "steps": int(steps[index]),
            "outcome": outcome[index] if outcome else ("win" if win else "loss"),
            "opponent": "heuristic",
        }
        for index, win in enumerate(wins)
    ]


def _flat_run(episodes=1200, rate=6):
    """A periodically flat win rate: 6 wins in every 10 episodes, forever."""
    return [1 if (index % 10) < rate else 0 for index in range(episodes)]


def _climbing_run(episodes=1200, seed=0):
    """A run whose win probability ramps across the whole run -- still rising at the last episode."""
    rng = np.random.default_rng(seed)
    probabilities = 0.05 + 0.85 * (np.arange(episodes) / episodes)
    return (rng.random(episodes) < probabilities).astype(int).tolist()


def test_binomial_half_width_matches_formula():
    """The drawn band is the 95% binomial CI of the windowed rate, in percentage points."""
    for rate, window in ((60.0, 50), (60.0, 200), (5.0, 100), (95.0, 200)):
        proportion = rate / 100.0
        expected = 1.96 * math.sqrt(proportion * (1 - proportion) / window) * 100
        assert _binomial_half_width(np.array([rate]), window)[0] == pytest.approx(expected)


def test_binomial_half_width_shrinks_with_the_window():
    """More episodes per point => tighter band. This is why convergence moved to a 200-window."""
    wide = _binomial_half_width(np.array([60.0]), 50)[0]
    narrow = _binomial_half_width(np.array([60.0]), CONVERGENCE_WINDOW)[0]
    assert narrow < wide
    assert narrow == pytest.approx(wide / math.sqrt(CONVERGENCE_WINDOW / 50))


def test_tolerance_is_a_sane_multiple_of_the_noise_floor():
    """The band is 2 sigma of the convergence window -- and the *old* 50-episode band was inside noise.

    Guards the original defect: at 50 episodes a ±5 pp band is ~0.7 sigma, so a converged agent
    failed the test routinely. The band used for a 60% run on a 200-episode window is 2*3.5 = 6.9 pp.
    """
    sigma_wide = binomial_sigma(0.60, 50)
    sigma_narrow = binomial_sigma(0.60, CONVERGENCE_WINDOW)
    band = max(CONVERGENCE_TOLERANCE_FLOOR, CONVERGENCE_SIGMA_MULTIPLE * sigma_narrow)
    assert band == pytest.approx(2 * sigma_narrow)
    # The band on the 200-window is exactly the old ±5 pp band's width relative to a 100-window...
    # ...but the old test ran on 50 episodes, where ±5 pp is only 0.7 sigma.
    assert CONVERGENCE_TOLERANCE_FLOOR / sigma_wide < 1.0
    assert band / sigma_narrow >= 1.3


def test_tolerance_floor_keeps_a_very_high_win_rate_testable():
    """At a 98% win rate the binomial sigma collapses; the ±5 pp floor stops the test going tautological."""
    band = max(CONVERGENCE_TOLERANCE_FLOOR, CONVERGENCE_SIGMA_MULTIPLE * binomial_sigma(0.98, CONVERGENCE_WINDOW))
    assert band == CONVERGENCE_TOLERANCE_FLOOR


def test_block_means_are_non_overlapping_and_drop_the_tail():
    """Trend fitting uses independent blocks: overlapping windows give a fake-precise slope CI."""
    values = np.arange(10, dtype=float)
    assert block_means(values, 5).tolist() == [2.0, 7.0]
    assert block_means(values, 4).tolist() == [1.5, 5.5]  # episodes 9 and 10 dropped
    assert block_means(values, 20).size == 0


def test_trend_ci_is_not_fake_precise():
    """A +15 pp/1000 drift must be reported in percentage points, with an honest few-pp CI.

    Guards two traps at once: a rolling-mean regression would report a CI of ~0.01 pp here (it
    shrinks with the smoothing, not with the evidence), and computing the slope on 0/1 win flags
    without converting to percentage points would report 0.15 instead of 15.
    """
    episodes = 3000
    drift = 0.00015  # proportion per episode == 15 pp per 1000 episodes; ends at a 95% win rate
    rate = 0.5 + drift * np.arange(episodes)
    rng = np.random.default_rng(11)
    wins = (rng.random(episodes) < rate).astype(int).tolist()
    metrics = compute_metrics(_rows(wins))
    assert metrics["trend_blocks_used"] >= 3
    assert 10.0 <= metrics["trend_slope_pp_per_1000"] <= 20.0
    assert 1.0 < metrics["trend_ci95_pp_per_1000"] < 8.0


def test_convergence_declared_for_a_flat_run():
    """A periodic run whose rate never moves converges, and reports no meaningful trend."""
    metrics = compute_metrics(_rows(_flat_run()))
    assert metrics["convergence_episode"] is not None
    assert metrics["convergence_episode"] <= CONVERGENCE_WINDOW + 10
    assert abs(metrics["performance_improvement"]) < 0.01
    # A perfectly periodic run has no trend and no residual spread; the CI-straddles-zero check
    # belongs to the noisy case, so here both quantities are simply negligible.
    assert abs(metrics["trend_slope_pp_per_1000"]) < 0.1
    assert metrics["trend_ci95_pp_per_1000"] < 0.1


def test_convergence_not_declared_when_still_climbing():
    """A run still improving at its final episode must not be called converged."""
    metrics = compute_metrics(_rows(_climbing_run()))
    assert metrics["convergence_episode"] is None
    assert metrics["trend_slope_pp_per_1000"] > 0
    assert metrics["trend_slope_pp_per_1000"] > metrics["trend_ci95_pp_per_1000"]  # CI excludes 0
    assert metrics["performance_improvement"] > 0.5


def test_metric_identities():
    """Counts, averages and window means agree with a hand computation."""
    rows = _rows(_climbing_run(episodes=800, seed=3))
    metrics = compute_metrics(rows)
    wins = np.array([row["win"] for row in rows], dtype=float)

    assert metrics["episodes"] == 800
    assert metrics["wins"] == int(wins.sum())
    assert metrics["success_rate"] == pytest.approx(wins.mean())
    assert metrics["final_window_success_rate"] == pytest.approx(wins[-200:].mean())
    assert metrics["early_window_success_rate"] == pytest.approx(wins[:100].mean())
    assert metrics["performance_improvement"] == pytest.approx(wins[-100:].mean() - wins[:100].mean())
    assert metrics["average_steps_per_episode"] == pytest.approx(25.0)
    assert metrics["windows"]["win_rate_convergence"] == CONVERGENCE_WINDOW


def test_cli_renders_both_formats_and_writes_metrics(tmp_path):
    """The whole offline path: episodes.jsonl in, PNG + HTML + metrics.json out."""
    run_dir = tmp_path / "synthetic_run"
    run_dir.mkdir()
    rows = _rows(_climbing_run(episodes=600, seed=7))
    (run_dir / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    assert main(["--run", str(run_dir), "--name", "curve_test"]) == 0

    assert (run_dir / "curve_test.png").stat().st_size > 10_000
    assert (run_dir / "curve_test.html").stat().st_size > 100_000  # plotly.js is inlined
    written = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert written["episodes"] == 600
    assert written["convergence_episode"] is None  # this run is still climbing
    assert "trend_slope_pp_per_1000" in written


def test_only_flag_renders_one_panel(tmp_path):
    """``--only success`` is what Phase 5 uses for the per-graph report assets."""
    run_dir = tmp_path / "synthetic_one_panel"
    run_dir.mkdir()
    (run_dir / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in _rows(_flat_run(episodes=500))) + "\n",
        encoding="utf-8",
    )
    assert main(["--run", str(run_dir), "--only", "success", "--name", "only_success"]) == 0
    assert (run_dir / "only_success.png").exists()
    assert not (run_dir / "only_success_reward.png").exists()


def test_comparison_plot_uses_percentage_points(tmp_path):
    """Regression: ``win_rate`` is already in percent -- scaling it again put the axis at 5400 %.

    The symptom was nasty: squashed curves, a parity line sitting on the floor, *and* an invisible
    CI band (the double-scaled rate clipped to 1.0 inside the binomial formula, collapsing the
    half-width to ~0). Caught by the vision QA pass, not by reading the code.
    """
    series, metrics = {}, {}
    for name, rate in (("run_a", 6), ("run_b", 4)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "episodes.jsonl").write_text(
            "\n".join(json.dumps(row) for row in _rows(_flat_run(400, rate=rate))) + "\n",
            encoding="utf-8",
        )
        series[name] = load_frame(run_dir)
        metrics[name] = compute_metrics(load_episodes(run_dir))

    figure = comparison_seaborn(series, metrics)
    win_axis = figure.axes[0]
    # the parity line is deliberately unlabelled in-axes (it appears in the shared figure legend)
    curves = [line for line in win_axis.lines if line.get_label().startswith(("run_a", "run_b"))]
    assert len(curves) == 2
    # the bug guard: a double-scaled rate showed up as 5400 % here
    assert max(float(np.max(line.get_ydata())) for line in curves) <= 100.0
    # steady state: the periodic runs settle at 60 % and 40 % (their first point is a partial window)
    finals = [float(line.get_ydata()[-1]) for line in curves]
    assert finals == pytest.approx([60.0, 40.0], abs=1.5)
    assert win_axis.get_ylim()[1] < 200.0  # not the squashed 0-10000 scale

    parity = [line for line in win_axis.lines if set(np.asarray(line.get_ydata(), dtype=float)) == {EVEN_MATCH_RATE}]
    assert parity, "the 50 % parity line is missing from the success-rate panel"

    assert len(win_axis.collections) == 2, "one CI band per run"


def test_compare_overlays_two_runs(tmp_path):
    """``--compare`` writes a figure plus a metrics table, and rejects a single run."""
    for name, rate in (("run_a", 6), ("run_b", 4)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "episodes.jsonl").write_text(
            "\n".join(json.dumps(row) for row in _rows(_flat_run(600, rate=rate))) + "\n",
            encoding="utf-8",
        )
    argv = [
        "--compare", str(tmp_path / "run_a"), str(tmp_path / "run_b"),
        "--out-dir", str(tmp_path), "--name", "ab",
    ]
    assert main(argv) == 0

    assert (tmp_path / "ab.png").stat().st_size > 10_000
    assert (tmp_path / "ab.html").exists()
    table = json.loads((tmp_path / "ab.json").read_text(encoding="utf-8"))["runs"]
    assert set(table) == {"run_a", "run_b"}
    assert table["run_a"]["success_rate"] == pytest.approx(0.6)
    assert table["run_b"]["success_rate"] == pytest.approx(0.4)

    with pytest.raises(ValueError, match="at least two"):
        main(["--compare", str(tmp_path / "run_a")])
    with pytest.raises(SystemExit, match="--run is required"):
        main([])


def test_moving_average_is_defined_from_the_first_episode():
    """Partial windows at the start: the curve has a value for episode 1 (used for early windows)."""
    smoothed = moving_average(np.array([1.0, 0.0, 1.0, 0.0]), window=2)
    assert smoothed[0] == pytest.approx(1.0)
    assert smoothed[1] == pytest.approx(0.5)
    assert smoothed[3] == pytest.approx(0.5)


@pytest.mark.parametrize("run_name", ["p3_100k"])
def test_real_run_metrics_if_present(run_name):
    """Sanity-check against a real logged run when one is on disk (skipped otherwise)."""
    run_dir = Path(__file__).resolve().parents[1] / "runs" / run_name
    if not (run_dir / "episodes.jsonl").exists():
        pytest.skip(f"{run_name} not on disk")
    metrics = compute_metrics(plots_module.load_episodes(run_dir))
    assert 0.0 <= metrics["success_rate"] <= 1.0
    assert 0.0 <= metrics["final_window_success_rate"] <= 1.0
    assert metrics["wins"] + metrics["losses"] + metrics["unfinished"] == metrics["episodes"]
