"""Phase 4 evaluation tests: the statistics, the counting, and one server-backed run.

Most of this file is offline -- the statistics and the bookkeeping are where a bug would silently
misreport a win rate, and they need no server. The one integration test trains a tiny checkpoint
and evaluates it end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lab_ca import evaluate as evaluate_module
from lab_ca.evaluate import (
    DEMO_AGENT_USERNAME,
    DEMO_OPPONENT_USERNAME,
    EvalConfig,
    build_parser,
    choose_actions,
    demo_out_path,
    group_replays,
    replay_result,
    run_battles,
    spectate_url,
    summarize_evals,
    summarize_outcomes,
    wilson_interval,
    write_log_view,
    write_payload,
    write_summary,
)


def test_wilson_interval_matches_known_values() -> None:
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(40.4, abs=0.5)
    assert high == pytest.approx(59.6, abs=0.5)

    # the case that motivated Wilson over the normal approximation: a perfect score must still
    # carry a visibly non-zero interval rather than a fake +/-0.0
    low, high = wilson_interval(100, 100)
    assert low == pytest.approx(96.3, abs=0.5)
    assert high == 100.0
    assert high - low > 3.0

    low, high = wilson_interval(0, 50)
    assert low == 0.0
    assert high == pytest.approx(7.1, abs=0.2)

    assert wilson_interval(0, 0) == (0.0, 100.0)  # nothing measured: no information


def test_summarize_outcomes_excludes_unfinished_battles_from_the_rate() -> None:
    outcomes = [
        {"opponent": "random", "won": True, "outcome": "win", "steps": 20},
        {"opponent": "random", "won": True, "outcome": "win", "steps": 24},
        {"opponent": "random", "won": False, "outcome": "loss", "steps": 30},
        {"opponent": "random", "won": False, "outcome": "unfinished", "steps": 300},
    ]
    summary = summarize_outcomes(outcomes)

    assert summary["battles"] == 4
    assert summary["decided"] == 3
    assert summary["wins"] == 2
    assert summary["losses"] == 1
    assert summary["unfinished"] == 1
    assert summary["win_rate_pct"] == pytest.approx(66.67, abs=0.01)
    low, high = summary["win_rate_ci95_pct"]
    assert low < summary["win_rate_pct"] < high
    # steps are averaged over decided battles only, so the 300-turn stall cannot dominate
    assert summary["steps_mean"] == pytest.approx(24.67, abs=0.01)
    assert summary["steps_max"] == 30


def test_cli_defaults_come_from_the_eval_config() -> None:
    """The parser must take its defaults from ``EvalConfig`` -- two copies of a default drift."""
    args = build_parser().parse_args([])
    default = EvalConfig()
    assert args.agent == default.agent
    assert args.battles == default.battles
    assert args.concurrent == default.concurrent
    assert args.seed == default.seed
    assert args.demo == default.demo
    assert args.demo_battles == default.demo_battles
    assert args.demo_delay == default.demo_delay
    assert tuple(args.opponents.split(",")) == default.opponents
    assert tuple(args.demo_opponents.split(",")) == default.demo_opponents


def test_random_agent_only_ever_picks_legal_actions() -> None:
    rng = np.random.default_rng(0)
    obs = {"action_mask": np.array([0, 1, 1, 0, 0, 0, 1, 0, 0, 0])}
    for _ in range(200):
        (action,) = choose_actions(None, obs, "random", rng)
        assert obs["action_mask"][int(action)] == 1


def test_random_agent_handles_a_batch_of_masks() -> None:
    """With a vec env the mask arrives batched; every row must be respected independently."""
    rng = np.random.default_rng(1)
    masks = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
    actions = choose_actions(None, {"action_mask": masks}, "random", rng)
    assert actions.tolist() == [0, 2, 1, int(actions[3])]
    assert all(masks[index][int(action)] for index, action in enumerate(actions))


def test_write_payload_merges_so_the_demo_cannot_clobber_the_metrics(tmp_path: Path) -> None:
    """Pass 1 writes metrics, Pass 2 appends replays to the same file: both must survive."""
    path = tmp_path / "eval.json"
    write_payload(path, {"results": {"heuristic": {"win_rate_pct": 61.0}}})
    write_payload(path, {"demo_replays": ["runs/x/demo/a.html"]})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["results"]["heuristic"]["win_rate_pct"] == 61.0
    assert payload["demo_replays"] == ["runs/x/demo/a.html"]
    assert "updated" in payload


def test_spectate_url_maps_a_battle_tag_to_a_room_url() -> None:
    assert spectate_url("battle-gen9customgame-12345") == "http://localhost:8000/gen9customgame-12345"
    assert spectate_url(">battle-gen9customgame-9") == "http://localhost:8000/gen9customgame-9"


class _FakeVecEnv:
    """A vec env that finishes ``per_step`` battles per call, to test the counting offline."""

    def __init__(self, per_step: int) -> None:
        self.per_step = per_step
        self.steps = 0

    def reset(self):
        self.steps = 0
        return {"action_mask": np.ones((self.per_step, 4), dtype=bool)}

    def step(self, actions):
        self.steps += 1
        infos = [
            {"episode": {"r": 1.0, "l": 25}, "win": True, "outcome": "win"}
            for _ in range(self.per_step)
        ]
        obs = {"action_mask": np.ones((self.per_step, 4), dtype=bool)}
        return obs, np.zeros(self.per_step), np.array([True] * self.per_step), infos


def test_run_battles_records_exactly_the_requested_count_when_battles_finish_in_parallel() -> None:
    """Eight parallel envs finish up to eight battles per step; the report must still say N."""
    env = _FakeVecEnv(per_step=3)
    outcomes = run_battles(env, agent="random", opponent="random", battles=4)
    assert len(outcomes) == 4
    assert all(outcome["won"] for outcome in outcomes)
    assert all(outcome["steps"] == 25 for outcome in outcomes)


def test_run_battles_stops_at_the_turn_cap_instead_of_hanging() -> None:
    """A battle that never ends must not hang the evaluation: it returns what it has."""

    class _StuckEnv(_FakeVecEnv):
        def step(self, actions):
            self.steps += 1
            obs = {"action_mask": np.ones((1, 4), dtype=bool)}
            return obs, np.zeros(1), np.array([False]), [{}]

    env = _StuckEnv(per_step=1)
    outcomes = run_battles(env, agent="random", opponent="random", battles=2)
    assert outcomes == []
    assert env.steps == evaluate_module.MAX_TURNS_PER_BATTLE * 2


def _write_eval(path: Path, run: str, rate: float, opponent: str = "heuristic") -> Path:
    payload = {
        "source_run": run,
        "agent": "dqn",
        "results": {
            opponent: {
                "battles": 20, "decided": 20, "wins": int(rate / 5), "losses": 20 - int(rate / 5),
                "unfinished": 0, "win_rate_pct": rate, "win_rate_ci95_pct": [rate - 5.0, rate + 5.0],
                "steps_mean": 25.0,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_summarize_evals_and_summary_artifacts(tmp_path: Path) -> None:
    paths = [
        _write_eval(tmp_path / "a.json", "p3_400k", 81.0),
        _write_eval(tmp_path / "b.json", "p3b_100k", 73.5),
    ]
    rows = summarize_evals(paths)
    assert [row["run"] for row in rows] == ["p3_400k", "p3b_100k"]
    assert rows[0]["win_rate_pct"] == 81.0
    assert rows[0]["battles"] == 20

    written = write_summary(rows, tmp_path / "phase4_eval")
    assert set(written) == {"markdown", "png", "html"}
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0
    table = written["markdown"].read_text(encoding="utf-8")
    assert "p3_400k" in table and "81.0%" in table and "76.0-86.0" in table
    assert table.count("\n|") == 4  # header, separator, two rows


def test_demo_writes_into_eval_json_and_merges_instead_of_clobbering(tmp_path: Path) -> None:
    """The plan puts ``demo_replays`` in ``eval.json``; merging is what makes sharing it safe."""
    checkpoint = tmp_path / "runs" / "p3_400k" / "model_final.zip"
    demo_dir = checkpoint.parent / "demo"
    config = EvalConfig(checkpoint=str(checkpoint), demo=True)
    target = checkpoint.parent / "eval.json"
    assert demo_out_path(config, demo_dir) == target

    # a pass-1 evaluation is already sitting there
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"results": {"heuristic": {"win_rate_pct": 83.0}}}), encoding="utf-8"
    )
    write_payload(target, {"demo_replays": ["a.html"]})
    merged = json.loads(target.read_text(encoding="utf-8"))
    assert merged["results"]["heuristic"]["win_rate_pct"] == 83.0
    assert merged["demo_replays"] == ["a.html"]

    explicit = EvalConfig(checkpoint=str(checkpoint), demo=True, out=str(tmp_path / "custom.json"))
    assert demo_out_path(explicit, demo_dir) == tmp_path / "custom.json"


def test_replays_group_by_battle_tag_and_report_their_own_winner(tmp_path: Path) -> None:
    """Both sides write a file, and the result is read from the file rather than from write order."""
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    win_line = "|win|" + DEMO_AGENT_USERNAME
    (demo_dir / f"{DEMO_AGENT_USERNAME} - battle-gen9customgame-1.html").write_text(
        f"<html>{win_line}</html>", encoding="utf-8"
    )
    (demo_dir / f"{DEMO_OPPONENT_USERNAME} - battle-gen9customgame-1.html").write_text(
        f"<html>{win_line}</html>", encoding="utf-8"
    )
    (demo_dir / f"{DEMO_AGENT_USERNAME} - battle-gen9customgame-2.html").write_text(
        f"<html>|forfeit||win|{DEMO_OPPONENT_USERNAME}</html>", encoding="utf-8"
    )

    grouped = group_replays(sorted(demo_dir.glob("*.html")))
    assert sorted(grouped) == ["battle-gen9customgame-1", "battle-gen9customgame-2"]
    assert len(grouped["battle-gen9customgame-1"]) == 2  # agent copy + opponent copy

    agent_copy = demo_dir / f"{DEMO_AGENT_USERNAME} - battle-gen9customgame-1.html"
    assert replay_result(agent_copy) == {"winner": DEMO_AGENT_USERNAME, "forfeit": False}
    stray = demo_dir / f"{DEMO_AGENT_USERNAME} - battle-gen9customgame-2.html"
    assert replay_result(stray) == {"winner": DEMO_OPPONENT_USERNAME, "forfeit": True}


def test_log_view_extracts_the_embedded_protocol_log(tmp_path: Path) -> None:
    """The replay renders via the public client's JS, so the transcript is the offline view."""
    replay = tmp_path / f"{DEMO_AGENT_USERNAME} - battle-gen9customgame-3.html"
    replay.write_text(
        '<script type="text/plain" class="battle-log-data">'
        "|player|p1|lab_ca_dqn|\n|turn|1\n|move|p1a: Gholdengo|Shadow Ball|<b>not markup</b>\n"
        "</script><script src='https://play.pokemonshowdown.com/js/replay-embed.js'></script>",
        encoding="utf-8",
    )
    out = write_log_view(replay, tmp_path / "battle-3-transcript.html")
    assert out is not None and out.exists()
    text = out.read_text(encoding="utf-8")
    assert "|turn|1" in text and "|move|p1a: Gholdengo|Shadow Ball|" in text
    # protocol lines are shown literally, not interpreted as markup
    assert "<b>not markup</b>" not in text and "&lt;b&gt;not markup&lt;/b&gt;" in text
    assert "replay-embed.js" not in text  # no dependency on the public client

    # a file without the log block is reported rather than written as an empty page
    plain = tmp_path / "plain.html"
    plain.write_text("<html>no log here</html>", encoding="utf-8")
    assert write_log_view(plain, tmp_path / "nope.html") is None
    assert not (tmp_path / "nope.html").exists()


def test_summary_figure_layout_keeps_the_legend_off_the_bars() -> None:
    """Measure the layout rather than trusting it: a legend covering a bar is a real defect."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rows = [
        {"run": "eval_random_floor", "agent": "random", "opponent": "heuristic", "battles": 200,
         "win_rate_pct": 0.0, "ci_low": 0.0, "ci_high": 1.9, "steps_mean": 26.6},
        {"run": "eval_random_floor", "agent": "random", "opponent": "random", "battles": 200,
         "win_rate_pct": 53.0, "ci_low": 46.1, "ci_high": 59.8, "steps_mean": 75.7},
        {"run": "p3_400k", "agent": "dqn", "opponent": "heuristic", "battles": 200,
         "win_rate_pct": 83.0, "ci_low": 77.2, "ci_high": 87.6, "steps_mean": 25.8},
    ]
    figure, axis, _frame = evaluate_module.eval_summary_figure(rows)
    try:
        assert axis.get_xlabel() == "run"
        assert axis.get_ylabel() == "Win rate (%)"
        assert "95% Wilson" in axis.get_title() and "50% parity" in axis.get_title()
        # short tick labels: the opponent is the hue, not part of every label
        labels = [text.get_text() for text in axis.get_xticklabels()]
        assert labels == ["eval_random_floor", "p3_400k"], labels
        # every bar carries its value, so a near-zero bar cannot read as a missing one
        values = [text.get_text() for text in axis.texts if text.get_text().endswith("%")]
        assert sorted(values) == ["0.0%", "53.0%", "83.0%"], values

        figure.canvas.draw()
        legend_box = axis.get_legend().get_window_extent()
        for patch in axis.patches:
            assert not legend_box.overlaps(patch.get_window_extent()), "legend covers a bar"
        parity = [text for text in axis.texts if text.get_text().startswith("parity")]
        assert parity, "the parity line needs a label"
        assert not legend_box.overlaps(parity[0].get_window_extent())
    finally:
        plt.close(figure)


@pytest.mark.integration
def test_evaluation_reports_a_win_rate_for_a_trained_checkpoint(server: None, tmp_path: Path) -> None:
    """End to end: train a tiny checkpoint, evaluate it frozen, read the artifact back."""
    from lab_ca.train import TrainingConfig, train

    run_dir = train(
        TrainingConfig(
            timesteps=300, opponent="random", seed=0, run_name="eval_src", checkpoint_every_steps=300
        ),
        runs_root=tmp_path,
    )
    out_path = evaluate_module.evaluate_checkpoint(
        EvalConfig(
            checkpoint=str(run_dir / "model_final.zip"),
            opponents=("random",),
            battles=4,
            concurrent=1,
            out=str(tmp_path / "eval.json"),
        )
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    result = payload["results"]["random"]

    assert result["decided"] + result["unfinished"] == 4
    assert 0.0 <= result["win_rate_pct"] <= 100.0
    assert len(result["battles_detail"]) == 4
    assert payload["checkpoint_steps"] is None  # model_final carries no step count
    assert payload["source_run"] == "eval_src"  # provenance read from the neighbouring run_config
    assert payload["config"]["concurrent"] == 1
