"""Curves and metrics from a run's episode log (spec §6).

    python -m lab_ca.plots --run runs/p3_100k

Writes, for the run it is pointed at:

* ``curve_check.png``  -- static, seaborn-rendered: the figure that goes in the report.
* ``curve_check.html`` -- the same three curves as one interactive plotly figure (hover,
  zoom, per-episode inspection) for whoever is reading the numbers.
* ``metrics.json``     -- the metrics summary (spec §6).

The three curves are cumulative reward, success rate and steps per episode, each raw plus a
moving average. ``--only`` renders a single one of them.

Convergence is defined here (spec §9 left the numbers open): the first episode after which the
rolling win rate stays inside ``PLATEAU_TOLERANCE`` of its final value for the rest of the run,
with at least one full window of episodes left to confirm it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: this module only ever writes files

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import seaborn as sns  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

REWARD_WINDOW = 25
STEPS_WINDOW = 25
WIN_WINDOW = 50
FINAL_WINDOW = 200  # episodes averaged for the "final" win rate
COMPARISON_WINDOW = 100  # early vs late window behind the improvement figure
PLATEAU_TOLERANCE = 0.05  # ±5 percentage points counts as "plateaued"
EVEN_MATCH_RATE = 50.0  # mirror match: 50% is parity, so it is the line to beat
GRID_COLOR = "#e6e6e6"


def load_episodes(run_dir: Path) -> List[Dict[str, Any]]:
    """Read ``episodes.jsonl`` into memory (one dict per episode)."""
    path = Path(run_dir) / "episodes.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"no episodes logged in {path}")
    return rows


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean that is defined from the first episode (partial windows at the start)."""
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    cumsum = np.concatenate(([0.0], np.cumsum(values)))
    out = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        out[index] = (cumsum[index + 1] - cumsum[start]) / (index - start + 1)
    return out


def compute_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The report's metrics summary (spec §6)."""
    wins = np.array([float(row["win"]) for row in rows])
    rewards = np.array([float(row["cumulative_reward"]) for row in rows])
    steps = np.array([float(row["steps"]) for row in rows])
    outcomes = [row.get("outcome", "unknown") for row in rows]
    count = len(rows)

    smoothed = moving_average(wins, WIN_WINDOW)
    final_rate = float(smoothed[-1])
    early = wins[:COMPARISON_WINDOW]
    late = wins[-COMPARISON_WINDOW:]

    in_band = np.abs(smoothed - final_rate) <= PLATEAU_TOLERANCE
    convergence: Optional[int] = None
    for index in range(count):
        if in_band[index:].all() and (count - index) >= WIN_WINDOW:
            convergence = index + 1  # 1-indexed episode number
            break

    return {
        "episodes": count,
        "opponent": rows[0].get("opponent", "unknown"),
        "cumulative_reward": float(rewards.sum()),
        "average_reward": float(rewards.mean()),
        "success_rate": float(wins.mean()),
        "final_window_success_rate": float(wins[-FINAL_WINDOW:].mean()),
        "early_window_success_rate": float(early.mean()),
        "late_window_success_rate": float(late.mean()),
        "performance_improvement": float(late.mean() - early.mean()),
        "average_steps_per_episode": float(steps.mean()),
        "wins": int(wins.sum()),
        "losses": int(sum(1 for outcome in outcomes if outcome == "loss")),
        "unfinished": int(sum(1 for outcome in outcomes if outcome == "unfinished")),
        "convergence_episode": convergence,
        "windows": {
            "reward": REWARD_WINDOW,
            "steps": STEPS_WINDOW,
            "win_rate": WIN_WINDOW,
            "final": FINAL_WINDOW,
            "plateau_tolerance": PLATEAU_TOLERANCE,
        },
    }


def load_frame(run_dir: Path) -> pd.DataFrame:
    """``episodes.jsonl`` as a DataFrame, with the smoothed columns the plots draw."""
    frame = pd.DataFrame(load_episodes(run_dir))
    frame["episode"] = np.arange(1, len(frame) + 1)
    frame["reward_smooth"] = moving_average(frame["cumulative_reward"], REWARD_WINDOW)
    frame["win_rate"] = 100.0 * moving_average(frame["win"].astype(float), WIN_WINDOW)
    frame["steps_smooth"] = moving_average(frame["steps"], STEPS_WINDOW)
    return frame


def _panel_specs(frame: pd.DataFrame, metrics: Dict[str, Any], which: str) -> List[Dict[str, Any]]:
    """The requested curves as neutral descriptors, shared by both renderers."""
    reward = {
        "key": "reward",
        "raw": frame["cumulative_reward"],
        "raw_kind": "line",
        "smooth": frame["reward_smooth"],
        "label": "Cumulative reward",
        "window": REWARD_WINDOW,
        "color": "#4C72B0",
        "hline": (0.0, "zero reward"),
        "hover": "%{y:.1f}",
    }
    success = {
        "key": "success",
        # the raw win/loss signal is binary: drawn as faint markers at 0/100 so this panel
        # carries a per-episode trace like the other two instead of looking like an omission
        "raw": 100.0 * frame["win"].astype(float),
        "raw_kind": "markers",
        "smooth": frame["win_rate"],
        "label": "Success rate (%)",
        "window": WIN_WINDOW,
        "color": "#55A868",
        "hline": (EVEN_MATCH_RATE, f"parity with {metrics['opponent']} ({EVEN_MATCH_RATE:.0f}%)"),
        "hover": "%{y:.1f}%",
    }
    steps = {
        "key": "steps",
        "raw": frame["steps"].astype(float),
        "raw_kind": "line",
        "smooth": frame["steps_smooth"],
        "label": "Steps per episode",
        "window": STEPS_WINDOW,
        "color": "#C44E52",
        "hline": None,
        "hover": "%{y:.0f}",
    }
    specs = {"reward": reward, "success": success, "steps": steps}
    return list(specs.values()) if which == "all" else [specs[which]]


def _summary_line(metrics: Dict[str, Any]) -> str:
    convergence = metrics["convergence_episode"]
    return (
        f"{metrics['episodes']} episodes   ·   "
        f"final win rate {100 * metrics['final_window_success_rate']:.1f}%   ·   "
        f"improvement {100 * metrics['performance_improvement']:+.1f} pp   ·   "
        f"convergence {('episode ' + str(convergence)) if convergence else 'not reached'}"
    )


def seaborn_figure(frame: pd.DataFrame, metrics: Dict[str, Any], specs: List[Dict[str, Any]]) -> "plt.Figure":
    """Static figure for the report: seaborn theme, raw + smoothed overlay."""
    sns.set_theme(style="whitegrid", context="talk", palette="deep")
    sns.set_context("talk", rc={"axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 10})

    figure, axes = plt.subplots(
        len(specs), 1, figsize=(12, 3.4 * len(specs) + 1.2), sharex=True, squeeze=False
    )
    convergence = metrics["convergence_episode"]

    for axis, spec in zip(axes[:, 0], specs):
        if spec["raw"] is not None:
            if spec["raw_kind"] == "markers":
                axis.scatter(frame["episode"], spec["raw"], color=spec["color"], alpha=0.24,
                             s=13, linewidths=0, label="per episode (win / loss)")
            else:
                axis.plot(frame["episode"], spec["raw"], color=spec["color"], alpha=0.35,
                          linewidth=0.9, label="per episode")
        axis.plot(frame["episode"], spec["smooth"], color=spec["color"], linewidth=2.6,
                  label=f"moving average (window {spec['window']})")
        if spec["hline"] is not None:
            value, label = spec["hline"]
            axis.axhline(value, color="0.45", linestyle=(0, (4, 3)), linewidth=1.2, label=label)
        if convergence is not None and spec["key"] == "success":
            axis.axvline(convergence, color="#8172B3", linestyle="--", linewidth=1.4,
                         label=f"convergence (episode {convergence})")
        axis.set_ylabel(spec["label"])
        axis.legend(loc="best", framealpha=0.92)
        axis.margins(x=0.01)

    axes[-1, 0].set_xlabel("Episode   (one episode = one battle)")
    figure.suptitle(
        f"DQN vs {metrics['opponent']} — {frame.attrs.get('run', 'run')}\n{_summary_line(metrics)}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def plotly_figure(frame: pd.DataFrame, metrics: Dict[str, Any], specs: List[Dict[str, Any]]) -> go.Figure:
    """Interactive twin of the static figure: zoom, hover, legend toggling."""
    figure = make_subplots(
        rows=len(specs),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[spec["label"] for spec in specs],
    )
    episodes = frame["episode"]

    for row, spec in enumerate(specs, start=1):
        if spec["raw"] is not None:
            if spec["raw_kind"] == "markers":
                figure.add_trace(
                    go.Scatter(
                        x=episodes, y=spec["raw"], mode="markers", name="per episode (win / loss)",
                        marker=dict(color=spec["color"], size=4, opacity=0.3),
                        legendgroup=spec["key"],
                        hovertemplate="episode %{x}<br>" + spec["hover"]
                        + "<extra>per episode</extra>",
                    ),
                    row=row, col=1,
                )
            else:
                figure.add_trace(
                    go.Scatter(
                        x=episodes, y=spec["raw"], mode="lines", name="per episode",
                        line=dict(color=spec["color"], width=1), opacity=0.38,
                        legendgroup=spec["key"],
                        hovertemplate="episode %{x}<br>" + spec["hover"]
                        + "<extra>per episode</extra>",
                    ),
                    row=row, col=1,
                )
        figure.add_trace(
            go.Scatter(
                x=episodes, y=spec["smooth"], mode="lines",
                name=f"moving average ({spec['window']})",
                line=dict(color=spec["color"], width=3), legendgroup=spec["key"],
                hovertemplate="episode %{x}<br>" + spec["hover"] + "<extra>moving average</extra>",
            ),
            row=row, col=1,
        )
        if spec["hline"] is not None:
            value, label = spec["hline"]
            figure.add_hline(
                y=value, line=dict(color="#808080", width=1.4, dash="dash"),
                annotation_text=label, annotation_position="top left", row=row, col=1,
            )
        if metrics["convergence_episode"] is not None and spec["key"] == "success":
            figure.add_vline(
                x=metrics["convergence_episode"],
                line=dict(color="#8172B3", width=1.6, dash="dash"),
                annotation_text=f"convergence (episode {metrics['convergence_episode']})",
                annotation_position="top right", row=row, col=1,
            )

    figure.update_layout(
        template="plotly_white",
        height=int(330 * len(specs) + 130),
        hovermode="x unified",
        title=dict(
            text=(
                f"DQN vs {metrics['opponent']} — {frame.attrs.get('run', 'run')}"
                f"<br><sup>{_summary_line(metrics)}</sup>"
            ),
            x=0.02,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=70, r=30, t=140, b=60),
    )
    figure.update_xaxes(title_text="Episode   (one episode = one battle)", row=len(specs), col=1)
    figure.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    figure.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    return figure


def render(
    run_dir: Path,
    out_dir: Optional[Path] = None,
    stem: str = "curve_check",
    which: str = "all",
    formats: Sequence[str] = ("png", "html"),
) -> Dict[str, Path]:
    """Render the curves for ``run_dir``; returns the paths written, keyed by format."""
    frame = load_frame(run_dir)
    frame.attrs["run"] = run_dir.name
    metrics = compute_metrics(load_episodes(run_dir))
    specs = _panel_specs(frame, metrics, which)
    out_dir = Path(out_dir) if out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    if "png" in formats:
        figure = seaborn_figure(frame, metrics, specs)
        path = out_dir / f"{stem}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        written["png"] = path
    if "html" in formats:
        path = out_dir / f"{stem}.html"
        plotly_figure(frame, metrics, specs).write_html(path, include_plotlyjs=True)
        written["html"] = path
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run directory holding episodes.jsonl")
    parser.add_argument("--out-dir", default=None, help="where to write (default: the run dir)")
    parser.add_argument("--name", default="curve_check", help="output stem")
    parser.add_argument(
        "--only", choices=("all", "reward", "success", "steps"), default="all",
        help="render a single curve instead of the three-panel figure",
    )
    parser.add_argument(
        "--formats", default="png,html", help="comma-separated: png, html"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    written = render(
        run_dir,
        out_dir=out_dir,
        stem=args.name,
        which=args.only,
        formats=[item.strip() for item in args.formats.split(",") if item.strip()],
    )
    metrics = compute_metrics(load_episodes(run_dir))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps({key: value for key, value in metrics.items() if key != "windows"}, indent=2))
    print("\nwrote:", ", ".join(str(path) for path in written.values()), f"{out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
