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
import math
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
WIN_WINDOW = 50  # window of the success-rate curve drawn in the figure
# Convergence is judged on a *separate, longer* window: at 50 episodes the binomial sigma of a
# 60% win rate is ~6.9 pp, so a fixed ±5 pp band sits inside the noise and a converged agent would
# fail the test routinely. At 200 episodes sigma is ~3.5 pp, and the band below is set from sigma.
CONVERGENCE_WINDOW = 200
CONVERGENCE_SIGMA_MULTIPLE = 2.0  # band = 2 sigma of the windowed rate: "within noise of the end"
CONVERGENCE_TOLERANCE_FLOOR = 0.05  # never demand a band tighter than ±5 pp
TREND_WINDOW = 2000  # episodes behind the "is it still climbing?" slope
TREND_BLOCK = CONVERGENCE_WINDOW  # non-overlapping blocks used for that fit (like-for-like)
FINAL_WINDOW = 200  # episodes averaged for the "final" win rate
COMPARISON_WINDOW = 100  # early vs late window behind the improvement figure
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


def block_means(values: np.ndarray, block: int) -> np.ndarray:
    """Non-overlapping block means (the last, possibly short, block is dropped).

    Used for trend fitting: consecutive points of a rolling mean are heavily autocorrelated, so
    regressing on them produces a meaningless confidence interval (it shrinks with the smoothing,
    not with the evidence). Independent blocks give an honest one.
    """
    values = np.asarray(values, dtype=float)
    blocks = len(values) // block
    if blocks == 0:
        return np.array([])
    return values[: blocks * block].reshape(blocks, block).mean(axis=1)


def binomial_sigma(rate: float, window: int) -> float:
    """Standard error of a win rate averaged over ``window`` episodes (as a proportion)."""
    rate = min(max(rate, 0.0), 1.0)
    return math.sqrt(rate * (1.0 - rate) / window)


def compute_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The report's metrics summary (spec §6).

    Convergence: the first episode after which the ``CONVERGENCE_WINDOW``-episode rolling win rate
    never again leaves the band around its final value, with a full window left to confirm it.
    The band is ``CONVERGENCE_SIGMA_MULTIPLE`` sigma of that window's binomial noise (floor ±5 pp),
    so "converged" means "stopped moving by more than sampling noise" rather than "stopped moving
    by more than an arbitrary constant".

    Because a plateau is a *negative* claim, the run is also summarised by the slope of its win
    rate over the last ``TREND_WINDOW`` episodes, fitted on independent ``TREND_BLOCK``-episode
    blocks -- so "plateaued" and "still climbing, just slowly" can be told apart instead of both
    reading as "not converged".
    """
    wins = np.array([float(row["win"]) for row in rows])
    rewards = np.array([float(row["cumulative_reward"]) for row in rows])
    steps = np.array([float(row["steps"]) for row in rows])
    outcomes = [row.get("outcome", "unknown") for row in rows]
    count = len(rows)

    plateau = moving_average(wins, CONVERGENCE_WINDOW)
    final_rate = float(plateau[-1])
    tolerance = max(
        CONVERGENCE_TOLERANCE_FLOOR, CONVERGENCE_SIGMA_MULTIPLE * binomial_sigma(final_rate, CONVERGENCE_WINDOW)
    )
    early = wins[:COMPARISON_WINDOW]
    late = wins[-COMPARISON_WINDOW:]

    in_band = np.abs(plateau - final_rate) <= tolerance
    convergence: Optional[int] = None
    for index in range(count):
        if in_band[index:].all() and (count - index) >= CONVERGENCE_WINDOW:
            convergence = index + 1  # 1-indexed episode number
            break

    blocks = block_means(wins, TREND_BLOCK)
    used_blocks = min(len(blocks), max(3, TREND_WINDOW // TREND_BLOCK))
    trend_slope = trend_ci = float("nan")
    if len(blocks) >= 3:
        tail = blocks[-used_blocks:]
        x = np.arange(len(tail), dtype=float)
        slope, intercept = np.polyfit(x, tail, 1)
        residual = tail - (slope * x + intercept)
        stderr = residual.std(ddof=2) / (x.std() * math.sqrt(len(tail)))
        per_1000 = 1000.0 / TREND_BLOCK * 100.0  # blocks -> percentage points per 1000 episodes
        trend_slope = float(slope * per_1000)
        trend_ci = float(1.96 * stderr * per_1000)

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
        "convergence_band_pp": float(tolerance * 100),
        "trend_slope_pp_per_1000": trend_slope,
        "trend_ci95_pp_per_1000": trend_ci,
        "trend_blocks_used": int(used_blocks) if len(blocks) >= 3 else 0,
        "windows": {
            "reward": REWARD_WINDOW,
            "steps": STEPS_WINDOW,
            "win_rate_plotted": WIN_WINDOW,
            "win_rate_convergence": CONVERGENCE_WINDOW,
            "convergence_sigma_multiple": CONVERGENCE_SIGMA_MULTIPLE,
            "trend": TREND_WINDOW,
            "trend_block": TREND_BLOCK,
            "final": FINAL_WINDOW,
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
        "label": "Win rate (%)",
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
    slope = metrics.get("trend_slope_pp_per_1000", float("nan"))
    slope_ci = metrics.get("trend_ci95_pp_per_1000", float("nan"))
    trend = f"trend {slope:+.1f} ± {slope_ci:.1f} pp/1k episodes" if slope == slope else "trend n/a"
    return (
        f"{metrics['episodes']} episodes   ·   "
        f"final win rate {100 * metrics['final_window_success_rate']:.1f}%   ·   "
        f"improvement {100 * metrics['performance_improvement']:+.1f} pp   ·   "
        f"convergence {('episode ' + str(convergence)) if convergence else 'not reached'}   ·   "
        f"{trend}"
    )


def _binomial_half_width(rate_pct: np.ndarray, window: int) -> np.ndarray:
    """±95% binomial half-width, in percentage points, of a rate averaged over ``window`` episodes.

    This is the noise floor of the plotted curve: two runs whose curves stay inside each other's
    bands are not distinguishable at this episode count, no matter how different they look.
    """
    proportion = np.clip(np.asarray(rate_pct, dtype=float) / 100.0, 0.0, 1.0)
    return 1.96 * np.sqrt(proportion * (1.0 - proportion) / window) * 100.0


def _rgba(hex_color: str, alpha: float) -> str:
    """``#4c72b0`` + 0.18 -> ``rgba(76,114,176,0.18)`` (plotly needs the string form)."""
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


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
                axis.scatter(frame["episode"], spec["raw"], color=spec["color"], alpha=0.16,
                             s=10, linewidths=0, label="per episode (win / loss)")
            else:
                axis.plot(frame["episode"], spec["raw"], color=spec["color"], alpha=0.35,
                          linewidth=0.9, label="per episode")
        axis.plot(frame["episode"], spec["smooth"], color=spec["color"], linewidth=2.6,
                  label=f"moving average (window {spec['window']})")
        if spec["key"] == "success":
            half = _binomial_half_width(spec["smooth"], spec["window"])
            axis.fill_between(
                frame["episode"], spec["smooth"] - half, spec["smooth"] + half,
                color=spec["color"], alpha=0.26, linewidth=0, label="±95% CI (binomial)",
            )
        if spec["hline"] is not None:
            value, label = spec["hline"]
            axis.axhline(value, color="0.45", linestyle=(0, (4, 3)), linewidth=1.2, label=label)
        if convergence is not None and spec["key"] == "success":
            axis.axvline(convergence, color="#8172B3", linestyle="--", linewidth=1.4,
                         label=f"convergence (episode {convergence})")
        axis.set_ylabel(spec["label"])
        # outside the axes, not `loc="best"`: "best" steers clear of lines and patches but not of
        # scatter collections, and a 14k-episode run fills every inside corner -- measured in the
        # report-asset audit, where an inside legend overlapped the data on all three panels
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
        axis.margins(x=0.01)

    axes[-1, 0].set_xlabel("Episode   (one battle ≈ 25 env steps)")
    figure.suptitle(
        f"DQN vs {metrics['opponent']} — {frame.attrs.get('run', 'run')}\n{_summary_line(metrics)}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 0.84, 0.95))  # right margin is the legends' column
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
        if spec["key"] == "success":
            half = _binomial_half_width(spec["smooth"], spec["window"])
            lower = np.asarray(spec["smooth"], dtype=float) - half
            upper = np.asarray(spec["smooth"], dtype=float) + half
            figure.add_trace(
                go.Scatter(x=episodes, y=lower, mode="lines", line=dict(width=0),
                           hoverinfo="skip", showlegend=False, legendgroup=spec["key"]),
                row=row, col=1,
            )
            figure.add_trace(
                go.Scatter(x=episodes, y=upper, mode="lines", line=dict(width=0),
                           fill="tonexty", fillcolor=_rgba(spec["color"], 0.18),
                           name="±95% CI (binomial)", hoverinfo="skip", legendgroup=spec["key"]),
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
    figure.update_xaxes(title_text="Episode   (one battle ≈ 25 env steps)", row=len(specs), col=1)
    figure.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    figure.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    return figure


COMPARISON_COLORS = ("#4C72B0", "#C44E52", "#55A868", "#8172B3", "#CCB974", "#64B5CD")


def comparison_series(run_dirs: Sequence[Path]) -> "Dict[str, pd.DataFrame]":
    """``{run_name: frame}`` for the overlay figure, preserving the order given."""
    series: Dict[str, pd.DataFrame] = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        frame = load_frame(run_dir)
        frame.attrs["run"] = run_dir.name
        series[run_dir.name] = frame
    return series


def _comparison_legend(name: str, metrics: Dict[str, Any]) -> str:
    return (
        f"{name} — final {100 * metrics['final_window_success_rate']:.1f}% "
        f"({metrics['trend_slope_pp_per_1000']:+.1f} ± {metrics['trend_ci95_pp_per_1000']:.1f} pp/1k)"
    )


def comparison_seaborn(series: "Dict[str, pd.DataFrame]", metrics: Dict[str, Dict[str, Any]]) -> "plt.Figure":
    """Win rate (with binomial CI band) and reward side by side for several runs."""
    sns.set_theme(style="whitegrid", context="talk", palette="deep")
    sns.set_context("talk", rc={"axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 9})

    figure, axes = plt.subplots(2, 1, figsize=(12, 9.2), sharex=True)
    rate_handles: list = []
    rate_labels: list = []
    for index, (name, frame) in enumerate(series.items()):
        color = COMPARISON_COLORS[index % len(COMPARISON_COLORS)]
        rate = frame["win_rate"]  # already percentage points (load_frame scales it once)
        half = _binomial_half_width(rate, WIN_WINDOW)
        axes[0].fill_between(
            frame["episode"], rate - half, rate + half, color=color, alpha=0.20, linewidth=0,
        )
        rate_handles.append(
            axes[0].plot(frame["episode"], rate, color=color, linewidth=2.6,
                         label=_comparison_legend(name, metrics[name]))[0]
        )
        rate_labels.append(_comparison_legend(name, metrics[name]))
        axes[1].plot(frame["episode"], frame["reward_smooth"], color=color, linewidth=2.0, label=name)

    parity = axes[0].axhline(EVEN_MATCH_RATE, color="0.45", linestyle=(0, (4, 3)), linewidth=1.3)
    axes[0].set_ylabel("Win rate (%)")
    zero = axes[1].axhline(0.0, color="0.45", linestyle=(0, (1, 3)), linewidth=1.3)
    axes[1].set_ylabel(f"Cumulative reward\n(moving average, window {REWARD_WINDOW})")
    axes[1].set_xlabel("Episode   (one battle ≈ 25 env steps)")
    for axis in axes:
        axis.margins(x=0.01)
    # one shared legend *outside* the axes: the curves cross every corner of these panels, so any
    # in-axes legend ends up on top of data (both colors are shared by the two panels)
    figure.legend(
        [*rate_handles, parity, zero],
        [*rate_labels, f"parity with the opponent ({EVEN_MATCH_RATE:.0f}%)",
         "zero reward (no better than chance)"],
        loc="upper center", bbox_to_anchor=(0.5, 0.885), ncol=3, framealpha=0.95, fontsize=10.5,
    )
    figure.suptitle(
        "Run comparison — win rate (with ±95% binomial CI) and reward\n"
        "overlapping bands mean the difference is not resolvable at this episode count",
        fontsize=13, y=0.98,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.855))
    return figure


def comparison_plotly(series: "Dict[str, pd.DataFrame]", metrics: Dict[str, Dict[str, Any]]) -> "go.Figure":
    """Interactive twin of :func:`comparison_seaborn`."""
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Win rate (%)", f"Cumulative reward (moving average, window {REWARD_WINDOW})"),
    )
    for index, (name, frame) in enumerate(series.items()):
        color = COMPARISON_COLORS[index % len(COMPARISON_COLORS)]
        rate = frame["win_rate"]  # already percentage points
        half = _binomial_half_width(rate, WIN_WINDOW)
        legend = _comparison_legend(name, metrics[name])
        figure.add_trace(
            go.Scatter(x=frame["episode"], y=rate - half, mode="lines", line=dict(width=0),
                       hoverinfo="skip", showlegend=False, legendgroup=name),
            row=1, col=1,
        )
        figure.add_trace(
            go.Scatter(x=frame["episode"], y=rate + half, mode="lines", line=dict(width=0),
                       fill="tonexty", fillcolor=_rgba(color, 0.12), hoverinfo="skip",
                       showlegend=False, legendgroup=name),
            row=1, col=1,
        )
        figure.add_trace(
            go.Scatter(x=frame["episode"], y=rate, mode="lines", name=legend,
                       line=dict(color=color, width=3), legendgroup=name,
                       hovertemplate="episode %{x}<br>%{y:.1f}%<extra>" + legend + "</extra>"),
            row=1, col=1,
        )
        figure.add_trace(
            go.Scatter(x=frame["episode"], y=frame["reward_smooth"], mode="lines", name=name,
                       line=dict(color=color, width=2), legendgroup=name, showlegend=False,
                       hovertemplate="episode %{x}<br>reward %{y:.1f}<extra>" + name + "</extra>"),
            row=2, col=1,
        )
    figure.add_hline(y=EVEN_MATCH_RATE, line=dict(color="#808080", width=1.4, dash="dash"),
                     annotation_text=f"parity ({EVEN_MATCH_RATE:.0f}%)", row=1, col=1)
    figure.update_layout(
        template="plotly_white", height=760, hovermode="x unified",
        title=dict(text="Run comparison — overlapping bands mean the difference is not resolvable", x=0.02),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
        margin=dict(l=80, r=30, t=150, b=60),
    )
    figure.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    figure.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    figure.update_xaxes(title_text="Episode   (one battle ≈ 25 env steps)", row=2, col=1)
    return figure


def compare(
    run_dirs: Sequence[Path],
    out_dir: Optional[Path] = None,
    stem: str = "comparison",
    formats: Sequence[str] = ("png", "html"),
) -> Dict[str, Path]:
    """Overlay several runs; writes the figure(s) plus ``<stem>.json`` with each run's metrics."""
    run_dirs = [Path(run_dir) for run_dir in run_dirs]
    if len(run_dirs) < 2:
        raise ValueError("compare needs at least two runs")
    series = comparison_series(run_dirs)
    metrics = {name: compute_metrics(load_episodes(Path(run_dir))) for name, run_dir in zip(series, run_dirs)}
    out_dir = Path(out_dir) if out_dir else run_dirs[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}
    written["json"] = out_dir / f"{stem}.json"
    written["json"].write_text(
        json.dumps({"runs": metrics, "windows": {"win_rate_plotted": WIN_WINDOW, "reward": REWARD_WINDOW}},
                   indent=2),
        encoding="utf-8",
    )
    if "png" in formats:
        figure = comparison_seaborn(series, metrics)
        written["png"] = out_dir / f"{stem}.png"
        figure.savefig(written["png"], dpi=150, bbox_inches="tight")
        plt.close(figure)
    if "html" in formats:
        written["html"] = out_dir / f"{stem}.html"
        comparison_plotly(series, metrics).write_html(written["html"], include_plotlyjs=True)
    return written


AGGREGATE_WINDOW = 50  # episodes per aggregated point (non-overlapping: independent blocks)


def _step_budget_label(timesteps: Any) -> str:
    """The step budget, as it appears in a config signature.

    A 100 k-step and a 200 k-step run are *not* repeat seeds of one experiment, so the budget has
    to be part of the signature or the aggregator averages across two different questions. Runs
    that never recorded it say so rather than being folded into the default budget.
    """
    if not isinstance(timesteps, (int, float)) or isinstance(timesteps, bool) or timesteps <= 0:
        return "steps unknown"
    if timesteps % 1000 == 0:
        return f"{timesteps / 1000:g}k steps"
    return f"{timesteps:g} steps"


def config_signature(run_dir: Path) -> str:
    """Label a run by the training config that produced it (read from ``run_config.json``).

    Two runs with the same signature are repeat seeds of the same experiment, which is what makes
    "is this config better, or was that a lucky seed" answerable.

    Missing keys fall back to the *documented* defaults (``TrainingConfig`` / SB3), because runs
    from before the override plumbing existed recorded fewer keys -- reconstructing them is
    truthful where editing the artifact would not be.
    """
    defaults = {
        "train_freq": 4, "gradient_steps": 1, "batch_size": 32,
        "target_update_interval": 10_000, "net_arch": [64, 64],
    }
    path = Path(run_dir) / "run_config.json"
    if not path.exists():
        return "unlabelled"
    recorded = json.loads(path.read_text(encoding="utf-8")).get("config", {})
    config = {**defaults, **{key: value for key, value in recorded.items() if value is not None}}
    updates = config["gradient_steps"] / max(1, config["train_freq"])
    arch = config["net_arch"]
    arch_text = ",".join(str(width) for width in arch) if isinstance(arch, list) else str(arch)
    return (
        f"{updates:g} upd/step · arch [{arch_text}] · batch {config['batch_size']} "
        f"· target {config['target_update_interval']} · {_step_budget_label(config.get('timesteps'))}"
    )


def aggregate_by_config(run_dirs: Sequence[Path]) -> "Dict[str, List[Path]]":
    """``{config signature: [run dirs]}``, ordered by signature for a stable legend."""
    groups: Dict[str, List[Path]] = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        groups.setdefault(config_signature(run_dir), []).append(run_dir)
    return dict(sorted(groups.items()))


def _block_frame(frame: "pd.DataFrame", window: int) -> "pd.DataFrame":
    """Collapse one run to non-overlapping ``window``-episode block means.

    Block means (rather than a rolling mean sampled every k episodes) keep the points independent,
    so the shaded spread across seeds is a real spread rather than a smoothed artefact.
    """
    blocks = len(frame) // window
    if blocks == 0:
        return frame.iloc[0:0]
    tail = frame.iloc[: blocks * window].copy()
    tail["block"] = np.arange(len(tail)) // window
    grouped = tail.groupby("block").agg(
        episode=("episode", "last"),
        win_rate=("win_rate", "mean"),  # already percentage points; equal to the block win rate
        reward=("cumulative_reward", "mean"),
        steps=("steps", "mean"),
    )
    grouped["epochs"] = window
    return grouped


def aggregate_table(run_dirs: Sequence[Path]) -> Dict[str, Any]:
    """Per-config summary: every seed's final window, plus the mean/spread across seeds."""
    table: Dict[str, Any] = {}
    for label, group in aggregate_by_config(run_dirs).items():
        metrics = [compute_metrics(load_episodes(run_dir)) for run_dir in group]
        finals = [100 * item["final_window_success_rate"] for item in metrics]
        trends = [item["trend_slope_pp_per_1000"] for item in metrics]
        table[label] = {
            "runs": [run_dir.name for run_dir in group],
            "seeds": [
                json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["config"].get("seed")
                if (run_dir / "run_config.json").exists() else None
                for run_dir in group
            ],
            "seeds_per_config": len(group),
            "episodes": [item["episodes"] for item in metrics],
            "final_window_success_rate_pct": {
                "per_seed": finals,
                "mean": float(np.mean(finals)),
                "min": float(np.min(finals)),
                "max": float(np.max(finals)),
                "spread": float(np.max(finals) - np.min(finals)),
            },
            "late_trend_pp_per_1000": {
                "per_seed": trends,
                "mean": float(np.mean(trends)),
                "min": float(np.min(trends)),
                "max": float(np.max(trends)),
            },
        }
    return table


def aggregate_series(run_dirs: Sequence[Path], window: int = AGGREGATE_WINDOW) -> Dict[str, Dict[str, Any]]:
    """Per config: the mean block-mean curve across seeds, with the min/max envelope."""
    series: Dict[str, Dict[str, Any]] = {}
    for label, group in aggregate_by_config(run_dirs).items():
        blocks = [_block_frame(load_frame(run_dir), window) for run_dir in group]
        blocks = [item for item in blocks if len(item)]
        if not blocks:
            continue
        common = min(len(item) for item in blocks)
        stacked = np.stack([item.iloc[:common]["win_rate"].to_numpy(dtype=float) for item in blocks])
        rewards = np.stack([item.iloc[:common]["reward"].to_numpy(dtype=float) for item in blocks])
        series[label] = {
            "episodes": blocks[0].iloc[:common]["episode"].to_numpy(dtype=float),
            "win_rate_mean": stacked.mean(axis=0),
            "win_rate_min": stacked.min(axis=0),
            "win_rate_max": stacked.max(axis=0),
            "reward_mean": rewards.mean(axis=0),
            "reward_min": rewards.min(axis=0),
            "reward_max": rewards.max(axis=0),
            "seeds": len(blocks),
        }
    return series


def _signature_parts(signature: str) -> "tuple[str, str]":
    """Split a config signature into (training knobs, step budget) so a legend can show both."""
    parts = [piece.strip() for piece in signature.split("·")]
    if len(parts) > 1 and "steps" in parts[-1]:
        return " · ".join(parts[:-1]), parts[-1]
    return signature, ""


def _shared_knobs(series: Dict[str, Dict[str, Any]]) -> str:
    """The knob signature shared by every group, or ``""`` if they differ.

    When all groups run the same hyperparameters (e.g. one config at two step budgets), repeating
    them in every legend entry wastes the densest space in the figure; the title carries them once.
    """
    knobs = {_signature_parts(label)[0] for label in series}
    return knobs.pop() if len(knobs) == 1 else ""


def _aggregate_legend(label: str, entry: Dict[str, Any], table: Dict[str, Any],
                      show_knobs: bool = True) -> str:
    """Short lines: the knobs (unless shared), how many runs back the curve, the headline number.

    A single-run group has no envelope, and saying so in the legend is the difference between a
    reader trusting the line and a reader mistaking an unshaded curve for a plotting bug.
    """
    stats = table.get(label, {}).get("final_window_success_rate_pct", {})
    if not stats:
        return label
    seeds = len(stats["per_seed"])
    knobs, budget = _signature_parts(label)
    head = f"{budget} — " if budget else ""
    if seeds == 1:
        scope = f"{head}n=1 (single run, no envelope)"
        tail = f"{stats['mean']:.1f}% final {FINAL_WINDOW}"
    else:
        scope = f"{head}n={seeds} seeds"
        tail = (f"{stats['mean']:.1f}% mean final {FINAL_WINDOW} "
                f"(min–max {stats['min']:.1f}–{stats['max']:.1f})")
    lines = ([knobs] if show_knobs else []) + [scope, tail]
    return "\n".join(lines)


X_END_RATIO = 0.8  # a "curves end here" mark needs a real gap, not a few percent of run-to-run spread
EPISODE_SPREAD_RATIO = 1.08  # ends differ by >8%: explain it, even without a "gap" mark


def _curve_end_note(series: Dict[str, Dict[str, Any]]) -> str:
    """Footnote for a figure whose groups ran to genuinely different episode counts."""
    notes = []
    ends: Dict[str, float] = {}
    for label, entry in series.items():
        episodes = entry.get("episodes")
        if episodes is not None and len(episodes):
            ends[label] = float(episodes[-1])
    if ends and min(ends.values()) <= X_END_RATIO * max(ends.values()):
        shortest = min(ends, key=ends.get)
        budget = _signature_parts(shortest)[1] or shortest
        notes.append(f"the {budget} curves end at ≈{ends[shortest]:,.0f} battles (that config ran out "
                     f"of episodes); beyond that point the figure shows the longer run only")
    elif len(ends) > 1 and max(ends.values()) > EPISODE_SPREAD_RATIO * min(ends.values()):
        # the same step budget is a different number of episodes per config, because a config that
        # takes longer per battle fits fewer battles into it. Curves therefore stop at different x
        # without anything being truncated -- say so, since the difference is plainly visible
        notes.append(
            f"the curves end at different episode counts ({min(ends.values()):,.0f}–"
            f"{max(ends.values()):,.0f}) at the same step budget: the configs differ in steps per "
            f"battle, so the same number of steps is a different number of battles"
        )
    if any(entry.get("seeds", 1) <= 1 for entry in series.values()):
        notes.append("a single-run curve has no seed variance behind it — treat its level as "
                     "indicative, not as a replicated mean")
    return " · ".join(notes)


def _mark_shortest_end(axis: "plt.Axes", series: Dict[str, Dict[str, Any]]) -> "Optional[float]":
    """Dotted line where a much shorter group stops; returns the x it marked, if any.

    Runs of the same config differ slightly in episode count (steps per battle varies), so only a
    real scale gap — one group stopping at well under ``X_END_RATIO`` of the longest — is annotated;
    otherwise every multi-seed figure would carry a spurious "ends here" mark.
    """
    ends = {
        label: float(entry["episodes"][-1])
        for label, entry in series.items()
        if entry.get("episodes") is not None and len(entry["episodes"])
    }
    if len(ends) < 2 or min(ends.values()) > X_END_RATIO * max(ends.values()):
        return None
    shortest = min(ends, key=ends.get)
    budget = _signature_parts(shortest)[1] or shortest
    axis.axvline(ends[shortest], color="0.55", linestyle=(0, (2, 3)), linewidth=1.1)
    # horizontal, just above the top spine: a rotated label inside the panel collides with the
    # curves and the seed band near the line
    axis.annotate(
        f"{budget} ends here", xy=(ends[shortest], 1.0), xycoords=("data", "axes fraction"),
        xytext=(0.0, 4.0), textcoords="offset points",
        fontsize=10.5, color="0.3", ha="center", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.4),
    )
    return ends[shortest]


def aggregate_seaborn(series: Dict[str, Dict[str, Any]], table: Dict[str, Any]) -> "plt.Figure":
    """Mean curve per config across seeds, shaded min-max; the honest read of a small sweep."""
    sns.set_theme(style="whitegrid", context="talk", palette="deep")
    sns.set_context("talk", rc={"axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 9})

    figure, axes = plt.subplots(2, 1, figsize=(12, 9.2), sharex=True)
    handles, labels = [], []
    shared = _shared_knobs(series)
    for index, (label, entry) in enumerate(series.items()):
        color = COMPARISON_COLORS[index % len(COMPARISON_COLORS)]
        # a single-run group is drawn lighter and thinner: its curve has no seed envelope behind it,
        # so it must not read as strongly supported as a multi-seed mean
        single = entry.get("seeds", 1) <= 1
        width, alpha = (2.0, 0.82) if single else (2.8, 1.0)
        legend_text = _aggregate_legend(label, entry, table, show_knobs=not shared)
        axes[0].fill_between(entry["episodes"], entry["win_rate_min"], entry["win_rate_max"],
                             color=color, alpha=0.24, linewidth=0)
        handle = axes[0].plot(entry["episodes"], entry["win_rate_mean"], color=color,
                              linewidth=width, alpha=alpha, label=legend_text)[0]
        handles.append(handle)
        labels.append(legend_text)
        axes[1].fill_between(entry["episodes"], entry["reward_min"], entry["reward_max"],
                             color=color, alpha=0.20, linewidth=0)
        axes[1].plot(entry["episodes"], entry["reward_mean"], color=color,
                     linewidth=width - 0.4, alpha=alpha)

    parity = axes[0].axhline(EVEN_MATCH_RATE, color="0.45", linestyle=(0, (4, 3)), linewidth=1.3)
    handles.append(parity)
    labels.append(f"parity with the opponent ({EVEN_MATCH_RATE:.0f}%)")
    # both reference lines carry a label: they share a dashed style, so an unlabelled zero line reads
    # as a second parity line rather than as "no better than chance"
    zero = axes[1].axhline(0.0, color="0.45", linestyle=(0, (1, 3)), linewidth=1.3)
    handles.append(zero)
    labels.append("zero reward (no better than chance)")
    marked = _mark_shortest_end(axes[0], series)
    if marked is not None:
        axes[1].axvline(marked, color="0.55", linestyle=(0, (2, 3)), linewidth=1.1)
    axes[0].set_ylabel(f"Win rate (%)\n{AGGREGATE_WINDOW}-episode rolling mean")
    axes[0].set_ylim(-2.0, 102.0)  # fixed 0-100 so panels stay comparable across figures
    axes[1].set_ylabel(f"Cumulative reward\n{AGGREGATE_WINDOW}-episode rolling mean")
    axes[1].set_xlabel("Episode   (one battle ≈ 25 env steps)")
    for axis in axes:
        axis.margins(x=0.01)

    # a tall legend (six groups) would run into the title in a single column, so wrap it
    columns = 1 if sum(label.count("\n") + 1 for label in labels) <= 6 else 2
    figure.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.008, 0.705),
                  ncol=columns, framealpha=0.95, fontsize=11.5)
    knobs_line = f"{shared}  ·  " if shared else ""
    lighter = " · single-run curves are drawn lighter" if any(
        entry.get("seeds", 1) <= 1 for entry in series.values()) else ""
    figure.suptitle(
        "Training curves by config — mean over seeds (shaded band: min–max across seeds, a range "
        "rather than a confidence interval)\n"
        f"{knobs_line}training-time win rate vs the scripted heuristic (moving policy, ε floor "
        f"0.02){lighter}",
        fontsize=13, y=0.98,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.70))
    note = _curve_end_note(series)
    if note:
        figure.text(0.012, 0.022, note, fontsize=11, color="0.32", ha="left", va="bottom",
                    wrap=True)
    return figure


def aggregate_plotly(series: Dict[str, Dict[str, Any]], table: Dict[str, Any]) -> "go.Figure":
    """Interactive twin of :func:`aggregate_seaborn`."""
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Win rate (%) — mean over seeds", "Cumulative reward — mean over seeds"),
    )
    for index, (label, entry) in enumerate(series.items()):
        color = COMPARISON_COLORS[index % len(COMPARISON_COLORS)]
        single = entry.get("seeds", 1) <= 1
        legend = _aggregate_legend(label, entry, table, show_knobs=not _shared_knobs(series))
        for row, key in ((1, "win_rate"), (2, "reward")):
            figure.add_trace(
                go.Scatter(x=entry["episodes"], y=entry[f"{key}_min"], mode="lines",
                           line=dict(width=0), hoverinfo="skip", showlegend=False, legendgroup=label),
                row=row, col=1,
            )
            figure.add_trace(
                go.Scatter(x=entry["episodes"], y=entry[f"{key}_max"], mode="lines",
                           line=dict(width=0), fill="tonexty", fillcolor=_rgba(color, 0.15),
                           hoverinfo="skip", showlegend=False, legendgroup=label),
                row=row, col=1,
            )
            figure.add_trace(
                go.Scatter(x=entry["episodes"], y=entry[f"{key}_mean"], mode="lines",
                           name=legend if row == 1 else label,
                           line=dict(color=color, width=2 if single else 3),
                           opacity=0.85 if single else 1.0,
                           legendgroup=label, showlegend=row == 1,
                           hovertemplate="episode %{x}<br>%{y:.1f}<extra>" + label + "</extra>"),
                row=row, col=1,
            )
    figure.add_hline(y=EVEN_MATCH_RATE, line=dict(color="#808080", width=1.4, dash="dash"),
                     annotation_text=f"parity ({EVEN_MATCH_RATE:.0f}%)", row=1, col=1)
    figure.add_hline(y=0.0, line=dict(color="#808080", width=1.2, dash="dot"), row=2, col=1,
                     annotation_text="zero reward", annotation_position="bottom right",
                     annotation_font_size=11)
    knobs_prefix = f"{_shared_knobs(series)} · " if _shared_knobs(series) else ""
    ends = {
        label: float(entry["episodes"][-1])
        for label, entry in series.items()
        if entry.get("episodes") is not None and len(entry["episodes"])
    }
    if len(ends) > 1 and min(ends.values()) <= X_END_RATIO * max(ends.values()):
        shortest = min(ends, key=ends.get)
        budget = _signature_parts(shortest)[1] or shortest
        for row in (1, 2):
            figure.add_vline(
                x=ends[shortest], line=dict(color="#8C8C8C", width=1.2, dash="dot"), row=row, col=1,
                annotation_text=f"{budget} ends here" if row == 1 else None,
                annotation_position="top", annotation_font_size=11,
            )
    note = _curve_end_note(series)
    if note:
        figure.add_annotation(text=note, xref="paper", yref="paper", x=0.0, y=-0.13,
                              showarrow=False, align="left", font=dict(size=11, color="#555555"))
    figure.update_layout(
        template="plotly_white", height=780, hovermode="x unified",
        title=dict(
            text=("Training curves by config — mean over seeds, shaded min–max across seeds<br>"
                  "<sup>" + knobs_prefix + "training-time win rate vs the scripted heuristic "
                  "(moving policy, ε floor 0.02)</sup>"),
            x=0.02,
        ),
        margin=dict(l=80, r=30, t=120, b=110),
        # horizontal legend between the title and the panels: the twin must not put it over the data
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0.0,
                    font=dict(size=11)),
    )
    figure.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    figure.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    figure.update_yaxes(range=[-2.0, 102.0], row=1, col=1)  # fixed 0-100 scale
    figure.update_yaxes(title_text=f"Win rate (%)\n{AGGREGATE_WINDOW}-episode rolling mean",
                        row=1, col=1)
    figure.update_yaxes(title_text=f"Cumulative reward\n{AGGREGATE_WINDOW}-episode rolling mean",
                        row=2, col=1)
    figure.update_xaxes(title_text="Episode   (one battle ≈ 25 env steps)", row=2, col=1)
    return figure


def aggregate(
    run_dirs: Sequence[Path],
    out_dir: Optional[Path] = None,
    stem: str = "aggregate",
    formats: Sequence[str] = ("png", "html"),
) -> Dict[str, Path]:
    """Group runs by training config, write the mean-curve figure(s) and the seed table."""
    run_dirs = [Path(run_dir) for run_dir in run_dirs]
    series = aggregate_series(run_dirs)
    table = aggregate_table(run_dirs)
    if not series:
        raise ValueError("no runs with a usable episodes.jsonl")
    out_dir = Path(out_dir) if out_dir else run_dirs[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}
    written["json"] = out_dir / f"{stem}.json"
    written["json"].write_text(
        json.dumps({"configs": table, "windows": {"aggregate": AGGREGATE_WINDOW}}, indent=2),
        encoding="utf-8",
    )
    if "png" in formats:
        figure = aggregate_seaborn(series, table)
        written["png"] = out_dir / f"{stem}.png"
        figure.savefig(written["png"], dpi=150, bbox_inches="tight")
        plt.close(figure)
    if "html" in formats:
        written["html"] = out_dir / f"{stem}.html"
        aggregate_plotly(series, table).write_html(written["html"], include_plotlyjs=True)
    return written


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
    parser.add_argument("--run", default=None, help="run directory holding episodes.jsonl")
    parser.add_argument(
        "--compare", nargs="+", default=None, metavar="RUN",
        help="two or more run directories: overlay their curves instead of plotting one run",
    )
    parser.add_argument(
        "--aggregate", nargs="+", default=None, metavar="RUN",
        help="two or more run directories: group by training config and average over seeds",
    )
    parser.add_argument("--out-dir", default=None, help="where to write (default: the run dir)")
    parser.add_argument("--name", default=None, help="output stem (default: curve_check / comparison)")
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
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]

    if args.aggregate:
        stem = args.name or "aggregate"
        out_dir = Path(args.out_dir) if args.out_dir else Path(args.aggregate[0]).parent
        written = aggregate(args.aggregate, out_dir=out_dir, stem=stem, formats=formats)
        table = json.loads((out_dir / f"{stem}.json").read_text(encoding="utf-8"))["configs"]
        for label, item in table.items():
            stats = item["final_window_success_rate_pct"]
            per_seed = ", ".join(f"{value:.1f}" for value in stats["per_seed"])
            print(
                f"{label}\n    final-200 win rate: {stats['mean']:.1f}% mean "
                f"({per_seed}) spread {stats['spread']:.1f} pp over {item['seeds_per_config']} seeds\n"
                f"    late trend: {item['late_trend_pp_per_1000']['mean']:+.1f} pp/1000 "
                f"(seeds: {', '.join(f'{value:+.1f}' for value in item['late_trend_pp_per_1000']['per_seed'])})"
            )
        print("\nwrote:", ", ".join(str(path) for path in written.values()))
        return 0

    if args.compare:
        stem = args.name or "comparison"
        out_dir = Path(args.out_dir) if args.out_dir else Path(args.compare[0])
        written = compare(args.compare, out_dir=out_dir, stem=stem, formats=formats)
        table = json.loads((out_dir / f"{stem}.json").read_text(encoding="utf-8"))["runs"]
        for name, metrics in table.items():
            print(f"{name:>14}  {_summary_line(metrics)}")
        print("\nwrote:", ", ".join(str(path) for path in written.values()))
        return 0

    if not args.run:
        raise SystemExit("--run is required (or use --compare with two or more runs)")
    run_dir = Path(args.run)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    written = render(
        run_dir,
        out_dir=out_dir,
        stem=args.name or "curve_check",
        which=args.only,
        formats=formats,
    )
    metrics = compute_metrics(load_episodes(run_dir))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps({key: value for key, value in metrics.items() if key != "windows"}, indent=2))
    print("\nwrote:", ", ".join(str(path) for path in written.values()), f"{out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
