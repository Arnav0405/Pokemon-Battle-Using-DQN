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
                axis.scatter(frame["episode"], spec["raw"], color=spec["color"], alpha=0.24,
                             s=13, linewidths=0, label="per episode (win / loss)")
            else:
                axis.plot(frame["episode"], spec["raw"], color=spec["color"], alpha=0.35,
                          linewidth=0.9, label="per episode")
        axis.plot(frame["episode"], spec["smooth"], color=spec["color"], linewidth=2.6,
                  label=f"moving average (window {spec['window']})")
        if spec["key"] == "success":
            half = _binomial_half_width(spec["smooth"], spec["window"])
            axis.fill_between(
                frame["episode"], spec["smooth"] - half, spec["smooth"] + half,
                color=spec["color"], alpha=0.18, linewidth=0, label="±95% CI (binomial)",
            )
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
    figure.update_xaxes(title_text="Episode   (one episode = one battle)", row=len(specs), col=1)
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
    axes[0].set_ylabel("Success rate (%)")
    axes[1].axhline(0.0, color="0.45", linestyle=(0, (4, 3)), linewidth=1.3)
    axes[1].set_ylabel(f"Cumulative reward\n(moving average, window {REWARD_WINDOW})")
    axes[1].set_xlabel("Episode   (one episode = one battle)")
    for axis in axes:
        axis.margins(x=0.01)
    # one shared legend *outside* the axes: the curves cross every corner of these panels, so any
    # in-axes legend ends up on top of data (both colors are shared by the two panels)
    figure.legend(
        [*rate_handles, parity],
        [*rate_labels, f"parity with the opponent ({EVEN_MATCH_RATE:.0f}%)"],
        loc="upper center", bbox_to_anchor=(0.5, 0.885), ncol=3, framealpha=0.95, fontsize=10.5,
    )
    figure.suptitle(
        "Run comparison — success rate (with ±95% binomial CI) and reward\n"
        "overlapping bands mean the difference is not resolvable at this episode count",
        fontsize=13, y=0.98,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.855))
    return figure


def comparison_plotly(series: "Dict[str, pd.DataFrame]", metrics: Dict[str, Dict[str, Any]]) -> "go.Figure":
    """Interactive twin of :func:`comparison_seaborn`."""
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Success rate (%)", f"Cumulative reward (moving average, window {REWARD_WINDOW})"),
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
    figure.update_xaxes(title_text="Episode   (one episode = one battle)", row=2, col=1)
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
