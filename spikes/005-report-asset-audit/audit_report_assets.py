"""Audit `report_assets/`: does `metrics.json` agree with the run log, and do the figures draw
the whole series with labelled axes?

Run from the repo root with the workspace venv:

    ../.venv/Scripts/python.exe spikes/005-report-asset-audit/audit_report_assets.py

Every check prints PASS/FAIL with the numbers behind it; a failure exits non-zero. Three things this
audit exists to catch, because none is visible by looking at a picture:

* **a series cut short** -- the drawn x-range is compared against the run's own episode count;
* **a degenerate uncertainty band** -- a CI or min-max envelope with ~zero width *looks* like
  "the seeds agree perfectly" while actually meaning the band was never drawn; measured off the
  drawn artists, not recomputed from the inputs, so a renderer bug is caught too;
* **a metric that disagrees with the log it claims to summarise** -- `metrics.json` is recomputed
  from `episodes.jsonl` here rather than trusted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from lab_ca import plots  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "report_assets"
RUN = REPO / "runs" / "p3_400k"

# the 100k sweep, all three seeds per arm -- the run set the arm decision was made on
SWEEP: List[Path] = [
    REPO / "runs" / name
    for name in (
        "p3_100k", "sweep_a_seed1", "sweep_a_seed2",
        "p3b_100k", "sweep_b_seed1", "sweep_b_seed2",
        "sweep_c_seed0", "sweep_c_seed1", "sweep_c_seed2",
    )
]

ADOPTED = "arch [64,64] · batch 128"  # arm B's signature: update budget + arch + batch

CHECKS: List[Tuple[str, bool, str]] = []


def check(name: str, ok: Any, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def line_xs(axis: Any) -> List[np.ndarray]:
    return [np.asarray(line.get_xdata(), dtype=float) for line in axis.lines]


def data_line_xs(axis: Any) -> List[np.ndarray]:
    """Line x-arrays that are data, not `axhline` stubs (whose x is 0..1 in axes fractions)."""
    return [xs for xs in line_xs(axis) if xs.size and float(xs.max()) > 100.0]


def scatter_xs(axis: Any) -> List[np.ndarray]:
    offsets = [
        np.asarray(collection.get_offsets(), dtype=float)
        for collection in axis.collections
        if getattr(collection, "get_offsets", None) is not None
        and np.asarray(collection.get_offsets()).size
    ]
    return [array[:, 0] for array in offsets]


def band_thickness(axis: Any, tail: float = 0.1) -> List[float]:
    """Median vertical thickness of each drawn `fill_between` band over the last `tail` of x.

    Measured off the drawn artists -- the polygon's vertices grouped by x -- so a band that was never
    drawn, or drawn collapsed, fails here even though the inputs it was built from look fine. One
    entry per band; a scatter of raw points is not a band.
    """
    thicknesses: List[float] = []
    for collection in axis.collections:
        if type(collection).__name__.startswith("PathCollection"):
            continue  # raw per-episode scatter, not a band
        columns: Dict[float, List[float]] = {}
        for path in getattr(collection, "get_paths", lambda: [])():
            vertices = np.asarray(getattr(path, "vertices", []), dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] != 2 or not vertices.size:
                continue
            for x, y in vertices:
                columns.setdefault(round(float(x), 6), []).append(float(y))
        if len(columns) < 2:
            continue
        xs = sorted(columns)
        tail_xs = xs[int(len(xs) * (1.0 - tail)):]
        thicknesses.append(float(np.median([max(columns[x]) - min(columns[x]) for x in tail_xs])))
    return thicknesses


# ---------------------------------------------------------------- metrics.json vs the run log
rows = plots.load_episodes(RUN)
metrics = json.loads((ASSETS / "metrics.json").read_text(encoding="utf-8"))
wins = np.array([row["win"] for row in rows], dtype=float)
steps = np.array([row["steps"] for row in rows], dtype=float)
window = plots.CONVERGENCE_WINDOW

check("metrics.episodes matches episodes.jsonl", metrics["episodes"] == len(rows),
      f'{metrics["episodes"]} vs {len(rows)} rows')
check("wins + losses + unfinished == episodes",
      metrics["wins"] + metrics["losses"] + metrics["unfinished"] == metrics["episodes"],
      f'{metrics["wins"]} + {metrics["losses"]} + {metrics["unfinished"]}')
check(f"final_window_success_rate == mean of the last {window} wins",
      abs(metrics["final_window_success_rate"] - wins[-window:].mean()) < 1e-9,
      f'{metrics["final_window_success_rate"]:.6f} vs {wins[-window:].mean():.6f}')
check(f"early_window_success_rate == mean of the first {window} wins",
      abs(metrics["early_window_success_rate"] - wins[:window].mean()) < 1e-9,
      f'{metrics["early_window_success_rate"]:.6f}')
check("success_rate == overall win fraction",
      abs(metrics["success_rate"] - wins.mean()) < 1e-9,
      f'{metrics["success_rate"]:.6f} vs {wins.mean():.6f}')
check("average_steps_per_episode == mean steps",
      abs(metrics["average_steps_per_episode"] - steps.mean()) < 1e-6,
      f'{metrics["average_steps_per_episode"]:.4f} vs {steps.mean():.4f}')
check("convergence_episode is inside the run",
      metrics["convergence_episode"] is None or 0 < metrics["convergence_episode"] <= len(rows),
      f'episode {metrics["convergence_episode"]} of {len(rows)}')
expected_band = max(
    plots.CONVERGENCE_TOLERANCE_FLOOR,
    plots.CONVERGENCE_SIGMA_MULTIPLE * plots.binomial_sigma(metrics["final_window_success_rate"], window) * 100.0,
)
check("convergence band == max(floor, 2 sigma of the window rate)",
      abs(metrics["convergence_band_pp"] - expected_band) < 1e-6,
      f'{metrics["convergence_band_pp"]:.3f} pp vs {expected_band:.3f} recomputed')

# ---------------------------------------------------------------- the three-panel curve figure
frame = plots.load_frame(RUN)
frame.attrs["run"] = RUN.name
specs = plots._panel_specs(frame, metrics, "all")
figure = plots.seaborn_figure(frame, metrics, specs)
axes = figure.axes
episodes = len(frame)

check("curve_check has three panels", len(axes) == 3, f"{len(axes)} axes")
check("every panel carries a y-axis label", all(axis.get_ylabel() for axis in axes),
      " | ".join(axis.get_ylabel() for axis in axes))
check("the bottom panel carries an x-axis label",
      "Episode" in axes[-1].get_xlabel(), repr(axes[-1].get_xlabel()))
check("the title names the opponent and the run",
      metrics["opponent"] in figure._suptitle.get_text() and RUN.name in figure._suptitle.get_text(),
      figure._suptitle.get_text().splitlines()[0])

for axis, spec in zip(axes, specs):
    key = spec["key"]
    xs = line_xs(axis) + scatter_xs(axis)
    reach = max(float(x.max()) for x in xs) if xs else 0.0
    smoothed = [line for line in axis.lines if "moving average" in line.get_label()]
    check(f"{key}: smoothed series drawn", len(smoothed) == 1, f"window {spec['window']}")
    check(f"{key}: raw per-episode series drawn",
          any(line.get_label() == "per episode" for line in axis.lines) or bool(scatter_xs(axis)),
          f"{len(scatter_xs(axis))} scatter series" if scatter_xs(axis) else "line")
    check(f"{key}: x reaches the last episode (no truncation)", reach == episodes,
          f"drawn to {reach:,.0f} of {episodes:,}")
    check(f"{key}: panel has a legend", axis.get_legend() is not None)

success = axes[1]
check("success panel marks parity with the opponent",
      any(np.allclose(np.asarray(line.get_ydata(), dtype=float), plots.EVEN_MATCH_RATE)
          for line in success.lines),
      f"y = {plots.EVEN_MATCH_RATE}")
check("success panel marks the convergence episode",
      any(line.get_linestyle() == "--"
          and np.allclose(np.asarray(line.get_xdata(), dtype=float), metrics["convergence_episode"])
          for line in success.lines),
      f"x = {metrics['convergence_episode']}")
success_bands = band_thickness(success)
check("success panel draws exactly one uncertainty band", len(success_bands) == 1, str(success_bands))
check("success CI band has real width at the end of the run (>5 pp thick)",
      bool(success_bands) and max(success_bands) > 5.0,
      f"{max(success_bands):.1f} pp thick" if success_bands else "no band drawn")
check("reward and steps panels draw no uncertainty band",
      not band_thickness(axes[0]) and not band_thickness(axes[2]),
      f"{band_thickness(axes[0])} / {band_thickness(axes[2])}")

# legends must sit outside the panels: `loc="best"` does not avoid scatter collections, so an inside
# legend covers data on a full-length run (a vision review caught this on all three panels)
figure.canvas.draw()
for axis, spec in zip(axes, specs):
    legend_box = axis.get_legend().get_window_extent()
    axes_box = axis.get_window_extent()
    check(f"{spec['key']}: legend lies outside the data area",
          legend_box.x0 >= axes_box.x1,
          f"legend x0 {legend_box.x0:.0f} vs axes right edge {axes_box.x1:.0f}")
    check(f"{spec['key']}: legend is inside the canvas",
          legend_box.x0 >= 0 and legend_box.x1 <= figure.get_window_extent().x1,
          f"legend x1 {legend_box.x1:.0f} vs canvas {figure.get_window_extent().x1:.0f}")
success_legend = [text.get_text() for text in success.get_legend().get_texts()]
check("success legend names the per-episode series (a dense scatter is not an axis line)",
      any("per episode" in text for text in success_legend), " | ".join(success_legend))

# ---------------------------------------------------------------- the arm-separation aggregate
table = json.loads((ASSETS / "sweep_aggregate.json").read_text(encoding="utf-8"))
configs: Dict[str, Dict[str, Any]] = table["configs"]
check("aggregate table has one entry per arm", len(configs) == 3, f"{len(configs)} configs")
check("every arm averages three seeds",
      all(entry["seeds_per_config"] == 3 for entry in configs.values()),
      str({label.split(" · ")[0]: entry["seeds_per_config"] for label, entry in configs.items()}))

adopted = [entry for label, entry in configs.items() if ADOPTED in label]
defaults = [entry for label, entry in configs.items() if "0.25 upd/step" in label]
capacity = [entry for label, entry in configs.items() if "arch [256,256]" in label]
check("each arm appears exactly once",
      len(adopted) == len(defaults) == len(capacity) == 1,
      f"adopted {len(adopted)}, defaults {len(defaults)}, capacity {len(capacity)}")
check("arm means reproduce the recorded sweep (49.2 / 51.3 / 73.5)",
      np.allclose(sorted(entry["final_window_success_rate_pct"]["mean"] for entry in configs.values()),
                  [49.2, 51.3, 73.5], atol=0.05),
      str(sorted(entry["final_window_success_rate_pct"]["mean"] for entry in configs.values())))
# the documented claim is arm A vs arm B: "A's best seed 57.5% < B's worst 67.0%" (plan.md)
check("arm separation per seed: the SB3-defaults arm's best seed < the adopted arm's worst seed",
      max(defaults[0]["final_window_success_rate_pct"]["per_seed"])
      < min(adopted[0]["final_window_success_rate_pct"]["per_seed"]),
      f'defaults best {max(defaults[0]["final_window_success_rate_pct"]["per_seed"]):.1f} vs '
      f'adopted worst {min(adopted[0]["final_window_success_rate_pct"]["per_seed"]):.1f}')
# arm C is rejected for spread, not for a mean shift: one seed lands inside the adopted arm's range
check("arm C is rejected for spread, which is what the write-up says",
      capacity[0]["final_window_success_rate_pct"]["spread"]
      > max(entry["final_window_success_rate_pct"]["spread"] for entry in adopted + defaults),
      f'spread {capacity[0]["final_window_success_rate_pct"]["spread"]:.1f} pp vs '
      f'adopted {adopted[0]["final_window_success_rate_pct"]["spread"]:.1f} pp, '
      f'defaults {defaults[0]["final_window_success_rate_pct"]["spread"]:.1f} pp')
check("arm C's lucky seed is why the sweep is read per seed, not per mean",
      max(capacity[0]["final_window_success_rate_pct"]["per_seed"])
      > min(adopted[0]["final_window_success_rate_pct"]["per_seed"]),
      f'capacity best {max(capacity[0]["final_window_success_rate_pct"]["per_seed"]):.1f} sits above '
      f'adopted worst {min(adopted[0]["final_window_success_rate_pct"]["per_seed"]):.1f}')

series = plots.aggregate_series(SWEEP)
aggregate = plots.aggregate_seaborn(series, table)
agg_axes = aggregate.axes
check("aggregate has a win-rate panel and a reward panel", len(agg_axes) == 2, f"{len(agg_axes)} axes")
check("aggregate panels are labelled",
      all(axis.get_ylabel() for axis in agg_axes) and "Episode" in agg_axes[-1].get_xlabel())
drawn_lengths = sorted(len(xs) for xs in data_line_xs(agg_axes[0]))
group_lengths = sorted(len(entry["episodes"]) for entry in series.values())
check("aggregate draws one mean curve per config, each to its own last episode",
      drawn_lengths == group_lengths,
      f"drawn {drawn_lengths} vs groups {group_lengths}")
envelopes = band_thickness(agg_axes[0])
check("aggregate draws one seed envelope per config", len(envelopes) == len(series),
      f"{len(envelopes)} bands for {len(series)} configs")
check("aggregate seed envelopes have real width at the end of the run",
      bool(envelopes) and max(envelopes) > 2.0,
      f"widest {max(envelopes):.1f} pp, narrowest {min(envelopes):.1f} pp" if envelopes
      else "no band drawn")
aggregate.canvas.draw()
agg_legend = aggregate.legends[0].get_window_extent()
check("aggregate legend sits clear of the panels",
      agg_legend.y0 >= agg_axes[0].get_window_extent().y1,
      f"legend y0 {agg_legend.y0:.0f} vs top panel {agg_axes[0].get_window_extent().y1:.0f}")
suptitle_box = aggregate._suptitle.get_window_extent()
check("aggregate legend does not run into the title",
      agg_legend.y1 <= suptitle_box.y0,
      f"legend top {agg_legend.y1:.0f} vs title bottom {suptitle_box.y0:.0f}")
agg_labels = " | ".join(text.get_text() for text in aggregate.legends[0].get_texts())
check("aggregate legend labels both reference lines",
      "parity with the opponent" in agg_labels and "zero reward" in agg_labels, agg_labels)
# the plotly twin must not drift from the static figure it accompanies
aggregate_html = (ASSETS / "sweep_aggregate.html").read_text(encoding="utf-8", errors="replace")
curve_html = (ASSETS / "curve_check.html").read_text(encoding="utf-8", errors="replace")
check("no figure or twin still says \"Success rate\"",
      not any("Success rate" in text for text in (aggregate_html, curve_html))
      and "Win rate (%)" in curve_html and "Win rate (%)" in aggregate_html,
      "both twins use the same rate terminology as the evaluation figure")

# ---------------------------------------------------------------- report
width = max(len(name) for name, _, _ in CHECKS)
for name, ok, detail in CHECKS:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
failures = [name for name, ok, _ in CHECKS if not ok]
print(f"\n{len(CHECKS) - len(failures)}/{len(CHECKS)} checks passed")
for name in failures:
    print(f"  FAILED: {name}")
sys.exit(1 if failures else 0)
