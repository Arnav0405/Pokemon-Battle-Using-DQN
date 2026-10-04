"""Structural check of the comparison figure: what the axes actually span, and what is plotted."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab_ca.plots import comparison_seaborn, comparison_series, compute_metrics, load_episodes  # noqa: E402

RUNS = [Path("runs/p3_100k"), Path("runs/p3b_100k")]
series = comparison_series(RUNS)
metrics = {path.name: compute_metrics(load_episodes(path)) for path in RUNS}
figure = comparison_seaborn(series, metrics)

for index, axis in enumerate(figure.axes):
    print(f"--- axes[{index}] ---")
    print(f"  xlim    : {axis.get_xlim()}")
    print(f"  ylim    : {axis.get_ylim()}")
    print(f"  ylabel  : {axis.get_ylabel()!r}")
    print(f"  lines   : {len(axis.lines)}  labels={[line.get_label() for line in axis.lines]}")
    print(f"  colls   : {len(axis.collections)}")
    print(f"  legend  : {axis.get_legend() is not None}")

for name, frame in series.items():
    print(
        f"{name}: episodes={len(frame)} last_rate={frame['win_rate'].iloc[-1] * 100:.1f}% "
        f"reward_smooth_last={frame['reward_smooth'].iloc[-1]:.1f}"
    )
