"""Structural check of the aggregate figure: does the legend really cover the data, and is the
"curves end here" line really in both panels?

The vision QA has flagged both; a vision model's reading of a rendered PNG is a hypothesis, and
this measures the claim against the figure objects instead of guessing from pixels.

Usage: python spikes/002-figure-audit/audit_aggregate.py <aggregate> <run> [<run> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from lab_ca.plots import (  # noqa: E402  (matplotlib backend must be set first)
    _curve_end_note,
    aggregate_seaborn,
    aggregate_series,
    aggregate_table,
)


def main() -> int:
    dirs = [Path(argument) for argument in sys.argv[2:]]
    series = aggregate_series(dirs)
    table = aggregate_table(dirs)
    figure = aggregate_seaborn(series, table)
    figure.canvas.draw()

    for index, axis in enumerate(figure.axes):
        verticals = []
        for line in axis.lines:
            x_data = line.get_xdata()
            # a vertical marker has a single x value repeated; matplotlib reports the dash pattern
            # as '--' regardless of the tuple we set, so match on geometry, not on linestyle text
            if len(x_data) and x_data[0] == x_data[-1]:
                verticals.append(float(x_data[0]))
        print(f"axes[{index}] vertical markers at x = {verticals}")

    note = _curve_end_note(series)
    print(f"footnote: {note or '(none)'}")

    if not figure.legends:
        print("no figure legend")
        return 0
    legend_box = figure.legends[0].get_window_extent()
    print(f"legend bbox  = {legend_box}")
    if figure._suptitle is not None:
        title_box = figure._suptitle.get_window_extent(figure.canvas.get_renderer())
        print(f"suptitle bbox = {title_box}")
        print(f"legend overlaps suptitle: {legend_box.overlaps(title_box)}")
        if legend_box.y1 > title_box.y0:
            print(f"  -> legend top is {legend_box.y1 - title_box.y0:.0f}px above the title's bottom")
    if figure.texts:
        for text in figure.texts:
            box = text.get_window_extent(figure.canvas.get_renderer())
            print(f"footnote bbox = {box} top={box.y1:.0f} figure_height={figure.bbox.height:.0f}")
    for index, axis in enumerate(figure.axes):
        axis_box = axis.get_window_extent()
        shade_hits = 0
        for collection in axis.collections:
            path = collection.get_paths()
            if path:
                points = path[0].vertices
                inside = (
                    (points[:, 0] >= legend_box.x0)
                    & (points[:, 0] <= legend_box.x1)
                    & (points[:, 1] >= legend_box.y0)
                    & (points[:, 1] <= legend_box.y1)
                )
                shade_hits += int(inside.sum())
        print(
            f"axes[{index}] bbox = {axis_box}  legend overlaps axes: "
            f"{legend_box.overlaps(axis_box)}  shaded-fill vertices under legend: {shade_hits}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
