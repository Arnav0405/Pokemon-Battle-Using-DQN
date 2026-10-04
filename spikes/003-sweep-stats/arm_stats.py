"""Per-arm statistics for the Phase-3 seed sweep (arms A/B/C, 3 seeds each).

Prints the table that the write-up and the report need: per-run metrics, per-arm mean/spread, and
a Welch t-test between arms (with the honest note that n=3 per arm limits what a p-value can say).
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab_ca.plots import compute_metrics, load_episodes  # noqa: E402

ARMS = {
    "A defaults": ["p3_100k", "sweep_a_seed1", "sweep_a_seed2"],
    "B 16x updates": ["p3b_100k", "sweep_b_seed1", "sweep_b_seed2"],
    "C B + 256x256": ["sweep_c_seed0", "sweep_c_seed1", "sweep_c_seed2"],
}
RUNS = Path("runs")


def welch(left, right):
    """Welch's t statistic + Satterthwaite df + two-sided p (scipy if present, else None)."""
    import statistics as st

    n1, n2 = len(left), len(right)
    m1, m2 = st.mean(left), st.mean(right)
    v1, v2 = st.variance(left), st.variance(right)
    se = math.sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / se
    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    )
    try:
        from scipy import stats

        p = float(2 * stats.t.sf(abs(t), df))
    except Exception:
        p = None
    return t, df, p


def permutation_test(left, right):
    """Exact two-sided permutation test on the difference of means (no distributional assumption).

    With 3 vs 3 runs there are only 20 ways to split the six values, so the smallest attainable
    two-sided p is 0.10 -- worth stating next to any t-test p-value, which is smaller only because
    it assumes normality of a 3-point sample.
    """
    from itertools import combinations

    pooled = list(left) + list(right)
    observed = abs(sum(left) / len(left) - sum(right) / len(right))
    total = 0
    at_least = 0
    for group in combinations(range(len(pooled)), len(left)):
        chosen = [pooled[index] for index in group]
        rest = [value for index, value in enumerate(pooled) if index not in group]
        total += 1
        if abs(sum(chosen) / len(chosen) - sum(rest) / len(rest)) >= observed - 1e-12:
            at_least += 1
    return observed, at_least, total, at_least / total


rows = {}
for arm, names in ARMS.items():
    for name in names:
        run_dir = RUNS / name
        metrics = compute_metrics(load_episodes(run_dir))
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["config"]
        rows.setdefault(arm, []).append(
            {
                "run": name,
                "seed": config.get("seed"),
                "episodes": metrics["episodes"],
                "steps_per_episode": metrics["average_steps_per_episode"],
                "final200": 100 * metrics["final_window_success_rate"],
                "overall": 100 * metrics["success_rate"],
                "convergence": metrics["convergence_episode"],
                "trend": metrics["trend_slope_pp_per_1000"],
            }
        )

print(f"{'arm':<16}{'run':<18}{'seed':>5}{'eps':>7}{'steps/ep':>10}{'overall%':>10}{'final200%':>11}{'conv':>7}{'trend':>8}")
for arm, items in rows.items():
    for item in items:
        print(
            f"{arm:<16}{item['run']:<18}{item['seed']:>5}{item['episodes']:>7}"
            f"{item['steps_per_episode']:>10.1f}{item['overall']:>10.1f}{item['final200']:>11.1f}"
            f"{(item['convergence'] if item['convergence'] else '-'):>7}{item['trend']:>+8.1f}"
        )

print("\nper-arm summary (final 200 episodes, %):")
finals = {arm: [item["final200"] for item in items] for arm, items in rows.items()}
for arm, values in finals.items():
    values_sorted = sorted(values)
    print(
        f"  {arm:<16} mean {sum(values) / len(values):5.1f}   seeds {values_sorted}   "
        f"spread {max(values) - min(values):5.1f}"
    )

print("\nWelch t-test and exact permutation test on the final-200 win rate (n=3 vs n=3):")
for left, right in (("A defaults", "B 16x updates"), ("B 16x updates", "C B + 256x256"),
                    ("A defaults", "C B + 256x256")):
    t, df, p = welch(finals[left], finals[right])
    observed, at_least, total, p_perm = permutation_test(finals[left], finals[right])
    p_text = f"p={p:.3f}" if p is not None else "p=n/a (no scipy)"
    print(
        f"  {left} vs {right}:\n"
        f"      Welch  t={t:+.2f} df={df:.1f} {p_text}\n"
        f"      exact  |diff|={observed:.1f} pp, {at_least}/{total} splits as extreme -> p={p_perm:.3f}"
    )

print("\noverlap check (does every run of one arm beat every run of the other?):")
for left, right in (("A defaults", "B 16x updates"), ("B 16x updates", "C B + 256x256")):
    lo, hi = finals[left], finals[right]
    print(
        f"  {left} max {max(lo):.1f} vs {right} min {min(hi):.1f} -> "
        f"{'COMPLETE SEPARATION' if max(lo) < min(hi) else 'overlapping'}"
    )

# Does the capacity arm's instability show up in the TD loss? (SB3 logs it in the console table.)
import re  # noqa: E402

print("\nTD loss over training (from the run logs; last logged value + max):")
for arm, items in rows.items():
    for item in items:
        log = RUNS / item["run"] / "train.log"
        if not log.exists():
            continue
        text = log.read_text(encoding="utf-8", errors="ignore")
        values = [float(match) for match in re.findall(r"\|\s+loss\s+\|\s+([0-9.eE+-]+)", text)]
        if not values:
            continue
        half = len(values) // 2
        print(
            f"  {arm:<16}{item['run']:<18} n={len(values):>4}  first-half mean {sum(values[:half]) / max(1, half):7.3f}"
            f"   last-half mean {sum(values[half:]) / max(1, len(values) - half):7.3f}"
            f"   max {max(values):9.3f}"
        )
