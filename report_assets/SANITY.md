# Report assets — what is here, and how it was checked

Everything in this folder is regenerated from a run's own log by the repo's own modules; nothing is
hand-edited. PNG for the printed report, self-contained HTML twin (plotly, zoom + hover) for reading
on screen, JSON where numbers should be quoted rather than read off a picture.

## Inventory

| File | What it shows | Regenerate with |
|---|---|---|
| `curve_check.png` / `.html` | the adopted 400k run: reward, win rate, steps per episode — raw + smoothed, binomial CI, parity and convergence markers | `python -m lab_ca.plots --run runs/p3_400k --out-dir report_assets` |
| `reward.png` / `.html`, `success.png` / `.html`, `steps.png` / `.html` | the same three curves as standalone panels, for placing one at a time | `... --only reward --name reward` (and `success`, `steps`) |
| `metrics.json` | every headline number of that run, in one place | written by the same single-run command |
| `sweep_aggregate.png` / `.html` / `.json` | the arm decision: mean win rate per config, shaded min–max across 3 seeds each, with the per-seed table | `... --aggregate runs/p3_100k runs/sweep_a_seed1 runs/sweep_a_seed2 runs/p3b_100k runs/sweep_b_seed1 runs/sweep_b_seed2 runs/sweep_c_seed0 runs/sweep_c_seed1 runs/sweep_c_seed2 --name sweep_aggregate` |
| `phase4_eval.png` / `.html` / `.md` | frozen-policy evaluation: win rate per checkpoint vs the scripted heuristic and vs random, Wilson 95% intervals, and the random-policy floor | `python -m lab_ca.evaluate --summarize runs/*/eval.json runs/eval_random_floor.json --out report_assets/phase4_eval` |

Headline numbers the figures carry: 14,519 episodes (400k steps), final-200 win rate **81.0 %**,
convergence declared at episode **13,781**; sweep means **51.3 / 73.5 / 49.2 %** for arms A / B / C;
frozen policy **87.5 %** vs the heuristic at 400k, with the scripted-random floor at 1.0 % (vs the
heuristic) and 49.0 % (vs random).

## The sanity pass

`spikes/005-report-asset-audit/audit_report_assets.py` — **52 checks, all passing**:

```bash
python spikes/005-report-asset-audit/audit_report_assets.py     # exits non-zero on any failure
```

Three things it does that looking at a picture cannot:

1. **Recomputes `metrics.json` from `episodes.jsonl`** — episode count, wins+losses+unfinished,
   final / early / overall win rates, mean steps, the convergence band — instead of trusting the file
   that claims to summarise the run.
2. **Checks every drawn series against the run's own episode count.** The "no truncation"
   requirement, made mechanical: a figure that silently stops early fails here. All three panels draw
   to episode 14,519 of 14,519.
3. **Measures the drawn uncertainty artists.** The win-rate CI band is 23 pp thick at the end of the
   run and the seed envelopes are 12–47 pp wide. A band that was never drawn (~0 width) *looks* like
   "the seeds agree perfectly" — the one figure defect that reads as a good result.

Also pinned: axis labels on every panel; a raw **and** a smoothed series per panel; the parity and
convergence references present; each legend outside the plot area, clear of the panels and the title,
and inside the canvas; the aggregate legend labelling both reference lines; and no figure or HTML twin
still using the old "success rate" wording.

## Visual QA (second model)

Every figure went to a separate vision model (opencode CLI, `fledge-alpha-free`) with a checklist, per
the standing rule for report figures. Two rounds were needed, because the first round found real
problems — in two figures:

| Figure | Round 1 (found) | Round 2 (verdict) |
|---|---|---|
| `phase4_eval.png` | missing x-axis label; the 1 % floor bar read as a *missing* bar; tick labels duplicating the legend | **report-ready** |
| `curve_check.png` | legends covering data on all three panels; CI band too faint against the dense win/loss scatter | **report-ready** |
| `sweep_aggregate.png` | legend running into the title; zero line unlabelled and styled like the parity line; "win rate" in the title vs "success rate" on the axes; curves ending at different episode counts unexplained | **report-ready** |

Fixes that came out of QA:

- **Legends moved outside the panels.** matplotlib's `loc="best"` steers clear of lines but *not* of
  scatter collections, and a 14.5k-episode run occupies every inside corner.
- **Scatter alpha reduced, CI band strengthened** so the band reads over the raw points.
- **The aggregate legend repositioned** below the title (it was overlapping the title's second line);
  pinned by an audit check on legend-vs-title bounding boxes.
- **The zero line is now labelled *and* dotted**, distinct from the dashed parity line, in both the
  PNG and the HTML twin.
- **"Success rate" → "Win rate" everywhere**, matching the evaluation figure and the README.
- **The aggregate says why the curves end at different episode counts** (the same step budget is a
  different number of battles per config) instead of leaving it to look like truncation.
- The aggregate title now calls the shaded band **a range, not a confidence interval**.

Three QA complaints were **not** acted on, with reasons:

- *"Thick solid green lines at 100 % and 0 % with no legend entry"* — these are the raw per-episode
  win/loss markers: 14.5k points at 0 % or 100 %, which at this episode count render as dense bands.
  They are labelled (`per episode (win / loss)`, asserted by the audit) and are the raw signal the
  spec asks for.
- *"Reward and steps panels have no uncertainty band"* — a binomial CI is defined for a proportion; it
  does not exist for reward or for episode length. The asymmetry is correct.
- *"The aggregate has no raw series"* — nine raw per-episode series on one panel is not a readable
  figure; the aggregate is deliberately mean-across-seeds with a min–max envelope.

Worth noting for anyone repeating this: the QA model reports real defects and misreads in one list
(in one round it called a bar both "the final bar" and a floor row). Treat each claim as a hypothesis
and check it against the figure object — that is what the audit is for.

## Caveats to carry into the report prose

- **The shaded band in the aggregate is a min–max range over 3 seeds, not a confidence interval.**
- **The win-rate CI band uses a normal approximation of a binomial**, so at extreme rates it can
  extend slightly outside 0–100 %. Left unclipped deliberately: clipping would imply a bounded
  interval.
- **"Win rate" in the figures is `final_window_success_rate` in `metrics.json`** — same quantity,
  different names in the data contract.
- **A longer battle against a random opponent is a statement about the opponent, not the policy** —
  the vs-random columns have longer episodes (33–60) for that reason.
- The 400k run is a **single seed** (n=1) at that budget. The 100k sweep is 3 seeds per arm, and its
  separation claim is arm A vs arm B; arm C is rejected for a 51 pp spread, and its single best seed
  (78.5 %) sits inside arm B's range.
