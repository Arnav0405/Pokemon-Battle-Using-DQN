# Spike 007 — changing the teams: what breaks, and what it costs

**Question.** Can the teams be swapped (agent and/or opponent), does anything need retraining, and
what is the measured cost?

## Why nothing crashes

`lab_ca.env.embed_battle` builds a fixed 37-float vector (`N_FEATURES = 37`; the README's "33" is
stale) containing **no species and no move identity**. Every feature is derived on the fly:

| Feature group | Count | Source |
|---|---|---|
| `move{0..3}_base_power` | 4 | `min(move.base_power, 250)/100`, `-1` when not selectable |
| `move{0..3}_effectiveness` | 4 | `move.type.damage_multiplier(opp types)` via the gen type chart |
| `team{0..5}_hp_fraction` | 6 | |
| `opp_active_hp_fraction` | 1 | |
| `our_status_*` / `opp_status_*` | 6 + 6 | one-hot |
| `our_boost_*` / `opp_boost_*` | 5 + 5 | `/6` |

A different team therefore lands in the same space with different values: the checkpoint loads, the
action space is still `Discrete(18)` (0-5 switch, 6-9 move, 10+ gimmick), and no vocabulary lookup
can miss. `make_env(agent_team=…, opponent_team=…)` already exists for this; `train.py` and
`evaluate.py` do **not** expose it as a CLI flag.

## The move-slot aliasing (the real consequence)

Base power is the only move-identity signal, so **every status move collapses to 0.00**:

```
Earthquake  BP=100 slot=1.00   Stealth Rock BP=0 slot=0.00   Recover  BP=0 slot=0.00
Make It Rain BP=120 slot=1.20  Nasty Plot   BP=0 slot=0.00   Protect  BP=0 slot=0.00
                               Swords Dance BP=0 slot=0.00   Calm Mind BP=0 slot=0.00
```

Two 0-BP moves *of the same type* are indistinguishable in every state. This is already true inside
the shipped canon team: **Garganacl's Recover and Protect are both Normal-type, 0 BP**, so their two
move slots (and their two action indices) carry identical features always. Verified pairwise against
Great Tusk (Ground/Fighting):

```
canon  : Make It Rain=(1.20,4.00) Shadow Ball=(0.80,0.00) Recover=(0.00,2.00) Nasty Plot=(0.00,2.00)
swapped: Make It Rain=(1.20,4.00) Shadow Ball=(0.80,0.00) Protect=(0.00,2.00) Swords Dance=(0.00,2.00)
  8 move features identical: True
```

So the agent can never learn "Recover when low, Protect to scout" -- it can only learn a slot-order
preference. The Protect/Recover stalling seen in the mechanism check is consistent with this.

## Measured transfer cost (frozen `runs/p3_400k/model_final.zip`, 200 battles/config)

| Configuration | Win rate vs scripted heuristic |
|---|---|
| canon vs canon (as shipped) | **90.0 %** [85.1, 93.4] |
| ALT vs canon (agent team swapped) | **1.5 %** [0.5, 4.3] |
| canon vs ALT (opponent team swapped) | **0.5 %** [0.1, 2.8] |
| ALT vs ALT (both swapped) | **0.0 %** [0.0, 1.9] |

The canon-vs-canon row reproduces the report's 87.5 % inside its CI, so the harness is measuring the
same thing it always did.

## Control: is that "team strength" or "no transfer"?

Same scripted heuristic on both sides, fair fight (`team_strength_control.py`, 200 battles):

| Pairing | Winner's rate |
|---|---|
| heuristic(ALT) vs heuristic(CANON) | 70.5 % [63.8, 76.4] — so the canon side wins 29.5 % |
| heuristic(CANON) vs heuristic(CANON) | 55.0 % [48.1, 61.7] — mirror, CI covers 50 % |

Team strength is worth roughly **20 pp** to a scripted player (50 % -> 29.5 % for the canon side).
The frozen policy loses **89.5 pp** (90 % -> 0.5 %) on the same change. So the collapse is not the
matchup; it is the policy, which is ~4.5x more brittle to an unseen opponent than the scripted
heuristic is.

## Mechanism, from the protocol

`mechanism_check.log`: real 27-turn battle, no forfeit, no harness fault. Clodsire sets Spikes on
turn 1, then Dondozo (Unaware, 504 HP) walls the canon team; the agent switches a **15/290 HP**
Iron Valiant in, and loops Protect/Recover on Garganacl. Unaware walls, Spikes and the whole
defensive structure of the ALT team are invisible to a feature set that knows only HP, status,
boosts and its own move types.

## Conclusions

1. Swapping teams is mechanically free — no code change required beyond plumbing a flag.
2. The frozen policy does **not** transfer. Treat a team change as a new environment: retrain.
3. Retraining alone cannot fix the status-move blindness. If the new team leans on setup/status
   moves, adding move identity (one-hot per known move id, or an embedding over the move
   vocabulary) is the prerequisite, otherwise two 0-BP same-type moves stay indistinguishable.
4. The report's "outplayed the scripted baseline" claim is a **mirror-match** claim. Change either
   side and the Phase 4/5 numbers no longer describe the system being demoed.

## Pitfalls hit here

- `cross_evaluate()` returned `n_finished_battles == 0` against the local `--no-security` server and
  then raised `ZeroDivisionError` in `Player.win_rate`. `await p1.battle_against(p2, n_battles=N)`
  works — use the primitive.
- With `DummyVecEnv`, several configurations in one process reuse poke-env's generated account name,
  so the server answers `|nametaken|` and the challenge never starts
  (`asyncio.TimeoutError: Agent is not challenging`). Give each configuration its own
  `AccountConfiguration`.
