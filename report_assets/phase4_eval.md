# Phase 4 -- frozen-policy evaluation

Greedy masked policy (epsilon disabled) vs the scripted baselines on the fixed roster.
50% is parity. Intervals are 95% Wilson score intervals over decided battles.

| run | agent | opponent | battles | win rate | 95% CI | mean turns |
|---|---|---|---|---|---|---|
| p3_100k | dqn | heuristic | 200 | 48.5% | 41.7-55.4 | 28.4 |
| p3_100k | dqn | random | 200 | 85.5% | 80.0-89.7 | 41.2 |
| p3_200k_2 | dqn | heuristic | 200 | 84.0% | 78.3-88.4 | 24.5 |
| p3_200k_2 | dqn | random | 200 | 72.0% | 65.4-77.8 | 60.1 |
| p3_400k | dqn | heuristic | 200 | 87.5% | 82.2-91.4 | 24.4 |
| p3_400k | dqn | random | 200 | 87.5% | 82.2-91.4 | 48.9 |
| p3b_100k | dqn | heuristic | 200 | 88.0% | 82.8-91.8 | 20.9 |
| p3b_100k | dqn | random | 200 | 86.0% | 80.5-90.1 | 48.8 |
| sweep_a_seed1 | dqn | heuristic | 200 | 51.5% | 44.6-58.3 | 26.1 |
| sweep_a_seed1 | dqn | random | 200 | 87.0% | 81.6-91.0 | 32.6 |
| sweep_a_seed2 | dqn | heuristic | 200 | 64.0% | 57.1-70.3 | 23.5 |
| sweep_a_seed2 | dqn | random | 200 | 84.0% | 78.3-88.4 | 36.9 |
| sweep_b_seed1 | dqn | heuristic | 200 | 75.0% | 68.6-80.5 | 21.9 |
| sweep_b_seed1 | dqn | random | 200 | 76.5% | 70.2-81.8 | 49.4 |
| sweep_b_seed2 | dqn | heuristic | 200 | 77.0% | 70.7-82.3 | 21.6 |
| sweep_b_seed2 | dqn | random | 200 | 74.0% | 67.5-79.6 | 53.1 |
| eval_random_floor | random | heuristic | 200 | 1.0% | 0.3-3.6 | 25.8 |
| eval_random_floor | random | random | 200 | 49.0% | 42.2-55.9 | 72.1 |
