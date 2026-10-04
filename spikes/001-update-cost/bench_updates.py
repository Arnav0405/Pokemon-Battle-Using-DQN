"""Microbenchmark: how expensive is one extra gradient step, relative to the server's step cost?

Grounds the Phase-3.5 plan: if the CPU cost of updates is small next to the Showdown round trip,
raising train_freq/gradient_steps buys sample efficiency for (almost) free.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch as th
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab_ca.policy import MaskedDQNPolicy  # noqa: E402

OBS = spaces.Dict(
    {
        "observation": spaces.Box(-1, 4, (33,), np.float32),
        "action_mask": spaces.Box(0, 1, (26,), np.int8),
    }
)
ACT = spaces.Discrete(26)
SERVER_STEP_MS = 1000 / 189.0  # measured throughput of the 100k run


def bench(net_arch, batch, iters=300):
    policy = MaskedDQNPolicy(OBS, ACT, lr_schedule=lambda _: 1e-3, net_arch=net_arch)
    net = policy.q_net
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    obs = {
        "observation": th.randn(batch, 33),
        "action_mask": (th.rand(batch, 26) > 0.3).to(th.int8),
    }
    obs["action_mask"][:, 0] = 1
    for _ in range(20):
        loss = net(obs).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    start = time.perf_counter()
    for _ in range(iters):
        loss = net(obs).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    per_update_ms = (time.perf_counter() - start) / iters * 1000
    params = sum(p.numel() for p in net.parameters())
    return per_update_ms, params


print(f"server env step (measured, 100k run): {SERVER_STEP_MS:.2f} ms\n")
print(f"{'net_arch':<12}{'batch':>7}{'ms/update':>11}{'params':>9}{'updates/env-step':>18}{'CPU ms/step':>13}{'% of server step':>18}")
for arch in ([64, 64], [128, 128], [256, 256]):
    for batch in (32, 128, 256):
        ms, params = bench(arch, batch)
        for updates_per_step in (0.25, 1.0, 4.0):  # default / train_freq=1,gsteps=1 / gsteps=4
            cpu = ms * updates_per_step
            print(
                f"{str(arch):<12}{batch:>7}{ms:>11.2f}{params:>9}{updates_per_step:>18.2f}"
                f"{cpu:>13.2f}{100 * cpu / SERVER_STEP_MS:>17.0f}%"
            )
    print()
