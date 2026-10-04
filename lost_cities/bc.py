"""Behavior cloning: trains the network to imitate the heuristic bot.

Generates heuristic vs heuristic games, stores (features, legal actions,
the heuristic's move, final round result) and trains:
    - the policy with cross-entropy against the heuristic's move
    - the value on the final point difference / REWARD_SCALE

The checkpoint is saved in PPO format, ready for `ppo --init`.

    .venv/bin/python -m lost_cities.bc --games 6000 --out runs/v3/u00000.pt
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .bots import HeuristicBot
from .engine import GameState
from .ppo import OBS_DIMS, REWARD_SCALE, PolicyNet, features_batch, legal_mask_batch


def generate(n_games: int, feat: int, seed: int, eps: float = 0.1, batch: int = 512):
    """Heuristic vs heuristic. With probability `eps` a random action is played
    (not stored as an example) so the dataset covers more varied states."""
    rng = random.Random(seed)
    bot = HeuristicBot()
    X, M, A, V = [], [], [], []
    for start in range(0, n_games, batch):
        games = [GameState.new_game(rng, starting_player=rng.randrange(2)) for _ in range(min(batch, n_games - start))]
        # per game: list of (index in X, player) to fill in the value at the end
        pending = [[] for _ in games]
        values = []
        active = list(range(len(games)))
        while active:
            states = [games[i] for i in active]
            players = [s.current for s in states]
            x = features_batch(states, players, feat).astype(np.float16)
            m = legal_mask_batch(states)
            for j, i in enumerate(active):
                s = games[i]
                a = bot.act(s, rng)
                if rng.random() < eps:
                    s.step(rng.choice(s.legal_actions()))
                    continue
                pending[i].append((len(values), players[j]))
                values.append(0.0)
                X.append(x[j])
                M.append(m[j])
                A.append(a)
                s.step(a)
            active = [i for i in active if not games[i].done]
        for i, g in enumerate(games):
            for k, p in pending[i]:
                values[k] = g.score_diff(p) / REWARD_SCALE
        V += values
    return np.stack(X), np.stack(M), np.array(A, dtype=np.int64), np.array(V, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6000)
    ap.add_argument("--feat", type=int, default=2, choices=sorted(OBS_DIMS))
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/v3/u00000.pt")
    args = ap.parse_args()

    t0 = time.time()
    X, M, A, V = generate(args.games, args.feat, args.seed)
    print(f"dataset: {len(A)} decisions from {args.games} games in {time.time() - t0:.0f}s", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    n = len(A)
    perm = np.random.default_rng(args.seed).permutation(n)
    n_val = n // 20
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xt = torch.from_numpy(X).to(device)
    Mt = torch.from_numpy(M).to(device)
    At = torch.from_numpy(A).to(device)
    Vt = torch.from_numpy(V).to(device)

    net = PolicyNet(args.hidden, args.feat).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * (len(tr_idx) // args.batch + 1))

    def run_val():
        net.eval()
        with torch.no_grad():
            idx = torch.from_numpy(val_idx).to(device)
            logits, v = net(Xt[idx].float(), Mt[idx])
            acc = (logits.argmax(-1) == At[idx]).float().mean().item()
            vl = F.mse_loss(v, Vt[idx]).item()
        net.train()
        return acc, vl

    for ep in range(args.epochs):
        order = torch.from_numpy(np.random.default_rng(ep).permutation(tr_idx)).to(device)
        for s in range(0, len(order), args.batch):
            idx = order[s:s + args.batch]
            logits, v = net(Xt[idx].float(), Mt[idx])
            loss = F.cross_entropy(logits, At[idx]) + 0.5 * F.mse_loss(v, Vt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        acc, vl = run_val()
        print(f"epoch {ep + 1}: move accuracy {acc:.1%}  value error {vl:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": net.state_dict(), "update": 0, "hidden": args.hidden, "feat": args.feat}, args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
