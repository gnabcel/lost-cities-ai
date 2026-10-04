"""Pits bots against each other.

    python -m lost_cities.arena heuristic random -n 200
    python -m lost_cities.arena ismcts heuristic -n 20 --iters 300
    .venv/bin/python -m lost_cities.arena ppo:runs/v1/latest.pt heuristic -n 1000
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from .bots import HeuristicBot, RandomBot
from .engine import GameState
from .ismcts import FlatMCBot, ISMCTSBot


def make_bot(name: str, iters: int):
    if name == "random":
        return RandomBot()
    if name == "heuristic":
        return HeuristicBot()
    if name == "flatmc":
        return FlatMCBot(samples=iters)
    if name == "ismcts":
        return ISMCTSBot(iterations=iters)
    if name.startswith("ppo:"):
        from .ppo import PolicyBot  # requires torch

        return PolicyBot.load(name[4:])
    raise ValueError(name)


def play_round(bots, rng: random.Random, starting_player: int = 0, verbose: bool = False) -> GameState:
    s = GameState.new_game(rng, starting_player)
    while not s.done:
        a = bots[s.current].act(s, rng)
        s.step(a)
    if verbose:
        print(s.render())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bot_a")
    ap.add_argument("bot_b")
    ap.add_argument("-n", type=int, default=100, help="rounds")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    a, b = make_bot(args.bot_a, args.iters), make_bot(args.bot_b, args.iters)
    diffs, wins, draws = [], 0, 0
    t0 = time.time()
    for i in range(args.n):
        # alternate seats and who starts
        seat_a = i % 2
        bots = [a, b] if seat_a == 0 else [b, a]
        s = play_round(bots, rng, starting_player=(i // 2) % 2)
        d = s.score_diff(seat_a)
        diffs.append(d)
        wins += d > 0
        draws += d == 0
    dt = time.time() - t0
    n = args.n
    print(f"{args.bot_a} vs {args.bot_b}: {n} rounds in {dt:.1f}s ({dt / n * 1000:.0f} ms/round)")
    print(f"  wins {wins / n:.1%}  draws {draws / n:.1%}")
    print(f"  mean score diff {statistics.mean(diffs):+.1f} ± {statistics.stdev(diffs) / n ** 0.5 if n > 1 else 0:.1f}")


if __name__ == "__main__":
    main()
