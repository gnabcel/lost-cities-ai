"""Play-time search on top of a trained policy.

RolloutSearchBot: for the network's best `top_k` moves, it samples `dets`
determinizations (opponent hand + deck consistent with what has been seen),
applies the move and plays out the round: the player itself with the network
(greedy) and the opponent with a model drawn per determinization (the same
network, or other opponents from `opponents`), so the search doesn't assume
the opponent plays like it does. It picks the move with the best mean
difference. The same determinizations (and simulated opponents) are used for
every candidate (common random numbers).

    .venv/bin/python -m lost_cities.search --model runs/v3/u00450.pt --rounds 200
"""

from __future__ import annotations

import argparse
import random
from multiprocessing import Pool

import numpy as np
import torch

from .bots import Bot
from .engine import DRAW_DECK, PHASE_DRAW, GameState

# Two deterministic policies can get stuck in a loop (discarding / picking up
# the same card forever). Past this cap, the draw phase always draws from the
# deck, so the round is guaranteed to end.
MAX_ROLLOUT_PLIES = 200

from .ppo import MAX_GAME_PLIES, REWARD_SCALE, PolicyBot, features_batch, legal_mask_batch  # noqa: E402,F401


def load_opponent(spec: str, device="cpu") -> Bot:
    """'heuristic' or path to a checkpoint (.pt), played greedily."""
    if spec == "heuristic":
        from .bots import HeuristicBot

        return HeuristicBot()
    return PolicyBot.load(spec, device=device, greedy=True)


def parse_mix(spec: str):
    """'heuristic:0.3,runs/v3/u00450.pt:0.2' -> [(spec, weight)]. The rest is 'self'."""
    out = []
    for part in filter(None, (spec or "").split(",")):
        name, w = part.rsplit(":", 1)
        out.append((name, float(w)))
    assert sum(w for _, w in out) <= 1.0 + 1e-9, "opponent weights add up to more than 1"
    return out


class RolloutSearchBot(Bot):
    name = "search"

    def __init__(self, policy: PolicyBot, dets: int = 16, top_k: int = 4, min_prob: float = 0.02,
                 opponents=None, prior: float = 0.0, depth: int = 0):
        """`opponents`: list [(bot, weight)] used to simulate the opponent; the weight
        left over up to 1 goes to the network itself."""
        self.policy = policy
        self.net = policy.net
        self.dets = dets
        self.top_k = top_k
        self.min_prob = min_prob
        self.prior = prior  # points per unit of log(network prob) when choosing
        # depth > 0: simulations are cut off after `depth` plies and the network's
        # value head estimates the rest of the round (less noise, cheaper)
        self.depth = depth
        self.opponents = [b for b, _ in (opponents or [])]
        self.opp_weights = [1.0 - sum(w for _, w in (opponents or []))] + [w for _, w in (opponents or [])]
        self.capped = 0  # rollouts that hit the cap (diagnostic)

    @torch.no_grad()
    def _logits(self, states):
        obs = torch.from_numpy(features_batch(states, [s.current for s in states], self.net.feat))
        mask = torch.from_numpy(legal_mask_batch(states))
        return self.net(obs, mask)[0]

    @torch.no_grad()
    def act(self, state: GameState, rng: random.Random) -> int:
        cands, means = self.search(state, rng)
        if self.prior > 0:
            means = means + self.prior * np.log(np.maximum(self.last_probs, 1e-6))
        return cands[int(np.argmax(means))]

    @torch.no_grad()
    def search(self, state: GameState, rng: random.Random):
        """Returns (candidates, mean point difference of each one).
        With a single candidate it doesn't simulate and its value is 0."""
        probs = self._logits([state])[0].softmax(-1)
        order = probs.argsort(descending=True).tolist()
        cands = [a for a in order[: self.top_k] if probs[a] >= self.min_prob] or order[:1]
        self.last_probs = probs[cands].numpy()  # network prior over the candidates
        if len(cands) == 1:
            return cands, np.zeros(1)
        me = state.current
        dets = [state.determinize(me, rng) for _ in range(self.dets)]
        # 0 = the network itself; k > 0 = self.opponents[k - 1]
        det_opp = rng.choices(range(len(self.opp_weights)), self.opp_weights, k=self.dets) if self.opponents else [0] * self.dets
        games, game_opp = [], []
        for a in cands:
            for d, o in zip(dets, det_opp):
                g = d.clone()
                g.step(a)
                games.append(g)
                game_opp.append(o)
        active = [i for i, g in enumerate(games) if not g.done]
        plies = 0
        while active and (self.depth <= 0 or plies < self.depth):
            plies += 1
            # who decides each move: the network (own side or mirror opponent) or an opponent from the pool
            by_model = {}
            for i in active:
                o = 0 if games[i].current == me else game_opp[i]
                by_model.setdefault(o, []).append(i)
            acts = {}
            for o, idxs in by_model.items():
                sts = [games[i] for i in idxs]
                if o == 0:
                    chosen = self._logits(sts).argmax(-1).tolist()
                elif isinstance(self.opponents[o - 1], PolicyBot):
                    chosen = self.opponents[o - 1].act_batch(sts, rng)
                else:
                    chosen = [self.opponents[o - 1].act(g, rng) for g in sts]
                acts.update(zip(idxs, chosen))
            for i in active:
                g, a = games[i], acts[i]
                if plies > MAX_ROLLOUT_PLIES and g.phase == PHASE_DRAW:
                    a = DRAW_DECK
                g.step(a)
            active = [i for i in active if not games[i].done]
        if plies > MAX_ROLLOUT_PLIES:
            self.capped += 1
        diffs = np.array([g.score_diff(me) if g.done else 0.0 for g in games], dtype=np.float64)
        if active:  # unfinished leaves: value for whoever is to move, seen from `me`
            sts = [games[i] for i in active]
            obs = torch.from_numpy(features_batch(sts, [g.current for g in sts], self.net.feat))
            v = self.net(obs, torch.from_numpy(legal_mask_batch(sts)))[1].double().numpy() * REWARD_SCALE
            diffs[active] = np.where([g.current == me for g in sts], v, -v)
        diffs = diffs.reshape(len(cands), self.dets)
        return cands, diffs.mean(1)


def _worker(job):
    model, opp, start, n, dets, top_k, seed, prior, depth = job
    torch.set_num_threads(1)
    from .bots import HeuristicBot

    pol = PolicyBot.load(model)
    me = RolloutSearchBot(pol, dets=dets, top_k=top_k, prior=prior, depth=depth)
    other = HeuristicBot() if opp == "heuristic" else PolicyBot.load(opp)
    rng = random.Random(seed + start)
    diffs = []
    capped_games = 0
    for i in range(start, start + n):
        # games mirrored per seed: same deal with each seat
        s = GameState.new_game(random.Random(seed * 100003 + i // 2), starting_player=(i // 4) % 2)
        seat = i % 2
        bots = [me, other] if seat == 0 else [other, me]
        plies = 0
        while not s.done:
            a = bots[s.current].act(s, rng)
            plies += 1
            if plies > MAX_GAME_PLIES and s.phase == PHASE_DRAW:
                a = DRAW_DECK
            s.step(a)
        capped_games += plies > MAX_GAME_PLIES
        diffs.append(s.score_diff(seat))
    return diffs, me.capped, capped_games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/v3/u00450.pt")
    ap.add_argument("--opponent", default=None, help="'heuristic' or .pt path (default: the same network without search)")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--dets", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--prior", type=float, default=0.0, help="weight of log(network prob) when choosing")
    ap.add_argument("--depth", type=int, default=0, help="plies per simulation before using the value head (0 = to the end)")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()
    opp = args.opponent or args.model
    per = args.rounds // args.procs
    jobs = [(args.model, opp, k * per, per, args.dets, args.top_k, args.seed, args.prior, args.depth) for k in range(args.procs)]
    with Pool(args.procs) as pool:
        parts = pool.map(_worker, jobs)
    diffs = np.array([d for part, _, _ in parts for d in part])
    print(f"rollouts that hit the cap: {sum(c for _, c, _ in parts)} · real rounds that hit it: {sum(g for _, _, g in parts)}")
    print(f"search({args.model}, dets={args.dets}, top_k={args.top_k}, prior={args.prior}, depth={args.depth}) vs {opp}: "
          f"{len(diffs)} rounds  win {(diffs > 0).mean():.1%}  diff {diffs.mean():+.1f} ± {diffs.std(ddof=1) / len(diffs) ** 0.5:.1f}")


if __name__ == "__main__":
    main()
