"""PPO with self-play for Lost Cities.

The agent plays one seat in many games in parallel. Each game's opponent is
drawn when the game starts from: the heuristic bot, the current version of the
network (frozen during the iteration) and earlier snapshots (the "league").

The reward is the point difference at the end of the round / REWARD_SCALE.

    .venv/bin/python -m lost_cities.ppo --run first --updates 2000
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bots import Bot, HeuristicBot
from .engine import CARDS_PER_COLOR, DRAW_DECK, NUM_ACTIONS, NUM_CARDS, NUM_COLORS, PHASE_DRAW, GameState, card_value

# Two greedy policies can get stuck in a loop (discarding / picking up the same
# card forever). Past this ply cap, the draw phase always draws from the
# deck, so the round is guaranteed to end.
MAX_GAME_PLIES = 400

REWARD_SCALE = 50.0
OBS_DIMS = {
    1: 6 * NUM_CARDS + 8 + NUM_COLORS * 8,
    2: 6 * NUM_CARDS + 8 + NUM_COLORS * 12 + 1,
}
OBS_DIM = OBS_DIMS[1]


RAW_DIM = 6 * NUM_CARDS + 8
_BASE = 6 * NUM_CARDS
_VALUES = np.array([card_value(c) for c in range(NUM_CARDS)], dtype=np.float64).reshape(NUM_COLORS, CARDS_PER_COLOR)
_IS_NUMBER = _VALUES > 0


def raw_obs_batch(states: List[GameState], players: List[int]) -> np.ndarray:
    """Same as GameState.observation(p) but for a batch, without intermediate lists."""
    out = np.zeros((len(states), RAW_DIM), dtype=np.float64)
    idx = []
    for b, (s, p) in enumerate(zip(states, players)):
        o = b * RAW_DIM
        opp = 1 - p
        idx += [o + c for c in s.hands[p]]
        for col in range(NUM_COLORS):
            idx += [o + NUM_CARDS + c for c in s.expeditions[p][col]]
            idx += [o + 2 * NUM_CARDS + c for c in s.expeditions[opp][col]]
            pile = s.discards[col]
            if pile:
                idx += [o + 3 * NUM_CARDS + c for c in pile]
                idx.append(o + 4 * NUM_CARDS + pile[-1])
        idx += [o + 5 * NUM_CARDS + c for c in s.known[opp]]
        idx.append(o + _BASE + 1 + s.phase)
        if s.just_discarded is not None:
            idx.append(o + _BASE + 3 + s.just_discarded)
        out[b, _BASE] = len(s.deck) / (NUM_CARDS - 16)
    out.flat[idx] = 1.0
    return out


def features_batch(states: List[GameState], players: List[int], version: int = 1) -> np.ndarray:
    """Raw engine observation + per-color features that help learning go faster.

    v2 adds, per color, the unseen cards (deck + opponent's hidden hand) that are
    still useful to each player, and the number of turns p has left.
    The derived features are computed vectorized from the raw bits.
    """
    raw = raw_obs_batch(states, players)
    B = len(states)
    blk = lambda i: raw[:, i * NUM_CARDS:(i + 1) * NUM_CARDS].reshape(B, NUM_COLORS, CARDS_PER_COLOR)
    hand, mine, theirs = blk(0), blk(1), blk(2)
    # expeditions are increasing, so the last number is the max (1 if there are no numbers)
    last_m = np.maximum((mine * _VALUES).max(-1), 1)
    last_t = np.maximum((theirs * _VALUES).max(-1), 1)
    per_color = [
        last_m / 10 * mine.any(-1),
        last_t / 10 * theirs.any(-1),
        (mine * _VALUES).sum(-1) / 54,
        (theirs * _VALUES).sum(-1) / 54,
        mine[..., :3].sum(-1) / 3,
        theirs[..., :3].sum(-1) / 3,
        hand.sum(-1) / 8,
        (hand * _VALUES).sum(-1) / 54,
    ]
    parts = [raw]
    if version >= 2:
        # the cards form a partition: what p can't see is everything else
        unseen = 1 - hand - mine - theirs - blk(3) - blk(5)
        up_m = unseen * _IS_NUMBER * (_VALUES > last_m[..., None])
        up_t = unseen * _IS_NUMBER * (_VALUES > last_t[..., None])
        per_color += [up_m.sum(-1) / 9, (up_m * _VALUES).sum(-1) / 54,
                      up_t.sum(-1) / 9, (up_t * _VALUES).sum(-1) / 54]
    parts.append(np.stack(per_color, -1).reshape(B, -1))
    if version >= 2:
        deck = np.rint(raw[:, _BASE] * (NUM_CARDS - 16))
        parts.append((((deck + 1) // 2) / 22)[:, None])
    return np.concatenate(parts, 1).astype(np.float32)


def features(s: GameState, p: int, version: int = 1) -> np.ndarray:
    return features_batch([s], [p], version)[0]


def legal_mask_batch(states: List[GameState]) -> np.ndarray:
    m = np.zeros((len(states), NUM_ACTIONS), dtype=bool)
    idx = []
    for b, s in enumerate(states):
        o = b * NUM_ACTIONS
        idx += [o + a for a in s.legal_actions()]
    m.flat[idx] = True
    return m


def legal_mask(s: GameState) -> np.ndarray:
    return legal_mask_batch([s])[0]


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 512, feat: int = 1):
        super().__init__()
        self.feat = feat
        self.body = nn.Sequential(
            nn.Linear(OBS_DIMS[feat], hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, NUM_ACTIONS)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs, mask):
        h = self.body(obs)
        logits = self.pi(h).masked_fill(~mask, -1e9)
        return logits, self.v(h).squeeze(-1)


class PolicyBot(Bot):
    """Uses a trained network as a bot (for the arena or as an opponent)."""

    name = "ppo"

    def __init__(self, net: PolicyNet, device="cpu", greedy: bool = True):
        self.net = net.to(device).eval()
        self.device = device
        self.greedy = greedy

    @classmethod
    def load(cls, path: str, device="cpu", greedy: bool = True) -> "PolicyBot":
        ckpt = torch.load(path, map_location=device)
        net = PolicyNet(ckpt.get("hidden", 512), ckpt.get("feat", 1))
        net.load_state_dict(ckpt["model"])
        return cls(net, device, greedy)

    @torch.no_grad()
    def act_batch(self, states: List[GameState], rng: random.Random) -> List[int]:
        obs = torch.from_numpy(features_batch(states, [s.current for s in states], self.net.feat)).to(self.device)
        mask = torch.from_numpy(legal_mask_batch(states)).to(self.device)
        logits, _ = self.net(obs, mask)
        if self.greedy:
            return logits.argmax(-1).tolist()
        return torch.distributions.Categorical(logits=logits).sample().tolist()

    def act(self, state, rng):
        return self.act_batch([state], rng)[0]


# ---------------------------------------------------------------------- envs
class Env:
    __slots__ = ("state", "seat", "opp")

    def __init__(self):
        self.state = None
        self.seat = 0
        self.opp = 0


class VecSelfPlay:
    """N games in parallel. Invariant after `advance`: it is the agent's turn in all of them."""

    def __init__(self, n: int, rng: random.Random):
        self.envs = [Env() for _ in range(n)]
        self.rng = rng
        self.opponents: List[Bot] = []
        self.opp_weights: List[float] = []

    def reset_env(self, e: Env):
        e.seat = self.rng.randrange(2)
        e.state = GameState.new_game(self.rng, starting_player=self.rng.randrange(2))
        e.opp = self.rng.choices(range(len(self.opponents)), self.opp_weights)[0]

    def advance(self, envs: List[Env]) -> List[float]:
        """Plays the opponent's turns until it is the agent's turn.
        Returns the terminal reward (or None) of each env and resets the finished ones."""
        rewards = [None] * len(envs)
        pending = list(range(len(envs)))
        while pending:
            still = []
            by_opp = {}
            for i in pending:
                e = envs[i]
                s = e.state
                if s.done:
                    rewards[i] = s.score_diff(e.seat) / REWARD_SCALE
                    self.reset_env(e)
                    s = e.state
                if s.current != e.seat:
                    by_opp.setdefault(e.opp, []).append(i)
                    still.append(i)
            for o, idxs in by_opp.items():
                bot = self.opponents[o]
                if isinstance(bot, PolicyBot):
                    acts = bot.act_batch([envs[i].state for i in idxs], self.rng)
                else:
                    acts = [bot.act(envs[i].state, self.rng) for i in idxs]
                for i, a in zip(idxs, acts):
                    envs[i].state.step(a)
            pending = still
        return rewards


# ------------------------------------------------------------------- evaluate
@torch.no_grad()
def evaluate(net: PolicyNet, device, opponent: Bot, rounds: int, seed: int = 12345):
    """Win rate and mean difference of the network (greedy) against `opponent`."""
    states = play_eval_rounds(net, device, opponent, rounds, seed)
    diffs = np.array([s.score_diff(seat) for s, seat in states])
    net.train()
    return float((diffs > 0).mean()), float(diffs.mean())


@torch.no_grad()
def play_eval_rounds(net: PolicyNet, device, opponent: Bot, rounds: int, seed: int = 12345, stats: dict | None = None):
    """Plays `rounds` rounds (greedy network vs `opponent`), alternating seat and who
    starts. Returns [(final_state, network_seat)]. If `stats` is given, it
    records how many rounds hit the ply cap."""
    rng = random.Random(seed)
    me = PolicyBot(net, device, greedy=True)
    states = []
    for i in range(rounds):
        states.append((GameState.new_game(rng, starting_player=(i // 2) % 2), i % 2))
    plies = [0] * rounds

    def step(i, a):
        s = states[i][0]
        plies[i] += 1
        if plies[i] > MAX_GAME_PLIES and s.phase == PHASE_DRAW:
            a = DRAW_DECK
        s.step(a)

    active = list(range(rounds))
    while active:
        mine = [i for i in active if states[i][0].current == states[i][1]]
        theirs = [i for i in active if states[i][0].current != states[i][1]]
        if mine:
            for i, a in zip(mine, me.act_batch([states[i][0] for i in mine], rng)):
                step(i, a)
        if isinstance(opponent, PolicyBot) and theirs:
            for i, a in zip(theirs, opponent.act_batch([states[i][0] for i in theirs], rng)):
                step(i, a)
        else:
            for i in theirs:
                step(i, opponent.act(states[i][0], rng))
        active = [i for i in active if not states[i][0].done]
    if stats is not None:
        stats["capped"] = sum(p > MAX_GAME_PLIES for p in plies)
        stats["plies"] = float(np.mean(plies))
    return states


# ---------------------------------------------------------------------- train
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    run_dir = os.path.join("runs", args.run)
    os.makedirs(run_dir, exist_ok=True)

    net = PolicyNet(args.hidden, args.feat).to(device)
    ref_net = None
    if args.init:
        init = torch.load(args.init, map_location=device)
        assert init.get("feat", 1) == args.feat and init.get("hidden", 512) == args.hidden, "incompatible init"
        net.load_state_dict(init["model"])
        print(f"initial weights from {args.init}")
        if args.ref_kl > 0:
            # frozen reference policy: penalizes drifting away from it early on
            ref_net = copy.deepcopy(net).eval()
            for prm in ref_net.parameters():
                prm.requires_grad_(False)
    obs_dim = OBS_DIMS[args.feat]
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    start_update = 0
    ckpt_path = os.path.join(run_dir, "latest.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_update = ckpt["update"]
        print(f"resuming from update {start_update}")

    heuristic = HeuristicBot()
    # opponent 0: heuristic, opponent 1: the current frozen network, 2..: league snapshots
    frozen = PolicyBot(copy.deepcopy(net), device, greedy=False)
    league: List[PolicyBot] = []
    vec = VecSelfPlay(args.envs, rng)

    def refresh_opponents():
        vec.opponents = [heuristic, frozen] + league
        w_league = args.p_league / len(league) if league else 0.0
        p_self = 1.0 - args.p_heuristic - (args.p_league if league else 0.0)
        vec.opp_weights = [args.p_heuristic, p_self] + [w_league] * len(league)

    refresh_opponents()
    for e in vec.envs:
        vec.reset_env(e)
    vec.advance(vec.envs)

    T, E = args.steps, args.envs
    obs_buf = torch.zeros(T, E, obs_dim, device=device)
    mask_buf = torch.zeros(T, E, NUM_ACTIONS, dtype=torch.bool, device=device)
    act_buf = torch.zeros(T, E, dtype=torch.long, device=device)
    logp_buf = torch.zeros(T, E, device=device)
    val_buf = torch.zeros(T, E, device=device)
    rew_buf = torch.zeros(T, E, device=device)
    done_buf = torch.zeros(T, E, device=device)

    log_path = os.path.join(run_dir, "log.csv")
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if start_update == 0:
        log.writerow(["update", "samples", "episodes", "mean_reward", "pi_loss", "v_loss", "entropy", "win_vs_heur", "diff_vs_heur", "sec"])

    t_start = time.time()
    for update in range(start_update + 1, start_update + args.updates + 1):
        frac = 1.0 - (update - 1) / (start_update + args.updates)
        progress = (update - 1) / (start_update + args.updates)
        kl_coef = args.ref_kl * max(0.0, 1.0 - progress / args.ref_kl_until) if ref_net is not None else 0.0
        for g in opt.param_groups:
            g["lr"] = args.lr * max(frac, 0.1)

        ep_rewards = []
        net.eval()
        for t in range(T):
            obs = features_batch([e.state for e in vec.envs], [e.seat for e in vec.envs], args.feat)
            mask = legal_mask_batch([e.state for e in vec.envs])
            obs_t = torch.from_numpy(obs).to(device)
            mask_t = torch.from_numpy(mask).to(device)
            with torch.no_grad():
                logits, value = net(obs_t, mask_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            obs_buf[t], mask_buf[t], act_buf[t] = obs_t, mask_t, action
            logp_buf[t], val_buf[t] = dist.log_prob(action), value
            for e, a in zip(vec.envs, action.tolist()):
                e.state.step(a)
            rewards = vec.advance(vec.envs)
            r = torch.tensor([x if x is not None else 0.0 for x in rewards], device=device)
            d = torch.tensor([x is not None for x in rewards], dtype=torch.float32, device=device)
            rew_buf[t], done_buf[t] = r, d
            ep_rewards += [x for x in rewards if x is not None]

        # bootstrap + GAE
        with torch.no_grad():
            obs = torch.from_numpy(features_batch([e.state for e in vec.envs], [e.seat for e in vec.envs], args.feat)).to(device)
            mask = torch.from_numpy(legal_mask_batch([e.state for e in vec.envs])).to(device)
            _, next_value = net(obs, mask)
            adv = torch.zeros_like(rew_buf)
            last = torch.zeros(E, device=device)
            for t in reversed(range(T)):
                nv = next_value if t == T - 1 else val_buf[t + 1]
                nonterm = 1.0 - done_buf[t]
                delta = rew_buf[t] + args.gamma * nv * nonterm - val_buf[t]
                last = delta + args.gamma * args.lam * nonterm * last
                adv[t] = last
            ret = adv + val_buf

        # PPO update
        net.train()
        b_obs, b_mask = obs_buf.reshape(T * E, -1), mask_buf.reshape(T * E, -1)
        b_act, b_logp = act_buf.reshape(-1), logp_buf.reshape(-1)
        b_adv, b_ret, b_val = adv.reshape(-1), ret.reshape(-1), val_buf.reshape(-1)
        n = T * E
        for _ in range(args.epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, args.minibatch):
                idx = perm[start:start + args.minibatch]
                logits, v = net(b_obs[idx], b_mask[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_act[idx])
                ratio = (logp - b_logp[idx]).exp()
                a = b_adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pi_loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                v_clipped = b_val[idx] + (v - b_val[idx]).clamp(-args.clip, args.clip)
                v_loss = 0.5 * torch.max(F.mse_loss(v, b_ret[idx], reduction="none"),
                                         F.mse_loss(v_clipped, b_ret[idx], reduction="none")).mean()
                # entropy over legal actions only
                p = logits.softmax(-1)
                ent = -(p * logits.log_softmax(-1)).masked_fill(~b_mask[idx], 0).sum(-1).mean()
                loss = pi_loss + args.vf_coef * v_loss - args.ent_coef * ent
                if kl_coef > 0:
                    with torch.no_grad():
                        ref_logp = ref_net(b_obs[idx], b_mask[idx])[0].log_softmax(-1)
                    kl = (p * (logits.log_softmax(-1) - ref_logp)).masked_fill(~b_mask[idx], 0).sum(-1).mean()
                    loss = loss + kl_coef * kl
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        frozen.net.load_state_dict(net.state_dict())

        if update % args.snapshot_every == 0:
            league.append(PolicyBot(copy.deepcopy(net).eval(), device, greedy=False))
            if len(league) > args.league_size:
                league.pop(0)
            refresh_opponents()

        win, diff = "", ""
        if update % args.eval_every == 0:
            win, diff = evaluate(net, device, heuristic, args.eval_rounds)
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "update": update, "hidden": args.hidden, "feat": args.feat}, ckpt_path)
            torch.save({"model": net.state_dict(), "update": update, "hidden": args.hidden, "feat": args.feat}, os.path.join(run_dir, f"u{update:05d}.pt"))
        el = time.time() - t_start
        mr = float(np.mean(ep_rewards)) * REWARD_SCALE if ep_rewards else float("nan")
        log.writerow([update, update * T * E, len(ep_rewards), f"{mr:.2f}", f"{pi_loss.item():.4f}", f"{v_loss.item():.4f}", f"{ent.item():.3f}", win, diff, f"{el:.0f}"])
        log_f.flush()
        msg = f"u{update:5d}  eps {len(ep_rewards):4d}  mean.diff(mix) {mr:+6.1f}  ent {ent.item():.2f}  vloss {v_loss.item():.3f}  {el:6.0f}s"
        if win != "":
            msg += f"  | vs heuristic: win {win:.1%} diff {diff:+.1f}"
        print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ppo")
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--feat", type=int, default=1, choices=sorted(OBS_DIMS))
    ap.add_argument("--p-heuristic", type=float, default=0.3)
    ap.add_argument("--p-league", type=float, default=0.3)
    ap.add_argument("--snapshot-every", type=int, default=50)
    ap.add_argument("--league-size", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-rounds", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=None, help="checkpoint to initialize the weights from (e.g. from bc.py)")
    ap.add_argument("--ref-kl", type=float, default=0.0, help="weight of the KL toward the initial policy")
    ap.add_argument("--ref-kl-until", type=float, default=0.5, help="fraction of the run at which the KL reaches 0")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
