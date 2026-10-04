"""Expert iteration (AlphaZero style): search -> distillation -> repeat.

Each iteration:
  1. Generates games in parallel with RolloutSearchBot playing against itself.
     For each decision it stores the features, the target distribution (softmax
     of each candidate's mean difference / --temp) and the round result.
  2. Trains the network (starting from the current base) to imitate that
     distribution and to predict the result.
  3. Pits the new network against the base. If it wins, it becomes the base.

Checkpoints are saved to runs/<run>/u<iter>.pt (the tracker and the UI pick them up).

    .venv/bin/python -m lost_cities.expert_iter --init runs/v4a/u01050.pt --iters 5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

from .bots import HeuristicBot
from .engine import DRAW_DECK, DRAW_DISCARD_OFFSET, NUM_ACTIONS, PHASE_DRAW, GameState, can_play_on
from .ppo import REWARD_SCALE, PolicyBot, PolicyNet, features_batch, legal_mask_batch, play_eval_rounds
from .search import MAX_GAME_PLIES, RolloutSearchBot, load_opponent, parse_mix


def _gen_worker(job):
    model, n_rounds, seed, dets, top_k, temp, eps, rollout_mix, game_mix, prior, depth = job
    torch.set_num_threads(1)
    loaded = {}
    get = lambda spec: loaded.setdefault(spec, load_opponent(spec))
    bot = RolloutSearchBot(PolicyBot.load(model), dets=dets, top_k=top_k, depth=depth,
                           opponents=[(get(sp), w) for sp, w in parse_mix(rollout_mix)])
    game_pool = parse_mix(game_mix)
    game_specs = ["self"] + [sp for sp, _ in game_pool]
    game_weights = [1.0 - sum(w for _, w in game_pool)] + [w for _, w in game_pool]
    feat = bot.net.feat
    rng = random.Random(seed)
    X, M, P, V = [], [], [], []
    for r in range(n_rounds):
        s = GameState.new_game(rng, starting_player=r % 2)
        spec = rng.choices(game_specs, game_weights)[0]
        # in self-play both seats are searched (and stored); against another opponent, only one
        search_seats = {0, 1} if spec == "self" else {r // 2 % 2}
        opp = None if spec == "self" else get(spec)
        rows = []
        plies = 0
        while not s.done:
            if s.current not in search_seats:
                a = opp.act(s, rng)
                plies += 1
                if plies > MAX_GAME_PLIES and s.phase == PHASE_DRAW:
                    a = DRAW_DECK
                s.step(a)
                continue
            cands, means = bot.search(s, rng)
            # with prior > 0: score = mean + prior * log(network prob), so search
            # noise doesn't pull the network away from its own move without evidence
            score = means + prior * np.log(np.maximum(bot.last_probs, 1e-6)) if prior > 0 else means
            target = np.zeros(NUM_ACTIONS, dtype=np.float32)
            if len(cands) == 1:
                w = np.ones(1)
            else:
                z = (score - score.max()) / temp
                w = np.exp(z) / np.exp(z).sum()
            target[cands] = w
            X.append(features_batch([s], [s.current], feat)[0].astype(np.float16))
            M.append(legal_mask_batch([s])[0])
            P.append(target.astype(np.float16))
            rows.append(s.current)
            a = cands[int(np.argmax(score))]
            if len(cands) > 1 and rng.random() < eps:
                a = rng.choices(cands, weights=w)[0]
            plies += 1
            if plies > MAX_GAME_PLIES and s.phase == PHASE_DRAW:
                a = DRAW_DECK
            s.step(a)
        V += [s.score_diff(p) / REWARD_SCALE for p in rows]
    return np.stack(X), np.stack(M), np.stack(P), np.array(V, dtype=np.float32)


def generate(model, rounds, procs, seed, dets, top_k, temp, eps, rollout_mix, game_mix, prior=0.0, depth=0):
    per = max(1, rounds // procs)
    jobs = [(model, per, seed * 1000 + k, dets, top_k, temp, eps, rollout_mix, game_mix, prior, depth) for k in range(procs)]
    with Pool(procs) as pool:
        parts = pool.map(_gen_worker, jobs)
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def distill(init_path, data, device, epochs, lr, batch, seed, value_epochs=None, value_probe=0, scratch=False):
    """Fits the network to the search targets. The value head is only trained
    together with the trunk for the first `value_epochs` epochs (default: all):
    with many epochs it memorizes each round's result, because positions from
    the same round share the same target. With `value_probe` > 0, at the end
    only the value output layer is refit on top of the already trained trunk
    (few parameters: it can't memorize).
    With `scratch`, the network starts from scratch (same architecture as
    init_path) and the lr follows a cycle (OneCycle). `data` is a tuple
    (X, M, P, V) or a list of tuples, concatenated directly on the GPU to avoid
    duplicating them in RAM."""
    parts = data if isinstance(data, list) else [data]
    Xt, Mt, Pt, Vt = (torch.cat([torch.from_numpy(d[i]).to(device) for d in parts]) for i in range(4))
    ck = torch.load(init_path, map_location=device)
    if scratch:
        torch.manual_seed(seed)
    net = PolicyNet(ck.get("hidden", 512), ck.get("feat", 1)).to(device)
    if not scratch:
        net.load_state_dict(ck["model"])
    n = len(Vt)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, n // 20)
    val, tr = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = None
    if scratch:
        steps = epochs * ((len(tr) + batch - 1) // batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)

    def losses(idx):
        logits, v = net(Xt[idx].float(), Mt[idx])
        logp = logits.log_softmax(-1).masked_fill(~Mt[idx], 0)
        pol = -(Pt[idx].float() * logp).sum(-1).mean()
        return pol, F.mse_loss(v, Vt[idx]), logits

    for ep in range(epochs):
        net.train()
        order = torch.from_numpy(rng.permutation(tr)).to(device)
        for s in range(0, len(order), batch):
            pol, vl, _ = losses(order[s:s + batch])
            loss = pol + 0.5 * vl if value_epochs is None or ep < value_epochs else pol
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
        net.eval()
        with torch.no_grad():
            idx = torch.from_numpy(val).to(device)
            pol, vl, logits = losses(idx)
            agree = (logits.argmax(-1) == Pt[idx].float().argmax(-1)).float().mean().item()
        print(f"    epoch {ep + 1}: policy loss {pol.item():.4f}  value {vl.item():.4f}  agrees with search {agree:.1%}", flush=True)
    if value_probe > 0:
        net.eval()
        with torch.no_grad():
            H = torch.cat([net.body(Xt[k:k + 8192].float()) for k in range(0, n, 8192)])
        opt_v = torch.optim.Adam(net.v.parameters(), lr=1e-3)
        tr_t, val_t = torch.from_numpy(tr).to(device), torch.from_numpy(val).to(device)
        for ep in range(value_probe):
            order = tr_t[torch.from_numpy(rng.permutation(len(tr))).to(device)]
            for s in range(0, len(order), batch):
                idx = order[s:s + batch]
                vl = F.mse_loss(net.v(H[idx]).squeeze(-1), Vt[idx])
                opt_v.zero_grad()
                vl.backward()
                opt_v.step()
        with torch.no_grad():
            vl = F.mse_loss(net.v(H[val_t]).squeeze(-1), Vt[val_t])
        print(f"    value layer refit ({value_probe} epochs): value {vl.item():.4f}", flush=True)
    return net, ck


@torch.no_grad()
def style_vs_heuristic(net, device, rounds=1000, seed=777):
    """Plays the network (greedy) against the heuristic and measures the stalling habit:
    cards drawn from a discard pile that it can't play on an expedition it has started."""
    rng = random.Random(seed)
    me = PolicyBot(net, device, greedy=True)
    heur = HeuristicBot()
    games = [(GameState.new_game(rng, starting_player=(i // 2) % 2), i % 2) for i in range(rounds)]
    useless = 0
    plies = [0] * rounds
    active = list(range(rounds))
    while active:
        mine = [i for i in active if games[i][0].current == games[i][1]]
        for i, a in zip(mine, me.act_batch([games[i][0] for i in mine], rng) if mine else []):
            s, seat = games[i]
            if a > DRAW_DECK:
                col = a - DRAW_DISCARD_OFFSET
                c, exp = s.discards[col][-1], s.expeditions[seat][col]
                useless += not exp or not can_play_on(exp, c)
            s.step(a)
            plies[i] += 1
        moved = set(mine)
        for i in active:
            s, seat = games[i]
            if not s.done and s.current != seat and i not in moved:
                s.step(heur.act(s, rng))
                plies[i] += 1
        active = [i for i in active if not games[i][0].done]
    diffs = np.array([s.score_diff(seat) for s, seat in games])
    return float(diffs.mean()), useless / rounds, float(np.mean(plies))


def head_to_head(net_a, path_b, rounds, seed, device):
    b = PolicyBot.load(path_b, device=device)
    stats = {}
    st = play_eval_rounds(net_a, device, b, rounds, seed, stats)
    d = np.array([s.score_diff(seat) for s, seat in st])
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), float((d > 0).mean()), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="runs/v4a/u01050.pt")
    ap.add_argument("--run", default="ei")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--start-iter", type=int, default=1, help="number of the first iteration (to continue a run)")
    ap.add_argument("--rounds", type=int, default=1400, help="self-play rounds with search per iteration")
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--dets", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=2.0, help="temperature (in points) of the target")
    ap.add_argument("--eps", type=float, default=0.15, help="probability of exploring by sampling from the target")
    ap.add_argument("--prior", type=float, default=0.0,
                    help="weight (in points) of log(network prob) when choosing and building the target; 0 = search only")
    ap.add_argument("--depth", type=int, default=0,
                    help="plies per simulation before using the value head (0 = to the end of the round)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--value-epochs", type=int, default=None, help="epochs during which the value head is trained (default: all)")
    ap.add_argument("--value-probe", type=int, default=0, help="epochs of a final refit of the value layer alone")
    ap.add_argument("--scratch", action="store_true",
                    help="each iteration trains a new network from scratch on the whole buffer (instead of fine-tuning the base)")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--eval-rounds", type=int, default=4000)
    ap.add_argument("--accept-se", type=float, default=0.0,
                    help="accept if the advantage over the base exceeds this many standard errors")
    ap.add_argument("--keep-data", type=int, default=2, help="how many iterations of data are reused")
    ap.add_argument("--rollout-opps", default="heuristic:0.3,runs/v3/u00450.pt:0.2",
                    help="opponents simulated in the search, 'spec:weight,...' (the rest is the network itself)")
    ap.add_argument("--game-opps", default="heuristic:0.3,runs/v3/u00450.pt:0.2",
                    help="real opponents in the generation games (the rest is self-play)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-data", action="store_true", help="saves each iteration's data to runs/<run>/data_*.npz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join("runs", args.run)
    os.makedirs(run_dir, exist_ok=True)
    base = args.init
    history = []
    buffer = []
    if args.save_data and args.keep_data > 1:
        # on relaunch, recover the saved data from the last iterations
        for path in sorted(glob.glob(os.path.join(run_dir, "data_*.npz")))[-(args.keep_data - 1):]:
            z = np.load(path)
            buffer.append(tuple(z[k] for k in ("X", "M", "P", "V")))
            print(f"recovered data: {path} ({len(z['V'])} decisions)", flush=True)
    t0 = time.time()
    # stalling-habit baseline with the initial network
    ck0 = torch.load(base, map_location=device)
    net0 = PolicyNet(ck0.get("hidden", 512), ck0.get("feat", 1)).to(device).eval()
    net0.load_state_dict(ck0["model"])
    h0, u0, p0 = style_vs_heuristic(net0, device)
    print(f"initial {base}: vs heuristic {h0:+.1f} · useless draws {u0:.2f}/round · {p0:.0f} plies/round", flush=True)
    for it in range(args.start_iter, args.start_iter + args.iters):
        print(f"\n=== iteration {it} · base {base} ===", flush=True)
        tg = time.time()
        data = generate(base, args.rounds, args.procs, args.seed * 100 + it, args.dets, args.top_k, args.temp, args.eps,
                        args.rollout_opps, args.game_opps, args.prior, args.depth)
        print(f"  data: {len(data[3])} decisions in {time.time() - tg:.0f}s", flush=True)
        if args.save_data:
            np.savez(os.path.join(run_dir, f"data_{it:05d}.npz"), X=data[0], M=data[1], P=data[2], V=data[3])
        buffer = (buffer + [data])[-args.keep_data:]

        td = time.time()
        net, ck = distill(base, list(buffer), device, args.epochs, args.lr, args.batch, args.seed + it,
                          args.value_epochs, args.value_probe, args.scratch)
        print(f"  distillation: {sum(len(d[3]) for d in buffer)} examples in {time.time() - td:.0f}s", flush=True)

        net.eval()
        d, se, win, hh = head_to_head(net, base, args.eval_rounds, 4242 + it, device)
        h_diff, h_useless, h_plies = style_vs_heuristic(net, device)
        accepted = d > args.accept_se * se
        path = os.path.join(run_dir, f"u{it:05d}.pt")
        torch.save({"model": net.state_dict(), "update": it, "hidden": ck.get("hidden", 512), "feat": ck.get("feat", 1)}, path)
        rec = {"iter": it, "base": base, "vs_base": d, "vs_base_se": se, "vs_base_win": win,
               "vs_heur": h_diff, "useless_draws": h_useless, "plies": h_plies,
               "vs_base_plies": hh["plies"], "vs_base_capped": hh["capped"] / args.eval_rounds,
               "accepted": bool(accepted), "sec": time.time() - t0}
        history.append(rec)
        with open(os.path.join(run_dir, "history.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  vs base: {hh['plies']:.0f} plies/round · {hh['capped'] / args.eval_rounds:.1%} of rounds hit the cap", flush=True)
        print(f"  RESULT it{it}: vs base {d:+.1f} ± {se:.1f} (win {win:.1%}) · vs heuristic {h_diff:+.1f} · "
              f"useless draws {h_useless:.2f}/round · {h_plies:.0f} plies/round · "
              f"{'ACCEPTED' if accepted else 'rejected'} · {(time.time() - t0) / 60:.0f} min", flush=True)
        if accepted:
            base = path


if __name__ == "__main__":
    main()
