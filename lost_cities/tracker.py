"""Evaluates every new checkpoint in runs/*/u*.pt against the heuristic and builds
runs/summary.json for the dashboard (dashboard/index.html).

    .venv/bin/python -m lost_cities.tracker            # continuous loop
    .venv/bin/python -m lost_cities.tracker --once     # single pass

Prints a "MILESTONE ..." line when a run beats the heuristic for the first
time (mean difference > 0) or breaks its difference record.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import time

import numpy as np
import torch

from .bots import HeuristicBot
from .engine import card_value
from .ppo import PolicyBot, play_eval_rounds

EVAL_SEED = 777  # different from the training evaluation's seed
BIN = 10
# Real progress of the expert iteration loop: each checkpoint head-to-head against
# a fixed network. The difference vs the heuristic rewards stalling the round and
# is useless for comparing strong networks with each other.
ANCHOR = "runs/ei/u00004.pt"
ANCHOR_ROUNDS = 16000
ANCHOR_SEED = 31337


def update_anchor(summary):
    """Evaluates the new runs/ei checkpoints against ANCHOR and adds the curve
    and the history of accepted iterations to the summary."""
    if not os.path.exists(ANCHOR):
        return
    from .expert_iter import head_to_head

    path_out = "runs/ei/anchor.jsonl"
    evals = []
    if os.path.exists(path_out):
        with open(path_out) as f:
            evals = [json.loads(line) for line in f if line.strip()]
    done = {e["update"] for e in evals}
    anchor_u = int(re.search(r"u(\d+)\.pt$", ANCHOR).group(1))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for path in sorted(glob.glob("runs/ei/u*.pt")):
        u = int(re.search(r"u(\d+)\.pt$", path).group(1))
        if u in done or u <= anchor_u:
            continue
        net = PolicyBot.load(path, device=dev).net.eval()
        d, se, win, _ = head_to_head(net, ANCHOR, ANCHOR_ROUNDS, ANCHOR_SEED, dev)
        e = {"update": u, "diff": d, "diff_se": se, "win": win}
        evals.append(e)
        with open(path_out, "a") as f:
            f.write(json.dumps(e) + "\n")
        print(f"head-to-head ei u{u} vs u{anchor_u}: {d:+.2f} ± {se:.2f}", flush=True)
    history = []
    if os.path.exists("runs/ei/history.jsonl"):
        with open("runs/ei/history.jsonl") as f:
            history = [json.loads(line) for line in f if line.strip()]
    summary["anchor"] = {
        "ref": anchor_u,
        "evals": sorted(evals, key=lambda e: e["update"]),
        "accepted": [anchor_u] + [r["iter"] for r in history if r.get("accepted") and r["iter"] > anchor_u],
    }


def style_stats(state, p):
    """Play-style statistics for p in a finished round."""
    exps = [e for e in state.expeditions[p] if e]
    return {
        "score": state.score(p),
        "expeditions": len(exps),
        "wagers": sum(1 for e in exps for c in e if card_value(c) == 0),
        "cards_played": sum(len(e) for e in exps),
        "bonus8": sum(1 for e in exps if len(e) >= 8),
        "negative_exps": sum(1 for e in state.expeditions[p] if e and sum(card_value(c) for c in e) < 20),
    }


def eval_checkpoint(path, rounds):
    bot = PolicyBot.load(path, device="cpu")
    ckpt_update = torch.load(path, map_location="cpu")["update"]
    states = play_eval_rounds(bot.net, "cpu", HeuristicBot(), rounds, EVAL_SEED)
    diffs = np.array([s.score_diff(seat) for s, seat in states])
    agg = lambda p_of: {k: float(np.mean([style_stats(s, p_of(seat))[k] for s, seat in states]))
                        for k in style_stats(states[0][0], 0)}
    lo, hi = -200, 200
    edges = list(range(lo, hi + BIN, BIN))
    hist, _ = np.histogram(np.clip(diffs, lo, hi - 1), bins=edges)
    return {
        "update": int(ckpt_update),
        "rounds": rounds,
        "win": float((diffs > 0).mean()),
        "draw": float((diffs == 0).mean()),
        "diff": float(diffs.mean()),
        "diff_se": float(diffs.std(ddof=1) / np.sqrt(len(diffs))),
        "diff_median": float(np.median(diffs)),
        "hist_edges": edges,
        "hist": hist.tolist(),
        "bot": agg(lambda seat: seat),
        "heuristic": agg(lambda seat: 1 - seat),
    }


def read_log(run_dir):
    path = os.path.join(run_dir, "log.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "update": int(r["update"]),
                "samples": int(r["samples"]),
                "mean_reward": float(r["mean_reward"]) if r["mean_reward"] != "nan" else None,
                "entropy": float(r["entropy"]),
                "v_loss": float(r["v_loss"]),
                "sec": float(r["sec"]),
            })
    return rows


def write_summary(summary):
    summary["generated"] = time.time()
    tmp = "runs/summary.json.tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f)
    os.replace(tmp, "runs/summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)

    records = {}  # run -> {"best": diff, "beat": bool}
    while True:
        summary = {"generated": time.time(), "runs": {}}
        for run_dir in sorted(glob.glob("runs/*/")):
            run = os.path.basename(run_dir.rstrip("/"))
            evals_path = os.path.join(run_dir, "evals.jsonl")
            evals = []
            if os.path.exists(evals_path):
                with open(evals_path) as f:
                    evals = [json.loads(line) for line in f if line.strip()]
            done = {e["update"] for e in evals}
            rec = records.setdefault(run, {
                "best": max((e["diff"] for e in evals), default=float("-inf")),
                "beat": any(e["diff"] > 0 for e in evals),
            })
            ckpts = sorted(glob.glob(os.path.join(run_dir, "u*.pt")))
            for path in ckpts:
                u = int(re.search(r"u(\d+)\.pt$", path).group(1))
                if u in done:
                    continue
                e = eval_checkpoint(path, args.rounds)
                evals.append(e)
                with open(evals_path, "a") as f:
                    f.write(json.dumps(e) + "\n")
                msg = f"{run} u{u}: win {e['win']:.1%}  diff {e['diff']:+.1f} ± {e['diff_se']:.1f}"
                if e["diff"] > 0 and not rec["beat"]:
                    rec["beat"] = True
                    print(f"MILESTONE {msg}  <-- beats the heuristic for the first time!", flush=True)
                elif e["diff"] > rec["best"]:
                    print(f"RECORD {msg}", flush=True)
                else:
                    print(f"eval {msg}", flush=True)
                rec["best"] = max(rec["best"], e["diff"])
                summary["runs"][run] = {"log": read_log(run_dir), "evals": sorted(evals, key=lambda e: e["update"])}
                write_summary(summary)
            evals.sort(key=lambda e: e["update"])
            summary["runs"][run] = {"log": read_log(run_dir), "evals": evals}
        update_anchor(summary)
        write_summary(summary)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
