"""Local server to play against the bot in the browser.

    .venv/bin/python -m lost_cities.play_server                 # best available checkpoint
    .venv/bin/python -m lost_cities.play_server --model runs/v3/u00500.pt
    .venv/bin/python -m lost_cities.play_server --model heuristic

Open http://localhost:8766/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from .bots import HeuristicBot
from .engine import (
    CARDS_PER_COLOR,
    COLOR_NAMES,
    DISCARD_OFFSET,
    DRAW_DECK,
    DRAW_DISCARD_OFFSET,
    NUM_COLORS,
    PHASE_PLAY,
    GameState,
    card_color,
    card_str,
    card_value,
    is_wager,
    score_expedition,
)

HUMAN, BOT = 0, 1
GAMES_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "human_games.jsonl")
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "play")
SHIPPED_CHAMPION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "champion.pt")


def champion_checkpoint(history="runs/ei/history.jsonl"):
    """The latest accepted expert iteration: each one beat the previous one
    head-to-head. That's a better criterion than the diff vs the heuristic, which
    rewards dragging the round out (it scores points off the heuristic, not off a network)."""
    if not os.path.exists(history):
        return None
    with open(history) as f:
        accepted = [json.loads(l) for l in f if l.strip()]
    accepted = [r for r in accepted if r.get("accepted")]
    if not accepted:
        return None
    path = os.path.join(os.path.dirname(history), f"u{accepted[-1]['iter']:05d}.pt")
    return path if os.path.exists(path) else None


def best_checkpoint(pattern: str) -> str:
    """The expert iteration champion if there is one; else models/champion.pt
    (all a fresh clone ships with); else the checkpoint with the best mean diff
    vs the heuristic per the tracker, across all runs matching `pattern` (e.g. runs/*)."""
    champ = champion_checkpoint()
    if champ:
        return champ
    if os.path.exists(SHIPPED_CHAMPION):
        return SHIPPED_CHAMPION
    best, best_path = None, None
    for run_dir in sorted(glob.glob(pattern)):
        evals_path = os.path.join(run_dir, "evals.jsonl")
        if not os.path.exists(evals_path):
            continue
        with open(evals_path) as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                path = os.path.join(run_dir, f"u{e['update']:05d}.pt")
                if os.path.exists(path) and (best is None or e["diff"] > best):
                    best, best_path = e["diff"], path
    if best_path is None:
        raise SystemExit(f"no evaluated checkpoints in {pattern}")
    return best_path


def card_json(c: int) -> dict:
    return {"id": c, "color": card_color(c), "value": card_value(c), "label": card_str(c)}


def card_name(c: int) -> str:
    color = COLOR_NAMES[card_color(c)].lower()
    return f"{color} wager" if is_wager(c) else f"{color} {card_value(c)}"


def action_text(a: int) -> str:
    if a < DISCARD_OFFSET:
        return f"play {card_name(a)}"
    if a < DRAW_DECK:
        return f"discard {card_name(a - DISCARD_OFFSET)}"
    if a == DRAW_DECK:
        return "draw from the deck"
    return f"draw from the {COLOR_NAMES[a - DRAW_DISCARD_OFFSET].lower()} discard pile"


def describe(a: int, who: str, drawn: int | None = None) -> str:
    """drawn: the card that was drawn, or None when the reader mustn't see it (the bot's deck draws)."""
    if a < DISCARD_OFFSET:
        return f"{who} played {card_name(a)}"
    if a < DRAW_DECK:
        return f"{who} discarded {card_name(a - DISCARD_OFFSET)}"
    if a == DRAW_DECK:
        return f"{who} drew from the deck" if drawn is None else f"{who} drew {card_name(drawn)} from the deck"
    return f"{who} took {card_name(drawn)} from the discard pile"


class Match:
    def __init__(self, bot, model_name: str, rounds: int = 3, human_starts: bool = True):
        self.bot = bot
        self.model_name = model_name
        self.rounds = rounds
        self.rng = random.Random()
        self.round = 0
        self.history = []  # [(human, bot)] per round
        self.first = HUMAN if human_starts else BOT
        self.log = []
        self.last_drawn = None
        self.new_round()

    def new_round(self):
        self.round += 1
        starter = self.first if self.round % 2 == 1 else 1 - self.first
        self.state = GameState.new_game(self.rng, starting_player=starter)
        # record to replay the round later (to analyze your matches)
        self.record = {"start": self.state.clone(), "actions": []}
        self.log.append(f"— Round {self.round}: {'you go' if starter == HUMAN else 'the bot goes'} first —")
        self.last_drawn = None

    def canonical(self, a: int) -> int:
        """Same-color wagers are interchangeable: map to the one the engine accepts."""
        s = self.state
        legal = s.legal_actions()
        if a in legal or a >= DRAW_DECK:
            return a
        c = a if a < DISCARD_OFFSET else a - DISCARD_OFFSET
        offset = a - c
        if is_wager(c):
            for w in sorted(s.hands[s.current]):
                if is_wager(w) and card_color(w) == card_color(c) and offset + w in legal:
                    return offset + w
        return a

    def apply(self, a: int, who: str):
        s = self.state
        p = s.current
        before = set(s.hands[p])
        self.record["actions"].append([p, a])
        s.step(a)
        drawn = next(iter(set(s.hands[p]) - before), None) if a >= DRAW_DECK else None
        if drawn is not None and p == HUMAN:
            self.last_drawn = drawn
        self.log.append(describe(a, who, None if a == DRAW_DECK and p == BOT else drawn))
        if s.done:
            h, b = s.score(HUMAN), s.score(BOT)
            self.history.append((h, b))
            self.log.append(f"End of round {self.round}: you {h:+d}, bot {b:+d}")
            self.save_round(h, b)

    def save_round(self, h: int, b: int):
        st = self.record["start"]
        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model_name,
            "round": self.round,
            "starter": st.current,
            "deck": st.deck,
            "hands": st.hands,
            "actions": self.record["actions"],
            "score_human": h,
            "score_bot": b,
        }
        try:
            with open(GAMES_LOG, "a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError as e:
            print(f"couldn't save the round: {e}", flush=True)

    def human_action(self, a: int):
        s = self.state
        if s.done or s.current != HUMAN:
            raise ValueError("not your turn")
        a = self.canonical(a)
        if a not in s.legal_actions():
            raise ValueError(f"illegal move: {action_text(a)}")
        self.apply(a, "You")

    def bot_step(self):
        s = self.state
        if s.done or s.current != BOT:
            return
        self.apply(self.bot.act(s, self.rng), "Bot")

    def hint(self):
        """What the bot would play in your place (top 3 with probabilities)."""
        s = self.state
        if s.done or s.current != HUMAN or not hasattr(self.bot, "net"):
            return []
        import torch

        from .ppo import features_batch, legal_mask_batch

        with torch.no_grad():
            obs = torch.from_numpy(features_batch([s], [HUMAN], self.bot.net.feat))
            mask = torch.from_numpy(legal_mask_batch([s]))
            logits, value = self.bot.net(obs, mask)
            probs = logits.softmax(-1)[0]
        top = probs.argsort(descending=True)[:3].tolist()
        return {
            "value": round(float(value[0]) * 50, 1),
            "options": [{"action": a, "text": action_text(a), "prob": round(float(probs[a]), 3)} for a in top if probs[a] > 0.005],
        }

    def to_json(self):
        s = self.state
        match_over = s.done and self.round >= self.rounds
        totals = [sum(r[0] for r in self.history), sum(r[1] for r in self.history)]
        hand = sorted(s.hands[HUMAN], key=lambda c: (card_color(c), card_value(c), c))
        return {
            "model": self.model_name,
            "round": self.round,
            "rounds": self.rounds,
            "history": self.history,
            "totals": totals,
            "turn": "human" if s.current == HUMAN else "bot",
            "phase": "play" if s.phase == PHASE_PLAY else "draw",
            "round_over": s.done,
            "match_over": match_over,
            "hand": [card_json(c) for c in hand],
            "last_drawn": self.last_drawn,
            "bot_hand_count": len(s.hands[BOT]),
            "bot_known": [card_json(c) for c in sorted(s.known[BOT])],
            "bot_hand": [card_json(c) for c in sorted(s.hands[BOT], key=lambda c: (card_color(c), card_value(c)))] if s.done else None,
            "expeditions": {
                who: [[card_json(c) for c in s.expeditions[p][col]] for col in range(NUM_COLORS)]
                for who, p in (("human", HUMAN), ("bot", BOT))
            },
            "exp_scores": {
                who: [score_expedition(s.expeditions[p][col]) for col in range(NUM_COLORS)]
                for who, p in (("human", HUMAN), ("bot", BOT))
            },
            "score": {"human": s.score(HUMAN), "bot": s.score(BOT)},
            "discards": [
                {"top": card_json(d[-1]) if d else None, "count": len(d), "cards": [card_json(c) for c in d]}
                for d in s.discards
            ],
            "deck": len(s.deck),
            "just_discarded": s.just_discarded,
            "legal": s.legal_actions() if s.current == HUMAN and not s.done else [],
            "log": self.log[-40:],
        }


class Handler(SimpleHTTPRequestHandler):
    server_version = "LostCities/0.1"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=STATIC_DIR, **kw)

    def log_message(self, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        app = self.server.app
        if self.path.startswith("/api/state"):
            with app["lock"]:
                return self._json(app["match"].to_json())
        if self.path.startswith("/api/hint"):
            with app["lock"]:
                return self._json(app["match"].hint())
        return super().do_GET()

    def do_POST(self):
        app = self.server.app
        n = int(self.headers.get("Content-Length") or 0)
        data = json.loads(self.rfile.read(n) or b"{}")
        with app["lock"]:
            m = app["match"]
            try:
                if self.path == "/api/new":
                    app["reload"]()
                    bot, name = app["strongest"]()
                    app["match"] = Match(bot, name, int(data.get("rounds", 3)),
                                         human_starts=data.get("starts", "human") == "human")
                elif self.path == "/api/action":
                    m.human_action(int(data["action"]))
                elif self.path == "/api/bot":
                    m.bot_step()
                elif self.path == "/api/next_round":
                    if m.state.done and m.round < m.rounds:
                        m.new_round()
                else:
                    return self._json({"error": "not found"}, 404)
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
            return self._json(app["match"].to_json())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best", help="'best' (expert iteration champion, else models/champion.pt, else the tracker's best), path to a .pt, or 'heuristic'")
    ap.add_argument("--run", default="runs/*", help="runs to search for the best checkpoint")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()

    app = {"lock": threading.Lock(), "bot": None, "model_name": None}

    def reload():
        """With --model best, every new match uses the current best checkpoint."""
        if args.model == "heuristic":
            if app["bot"] is None:
                app["bot"], app["model_name"] = HeuristicBot(), "heuristic"
            return
        from .ppo import PolicyBot

        path = os.path.relpath(best_checkpoint(args.run) if args.model == "best" else args.model)
        if path != app["model_name"]:
            app["bot"], app["model_name"] = PolicyBot.load(path, device="cpu", greedy=True), path
            print(f"Bot: {path}", flush=True)

    def strongest():
        """A single difficulty: the champion plus the search setup that beats the bare
        network in the arena (with 16 dets and no prior, search played worse than the network)."""
        bot, name = app["bot"], app["model_name"]
        if hasattr(bot, "net"):
            from .search import RolloutSearchBot

            bot, name = RolloutSearchBot(bot, dets=64, top_k=4, prior=2.0), f"{name} + search"
        return bot, name

    app["reload"] = reload
    app["strongest"] = strongest
    reload()
    app["match"] = Match(*strongest())
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.app = app
    print(f"Open http://localhost:{args.port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
