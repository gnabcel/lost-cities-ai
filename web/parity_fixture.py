"""Dumps states and full rounds from the Python engine so web/test_parity.js can
check that the JavaScript port computes the same legal moves, network inputs and
scores.

    .venv/bin/python web/parity_fixture.py && node web/test_parity.js
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lost_cities.engine import GameState  # noqa: E402
from lost_cities.ppo import PolicyBot, features_batch  # noqa: E402


def state_json(s):
    return {"deck": s.deck, "hands": s.hands, "expeditions": s.expeditions, "discards": s.discards,
            "current": s.current, "phase": s.phase, "justDiscarded": s.just_discarded,
            "known": [sorted(k) for k in s.known], "done": s.done}


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sampler = PolicyBot.load(os.path.join(root, "models", "champion.pt"))
    sampler.greedy = False
    rng = random.Random(1)
    states, rounds = [], []
    for g in range(40):
        s = GameState.new_game(rng, starting_player=g % 2)
        start, actions, legal = state_json(s.clone()), [], []
        while not s.done:
            if rng.random() < 0.08:
                states.append(s.clone())
            legal.append(s.legal_actions())
            a = sampler.act(s, rng)
            actions.append(a)
            s.step(a)
        rounds.append({"start": start, "actions": actions, "legal": legal, "scores": [s.score(0), s.score(1)]})
    players = [rng.randrange(2) for _ in states]
    feats = features_batch(states, players, 2)
    out = {
        "states": [dict(state_json(s), player=p, features=f.tolist(), legal=s.legal_actions())
                   for s, p, f in zip(states, players, feats)],
        "rounds": rounds,
    }
    path = os.path.join(root, "web", "parity_fixture.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"{len(states)} states and {len(rounds)} rounds -> {path}")


if __name__ == "__main__":
    main()
