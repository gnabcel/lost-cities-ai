"""Single-Observer Information Set MCTS (Cowling, Powley & Whitehouse 2012).

Each iteration samples a determinization (opponent's hand + deck order
consistent with what the player sees) and walks a single shared tree,
using availability counts in UCB because the legal actions change
between determinizations.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Optional

from .bots import Bot, HeuristicBot
from .engine import GameState


class _Node:
    __slots__ = ("parent", "action", "player", "children", "visits", "avail", "value")

    def __init__(self, parent: Optional["_Node"], action: Optional[int], player: int):
        self.parent = parent
        self.action = action
        self.player = player  # who took the action leading to this node
        self.children: Dict[int, _Node] = {}
        self.visits = 0
        self.avail = 0
        self.value = 0.0


def _outcome(state: GameState, player: int, scale: float) -> float:
    """Outcome in [-1, 1] from the perspective of `player`."""
    return math.tanh(state.score_diff(player) / scale)


class ISMCTSBot(Bot):
    name = "ismcts"

    def __init__(self, iterations: int = 500, c: float = 0.7, rollout_bot: Optional[Bot] = None, scale: float = 40.0):
        self.iterations = iterations
        self.c = c
        self.rollout_bot = rollout_bot or HeuristicBot()
        self.scale = scale

    def act(self, state: GameState, rng: random.Random) -> int:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        me = state.current
        root = _Node(None, None, 1 - me)

        for _ in range(self.iterations):
            s = state.determinize(me, rng)
            node = root

            # selection + expansion
            while not s.done:
                acts = s.legal_actions()
                untried = [a for a in acts if a not in node.children]
                for a in acts:
                    ch = node.children.get(a)
                    if ch is not None:
                        ch.avail += 1
                if untried:
                    a = rng.choice(untried)
                    player = s.current
                    s.step(a)
                    child = _Node(node, a, player)
                    child.avail = 1
                    node.children[a] = child
                    node = child
                    break
                log_parent = None
                best, best_ucb = None, -1e9
                for a in acts:
                    ch = node.children[a]
                    ucb = ch.value / ch.visits + self.c * math.sqrt(math.log(ch.avail) / ch.visits)
                    if ucb > best_ucb:
                        best, best_ucb = ch, ucb
                s.step(best.action)
                node = best

            # rollout with the heuristic policy
            while not s.done:
                s.step(self.rollout_bot.act(s, rng))

            # backprop
            r0 = _outcome(s, 0, self.scale)
            while node is not None:
                node.visits += 1
                node.value += r0 if node.player == 0 else -r0
                node = node.parent

        best = max(root.children.values(), key=lambda ch: ch.visits)
        return best.action


class FlatMCBot(Bot):
    """Flat Monte Carlo over determinizations (one-level PIMC).

    Each legal action is evaluated on the same `samples` determinizations
    (common random numbers, less variance), finishing the round with the
    heuristic policy. It is a policy improvement over the heuristic.
    """

    name = "flatmc"

    def __init__(self, samples: int = 16, rollout_bot: Optional[Bot] = None, scale: float = 40.0):
        self.samples = samples
        self.rollout_bot = rollout_bot or HeuristicBot()
        self.scale = scale

    def act(self, state: GameState, rng: random.Random) -> int:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        me = state.current
        dets = [(state.determinize(me, rng), rng.getrandbits(32)) for _ in range(self.samples)]
        best, best_val = legal[0], -1e9
        for a in legal:
            total = 0.0
            for d, seed in dets:
                s = d.clone()
                s.step(a)
                r = random.Random(seed)
                while not s.done:
                    s.step(self.rollout_bot.act(s, r))
                total += s.score_diff(me)
            if total > best_val:
                best, best_val = a, total
        return best
