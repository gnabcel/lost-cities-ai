"""Reference bots. All of them use only the information visible to the player to move."""

from __future__ import annotations

import random
from typing import List

from .engine import (
    CARDS_PER_COLOR,
    DISCARD_OFFSET,
    DRAW_DECK,
    DRAW_DISCARD_OFFSET,
    NUM_COLORS,
    PHASE_PLAY,
    GameState,
    can_play_on,
    card_value,
)


class Bot:
    name = "bot"

    def act(self, state: GameState, rng: random.Random) -> int:
        raise NotImplementedError


class RandomBot(Bot):
    name = "random"

    def act(self, state, rng):
        return rng.choice(state.legal_actions())


def _last_value(expedition: List[int]) -> int:
    """Last number played (1 if the expedition is empty or has only wagers)."""
    for c in reversed(expedition):
        v = card_value(c)
        if v:
            return v
    return 1


class HeuristicBot(Bot):
    """Simple rules of an intermediate player.

    - On started expeditions, plays the lowest card if the gap is small.
    - Starts expeditions only with enough points of that color in hand.
    - Discards whatever is least useful to itself and to the opponent.
    - Draws from the discard pile only if the card fits with almost no gap.
    """

    name = "heuristic"

    def __init__(self, max_gap: int = 2, start_threshold: int = 18):
        self.max_gap = max_gap
        self.start_threshold = start_threshold

    def _hand_by_color(self, hand):
        by_color = [[] for _ in range(NUM_COLORS)]
        for c in hand:
            by_color[c // CARDS_PER_COLOR].append(c)
        for cs in by_color:
            cs.sort()
        return by_color

    def act(self, state: GameState, rng: random.Random) -> int:
        p = state.current
        opp = 1 - p
        my_exp = state.expeditions[p]
        opp_exp = state.expeditions[opp]
        deck_left = len(state.deck)
        my_turns_left = (deck_left + 1) // 2

        if state.phase != PHASE_PLAY:
            best, best_gap = DRAW_DECK, None
            for a in state.legal_actions():
                if a == DRAW_DECK:
                    continue
                col = a - DRAW_DISCARD_OFFSET
                c = state.discards[col][-1]
                if not my_exp[col] or not can_play_on(my_exp[col], c):
                    continue
                gap = card_value(c) - _last_value(my_exp[col]) - 1 if card_value(c) else 0
                if gap <= 1 and (best_gap is None or gap < best_gap):
                    best, best_gap = a, gap
            return best

        by_color = self._hand_by_color(state.hands[p])
        legal = set(state.legal_actions())

        # 1) Play on expeditions already started
        best_play, best_gap = None, None
        for col in range(NUM_COLORS):
            if not my_exp[col] or not by_color[col]:
                continue
            last = _last_value(my_exp[col])
            playable = [c for c in by_color[col] if card_value(c) > last]
            if not playable:
                continue
            c = playable[0]
            gap = card_value(c) - last - 1
            # near the end, any playable card adds points
            limit = self.max_gap if len(by_color[col]) < my_turns_left else 99
            if gap <= limit and (best_gap is None or gap < best_gap):
                best_play, best_gap = c, gap
        if best_play is not None:
            return best_play

        # 2) Start a new expedition
        best_start, best_sum = None, 0
        for col in range(NUM_COLORS):
            if my_exp[col] or not by_color[col]:
                continue
            numbers = [c for c in by_color[col] if card_value(c)]
            wagers = [c for c in by_color[col] if not card_value(c)]
            total = sum(card_value(c) for c in numbers)
            if len(by_color[col]) > my_turns_left:
                continue
            threshold = self.start_threshold + (6 if deck_left < 20 else 0)
            if total >= threshold and total > best_sum:
                best_sum = total
                # open with a wager if the hand is strong and plenty of deck is left
                if wagers and total >= threshold + 6 and deck_left > 20:
                    best_start = wagers[0]
                else:
                    best_start = numbers[0]
        if best_start is not None and best_start in legal:
            return best_start

        # 3) Discard the least valuable card
        best_disc, best_cost = None, None
        for c in state.hands[p]:
            if DISCARD_OFFSET + c not in legal:
                continue
            col = c // CARDS_PER_COLOR
            v = card_value(c)
            # value to me
            if my_exp[col]:
                mine = 0 if not can_play_on(my_exp[col], c) else 6 + v
            else:
                mine = sum(card_value(x) for x in by_color[col]) / 4 + (3 if v == 0 else 0)
            # value to the opponent
            if opp_exp[col]:
                theirs = 0 if not can_play_on(opp_exp[col], c) else 5 + v
            else:
                theirs = v / 3
            cost = mine + theirs
            if best_disc is None or cost < best_cost:
                best_disc, best_cost = c, cost
        return DISCARD_OFFSET + best_disc
