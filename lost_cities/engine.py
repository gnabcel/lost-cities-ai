"""Lost Cities engine (classic edition, 5 colors).

Cards: 60 in total, ids 0..59. id = color * 12 + rank
    rank 0..2  -> wager cards (handshakes)
    rank 3..11 -> numbers 2..10

A turn has two phases:
    PLAY phase: play a card to your expedition or discard it onto its color's pile
    DRAW phase: draw from the deck or from the top of a discard pile
                (you can't draw the card you just discarded)

The round ends as soon as the last card of the deck is drawn.

Flat action space (126 actions):
    0..59    play card c to its expedition
    60..119  discard card c
    120      draw from the deck
    121..125 draw from the discard pile of color (a - 121)
"""

from __future__ import annotations

import random
from typing import List, Optional

NUM_COLORS = 5
CARDS_PER_COLOR = 12
NUM_CARDS = NUM_COLORS * CARDS_PER_COLOR
HAND_SIZE = 8
COLOR_NAMES = ["Yellow", "Blue", "White", "Green", "Red"]
COLOR_SHORT = ["Y", "B", "W", "G", "R"]

PLAY_OFFSET = 0
DISCARD_OFFSET = NUM_CARDS
DRAW_DECK = 2 * NUM_CARDS
DRAW_DISCARD_OFFSET = DRAW_DECK + 1
NUM_ACTIONS = DRAW_DISCARD_OFFSET + NUM_COLORS

PHASE_PLAY = 0
PHASE_DRAW = 1


def card_color(c: int) -> int:
    return c // CARDS_PER_COLOR


def card_rank(c: int) -> int:
    return c % CARDS_PER_COLOR


def is_wager(c: int) -> bool:
    return c % CARDS_PER_COLOR < 3


def card_value(c: int) -> int:
    """Numeric value (0 for wagers)."""
    r = c % CARDS_PER_COLOR
    return 0 if r < 3 else r - 1


def card_str(c: int) -> str:
    v = card_value(c)
    return f"{COLOR_SHORT[card_color(c)]}{'$' if v == 0 else v}"


def action_str(a: int) -> str:
    if a < DISCARD_OFFSET:
        return f"play {card_str(a)}"
    if a < DRAW_DECK:
        return f"discard {card_str(a - DISCARD_OFFSET)}"
    if a == DRAW_DECK:
        return "draw from deck"
    return f"draw from discard {COLOR_SHORT[a - DRAW_DISCARD_OFFSET]}"


def score_expedition(cards: List[int]) -> int:
    if not cards:
        return 0
    wagers = 0
    total = 0
    for c in cards:
        v = card_value(c)
        if v == 0:
            wagers += 1
        else:
            total += v
    score = (total - 20) * (wagers + 1)
    if len(cards) >= 8:
        score += 20
    return score


def can_play_on(expedition: List[int], c: int) -> bool:
    if not expedition:
        return True
    last = card_value(expedition[-1])
    v = card_value(c)
    if v == 0:
        return last == 0  # wagers only before any number
    return v > last


class GameState:
    __slots__ = (
        "deck",
        "hands",
        "expeditions",
        "discards",
        "current",
        "phase",
        "just_discarded",
        "known",
        "done",
    )

    def __init__(self):
        self.deck: List[int] = []
        self.hands: List[List[int]] = [[], []]
        self.expeditions: List[List[List[int]]] = [
            [[] for _ in range(NUM_COLORS)] for _ in range(2)
        ]
        self.discards: List[List[int]] = [[] for _ in range(NUM_COLORS)]
        self.current = 0
        self.phase = PHASE_PLAY
        self.just_discarded: Optional[int] = None  # color discarded this turn
        # known[p]: cards in p's hand that the opponent knows (drawn from a discard pile)
        self.known: List[set] = [set(), set()]
        self.done = False

    # ------------------------------------------------------------------ setup
    @classmethod
    def new_game(cls, rng: Optional[random.Random] = None, starting_player: int = 0) -> "GameState":
        rng = rng or random.Random()
        s = cls()
        s.deck = list(range(NUM_CARDS))
        rng.shuffle(s.deck)
        for _ in range(HAND_SIZE):
            for p in (0, 1):
                s.hands[p].append(s.deck.pop())
        s.current = starting_player
        return s

    def clone(self) -> "GameState":
        s = GameState.__new__(GameState)
        s.deck = self.deck[:]
        s.hands = [self.hands[0][:], self.hands[1][:]]
        s.expeditions = [[e[:] for e in self.expeditions[0]], [e[:] for e in self.expeditions[1]]]
        s.discards = [d[:] for d in self.discards]
        s.current = self.current
        s.phase = self.phase
        s.just_discarded = self.just_discarded
        s.known = [set(self.known[0]), set(self.known[1])]
        s.done = self.done
        return s

    # ------------------------------------------------------------------ rules
    def legal_actions(self) -> List[int]:
        if self.done:
            return []
        p = self.current
        if self.phase == PHASE_PLAY:
            actions = []
            seen_wager = [False] * NUM_COLORS
            for c in sorted(self.hands[p]):
                col = c // CARDS_PER_COLOR
                if c % CARDS_PER_COLOR < 3:
                    # wagers of the same color are interchangeable: a single action
                    if seen_wager[col]:
                        continue
                    seen_wager[col] = True
                if can_play_on(self.expeditions[p][col], c):
                    actions.append(PLAY_OFFSET + c)
                actions.append(DISCARD_OFFSET + c)
            return actions
        actions = [DRAW_DECK]
        for col in range(NUM_COLORS):
            if self.discards[col] and col != self.just_discarded:
                actions.append(DRAW_DISCARD_OFFSET + col)
        return actions

    def step(self, a: int) -> None:
        assert not self.done
        p = self.current
        if self.phase == PHASE_PLAY:
            if a < DISCARD_OFFSET:
                c = a
                col = c // CARDS_PER_COLOR
                assert c in self.hands[p] and can_play_on(self.expeditions[p][col], c), action_str(a)
                self.hands[p].remove(c)
                self.expeditions[p][col].append(c)
                self.just_discarded = None
            else:
                c = a - DISCARD_OFFSET
                assert DISCARD_OFFSET <= a < DRAW_DECK and c in self.hands[p], action_str(a)
                col = c // CARDS_PER_COLOR
                self.hands[p].remove(c)
                self.discards[col].append(c)
                self.just_discarded = col
            self.known[p].discard(c)
            self.phase = PHASE_DRAW
            return

        if a == DRAW_DECK:
            self.hands[p].append(self.deck.pop())
        else:
            col = a - DRAW_DISCARD_OFFSET
            assert 0 <= col < NUM_COLORS and self.discards[col] and col != self.just_discarded, action_str(a)
            c = self.discards[col].pop()
            self.hands[p].append(c)
            self.known[p].add(c)
        self.just_discarded = None
        self.phase = PHASE_PLAY
        self.current = 1 - p
        if not self.deck:
            self.done = True

    # ---------------------------------------------------------------- scoring
    def score(self, p: int) -> int:
        return sum(score_expedition(e) for e in self.expeditions[p])

    def score_diff(self, p: int = 0) -> int:
        return self.score(p) - self.score(1 - p)

    # ------------------------------------------------------ hidden information
    def unseen_cards(self, p: int) -> List[int]:
        """Cards p can't see: the opponent's hand (except known cards) + the deck."""
        opp = 1 - p
        return [c for c in self.hands[opp] if c not in self.known[opp]] + self.deck

    def determinize(self, p: int, rng: random.Random) -> "GameState":
        """Copy of the state with the info hidden from p randomly resampled,
        consistent with everything p has observed."""
        s = self.clone()
        opp = 1 - p
        known = [c for c in s.hands[opp] if c in s.known[opp]]
        pool = self.unseen_cards(p)
        rng.shuffle(pool)
        n_hidden = len(s.hands[opp]) - len(known)
        s.hands[opp] = known + pool[:n_hidden]
        s.deck = pool[n_hidden:]
        return s

    def observation(self, p: int) -> List[float]:
        """Observation vector from p's perspective (6*60 + 8 = 368 floats).

        Blocks of 60: own hand, own expeditions, opponent's expeditions,
        cards in discard piles, discard pile tops, known cards in the opponent's hand.
        Then: deck size, phase (2), color that can't be drawn from (5).
        """
        opp = 1 - p
        obs = [0.0] * (6 * NUM_CARDS + 8)
        for c in self.hands[p]:
            obs[c] = 1.0
        for col in range(NUM_COLORS):
            for c in self.expeditions[p][col]:
                obs[NUM_CARDS + c] = 1.0
            for c in self.expeditions[opp][col]:
                obs[2 * NUM_CARDS + c] = 1.0
            pile = self.discards[col]
            for c in pile:
                obs[3 * NUM_CARDS + c] = 1.0
            if pile:
                obs[4 * NUM_CARDS + pile[-1]] = 1.0
        for c in self.known[opp]:
            obs[5 * NUM_CARDS + c] = 1.0
        base = 6 * NUM_CARDS
        obs[base] = len(self.deck) / (NUM_CARDS - 2 * HAND_SIZE)
        obs[base + 1 + self.phase] = 1.0
        if self.just_discarded is not None:
            obs[base + 3 + self.just_discarded] = 1.0
        return obs

    # ---------------------------------------------------------------- display
    def render(self, p: Optional[int] = None) -> str:
        lines = [f"Deck: {len(self.deck)}  |  Turn: P{self.current}  |  Phase: {'play' if self.phase == PHASE_PLAY else 'draw'}"]
        for q in (1, 0):
            exps = "  ".join(
                f"{COLOR_SHORT[col]}:[{' '.join(card_str(c) for c in self.expeditions[q][col])}]"
                for col in range(NUM_COLORS)
            )
            lines.append(f"P{q} ({self.score(q):+d})  {exps}")
        lines.append(
            "Discards: "
            + "  ".join(
                f"{COLOR_SHORT[col]}:{card_str(self.discards[col][-1]) if self.discards[col] else '-'}"
                for col in range(NUM_COLORS)
            )
        )
        for q in (0, 1):
            if p is None or p == q:
                lines.append(f"Hand P{q}: {' '.join(card_str(c) for c in sorted(self.hands[q]))}")
        return "\n".join(lines)
