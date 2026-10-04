import random

from lost_cities.bots import HeuristicBot, RandomBot
from lost_cities.engine import (
    DISCARD_OFFSET,
    DRAW_DECK,
    DRAW_DISCARD_OFFSET,
    HAND_SIZE,
    NUM_CARDS,
    PHASE_DRAW,
    GameState,
    can_play_on,
    score_expedition,
)


def c(color, value):
    """value 0 = wager (uses the first one), 2..10 = number."""
    return color * 12 + (0 if value == 0 else value + 1)


def test_scoring():
    assert score_expedition([]) == 0
    assert score_expedition([c(0, 2)]) == -18
    assert score_expedition([c(0, 0), c(0, 5)]) == (5 - 20) * 2
    three_wagers = [0, 1, 2]
    assert score_expedition(three_wagers) == -80
    full = three_wagers + [c(0, v) for v in (2, 3, 4, 5, 6)]  # 8 cards -> bonus
    assert score_expedition(full) == (20 - 20) * 4 + 20
    assert score_expedition([c(1, v) for v in range(2, 11)]) == 34 + 20


def test_play_order():
    assert can_play_on([], c(0, 0))
    assert can_play_on([c(0, 0)], c(0, 0))
    assert can_play_on([c(0, 3)], c(0, 5))
    assert not can_play_on([c(0, 5)], c(0, 3))
    assert not can_play_on([c(0, 5)], c(0, 5))
    assert not can_play_on([c(0, 2)], c(0, 0))


def test_card_conservation_and_termination():
    rng = random.Random(1)
    for _ in range(50):
        s = GameState.new_game(rng)
        assert len(s.hands[0]) == len(s.hands[1]) == HAND_SIZE
        while not s.done:
            s.step(rng.choice(s.legal_actions()))
            cards = s.deck + s.hands[0] + s.hands[1]
            cards += [x for d in s.discards for x in d]
            cards += [x for p in (0, 1) for e in s.expeditions[p] for x in e]
            assert sorted(cards) == list(range(NUM_CARDS))
        assert not s.deck


def test_cannot_redraw_own_discard():
    s = GameState.new_game(random.Random(2))
    card = s.hands[0][0]
    s.step(DISCARD_OFFSET + card)
    assert s.phase == PHASE_DRAW
    col = card // 12
    assert DRAW_DISCARD_OFFSET + col not in s.legal_actions()
    assert DRAW_DECK in s.legal_actions()


def test_known_cards_and_determinize():
    rng = random.Random(3)
    s = GameState.new_game(rng)
    card = s.hands[0][0]
    s.step(DISCARD_OFFSET + card)
    s.step(DRAW_DECK)
    # P1 draws the discarded card: P0 knows it
    s.step(s.legal_actions()[-1])  # P1 plays/discards something
    if DRAW_DISCARD_OFFSET + card // 12 in s.legal_actions():
        s.step(DRAW_DISCARD_OFFSET + card // 12)
        assert card in s.known[1]
        for _ in range(20):
            d = s.determinize(0, rng)
            assert card in d.hands[1]
            assert d.hands[0] == s.hands[0]
            assert len(d.deck) == len(s.deck)
            assert sorted(d.deck + d.hands[1]) == sorted(s.deck + s.hands[1])


def test_bots_only_legal():
    rng = random.Random(4)
    for bot in (RandomBot(), HeuristicBot()):
        for _ in range(20):
            s = GameState.new_game(rng)
            while not s.done:
                a = bot.act(s, rng)
                assert a in s.legal_actions()
                s.step(a)


def test_observation_size():
    s = GameState.new_game(random.Random(5))
    assert len(s.observation(0)) == 6 * NUM_CARDS + 8
