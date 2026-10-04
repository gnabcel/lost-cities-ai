import random

import pytest

torch = pytest.importorskip("torch")

from lost_cities.engine import GameState  # noqa: E402
from lost_cities.ppo import OBS_DIM, PolicyBot, PolicyNet, VecSelfPlay, features, legal_mask  # noqa: E402
from lost_cities.bots import HeuristicBot  # noqa: E402


def test_features_shape():
    s = GameState.new_game(random.Random(0))
    assert features(s, 0).shape == (OBS_DIM,)


def test_policy_only_legal():
    rng = random.Random(1)
    bot = PolicyBot(PolicyNet(64), greedy=False)
    for _ in range(3):
        s = GameState.new_game(rng)
        while not s.done:
            a = bot.act(s, rng)
            assert legal_mask(s)[a]
            s.step(a)


def test_vec_advance_invariant():
    rng = random.Random(2)
    vec = VecSelfPlay(16, rng)
    vec.opponents = [HeuristicBot(), PolicyBot(PolicyNet(64), greedy=False)]
    vec.opp_weights = [0.5, 0.5]
    for e in vec.envs:
        vec.reset_env(e)
    vec.advance(vec.envs)
    for _ in range(200):
        for e in vec.envs:
            assert e.state.current == e.seat and not e.state.done
            e.state.step(rng.choice(e.state.legal_actions()))
        vec.advance(vec.envs)


def test_features_v2_shape():
    from lost_cities.ppo import OBS_DIMS

    s = GameState.new_game(random.Random(0))
    assert features(s, 0, 2).shape == (OBS_DIMS[2],)
    bot = PolicyBot(PolicyNet(64, feat=2), greedy=False)
    assert legal_mask(s)[bot.act(s, random.Random(0))]


def _features_reference(s, p, version):
    """Original (slow, card-by-card) implementation of the features."""
    import numpy as np

    from lost_cities.bots import _last_value
    from lost_cities.engine import CARDS_PER_COLOR, NUM_CARDS, NUM_COLORS, card_value

    opp = 1 - p
    extra = []
    if version >= 2:
        unseen = [False] * NUM_CARDS
        for c in s.unseen_cards(p):
            unseen[c] = True
    for col in range(NUM_COLORS):
        mine, theirs = s.expeditions[p][col], s.expeditions[opp][col]
        hand = [c for c in s.hands[p] if c // CARDS_PER_COLOR == col]
        extra += [
            _last_value(mine) / 10 if mine else 0.0,
            _last_value(theirs) / 10 if theirs else 0.0,
            sum(card_value(c) for c in mine) / 54,
            sum(card_value(c) for c in theirs) / 54,
            sum(1 for c in mine if card_value(c) == 0) / 3,
            sum(1 for c in theirs if card_value(c) == 0) / 3,
            len(hand) / 8,
            sum(card_value(c) for c in hand) / 54,
        ]
        if version >= 2:
            base = col * CARDS_PER_COLOR
            my_last = _last_value(mine) if mine else 1
            opp_last = _last_value(theirs) if theirs else 1
            mine_up = [card_value(c) for c in range(base + 3, base + 12) if unseen[c] and card_value(c) > my_last]
            opp_up = [card_value(c) for c in range(base + 3, base + 12) if unseen[c] and card_value(c) > opp_last]
            extra += [len(mine_up) / 9, sum(mine_up) / 54, len(opp_up) / 9, sum(opp_up) / 54]
    if version >= 2:
        extra.append(((len(s.deck) + 1) // 2) / 22)
    return np.asarray(s.observation(p) + extra, dtype=np.float32)


def test_batched_features_match_reference():
    import numpy as np

    from lost_cities.ppo import features_batch, legal_mask_batch

    rng = random.Random(7)
    bot = HeuristicBot()
    states, players = [], []
    for _ in range(40):
        s = GameState.new_game(rng)
        while not s.done:
            states.append(s.clone())
            players.append(rng.randrange(2))
            s.step(bot.act(s, rng) if rng.random() < 0.7 else rng.choice(s.legal_actions()))
    for v in (1, 2):
        got = features_batch(states, players, v)
        want = np.stack([_features_reference(s, p, v) for s, p in zip(states, players)])
        np.testing.assert_array_equal(got, want)
    masks = legal_mask_batch(states)
    for s, m in zip(states, masks):
        assert sorted(np.flatnonzero(m).tolist()) == sorted(s.legal_actions())
