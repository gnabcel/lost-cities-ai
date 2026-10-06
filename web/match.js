// A match against the bot, entirely in the browser: a port of the Match class in
// lost_cities/play_server.py. api(path, body) answers the same requests the Python
// server does, so the UI is the same page as play/index.html.
// Requires engine.js and bot.js.
(() => {
const HUMAN = 0, BOT = 1;
const SEARCH = { dets: 64, topK: 4, prior: 2 };

const cardJson = c => ({ id: c, color: LC.cardColor(c), value: LC.cardValue(c), label: LC.cardStr(c) });

function cardName(c) {
  const color = LC.COLOR_NAMES[LC.cardColor(c)].toLowerCase();
  return LC.isWager(c) ? `${color} wager` : `${color} ${LC.cardValue(c)}`;
}

function actionText(a) {
  if (a < LC.DISCARD_OFFSET) return `play ${cardName(a)}`;
  if (a < LC.DRAW_DECK) return `discard ${cardName(a - LC.DISCARD_OFFSET)}`;
  if (a === LC.DRAW_DECK) return 'draw from the deck';
  return `draw from the ${LC.COLOR_NAMES[a - LC.DRAW_DISCARD_OFFSET].toLowerCase()} discard pile`;
}

// drawn: the card that was drawn, or null when the reader mustn't see it (the bot's deck draws)
function describe(a, who, drawn = null) {
  if (a < LC.DISCARD_OFFSET) return `${who} played ${cardName(a)}`;
  if (a < LC.DRAW_DECK) return `${who} discarded ${cardName(a - LC.DISCARD_OFFSET)}`;
  if (a === LC.DRAW_DECK) return drawn === null ? `${who} drew from the deck` : `${who} drew ${cardName(drawn)} from the deck`;
  return `${who} took ${cardName(drawn)} from the discard pile`;
}

const byColorValue = (a, b) => LC.cardColor(a) - LC.cardColor(b) || LC.cardValue(a) - LC.cardValue(b) || a - b;

class Match {
  constructor(bot, modelName, rounds = 3, humanStarts = true) {
    this.bot = bot;
    this.modelName = modelName;
    this.rounds = rounds;
    this.round = 0;
    this.history = [];  // [human, bot] per round
    this.first = humanStarts ? HUMAN : BOT;
    this.log = [];
    this.lastDrawn = null;
    this.newRound();
  }

  newRound() {
    this.round++;
    const starter = this.round % 2 === 1 ? this.first : 1 - this.first;
    this.state = LC.GameState.newGame(Math.random, starter);
    this.log.push(`— Round ${this.round}: ${starter === HUMAN ? 'you go' : 'the bot goes'} first —`);
    this.lastDrawn = null;
  }

  // same-color wagers are interchangeable: map to the one the engine accepts
  canonical(a) {
    const s = this.state, legal = s.legalActions();
    if (legal.includes(a) || a >= LC.DRAW_DECK) return a;
    const c = a < LC.DISCARD_OFFSET ? a : a - LC.DISCARD_OFFSET, offset = a - c;
    if (LC.isWager(c))
      for (const w of s.hands[s.current].slice().sort((x, y) => x - y))
        if (LC.isWager(w) && LC.cardColor(w) === LC.cardColor(c) && legal.includes(offset + w)) return offset + w;
    return a;
  }

  apply(a, who) {
    const s = this.state, p = s.current, before = new Set(s.hands[p]);
    s.step(a);
    const drawn = a >= LC.DRAW_DECK ? s.hands[p].find(c => !before.has(c)) ?? null : null;
    if (drawn !== null && p === HUMAN) this.lastDrawn = drawn;
    this.log.push(describe(a, who, a === LC.DRAW_DECK && p === BOT ? null : drawn));
    if (s.done) {
      const h = s.score(HUMAN), b = s.score(BOT);
      this.history.push([h, b]);
      const sign = x => (x >= 0 ? '+' : '') + x;
      this.log.push(`End of round ${this.round}: you ${sign(h)}, bot ${sign(b)}`);
    }
  }

  humanAction(a) {
    const s = this.state;
    if (s.done || s.current !== HUMAN) throw new Error('not your turn');
    a = this.canonical(a);
    if (!s.legalActions().includes(a)) throw new Error(`illegal move: ${actionText(a)}`);
    this.apply(a, 'You');
  }

  async botStep() {
    const s = this.state;
    if (s.done || s.current !== BOT) return;
    this.apply(await this.bot.search(s, SEARCH), 'Bot');
  }

  // what the bot would play in your place (top 3 with probabilities)
  async hint() {
    const s = this.state;
    if (s.done || s.current !== HUMAN) return [];
    const { probs, value } = await this.bot.policy(s);
    const top = [...probs.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3).filter(([, p]) => p > 0.005);
    return {
      value: Math.round(value * 10) / 10,
      options: top.map(([a, p]) => ({ action: a, text: actionText(a), prob: Math.round(p * 1000) / 1000 })),
    };
  }

  toJson() {
    const s = this.state;
    const sides = [['human', HUMAN], ['bot', BOT]];
    return {
      model: this.modelName,
      round: this.round,
      rounds: this.rounds,
      history: this.history,
      totals: [this.history.reduce((t, r) => t + r[0], 0), this.history.reduce((t, r) => t + r[1], 0)],
      turn: s.current === HUMAN ? 'human' : 'bot',
      phase: s.phase === LC.PHASE_PLAY ? 'play' : 'draw',
      round_over: s.done,
      match_over: s.done && this.round >= this.rounds,
      hand: s.hands[HUMAN].slice().sort(byColorValue).map(cardJson),
      last_drawn: this.lastDrawn,
      bot_hand_count: s.hands[BOT].length,
      bot_known: [...s.known[BOT]].sort((a, b) => a - b).map(cardJson),
      bot_hand: s.done ? s.hands[BOT].slice().sort(byColorValue).map(cardJson) : null,
      expeditions: Object.fromEntries(sides.map(([who, p]) => [who, s.expeditions[p].map(e => e.map(cardJson))])),
      exp_scores: Object.fromEntries(sides.map(([who, p]) => [who, s.expeditions[p].map(LC.scoreExpedition)])),
      score: { human: s.score(HUMAN), bot: s.score(BOT) },
      discards: s.discards.map(d => ({ top: d.length ? cardJson(d[d.length - 1]) : null, count: d.length, cards: d.map(cardJson) })),
      deck: s.deck.length,
      just_discarded: s.justDiscarded,
      legal: s.current === HUMAN && !s.done ? s.legalActions() : [],
      log: this.log.slice(-40),
    };
  }
}

// Same requests as the Python server (see Handler in play_server.py).
const local = { bot: null, match: null, modelName: 'champion (ONNX) + search' };

async function api(path, body) {
  if (!local.bot) {
    local.bot = await Bot.load('model.onnx');
    local.match = new Match(local.bot, local.modelName);
  }
  const m = local.match;
  if (path === '/api/state') return m.toJson();
  if (path === '/api/hint') return m.hint();
  if (path === '/api/new') {
    local.match = new Match(local.bot, local.modelName, +(body.rounds ?? 3), (body.starts ?? 'human') === 'human');
  } else if (path === '/api/action') m.humanAction(+body.action);
  else if (path === '/api/bot') await m.botStep();
  else if (path === '/api/next_round') { if (m.state.done && m.round < m.rounds) m.newRound(); }
  else throw new Error('not found');
  return local.match.toJson();
}

globalThis.api = api;
globalThis.Match = Match;
})();
