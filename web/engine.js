// Lost Cities engine, a line-by-line port of lost_cities/engine.py, plus the
// network's input features (ppo.features_batch, version 2). Both must match the
// Python side exactly: web/test_parity.js checks them against Python dumps.
(() => {
const NUM_COLORS = 5, CARDS_PER_COLOR = 12, NUM_CARDS = 60, HAND_SIZE = 8;
const PLAY_OFFSET = 0, DISCARD_OFFSET = 60, DRAW_DECK = 120, DRAW_DISCARD_OFFSET = 121, NUM_ACTIONS = 126;
const PHASE_PLAY = 0, PHASE_DRAW = 1;
const OBS_DIM = 6 * NUM_CARDS + 8 + NUM_COLORS * 12 + 1;  // 429
const COLOR_NAMES = ['Yellow', 'Blue', 'White', 'Green', 'Red'];
const COLOR_SHORT = ['Y', 'B', 'W', 'G', 'R'];

const cardColor = c => Math.floor(c / CARDS_PER_COLOR);
const isWager = c => c % CARDS_PER_COLOR < 3;
const cardValue = c => { const r = c % CARDS_PER_COLOR; return r < 3 ? 0 : r - 1; };
const cardStr = c => `${COLOR_SHORT[cardColor(c)]}${cardValue(c) === 0 ? '$' : cardValue(c)}`;

function scoreExpedition(cards) {
  if (!cards.length) return 0;
  let wagers = 0, total = 0;
  for (const c of cards) { const v = cardValue(c); if (v === 0) wagers++; else total += v; }
  let score = (total - 20) * (wagers + 1);
  if (cards.length >= 8) score += 20;
  return score;
}

function canPlayOn(expedition, c) {
  if (!expedition.length) return true;
  const last = cardValue(expedition[expedition.length - 1]), v = cardValue(c);
  if (v === 0) return last === 0;  // wagers only before any number
  return v > last;
}

// Fisher-Yates with a [0, 1) random source
function shuffle(arr, rand = Math.random) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

class GameState {
  constructor() {
    this.deck = [];
    this.hands = [[], []];
    this.expeditions = [[[], [], [], [], []], [[], [], [], [], []]];
    this.discards = [[], [], [], [], []];
    this.current = 0;
    this.phase = PHASE_PLAY;
    this.justDiscarded = null;  // color discarded this turn
    this.known = [new Set(), new Set()];  // cards in p's hand the opponent knows (drawn from a discard pile)
    this.done = false;
  }

  static newGame(rand = Math.random, startingPlayer = 0) {
    const s = new GameState();
    s.deck = shuffle([...Array(NUM_CARDS).keys()], rand);
    for (let i = 0; i < HAND_SIZE; i++) for (const p of [0, 1]) s.hands[p].push(s.deck.pop());
    s.current = startingPlayer;
    return s;
  }

  clone() {
    const s = new GameState();
    s.deck = this.deck.slice();
    s.hands = [this.hands[0].slice(), this.hands[1].slice()];
    s.expeditions = this.expeditions.map(side => side.map(e => e.slice()));
    s.discards = this.discards.map(d => d.slice());
    s.current = this.current;
    s.phase = this.phase;
    s.justDiscarded = this.justDiscarded;
    s.known = [new Set(this.known[0]), new Set(this.known[1])];
    s.done = this.done;
    return s;
  }

  legalActions() {
    if (this.done) return [];
    const p = this.current;
    if (this.phase === PHASE_PLAY) {
      const actions = [], seenWager = [false, false, false, false, false];
      for (const c of this.hands[p].slice().sort((a, b) => a - b)) {
        const col = cardColor(c);
        if (isWager(c)) {
          // wagers of the same color are interchangeable: a single action
          if (seenWager[col]) continue;
          seenWager[col] = true;
        }
        if (canPlayOn(this.expeditions[p][col], c)) actions.push(PLAY_OFFSET + c);
        actions.push(DISCARD_OFFSET + c);
      }
      return actions;
    }
    const actions = [DRAW_DECK];
    for (let col = 0; col < NUM_COLORS; col++)
      if (this.discards[col].length && col !== this.justDiscarded) actions.push(DRAW_DISCARD_OFFSET + col);
    return actions;
  }

  step(a) {
    if (this.done) throw new Error('round is over');
    const p = this.current;
    if (this.phase === PHASE_PLAY) {
      let c;
      if (a < DISCARD_OFFSET) {
        c = a;
        const col = cardColor(c);
        if (!this.hands[p].includes(c) || !canPlayOn(this.expeditions[p][col], c)) throw new Error(`illegal move ${a}`);
        this.hands[p].splice(this.hands[p].indexOf(c), 1);
        this.expeditions[p][col].push(c);
        this.justDiscarded = null;
      } else {
        c = a - DISCARD_OFFSET;
        if (a >= DRAW_DECK || !this.hands[p].includes(c)) throw new Error(`illegal move ${a}`);
        const col = cardColor(c);
        this.hands[p].splice(this.hands[p].indexOf(c), 1);
        this.discards[col].push(c);
        this.justDiscarded = col;
      }
      this.known[p].delete(c);
      this.phase = PHASE_DRAW;
      return;
    }
    if (a === DRAW_DECK) {
      this.hands[p].push(this.deck.pop());
    } else {
      const col = a - DRAW_DISCARD_OFFSET;
      if (!(col >= 0 && col < NUM_COLORS) || !this.discards[col].length || col === this.justDiscarded)
        throw new Error(`illegal move ${a}`);
      const c = this.discards[col].pop();
      this.hands[p].push(c);
      this.known[p].add(c);
    }
    this.justDiscarded = null;
    this.phase = PHASE_PLAY;
    this.current = 1 - p;
    if (!this.deck.length) this.done = true;
  }

  score(p) { return this.expeditions[p].reduce((t, e) => t + scoreExpedition(e), 0); }
  scoreDiff(p = 0) { return this.score(p) - this.score(1 - p); }

  // cards p can't see: the opponent's hand (except known cards) + the deck
  unseenCards(p) {
    const opp = 1 - p;
    return this.hands[opp].filter(c => !this.known[opp].has(c)).concat(this.deck);
  }

  // copy with the info hidden from p resampled, consistent with what p has seen
  determinize(p, rand = Math.random) {
    const s = this.clone(), opp = 1 - p;
    const known = s.hands[opp].filter(c => s.known[opp].has(c));
    const pool = shuffle(this.unseenCards(p), rand);
    const nHidden = s.hands[opp].length - known.length;
    s.hands[opp] = known.concat(pool.slice(0, nHidden));
    s.deck = pool.slice(nHidden);
    return s;
  }
}

// Network input from p's point of view, written into out[offset .. offset + 429).
function writeFeatures(s, p, out, offset = 0) {
  const opp = 1 - p, base = 6 * NUM_CARDS;
  const raw = new Float64Array(base + 8);
  for (const c of s.hands[p]) raw[c] = 1;
  for (let col = 0; col < NUM_COLORS; col++) {
    for (const c of s.expeditions[p][col]) raw[NUM_CARDS + c] = 1;
    for (const c of s.expeditions[opp][col]) raw[2 * NUM_CARDS + c] = 1;
    const pile = s.discards[col];
    for (const c of pile) raw[3 * NUM_CARDS + c] = 1;
    if (pile.length) raw[4 * NUM_CARDS + pile[pile.length - 1]] = 1;
  }
  for (const c of s.known[opp]) raw[5 * NUM_CARDS + c] = 1;
  raw[base] = s.deck.length / (NUM_CARDS - 16);
  raw[base + 1 + s.phase] = 1;
  if (s.justDiscarded !== null) raw[base + 3 + s.justDiscarded] = 1;
  for (let i = 0; i < raw.length; i++) out[offset + i] = raw[i];

  const blk = (i, col, r) => raw[i * NUM_CARDS + col * CARDS_PER_COLOR + r];
  let o = offset + raw.length;
  for (let col = 0; col < NUM_COLORS; col++) {
    let anyM = 0, anyT = 0, maxM = 0, maxT = 0, sumM = 0, sumT = 0, wagM = 0, wagT = 0, handN = 0, handV = 0;
    for (let r = 0; r < CARDS_PER_COLOR; r++) {
      const v = r < 3 ? 0 : r - 1, h = blk(0, col, r), m = blk(1, col, r), t = blk(2, col, r);
      anyM = Math.max(anyM, m); anyT = Math.max(anyT, t);
      maxM = Math.max(maxM, m * v); maxT = Math.max(maxT, t * v);
      sumM += m * v; sumT += t * v;
      if (r < 3) { wagM += m; wagT += t; }
      handN += h; handV += h * v;
    }
    // expeditions are increasing, so the last number is the max (1 if there are no numbers)
    const lastM = Math.max(maxM, 1), lastT = Math.max(maxT, 1);
    // the cards form a partition: what p can't see is everything else
    let upMN = 0, upMV = 0, upTN = 0, upTV = 0;
    for (let r = 3; r < CARDS_PER_COLOR; r++) {
      const v = r - 1;
      const unseen = 1 - blk(0, col, r) - blk(1, col, r) - blk(2, col, r) - blk(3, col, r) - blk(5, col, r);
      if (v > lastM) { upMN += unseen; upMV += unseen * v; }
      if (v > lastT) { upTN += unseen; upTV += unseen * v; }
    }
    const f = [lastM / 10 * anyM, lastT / 10 * anyT, sumM / 54, sumT / 54, wagM / 3, wagT / 3, handN / 8, handV / 54,
               upMN / 9, upMV / 54, upTN / 9, upTV / 54];
    for (let k = 0; k < 12; k++) out[o + col * 12 + k] = f[k];
  }
  o += NUM_COLORS * 12;
  out[o] = Math.floor((s.deck.length + 1) / 2) / 22;
}

const LC = {
  NUM_COLORS, NUM_CARDS, NUM_ACTIONS, OBS_DIM, PLAY_OFFSET, DISCARD_OFFSET, DRAW_DECK, DRAW_DISCARD_OFFSET,
  PHASE_PLAY, PHASE_DRAW, COLOR_NAMES, COLOR_SHORT, cardColor, cardValue, isWager, cardStr, scoreExpedition,
  canPlayOn, shuffle, GameState, writeFeatures,
};
if (typeof module !== 'undefined') module.exports = LC;
else globalThis.LC = LC;
})();
