// The bot in the browser: the policy network (ONNX, run with onnxruntime-web) plus
// the same rollout search used in Python (lost_cities/search.py, RolloutSearchBot).
// Requires engine.js and onnxruntime-web (global `ort`).
(() => {
const MAX_ROLLOUT_PLIES = 200;

class Bot {
  constructor(session) { this.session = session; }

  static async load(url) {
    return new Bot(await ort.InferenceSession.create(url, { executionProviders: ['wasm'] }));
  }

  // logits[i * 126 + a] and value[i] for each state, from the point of view of the player to move
  async evaluate(states) {
    const n = states.length, x = new Float32Array(n * LC.OBS_DIM);
    states.forEach((s, i) => LC.writeFeatures(s, s.current, x, i * LC.OBS_DIM));
    const out = await this.session.run({ obs: new ort.Tensor('float32', x, [n, LC.OBS_DIM]) });
    return { logits: out.logits.data, value: out.value.data };
  }

  static bestLegal(s, logits, i) {
    let best = -1, bestL = -Infinity;
    for (const a of s.legalActions()) {
      const l = logits[i * LC.NUM_ACTIONS + a];
      if (l > bestL) { bestL = l; best = a; }
    }
    return best;
  }

  // network probabilities over the legal moves of s, plus its value estimate in points
  async policy(s) {
    const { logits, value } = await this.evaluate([s]);
    const legal = s.legalActions(), m = Math.max(...legal.map(a => logits[a]));
    const exps = legal.map(a => Math.exp(logits[a] - m)), z = exps.reduce((t, e) => t + e, 0);
    const probs = new Map(legal.map((a, k) => [a, exps[k] / z]));
    return { probs, value: value[0] * 50 };
  }

  async greedy(s) {
    const { logits } = await this.evaluate([s]);
    return Bot.bestLegal(s, logits, 0);
  }

  // Search: for the network's top candidates, sample determinizations of the hidden
  // cards, play each candidate and finish the round with the network on both sides.
  // Score = mean final difference + prior * log(prob), same determinizations for all.
  async search(s, { dets = 64, topK = 4, minProb = 0.02, prior = 2, rand = Math.random } = {}) {
    const { probs } = await this.policy(s);
    const order = [...probs.entries()].sort((a, b) => b[1] - a[1]);
    let cands = order.slice(0, topK).filter(([, p]) => p >= minProb);
    if (!cands.length) cands = order.slice(0, 1);
    if (cands.length === 1) return cands[0][0];
    const me = s.current;
    const worlds = Array.from({ length: dets }, () => s.determinize(me, rand));
    const games = [];
    for (const [a] of cands) for (const w of worlds) { const g = w.clone(); g.step(a); games.push(g); }
    let active = games.filter(g => !g.done), plies = 0;
    while (active.length) {
      plies++;
      const { logits } = await this.evaluate(active);
      active.forEach((g, i) => {
        let a = Bot.bestLegal(g, logits, i);
        if (plies > MAX_ROLLOUT_PLIES && g.phase === LC.PHASE_DRAW) a = LC.DRAW_DECK;
        g.step(a);
      });
      active = active.filter(g => !g.done);
    }
    let best = null, bestScore = -Infinity;
    cands.forEach(([a, p], k) => {
      let sum = 0;
      for (let d = 0; d < dets; d++) sum += games[k * dets + d].scoreDiff(me);
      const score = sum / dets + prior * Math.log(Math.max(p, 1e-6));
      if (score > bestScore) { bestScore = score; best = a; }
    });
    return best;
  }
}

if (typeof module !== 'undefined') module.exports = { Bot };
else globalThis.Bot = Bot;
})();
