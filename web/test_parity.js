// Checks the JavaScript engine against Python dumps (see web/parity_fixture.py).
// Network inputs are compared after rounding both sides to float32.
const fs = require('fs');
const path = require('path');
const LC = require('./engine.js');

const fx = JSON.parse(fs.readFileSync(path.join(__dirname, 'parity_fixture.json')));

function load(j) {
  const s = new LC.GameState();
  s.deck = j.deck.slice();
  s.hands = j.hands.map(h => h.slice());
  s.expeditions = j.expeditions.map(side => side.map(e => e.slice()));
  s.discards = j.discards.map(d => d.slice());
  s.current = j.current;
  s.phase = j.phase;
  s.justDiscarded = j.justDiscarded;
  s.known = j.known.map(k => new Set(k));
  s.done = j.done;
  return s;
}

const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
let fails = 0, worst = 0;

for (const [i, j] of fx.states.entries()) {
  const s = load(j);
  if (!same(s.legalActions(), j.legal)) { fails++; console.log(`state ${i}: legal moves differ`); }
  const f = new Float32Array(LC.OBS_DIM);
  LC.writeFeatures(s, j.player, f);
  const py = Float32Array.from(j.features);
  if (py.length !== f.length) { fails++; console.log(`state ${i}: ${f.length} features vs ${py.length}`); continue; }
  for (let k = 0; k < f.length; k++) {
    const d = Math.abs(f[k] - py[k]);
    worst = Math.max(worst, d);
    if (d > 0) { fails++; console.log(`state ${i}: feature ${k} is ${f[k]} vs ${py[k]}`); break; }
  }
}

for (const [i, r] of fx.rounds.entries()) {
  const s = load(r.start);
  for (const [t, a] of r.actions.entries()) {
    if (!same(s.legalActions(), r.legal[t])) { fails++; console.log(`round ${i} ply ${t}: legal moves differ`); break; }
    s.step(a);
  }
  if (!s.done || s.score(0) !== r.scores[0] || s.score(1) !== r.scores[1]) {
    fails++; console.log(`round ${i}: scores ${s.score(0)}/${s.score(1)} vs ${r.scores}`);
  }
}

const plies = fx.rounds.reduce((t, r) => t + r.actions.length, 0);
console.log(`${fx.states.length} states, ${fx.rounds.length} rounds (${plies} plies): ` +
            `${fails ? fails + ' mismatches' : 'all match'} (max feature diff ${worst})`);
process.exit(fails ? 1 : 0);
