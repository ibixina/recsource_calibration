// Monte Carlo reliability sweep. Usage: node sweep.js [--quick]
// Writes results.json and results.js (the same data, loadable by the page).
const { Worker, isMainThread, parentPort, workerData } = require('node:worker_threads');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const E = require('./engine.js');

const LOST = 0.05; // RMS error, as a fraction of the arena side, above which a frame counts as failed.
const quick = process.argv.includes('--quick');
const scale = quick ? 0.1 : 1;
const count = (v) => Math.max(4, Math.round(v * scale));

const quantile = (values, q) => {
  if (!values.length) return null;
  const v = Float64Array.from(values).sort();
  return v[Math.min(v.length - 1, Math.floor(q * v.length))];
};
const round = (v, digits = 5) => (v === null || !Number.isFinite(v) ? null : +v.toFixed(digits));

// ---------- cells ----------
function rigidityCell({ n, k, dim, seed }) {
  const rng = E.Rng(seed);
  const p = E.problem({ n, k, dim, sigma: 0 });
  const r = E.rigidity(p, E.randomPositions(p, rng));
  let solutions = null;
  if (r.rigid && n + k >= dim + 2 && n * k <= 49) {
    const counts = [];
    for (let t = 0; t < count(6); t++) {
      const x = E.randomPositions(p, rng);
      const y = E.predict(p, x, new Float64Array(p.m));
      counts.push(E.countSolutions(p, y, rng, { restarts: dim === 2 ? 300 : 600 }).length);
    }
    solutions = { min: Math.min(...counts), max: Math.max(...counts), median: quantile(counts, 0.5) };
  }
  return { n, k, dim, rigid: r.rigid, free: r.freeMotions, redundancy: r.redundancy, solutions };
}

function coldCell({ n, k, dim, model, sigma, trials, seed }) {
  const rng = E.Rng(seed);
  const p = E.problem({ n, k, dim, model, sigma });
  const errs = [], bounds = [];
  let tries = 0, ms = 0;
  for (let t = 0; t < trials; t++) {
    const x = E.randomPositions(p, rng);
    const y = E.measure(p, x, rng);
    const t0 = performance.now();
    const sol = E.coldStart(p, y, rng);
    ms += performance.now() - t0;
    tries += sol.tries;
    errs.push(E.shapeRms(sol.x, x, dim, { reflect: true }));
    bounds.push(E.crlbRms(p, x));
  }
  const within = (lim) => round(errs.filter((e) => e < lim).length / trials, 4);
  return {
    n, k, dim, model, sigma,
    ok2: within(0.02), ok5: within(0.05), ok10: within(0.1),
    median: round(quantile(errs, 0.5)), p90: round(quantile(errs, 0.9)),
    boundMedian: round(quantile(bounds, 0.5)),
    boundOk5: round(bounds.filter((b) => b < LOST).length / trials, 4),
    tries: round(tries / trials, 2), ms: round(ms / trials, 2),
  };
}

function trackCell({ n, k, dim = 2, model, sigma, speed, kind, trials, T, seed, start = 'known', adaptive = true, relock = true, anchorsN = 0 }) {
  const rng = E.Rng(seed);
  let frames = 0, correct = 0, mirrored = 0, wrong = 0, silentWrong = 0, falseAlarm = 0, cleanRuns = 0, relocks = 0;
  const errs = [], abs = [];
  const p = E.problem({ n, k, dim, model, sigma });
  const anchors = Array.from({ length: anchorsN }, (_, i) => n + i);
  let msTotal = 0;
  for (let t = 0; t < trials; t++) {
    const x = E.randomPositions(p, rng);
    let init = x, initialStd = 1e-4;
    if (start === 'cold') {
      init = E.aligned(E.coldStart(p, E.measure(p, x, rng), rng).x, x, dim, { reflect: true });
      initialStd = Math.max(0.02, 2 * sigma);
    }
    const tracker = new E.Tracker(p, init, { rng, adaptive, relock, anchors, initialStd, motion: adaptive ? 0.02 : Math.max(speed, 0.001) });
    const move = E.makeMotion(kind, p, x, rng, { speed, anchors });
    let clean = true;
    for (let f = 0; f < T; f++) {
      move();
      const y = E.measure(p, x, rng);
      const t0 = performance.now();
      tracker.step(y);
      msTotal += performance.now() - t0;
      const e = E.shapeRms(tracker.x, x, dim);
      const em = E.shapeRms(tracker.x, x, dim, { reflect: true });
      frames++;
      errs.push(e);
      if (e < LOST) { correct++; if (tracker.flagged) falseAlarm++; }
      else if (em < LOST) { mirrored++; clean = false; }
      else { wrong++; clean = false; if (!tracker.flagged) silentWrong++; }
    }
    if (clean) cleanRuns++;
    relocks += tracker.relocks;
    abs.push(E.absoluteRms(tracker.x, x, dim));
  }
  return {
    n, k, dim, model, sigma, speed, kind, start, adaptive, relock, anchors: anchorsN,
    correct: round(correct / frames, 4), mirrored: round(mirrored / frames, 4), wrong: round(wrong / frames, 4),
    silentWrong: round(silentWrong / frames, 4), falseAlarm: round(falseAlarm / Math.max(correct, 1), 4),
    cleanRuns: round(cleanRuns / trials, 4), relocks: round(relocks / trials, 2),
    median: round(quantile(errs, 0.5)), p95: round(quantile(errs, 0.95)),
    absEnd: round(quantile(abs, 0.5)), absEndP90: round(quantile(abs, 0.9)),
    msPerStep: round(msTotal / frames, 3),
  };
}

function driftCell({ n, k, sigma, speed, kind, anchorsN, trials, marks, seed }) {
  const rng = E.Rng(seed);
  const p = E.problem({ n, k, sigma });
  const anchors = Array.from({ length: anchorsN }, (_, i) => n + i);
  const byMark = marks.map(() => []);
  const T = marks[marks.length - 1];
  let mirroredEnd = 0;
  for (let t = 0; t < trials; t++) {
    const x = E.randomPositions(p, rng);
    const tracker = new E.Tracker(p, x, { rng, anchors });
    const move = E.makeMotion(kind, p, x, rng, { speed, anchors });
    let m = 0;
    for (let f = 1; f <= T; f++) {
      move();
      tracker.step(E.measure(p, x, rng));
      if (f === marks[m]) { byMark[m].push(E.absoluteRms(tracker.x, x, 2)); m++; }
    }
    if (E.shapeRms(tracker.x, x, 2) > LOST && E.shapeRms(tracker.x, x, 2, { reflect: true }) <= LOST) mirroredEnd++;
  }
  return {
    n, k, sigma, speed, kind, anchors: anchorsN, marks,
    median: byMark.map((v) => round(quantile(v, 0.5))),
    p90: byMark.map((v) => round(quantile(v, 0.9))),
    mirroredEnd: round(mirroredEnd / trials, 4),
  };
}

const layouts = {
  random: (n, k, rng) => {
    const out = [];
    for (let i = 0; i < n + k; i++) out.push([rng.uniform(), rng.uniform()]);
    return out;
  },
  'shared circle': (n, k, rng) => {
    const out = [];
    const off = rng.uniform() * 6.283;
    for (let i = 0; i < n + k; i++) {
      const a = off + (rng.uniform() * 6.283);
      out.push([0.5 + 0.45 * Math.cos(a), 0.5 + 0.45 * Math.sin(a)]);
    }
    return out;
  },
  'sources in a line': (n, k, rng) => {
    const out = [];
    for (let i = 0; i < n; i++) out.push([rng.uniform(), rng.uniform()]);
    for (let j = 0; j < k; j++) out.push([rng.uniform(), 0.5]);
    return out;
  },
  'two parallel lines': (n, k, rng) => {
    const out = [];
    for (let i = 0; i < n; i++) out.push([rng.uniform(), 0.1]);
    for (let j = 0; j < k; j++) out.push([rng.uniform(), 0.9]);
    return out;
  },
  'two far clusters': (n, k, rng) => {
    const out = [];
    for (let i = 0; i < n; i++) out.push([0.05 + 0.2 * rng.uniform(), 0.05 + 0.2 * rng.uniform()]);
    for (let j = 0; j < k; j++) out.push([0.75 + 0.2 * rng.uniform(), 0.75 + 0.2 * rng.uniform()]);
    return out;
  },
  'sources around, receivers inside': (n, k, rng) => {
    const out = [];
    for (let i = 0; i < n; i++) out.push([0.3 + 0.4 * rng.uniform(), 0.3 + 0.4 * rng.uniform()]);
    const off = rng.uniform() * 6.283;
    for (let j = 0; j < k; j++) {
      const a = off + (j / k) * 6.283;
      out.push([0.5 + 0.45 * Math.cos(a), 0.5 + 0.45 * Math.sin(a)]);
    }
    return out;
  },
};

function placementCell({ n, k, name, sigma, trials, seed }) {
  const rng = E.Rng(seed);
  const p = E.problem({ n, k, sigma });
  const errs = [], bounds = [];
  let rigidCount = 0;
  for (let t = 0; t < trials; t++) {
    const pts = layouts[name](n, k, rng);
    const x = Float64Array.from(pts.flat());
    if (E.rigidity(p, x).rigid) rigidCount++;
    bounds.push(E.crlbRms(p, x));
    errs.push(E.shapeRms(E.coldStart(p, E.measure(p, x, rng), rng).x, x, 2, { reflect: true }));
  }
  return {
    n, k, name, sigma,
    rigid: round(rigidCount / trials, 3),
    ok5: round(errs.filter((e) => e < LOST).length / trials, 4),
    median: round(quantile(errs, 0.5)),
    boundMedian: round(quantile(bounds, 0.5)),
  };
}

const cellFns = { rigidityCell, coldCell, trackCell, driftCell, placementCell };

// ---------- worker ----------
if (!isMainThread) {
  parentPort.on('message', ({ id, fn, args }) => {
    parentPort.postMessage({ id, result: cellFns[fn](args) });
  });
  return;
}

// ---------- job list ----------
const jobs = [];
let seedCounter = 1000;
const add = (group, fn, args) => jobs.push({ group, fn, args: { ...args, seed: seedCounter++ } });

for (const dim of [2, 3]) {
  for (let n = 1; n <= 8; n++) for (let k = 1; k <= 8; k++) add('rigidity', 'rigidityCell', { n, k, dim });
}

const noiseLevels = { range: [0.002, 0.005, 0.01, 0.02, 0.05], strength: [0.02, 0.05, 0.1, 0.2, 0.4] };
const sizes = [2, 3, 4, 5, 6, 8];
for (const model of ['range', 'strength']) {
  for (const sigma of noiseLevels[model]) {
    for (const n of sizes) for (const k of sizes) add('cold', 'coldCell', { n, k, dim: 2, model, sigma, trials: count(200) });
  }
}
for (const sigma of [0.005, 0.02]) {
  for (const n of [4, 5, 6, 8]) for (const k of [4, 5, 6, 8]) add('cold3d', 'coldCell', { n, k, dim: 3, model: 'range', sigma, trials: count(100) });
}

const trackNoise = { range: [0.002, 0.01, 0.03], strength: [0.02, 0.1, 0.3] };
const pairs = [[2, 2], [3, 3], [3, 4], [4, 4], [5, 5], [6, 6], [8, 8]];
const speeds = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3];
for (const model of ['range', 'strength']) {
  for (const sigma of trackNoise[model]) {
    for (const [n, k] of pairs) {
      const trials = count(n >= 8 ? 30 : 60);
      for (const speed of speeds) {
        for (const kind of ['all', 'one']) add('track', 'trackCell', { n, k, model, sigma, speed, kind, trials, T: 300 });
      }
      add('track', 'trackCell', { n, k, model, sigma, speed: 0, kind: 'jump', trials, T: 300 });
    }
  }
}

// Does it help to be told the starting layout? Does relock earn its place?
for (const [n, k] of [[3, 4], [4, 4], [6, 6]]) {
  for (const speed of [0.003, 0.01, 0.03, 0.1]) {
    add('startRelock', 'trackCell', { n, k, model: 'range', sigma: 0.01, speed, kind: 'all', trials: count(60), T: 300, start: 'known', relock: true });
    add('startRelock', 'trackCell', { n, k, model: 'range', sigma: 0.01, speed, kind: 'all', trials: count(60), T: 300, start: 'known', relock: false });
    add('startRelock', 'trackCell', { n, k, model: 'range', sigma: 0.01, speed, kind: 'all', trials: count(60), T: 300, start: 'cold', relock: true });
  }
}

// Three devices pinned: does a mirror flip still happen?
for (const anchorsN of [0, 3]) {
  for (const speed of [0.01, 0.03, 0.1]) {
    add('anchorFlip', 'trackCell', { n: 4, k: 4, model: 'range', sigma: 0.01, speed, kind: 'all', trials: count(60), T: 300, anchorsN });
    add('anchorFlip', 'trackCell', { n: 6, k: 6, model: 'range', sigma: 0.01, speed, kind: 'all', trials: count(60), T: 300, anchorsN });
  }
}

const marks = [10, 30, 100, 300, 1000];
for (const [n, k] of [[4, 4], [6, 6]]) {
  for (const anchorsN of [0, 1, 2, 3]) {
    for (const kind of ['all', 'one']) add('drift', 'driftCell', { n, k, sigma: 0.01, speed: 0.01, kind, anchorsN, trials: count(50), marks });
  }
}

for (const [n, k] of [[3, 3], [4, 4], [6, 6]]) {
  for (const name of Object.keys(layouts)) add('placement', 'placementCell', { n, k, name, sigma: 0.01, trials: count(200) });
}

// ---------- run ----------
const results = {};
let done = 0, next = 0;
const started = Date.now();
const workers = [];
const finish = () => {
  for (const w of workers) w.terminate();
  const out = {
    meta: {
      generated: new Date().toISOString(), quick, lostThreshold: LOST,
      seconds: Math.round((Date.now() - started) / 1000), jobs: jobs.length,
      arena: 'unit square; range sigma is a fraction of the side, strength sigma is fractional distance error',
    },
    ...results,
  };
  const json = JSON.stringify(out);
  const dir = __dirname;
  fs.writeFileSync(path.join(dir, quick ? 'results.quick.json' : 'results.json'), json);
  if (!quick) fs.writeFileSync(path.join(dir, 'results.js'), `window.RESULTS = ${json};\n`);
  console.log(`done: ${jobs.length} cells in ${out.meta.seconds}s`);
};
const size = Math.max(1, os.cpus().length - 1);
for (let w = 0; w < size; w++) {
  const worker = new Worker(__filename, { argv: process.argv.slice(2) });
  worker.on('message', ({ id, result }) => {
    (results[jobs[id].group] ||= []).push(result);
    done++;
    if (done % 100 === 0) console.log(`${done}/${jobs.length} after ${Math.round((Date.now() - started) / 1000)}s`);
    if (done === jobs.length) finish();
    else if (next < jobs.length) { const j = next++; worker.postMessage({ id: j, fn: jobs[j].fn, args: jobs[j].args }); }
  });
  workers.push(worker);
  if (next < jobs.length) { const j = next++; worker.postMessage({ id: j, fn: jobs[j].fn, args: jobs[j].args }); }
}
console.log(`${jobs.length} cells on ${size} threads${quick ? ' (quick)' : ''}`);
