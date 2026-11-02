const test = require('node:test');
const assert = require('node:assert');
const E = require('./engine.js');

const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg || ''} got ${a}, expected ${b} +/- ${tol}`);

test('jacobian matches finite differences for both signal models', () => {
  for (const model of ['range', 'strength']) {
    for (const dim of [2, 3]) {
      const p = E.problem({ n: 3, k: 4, dim, model, sigma: 0.01 });
      const x = E.randomPositions(p, E.Rng(1));
      const J = E.jacobian(p, x, new Float64Array(p.m * p.N));
      const up = new Float64Array(p.m), dn = new Float64Array(p.m);
      for (let c = 0; c < p.N; c++) {
        const xp = Float64Array.from(x), xm = Float64Array.from(x);
        xp[c] += 1e-6; xm[c] -= 1e-6;
        E.predict(p, xp, up); E.predict(p, xm, dn);
        for (let a = 0; a < p.m; a++) near(J[a * p.N + c], (up[a] - dn[a]) / 2e-6, 1e-5, `${model} dim${dim}`);
      }
    }
  }
});

test('moving everything rigidly leaves every reading exactly unchanged', () => {
  const p = E.problem({ n: 4, k: 3, dim: 2, sigma: 0 });
  const x = E.randomPositions(p, E.Rng(2));
  const a = 1.1, c = Math.cos(a), s = Math.sin(a);
  const moved = new Float64Array(p.N);
  for (let d = 0; d < p.N / 2; d++) {
    moved[2 * d] = c * x[2 * d] - s * x[2 * d + 1] + 5;
    moved[2 * d + 1] = s * x[2 * d] + c * x[2 * d + 1] - 3;
  }
  const y0 = E.predict(p, x, new Float64Array(p.m));
  const y1 = E.predict(p, moved, new Float64Array(p.m));
  for (let m = 0; m < p.m; m++) near(y0[m], y1[m], 1e-12);
});

test('alignment undoes a known rotation, translation and mirror', () => {
  for (const dim of [2, 3]) {
    const p = E.problem({ n: 3, k: 4, dim });
    const x = E.randomPositions(p, E.Rng(3));
    const moved = Float64Array.from(x);
    for (let d = 0; d < 7; d++) {
      const a = x[d * dim], b = x[d * dim + 1];
      moved[d * dim] = 0.6 * a - 0.8 * b + 2;
      moved[d * dim + 1] = 0.8 * a + 0.6 * b - 1;
    }
    near(E.shapeRms(moved, x, dim), 0, 1e-9, 'rotation');
    const mirrored = Float64Array.from(moved);
    for (let d = 0; d < 7; d++) mirrored[d * dim] *= -1;
    near(E.shapeRms(mirrored, x, dim, { reflect: true }), 0, 1e-9, 'mirror allowed');
    assert.ok(E.shapeRms(mirrored, x, dim) > 0.01, 'mirror must count as error when not allowed');
  }
});

test('rigidity counts match known cases', () => {
  const rigid = (n, k, dim) => {
    const p = E.problem({ n, k, dim });
    return E.rigidity(p, E.randomPositions(p, E.Rng(100 + n * 10 + k))).rigid;
  };
  assert.equal(rigid(2, 2, 2), false);
  assert.equal(rigid(2, 8, 2), false);
  assert.equal(rigid(3, 3, 2), true);
  assert.equal(rigid(3, 8, 3), false);
  assert.equal(rigid(4, 5, 3), false);
  assert.equal(rigid(4, 6, 3), true);
  assert.equal(rigid(5, 5, 3), true);
});

test('3x3 on a shared circle loses rigidity (conic singularity)', () => {
  const p = E.problem({ n: 3, k: 3, dim: 2 });
  const x = new Float64Array(p.N);
  for (let a = 0; a < 6; a++) {
    const th = 0.3 + a * 0.97;
    x[2 * a] = 0.5 + 0.4 * Math.cos(th);
    x[2 * a + 1] = 0.5 + 0.4 * Math.sin(th);
  }
  assert.equal(E.rigidity(p, x).rigid, false);
});

test('noise-free cold start recovers the layout', () => {
  const rng = E.Rng(5);
  for (const model of ['range', 'strength']) {
    for (const [n, k] of [[3, 4], [4, 4], [6, 6], [8, 8]]) {
      const p = E.problem({ n, k, model, sigma: 1e-6 });
      let good = 0;
      for (let t = 0; t < 10; t++) {
        const x = E.randomPositions(p, rng);
        const y = E.measure(p, x, rng);
        const sol = E.coldStart(p, y, rng);
        if (E.shapeRms(sol.x, x, 2, { reflect: true }) < 1e-3) good++;
      }
      assert.ok(good >= 9, `${model} ${n}x${k}: ${good}/10`);
    }
  }
});

test('cold start error matches the Cramer-Rao bound at high signal-to-noise', () => {
  const rng = E.Rng(6);
  for (const model of ['range', 'strength']) {
    const p = E.problem({ n: 5, k: 5, model, sigma: 1e-3 });
    let x;
    do { x = E.randomPositions(p, rng); } while (E.crlbRms(p, x) > 3e-3);
    const bound = E.crlbRms(p, x);
    let s = 0;
    const trials = 400;
    for (let t = 0; t < trials; t++) {
      const y = E.measure(p, x, rng);
      const sol = E.coldStart(p, y, rng, { seeds: [x] });
      // A mirror image gives identical readings, so one-shot error has to allow for it.
      const e = E.shapeRms(sol.x, x, 2, { reflect: true });
      s += e * e;
    }
    const rms = Math.sqrt(s / trials);
    const ratio = rms / bound;
    assert.ok(ratio > 0.93 && ratio < 1.07, `${model}: error/bound ${ratio.toFixed(3)}`);
  }
});

test('tracker follows noise-free motion from a known start', () => {
  const rng = E.Rng(7);
  const p = E.problem({ n: 4, k: 4, sigma: 1e-5 });
  const x = E.randomPositions(p, rng);
  const tracker = new E.Tracker(p, x, { rng });
  const move = E.makeMotion('all', p, x, rng, { speed: 0.01 });
  let worst = 0;
  for (let t = 0; t < 200; t++) {
    move();
    tracker.step(E.measure(p, x, rng));
    worst = Math.max(worst, E.shapeRms(tracker.x, x, 2));
  }
  assert.ok(worst < 2e-3, `worst shape error ${worst}`);
});

test('tracker does not move on a rigid move: readings are identical, so belief stays put', () => {
  const rng = E.Rng(8);
  const p = E.problem({ n: 4, k: 4, sigma: 1e-5 });
  const x = E.randomPositions(p, rng);
  const start = Float64Array.from(x);
  const tracker = new E.Tracker(p, x, { rng });
  const move = E.makeMotion('rigid', p, x, rng, { speed: 0.01 });
  for (let t = 0; t < 100; t++) { move(); tracker.step(E.measure(p, x, rng)); }
  assert.ok(E.absoluteRms(tracker.x, start, 2) < 1e-3, 'belief should not have moved');
  assert.ok(E.absoluteRms(tracker.x, x, 2) > 0.2, 'so absolute error should be large');
  assert.ok(E.shapeRms(tracker.x, x, 2) < 1e-3, 'but the shape is still right');
});

test('anchors stop absolute drift', () => {
  const run = (anchors) => {
    const rng = E.Rng(9);
    let total = 0;
    for (let trial = 0; trial < 10; trial++) {
      const p = E.problem({ n: 4, k: 4, sigma: 0.005 });
      const x = E.randomPositions(p, rng);
      const tracker = new E.Tracker(p, x, { rng, anchors });
      const move = E.makeMotion('all', p, x, rng, { speed: 0.01, anchors });
      for (let t = 0; t < 400; t++) { move(); tracker.step(E.measure(p, x, rng)); }
      total += E.absoluteRms(tracker.x, x, 2);
    }
    return total / 10;
  };
  const free = run([]);
  const pinned = run([4, 5, 6]);
  assert.ok(pinned < free * 0.6, `pinned ${pinned} vs free ${free}`);
});

test('relock recovers after a device teleports', () => {
  const rng = E.Rng(10);
  let recovered = 0;
  const trials = 20;
  for (let trial = 0; trial < trials; trial++) {
    const p = E.problem({ n: 5, k: 5, sigma: 0.002 });
    const x = E.randomPositions(p, rng);
    const tracker = new E.Tracker(p, x, { rng });
    for (let t = 0; t < 5; t++) tracker.step(E.measure(p, x, rng));
    for (let c = 0; c < 2; c++) x[2 * 2 + c] = rng.uniform();
    for (let t = 0; t < 10; t++) tracker.step(E.measure(p, x, rng));
    if (E.shapeRms(tracker.x, x, 2) < 0.02) recovered++;
  }
  assert.ok(recovered >= 18, `recovered ${recovered}/${trials}`);
});

test('a still layout averages over time, beating what one snapshot allows', () => {
  const rng = E.Rng(12);
  for (const [model, sigma] of [['range', 0.03], ['strength', 0.3]]) {
    const p = E.problem({ n: 6, k: 6, model, sigma });
    const ratios = [];
    for (let trial = 0; trial < 8; trial++) {
      const x = E.randomPositions(p, rng);
      const tracker = new E.Tracker(p, x, { rng });
      for (let t = 0; t < 200; t++) tracker.step(E.measure(p, x, rng));
      ratios.push(E.shapeRms(tracker.x, x, 2) / E.crlbRms(p, x));
    }
    ratios.sort((a, b) => a - b);
    assert.ok(ratios[4] < 0.6, `${model}: median error/snapshot bound ${ratios[4].toFixed(2)}`);
  }
});

test('the fit check rarely fires while tracking is correct', () => {
  const rng = E.Rng(13);
  for (const [n, k, speed] of [[4, 4, 0.003], [4, 4, 0.01], [6, 6, 0.01], [6, 6, 0.03]]) {
    const p = E.problem({ n, k, sigma: 0.01 });
    let alarms = 0, frames = 0;
    for (let trial = 0; trial < 10; trial++) {
      const x = E.randomPositions(p, rng);
      const tracker = new E.Tracker(p, x, { rng });
      const move = E.makeMotion('all', p, x, rng, { speed });
      for (let t = 0; t < 200; t++) {
        move();
        tracker.step(E.measure(p, x, rng));
        if (E.shapeRms(tracker.x, x, 2) < 0.05) { frames++; if (tracker.flagged) alarms++; }
      }
    }
    assert.ok(alarms / frames < 0.02, `${n}x${k} v=${speed}: false alarm rate ${(alarms / frames).toFixed(3)}`);
  }
});
