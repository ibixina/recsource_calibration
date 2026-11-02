// Position-from-signals engine. No dependencies; runs in Node and the browser.
//
// World: n receivers and k sources in the unit square/cube. Each receiver gets
// one reading per source, and that reading depends only on the distance between
// them. The estimators below see nothing except the n x k table of readings.
//
// State layout: x = [r_0 ... r_{n-1}, s_0 ... s_{k-1}], flattened, `dim` per device.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Engine = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ---------- random numbers ----------
  function Rng(seed) {
    let a = seed >>> 0;
    let spare = null;
    const u = () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    return {
      uniform: u,
      gauss() {
        if (spare !== null) { const s = spare; spare = null; return s; }
        let p, q, w;
        do { p = 2 * u() - 1; q = 2 * u() - 1; w = p * p + q * q; } while (w >= 1 || w === 0);
        const f = Math.sqrt(-2 * Math.log(w) / w);
        spare = q * f;
        return p * f;
      },
      int: (m) => Math.floor(u() * m),
    };
  }

  // ---------- problem and signal model ----------
  // model 'range':    y = distance + noise             (timing-like, sigma in arena units)
  // model 'strength': y = ln(distance) + noise         (amplitude-like, sigma is fractional)
  function problem({ n, k, dim = 2, model = 'range', sigma = 0.01 }) {
    return { n, k, dim, model, sigma, m: n * k, N: (n + k) * dim, soft2: model === 'strength' ? 1e-4 : 1e-12 };
  }

  function predict(p, x, out) {
    const { n, k, dim, soft2 } = p;
    const log = p.model === 'strength';
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < k; j++) {
        let d2 = soft2;
        for (let c = 0; c < dim; c++) { const e = x[i * dim + c] - x[(n + j) * dim + c]; d2 += e * e; }
        out[i * k + j] = log ? 0.5 * Math.log(d2) : Math.sqrt(d2);
      }
    }
    return out;
  }

  // Dense Jacobian, m rows by N columns.
  function jacobian(p, x, J) {
    const { n, k, dim, soft2, N } = p;
    const log = p.model === 'strength';
    J.fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < k; j++) {
        const row = (i * k + j) * N;
        let d2 = soft2;
        for (let c = 0; c < dim; c++) { const e = x[i * dim + c] - x[(n + j) * dim + c]; d2 += e * e; }
        const w = log ? 1 / d2 : 1 / Math.sqrt(d2);
        for (let c = 0; c < dim; c++) {
          const v = (x[i * dim + c] - x[(n + j) * dim + c]) * w;
          J[row + i * dim + c] = v;
          J[row + (n + j) * dim + c] = -v;
        }
      }
    }
    return J;
  }

  function measure(p, x, rng, out) {
    const y = out || new Float64Array(p.m);
    predict(p, x, y);
    if (p.sigma > 0) for (let a = 0; a < p.m; a++) y[a] += p.sigma * rng.gauss();
    return y;
  }

  function randomPositions(p, rng) {
    const x = new Float64Array(p.N);
    for (let a = 0; a < p.N; a++) x[a] = rng.uniform();
    return x;
  }

  // ---------- small linear algebra ----------
  // Cholesky solve of a symmetric positive definite matrix, in place on copies.
  // Returns false when the matrix is not positive definite.
  function cholesky(A, N) {
    for (let i = 0; i < N; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i * N + j];
        for (let q = 0; q < j; q++) s -= A[i * N + q] * A[j * N + q];
        if (i === j) {
          if (s <= 0) return false;
          A[i * N + i] = Math.sqrt(s);
        } else {
          A[i * N + j] = s / A[j * N + j];
        }
      }
    }
    return true;
  }

  function choleskySolve(L, N, b) {
    for (let i = 0; i < N; i++) {
      let s = b[i];
      for (let q = 0; q < i; q++) s -= L[i * N + q] * b[q];
      b[i] = s / L[i * N + i];
    }
    for (let i = N - 1; i >= 0; i--) {
      let s = b[i];
      for (let q = i + 1; q < N; q++) s -= L[q * N + i] * b[q];
      b[i] = s / L[i * N + i];
    }
    return b;
  }

  // Jacobi eigendecomposition of a symmetric matrix. Returns { values, vectors } with
  // vectors stored by column.
  function symEig(Ain, N) {
    const A = Float64Array.from(Ain);
    const V = new Float64Array(N * N);
    for (let i = 0; i < N; i++) V[i * N + i] = 1;
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0, diag = 0;
      for (let i = 0; i < N; i++) {
        diag += A[i * N + i] * A[i * N + i];
        for (let j = i + 1; j < N; j++) off += A[i * N + j] * A[i * N + j];
      }
      if (off <= 1e-28 * (diag + 1e-300)) break;
      for (let p = 0; p < N; p++) {
        for (let q = p + 1; q < N; q++) {
          const apq = A[p * N + q];
          if (Math.abs(apq) < 1e-300) continue;
          const theta = (A[q * N + q] - A[p * N + p]) / (2 * apq);
          const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let r = 0; r < N; r++) {
            const arp = A[r * N + p], arq = A[r * N + q];
            A[r * N + p] = c * arp - s * arq;
            A[r * N + q] = s * arp + c * arq;
          }
          for (let r = 0; r < N; r++) {
            const apr = A[p * N + r], aqr = A[q * N + r];
            A[p * N + r] = c * apr - s * aqr;
            A[q * N + r] = s * apr + c * aqr;
          }
          for (let r = 0; r < N; r++) {
            const vrp = V[r * N + p], vrq = V[r * N + q];
            V[r * N + p] = c * vrp - s * vrq;
            V[r * N + q] = s * vrp + c * vrq;
          }
        }
      }
    }
    const values = new Float64Array(N);
    for (let i = 0; i < N; i++) values[i] = A[i * N + i];
    return { values, vectors: V };
  }

  // ---------- alignment (Procrustes) ----------
  // Best rotation + translation (optionally reflection) taking `est` onto `ref`.
  // `mask` optionally restricts which devices drive the fit.
  function alignTransform(est, ref, dim, { reflect = false, mask = null } = {}) {
    const count = est.length / dim;
    const ce = new Float64Array(dim), cr = new Float64Array(dim);
    let used = 0;
    for (let a = 0; a < count; a++) {
      if (mask && !mask[a]) continue;
      used++;
      for (let c = 0; c < dim; c++) { ce[c] += est[a * dim + c]; cr[c] += ref[a * dim + c]; }
    }
    for (let c = 0; c < dim; c++) { ce[c] /= used; cr[c] /= used; }
    const H = new Float64Array(dim * dim);
    for (let a = 0; a < count; a++) {
      if (mask && !mask[a]) continue;
      for (let r = 0; r < dim; r++) {
        for (let c = 0; c < dim; c++) {
          H[r * dim + c] += (est[a * dim + r] - ce[r]) * (ref[a * dim + c] - cr[c]);
        }
      }
    }
    const R = rotationFromCovariance(H, dim, reflect);
    return { R, ce, cr };
  }

  // R maximising trace(R H), via H^T H = V S^2 V^T and U = H V / S.
  function rotationFromCovariance(H, dim, reflect) {
    const HtH = new Float64Array(dim * dim);
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) {
        for (let q = 0; q < dim; q++) HtH[i * dim + j] += H[q * dim + i] * H[q * dim + j];
      }
    }
    const { values, vectors: V } = symEig(HtH, dim);
    const order = [...values.keys()].sort((a, b) => values[b] - values[a]);
    // Columns of U, ordered by singular value, completed to an orthonormal basis.
    const U = [];
    const Vs = order.map((o) => { const v = new Float64Array(dim); for (let r = 0; r < dim; r++) v[r] = V[r * dim + o]; return v; });
    const top = Math.sqrt(Math.max(values[order[0]], 0));
    for (let idx = 0; idx < dim; idx++) {
      const s = Math.sqrt(Math.max(values[order[idx]], 0));
      let u = new Float64Array(dim);
      if (s > 1e-10 * (top + 1e-300)) {
        for (let r = 0; r < dim; r++) for (let c = 0; c < dim; c++) u[r] += H[r * dim + c] * Vs[idx][c];
        for (let r = 0; r < dim; r++) u[r] /= s;
      } else {
        for (let e = 0; e < dim; e++) {
          u.fill(0); u[e] = 1;
          for (const w of U) {
            let dp = 0;
            for (let r = 0; r < dim; r++) dp += u[r] * w[r];
            for (let r = 0; r < dim; r++) u[r] -= dp * w[r];
          }
          let nrm = 0;
          for (let r = 0; r < dim; r++) nrm += u[r] * u[r];
          if (nrm > 1e-6) { nrm = Math.sqrt(nrm); for (let r = 0; r < dim; r++) u[r] /= nrm; break; }
        }
      }
      U.push(u);
    }
    const build = (signLast) => {
      const R = new Float64Array(dim * dim);
      for (let r = 0; r < dim; r++) {
        for (let c = 0; c < dim; c++) {
          let t = 0;
          for (let idx = 0; idx < dim; idx++) t += (idx === dim - 1 ? signLast : 1) * Vs[idx][r] * U[idx][c];
          R[r * dim + c] = t;
        }
      }
      return R;
    };
    const R = build(1);
    if (!reflect && det(R, dim) < 0) return build(-1);
    return R;
  }

  function det(M, dim) {
    if (dim === 2) return M[0] * M[3] - M[1] * M[2];
    return M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) + M[2] * (M[3] * M[7] - M[4] * M[6]);
  }

  // Applies the transform so that `est` lands on `ref`'s frame.
  function aligned(est, ref, dim, opts) {
    const { R, ce, cr } = alignTransform(est, ref, dim, opts);
    const count = est.length / dim;
    const out = new Float64Array(est.length);
    for (let a = 0; a < count; a++) {
      for (let r = 0; r < dim; r++) {
        let t = cr[r];
        for (let c = 0; c < dim; c++) t += R[r * dim + c] * (est[a * dim + c] - ce[c]);
        out[a * dim + r] = t;
      }
    }
    return out;
  }

  function rmsPerDevice(a, b, dim) {
    let s = 0;
    for (let c = 0; c < a.length; c++) { const e = a[c] - b[c]; s += e * e; }
    return Math.sqrt(s / (a.length / dim));
  }

  // Error once rotation and translation are forgiven: did the system get the layout right?
  function shapeRms(est, truth, dim, { reflect = false } = {}) {
    return rmsPerDevice(aligned(est, truth, dim, { reflect }), truth, dim);
  }

  // Error with nothing forgiven: is each device where the system says it is?
  function absoluteRms(est, truth, dim) { return rmsPerDevice(est, truth, dim); }

  // ---------- observability ----------
  function maxRigidityRank(count, dim) {
    if (count <= dim + 1) return (count * (count - 1)) / 2;
    return dim * count - (dim * (dim + 1)) / 2;
  }

  function matrixRank(JtJ, N) {
    const { values } = symEig(JtJ, N);
    let top = 0;
    for (let i = 0; i < N; i++) top = Math.max(top, Math.abs(values[i]));
    let r = 0;
    for (let i = 0; i < N; i++) if (Math.abs(values[i]) > 1e-9 * top) r++;
    return r;
  }

  function JtJ(p, x) {
    const { m, N } = p;
    const J = jacobian(p, x, new Float64Array(m * N));
    const A = new Float64Array(N * N);
    for (let a = 0; a < m; a++) {
      for (let i = 0; i < N; i++) {
        const v = J[a * N + i];
        if (v === 0) continue;
        for (let j = 0; j < N; j++) A[i * N + j] += v * J[a * N + j];
      }
    }
    return A;
  }

  // Infinitesimal rigidity of the layout `x`. rigid === true means no device can
  // shift without changing some reading, apart from moving everything together.
  function rigidity(p, x) {
    const rank = matrixRank(JtJ(p, x), p.N);
    const need = maxRigidityRank(p.n + p.k, p.dim);
    return { rank, need, rigid: rank === need, freeMotions: need - rank, redundancy: p.m - rank };
  }

  // Cramer-Rao bound on per-device RMS error once the global frame is forgiven.
  // The Moore-Penrose inverse of the Fisher information gives the frame-free bound.
  function crlbRms(p, x) {
    const A = JtJ(p, x);
    const { values } = symEig(A, p.N);
    let top = 0;
    for (let i = 0; i < p.N; i++) top = Math.max(top, values[i]);
    const need = maxRigidityRank(p.n + p.k, p.dim);
    const sorted = [...values].sort((a, b) => b - a);
    let trace = 0;
    for (let i = 0; i < need; i++) {
      if (sorted[i] <= 1e-10 * top) return Infinity;
      trace += 1 / sorted[i];
    }
    return p.sigma * Math.sqrt(trace / (p.n + p.k));
  }

  // ---------- one-shot solver: Levenberg-Marquardt ----------
  function buildNormal(p, x, y, A, g, h) {
    const { n, k, dim, N, soft2 } = p;
    const log = p.model === 'strength';
    A.fill(0); g.fill(0);
    let cost = 0;
    const u = new Float64Array(dim);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < k; j++) {
        const ri = i * dim, sj = (n + j) * dim;
        let d2 = soft2;
        for (let c = 0; c < dim; c++) { u[c] = x[ri + c] - x[sj + c]; d2 += u[c] * u[c]; }
        const pred = log ? 0.5 * Math.log(d2) : Math.sqrt(d2);
        const w = log ? 1 / d2 : 1 / Math.sqrt(d2);
        for (let c = 0; c < dim; c++) u[c] *= w;
        const r = y[i * k + j] - pred;
        cost += r * r;
        for (let c = 0; c < dim; c++) {
          g[ri + c] += u[c] * r;
          g[sj + c] -= u[c] * r;
          for (let e = 0; e < dim; e++) {
            const v = u[c] * u[e];
            A[(ri + c) * N + ri + e] += v;
            A[(sj + c) * N + sj + e] += v;
            A[(ri + c) * N + sj + e] -= v;
            A[(sj + c) * N + ri + e] -= v;
          }
        }
      }
    }
    return cost;
  }

  function costOf(p, x, y, h) {
    predict(p, x, h);
    let c = 0;
    for (let a = 0; a < p.m; a++) { const r = y[a] - h[a]; c += r * r; }
    return c;
  }

  function levenberg(p, y, x0, { iters = 80, tol = 1e-12 } = {}) {
    const { N, m } = p;
    const x = Float64Array.from(x0);
    const A = new Float64Array(N * N), L = new Float64Array(N * N), g = new Float64Array(N);
    const d = new Float64Array(N), trial = new Float64Array(N), h = new Float64Array(m);
    let lambda = 1e-3;
    let cost = buildNormal(p, x, y, A, g, h);
    for (let it = 0; it < iters; it++) {
      let accepted = false;
      for (let attempt = 0; attempt < 12; attempt++) {
        L.set(A);
        for (let i = 0; i < N; i++) L[i * N + i] += lambda * (A[i * N + i] + 1e-6);
        if (!cholesky(L, N)) { lambda *= 4; continue; }
        d.set(g);
        choleskySolve(L, N, d);
        for (let i = 0; i < N; i++) trial[i] = x[i] + d[i];
        const c = costOf(p, trial, y, h);
        if (c < cost) {
          const drop = cost - c;
          x.set(trial);
          cost = buildNormal(p, x, y, A, g, h);
          lambda = Math.max(lambda / 3, 1e-10);
          accepted = true;
          if (drop < tol * (cost + 1e-30) || cost < 1e-24) return { x, cost };
          break;
        }
        lambda *= 4;
      }
      if (!accepted) break;
    }
    return { x, cost };
  }

  function acceptableChi2(nu) { return nu + 3 * Math.sqrt(2 * nu) + 1e-9; }

  // Multi-start solve from one table of readings. Random restarts sized from the data.
  // The result has an arbitrary global frame.
  function coldStart(p, y, rng, { restarts = 100, seeds = [] } = {}) {
    const need = maxRigidityRank(p.n + p.k, p.dim);
    const nu = Math.max(p.m - need, 0);
    const sigma2 = p.sigma * p.sigma || 1e-16;
    const accept = acceptableChi2(nu) * sigma2;
    let mean = 0;
    for (let a = 0; a < p.m; a++) mean += p.model === 'strength' ? Math.exp(y[a]) : Math.abs(y[a]);
    mean /= p.m;
    const box = 1.5 * mean;
    let best = null;
    const tryFrom = (x0) => {
      const res = levenberg(p, y, x0);
      if (!best || res.cost < best.cost) best = res;
      return best.cost <= accept;
    };
    for (const s of seeds) if (tryFrom(s)) return { ...best, tries: 1, chi2: best.cost / sigma2 };
    let tries = 0;
    const x0 = new Float64Array(p.N);
    for (let t = 0; t < restarts; t++) {
      for (let a = 0; a < p.N; a++) x0[a] = box * rng.uniform();
      tries++;
      if (tryFrom(x0)) break;
    }
    return { x: best.x, cost: best.cost, tries, chi2: best.cost / sigma2 };
  }

  // Counts the distinct layouts that explain noise-free readings exactly, up to
  // rotation, translation and mirror image.
  function countSolutions(p, y, rng, { restarts = 200, fit = 1e-16, sep = 1e-3 } = {}) {
    const found = [];
    let mean = 0;
    for (let a = 0; a < p.m; a++) mean += Math.abs(y[a]);
    const box = 1.5 * (mean / p.m);
    const x0 = new Float64Array(p.N);
    for (let t = 0; t < restarts; t++) {
      for (let a = 0; a < p.N; a++) x0[a] = box * rng.uniform();
      const { x, cost } = levenberg(p, y, x0, { iters: 150, tol: 1e-16 });
      if (cost > fit) continue;
      let isNew = true;
      for (const f of found) {
        if (shapeRms(x, f, p.dim, { reflect: true }) < sep) { isNew = false; break; }
      }
      if (isNew) found.push(Float64Array.from(x));
    }
    return found;
  }

  // ---------- continuous tracker: iterated Kalman filter + relock ----------
  // Each sample it assumes devices stayed close to where they were, with a motion
  // scale per device. With adaptive motion on, it re-estimates that scale per device
  // from the update itself, so still devices hold position and moving ones are free.
  class Tracker {
    constructor(p, x0, opts = {}) {
      const { motion = 0.02, minMotion = 0.0005, stillTest = 6, learnRate = 0.05, motionMargin = 3, initialStd = 1e-4, anchors = [], adaptive = true, relock = true, innerIters = 4, rng = Rng(1) } = opts;
      this.p = p;
      this.x = Float64Array.from(x0);
      this.motion = motion;
      this.minMotion = minMotion;
      this.stillTest = stillTest;
      this.learnRate = learnRate;
      this.motionMargin = motionMargin;
      this.learned = new Float64Array(p.n + p.k);
      this.adaptive = adaptive;
      this.relock = relock;
      this.innerIters = innerIters;
      this.rng = rng;
      this.anchored = new Uint8Array(p.n + p.k);
      for (const a of anchors) this.anchored[a] = 1;
      this.anchorCount = anchors.length;
      const { N, m } = p;
      this.P = new Float64Array(N * N);
      for (let a = 0; a < p.n + p.k; a++) {
        for (let c = 0; c < p.dim; c++) {
          const i = a * p.dim + c;
          this.P[i * N + i] = this.anchored[a] ? 0 : initialStd * initialStd;
        }
      }
      this.H = new Float64Array(m * N);
      this.PHt = new Float64Array(N * m);
      this.S = new Float64Array(m * m);
      this.h = new Float64Array(m);
      this.work = new Float64Array(m);
      this.Pp = new Float64Array(N * N);
      this.need = maxRigidityRank(p.n + p.k, p.dim);
      this.nu = Math.max(m - this.need, 0);
      this.lastGood = Float64Array.from(x0);
      this.badStreak = 0;
      this.relocks = 0;
      this.chi2 = 0;
      this.flagged = false;
    }

    // One iterated update under per-device motion variances q.
    update(xPrior, Pprior, y, q) {
      const { p, H, PHt, S, h, work } = this;
      const { N, m, dim } = p;
      const sigma2 = p.sigma * p.sigma || 1e-12;
      const Pp = this.Pp;
      Pp.set(Pprior);
      for (let a = 0; a < p.n + p.k; a++) {
        if (this.anchored[a]) continue;
        for (let c = 0; c < dim; c++) { const i = a * dim + c; Pp[i * N + i] += q[a]; }
      }
      const x = Float64Array.from(xPrior);
      const next = new Float64Array(N);
      let cost = costOf(p, x, y, h);
      for (let iter = 0; iter < this.innerIters; iter++) {
        jacobian(p, x, H);
        predict(p, x, h);
        // PHt = P H^T
        for (let i = 0; i < N; i++) {
          for (let a = 0; a < m; a++) {
            let s = 0;
            const ha = a * N;
            for (let c = 0; c < N; c++) { const v = H[ha + c]; if (v !== 0) s += Pp[i * N + c] * v; }
            PHt[i * m + a] = s;
          }
        }
        // S = H P H^T + R
        for (let a = 0; a < m; a++) {
          for (let b = 0; b <= a; b++) {
            let s = 0;
            const ha = a * N;
            for (let c = 0; c < N; c++) { const v = H[ha + c]; if (v !== 0) s += v * PHt[c * m + b]; }
            S[a * m + b] = s;
            S[b * m + a] = s;
          }
          S[a * m + a] += sigma2;
        }
        if (!cholesky(S, m)) return null;
        // Innovation, linearised at the current iterate.
        for (let a = 0; a < m; a++) {
          let s = y[a] - h[a];
          const ha = a * N;
          for (let c = 0; c < N; c++) s -= H[ha + c] * (xPrior[c] - x[c]);
          work[a] = s;
        }
        choleskySolve(S, m, work);
        for (let i = 0; i < N; i++) {
          let s = xPrior[i];
          for (let a = 0; a < m; a++) s += PHt[i * m + a] * work[a];
          next[i] = s;
        }
        let alpha = 1, moved = 0;
        const trial = new Float64Array(N);
        for (let back = 0; back < 5; back++) {
          for (let i = 0; i < N; i++) trial[i] = x[i] + alpha * (next[i] - x[i]);
          const c = costOf(p, trial, y, h);
          if (c <= cost || back === 4) { cost = c; break; }
          alpha *= 0.5;
        }
        for (let i = 0; i < N; i++) { const e = trial[i] - x[i]; moved += e * e; x[i] = trial[i]; }
        if (moved < 1e-14) break;
      }
      return { x, cost };
    }

    // Squared update each device would get from noise alone, read off the gain of the
    // most recent update.
    expectedUpdate() {
      const { p, PHt, S } = this;
      const { N, m, dim } = p;
      const sigma2 = p.sigma * p.sigma || 1e-12;
      const out = new Float64Array(p.n + p.k);
      const col = new Float64Array(m);
      for (let i = 0; i < N; i++) {
        for (let a = 0; a < m; a++) col[a] = PHt[i * m + a];
        choleskySolve(S, m, col);
        let s = 0;
        for (let a = 0; a < m; a++) s += sigma2 * col[a] * col[a];
        out[Math.floor(i / dim)] += s;
      }
      return out;
    }

    // Covariance at the converged iterate (Joseph form).
    finishCovariance(x) {
      const { p, H, PHt, S, Pp } = this;
      const { N, m } = p;
      const sigma2 = p.sigma * p.sigma || 1e-12;
      jacobian(p, x, H);
      for (let i = 0; i < N; i++) {
        for (let a = 0; a < m; a++) {
          let s = 0;
          const ha = a * N;
          for (let c = 0; c < N; c++) { const v = H[ha + c]; if (v !== 0) s += Pp[i * N + c] * v; }
          PHt[i * m + a] = s;
        }
      }
      for (let a = 0; a < m; a++) {
        for (let b = 0; b <= a; b++) {
          let s = 0;
          const ha = a * N;
          for (let c = 0; c < N; c++) { const v = H[ha + c]; if (v !== 0) s += v * PHt[c * m + b]; }
          S[a * m + b] = s;
          S[b * m + a] = s;
        }
        S[a * m + a] += sigma2;
      }
      if (!cholesky(S, m)) return;
      // With a tight prior the fit cannot absorb noise, so the leftover error is larger
      // than with no prior. Its expected size is sigma^2 tr(S^-1).
      const unit = new Float64Array(m);
      let trace = 0;
      for (let a = 0; a < m; a++) {
        unit.fill(0); unit[a] = 1;
        choleskySolve(S, m, unit);
        trace += unit[a];
      }
      this.nuEff = Math.max(this.nu, Math.min(m, sigma2 * trace));
      // P+ = P - PHt S^-1 PHt^T, symmetrised, with a floor to stop it collapsing.
      const col = new Float64Array(m);
      const out = new Float64Array(N * N);
      out.set(Pp);
      for (let i = 0; i < N; i++) {
        for (let a = 0; a < m; a++) col[a] = PHt[i * m + a];
        choleskySolve(S, m, col);
        for (let j = 0; j <= i; j++) {
          let s = 0;
          for (let a = 0; a < m; a++) s += col[a] * PHt[j * m + a];
          out[i * N + j] -= s;
          out[j * N + i] = out[i * N + j];
        }
      }
      for (let i = 0; i < N; i++) if (out[i * N + i] < 1e-14) out[i * N + i] = 1e-14;
      this.P.set(out);
    }

    step(y) {
      const { p } = this;
      const { dim } = p;
      const count = p.n + p.k;
      const q = new Float64Array(count).fill(this.motion * this.motion);
      let res = this.update(this.x, this.P, y, q);
      if (!res) return this.afterUpdate(y, this.x, costOf(p, this.x, y, this.h));
      if (this.adaptive) {
        // Learn each device's motion scale from updates larger than noise alone would
        // cause. Still devices settle near zero and average over time, steady movers
        // settle at their real speed, and a sudden jump is let through at once.
        // The margin allows for steady headings, which a random-step model underrates.
        const expected = this.expectedUpdate();
        for (let a = 0; a < count; a++) {
          let d2 = 0;
          for (let c = 0; c < dim; c++) { const e = res.x[a * dim + c] - this.x[a * dim + c]; d2 += e * e; }
          this.learned[a] += this.learnRate * (d2 - expected[a] - this.learned[a]);
          let var1 = this.minMotion * this.minMotion + (this.motionMargin * Math.max(this.learned[a], 0)) / dim;
          if (d2 > this.stillTest * expected[a]) var1 = Math.max(var1, (d2 * 2) / dim);
          q[a] = var1;
        }
        const again = this.update(this.x, this.P, y, q);
        if (again) res = again;
      }
      this.finishCovariance(res.x);
      return this.afterUpdate(y, res.x, res.cost);
    }

    afterUpdate(y, x, cost) {
      const { p } = this;
      const sigma2 = p.sigma * p.sigma || 1e-12;
      this.x.set(x);
      this.chi2 = cost / sigma2;
      const dof = this.nuEff === undefined ? this.nu : this.nuEff;
      const limit = dof + 3 * Math.sqrt(2 * Math.max(dof, 1)) + 2;
      this.flagged = this.nu > 0 && this.chi2 > limit;
      this.badStreak = this.flagged ? this.badStreak + 1 : 0;
      if (!this.flagged) this.lastGood.set(this.x);
      if (this.relock && this.badStreak >= 2) {
        this.tryRelock(y);
        if (!this.flagged) this.lastGood.set(this.x);
      }
      return this.x;
    }

    tryRelock(y) {
      const { p } = this;
      const { dim, N } = p;
      const count = p.n + p.k;
      const sigma2 = p.sigma * p.sigma || 1e-12;
      const sol = coldStart(p, y, this.rng, { restarts: 16, seeds: [this.x] });
      this.badStreak = 0;
      if (sol.cost >= 0.5 * this.chi2 * sigma2) return;
      // The readings cannot say which way round the layout is. Continuity can, so
      // both orientations are compared with the last estimate that fitted well.
      const ref = this.lastGood;
      const hasAnchors = this.anchorCount >= dim;
      const mirrored = Float64Array.from(sol.x);
      for (let a = 0; a < count; a++) mirrored[a * dim] *= -1;
      // Orientation is chosen over all devices, so neither candidate can cherry-pick.
      const mask0 = hasAnchors ? this.anchored : null;
      const a = aligned(sol.x, ref, dim, { mask: mask0 });
      const b = aligned(mirrored, ref, dim, { mask: mask0 });
      const best = rmsPerDevice(a, ref, dim) <= rmsPerDevice(b, ref, dim) ? sol.x : mirrored;
      let next = rmsPerDevice(a, ref, dim) <= rmsPerDevice(b, ref, dim) ? a : b;
      if (!hasAnchors) {
        // Then the frame is refit to the devices that moved least, so one large move
        // does not drag everything else along.
        const keep = Math.max(dim + 1, Math.ceil(count / 2));
        const err = perDeviceError(next, ref, dim);
        const order = [...err.keys()].sort((u, v) => err[u] - err[v]);
        const mask = new Uint8Array(count);
        for (let i = 0; i < keep; i++) mask[order[i]] = 1;
        next = aligned(best, ref, dim, { mask });
      }
      for (let d = 0; d < count; d++) {
        if (!this.anchored[d]) continue;
        for (let c = 0; c < dim; c++) next[d * dim + c] = this.x[d * dim + c];
      }
      this.x.set(next);
      this.P.fill(0);
      for (let d = 0; d < count; d++) {
        if (this.anchored[d]) continue;
        for (let c = 0; c < dim; c++) { const i = d * dim + c; this.P[i * N + i] = 0.02 * 0.02; }
      }
      this.relocks++;
      this.chi2 = sol.chi2;
      this.flagged = false;
    }
  }

  function perDeviceError(a, b, dim) {
    const count = a.length / dim;
    const out = new Float64Array(count);
    for (let d = 0; d < count; d++) {
      let s = 0;
      for (let c = 0; c < dim; c++) { const e = a[d * dim + c] - b[d * dim + c]; s += e * e; }
      out[d] = Math.sqrt(s);
    }
    return out;
  }

  // ---------- motion scripts ----------
  // Moves `x` in place, inside the unit box. Speed is distance per sample.
  function makeMotion(kind, p, x, rng, { speed = 0.01, anchors = [], every = 30 } = {}) {
    const { dim } = p;
    const count = p.n + p.k;
    const fixed = new Uint8Array(count);
    for (const a of anchors) fixed[a] = 1;
    const free = [...Array(count).keys()].filter((a) => !fixed[a]);
    const vel = new Float64Array(count * dim);
    const unit = (a) => {
      let s = 0;
      for (let c = 0; c < dim; c++) s += vel[a * dim + c] * vel[a * dim + c];
      s = Math.sqrt(s) || 1;
      for (let c = 0; c < dim; c++) vel[a * dim + c] /= s;
    };
    for (let a = 0; a < count; a++) {
      for (let c = 0; c < dim; c++) vel[a * dim + c] = rng.gauss();
      unit(a);
    }
    const walk = (a) => {
      for (let c = 0; c < dim; c++) vel[a * dim + c] += 0.3 * rng.gauss();
      unit(a);
      for (let c = 0; c < dim; c++) {
        const i = a * dim + c;
        x[i] += vel[a * dim + c] * speed;
        if (x[i] < 0) { x[i] = -x[i]; vel[a * dim + c] *= -1; }
        if (x[i] > 1) { x[i] = 2 - x[i]; vel[a * dim + c] *= -1; }
      }
    };
    let t = 0;
    let active = free.length ? free[rng.int(free.length)] : -1;
    return function step() {
      t++;
      if (kind === 'all') {
        for (const a of free) walk(a);
      } else if (kind === 'one') {
        if (t % every === 0 && free.length) active = free[rng.int(free.length)];
        if (active >= 0) walk(active);
      } else if (kind === 'jump') {
        if (t % every === 0 && free.length) {
          const a = free[rng.int(free.length)];
          for (let c = 0; c < dim; c++) x[a * dim + c] = rng.uniform();
        }
      } else if (kind === 'rigid') {
        // Everything translates together along a drifting heading.
        for (let c = 0; c < dim; c++) vel[c] += 0.3 * rng.gauss();
        unit(0);
        for (let a = 0; a < count; a++) for (let c = 0; c < dim; c++) x[a * dim + c] += vel[c] * speed;
      }
      return x;
    };
  }

  return {
    Rng, problem, predict, jacobian, measure, randomPositions,
    cholesky, choleskySolve, symEig, JtJ,
    aligned, alignTransform, shapeRms, absoluteRms, rmsPerDevice,
    maxRigidityRank, rigidity, crlbRms,
    levenberg, coldStart, countSolutions,
    Tracker, makeMotion,
  };
});
