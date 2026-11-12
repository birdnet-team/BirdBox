// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * BirdBox window fusion in JS, a float32 port of merge_raw_songs() with
 * merge_method "fusion" in src/inference/onnx_export.py.
 *
 * Input are the per-window boxes of the graph (its raw_all tensor, 9 columns:
 * clip, t0, t1, f_lo, f_hi, score, class, clip start, file end), computed once
 * at the lowest confidence. NMS only lets higher-scoring boxes suppress lower
 * ones, so keeping the rows above a threshold equals a graph run at that
 * threshold, and fusing them here reproduces the graph's output for any
 * (confidence, phrase gap) setting.
 *
 * Steps, as in the graph:
 *   1. Window ownership: a box counts only in the window whose middle half
 *      holds its centre; boxes of one call seen by overlapping windows are
 *      averaged (weighted box fusion, IoU >= 0.4).
 *   2. Boxes lying at least 80% inside a larger box of the same class go,
 *      before the join, so a dropped box cannot reshape a joined one.
 *   3. Join: phrase gap > 0 links boxes closer than the gap; 0 keeps calls
 *      and links only boxes that overlap by 0.05 s. Both need frequency
 *      IoU >= 0.8. Linked groups become one box, repeated until none joins.
 *
 * Each step only links boxes that overlap in time (1, 2) or lie within the
 * phrase gap (3). So where the rows have a wider gap in time, the boxes on
 * either side are fused on their own, with the same result: a long recording
 * is fused part by part, and the parts of a detection in progress that later
 * windows cannot reach are fused once and kept (cache in fuseBoxes).
 */

const f32 = Math.fround;
const COLS = 9;
const CLIP = 3;
const WBF_IOU = f32(0.4);
const JOIN_GAP = f32(-0.05);
const JOIN_FREQ_IOU = f32(0.8);
const NESTED_RATIO = f32(0.8);
// Parts are split where boxes are this much further apart than any step links: fused
// boxes are float32 means of their rows, so they may reach a little past them
const CUT_MARGIN = 0.01;

/** Stable argsort of 0..n-1 by key (ascending). */
const argsort = (n, key) => Array.from({ length: n }, (_, i) => i).sort((a, b) => key(a) - key(b) || a - b);

/** Step 1 on rows from..to-1: rows [t0, t1, flo, fhi, score, cls] plus the raw boxes behind each. */
function fuseWindows(raw, from, to, conf) {
  const c = f32(conf), quarter = CLIP / 4, dets = [];
  for (let r = from; r < to; r++) {
    const d = raw.subarray(r * COLS, r * COLS + COLS);
    if (!(d[5] > c)) continue;
    const clipStart = d[7], clipEnd = f32(clipStart + CLIP), centre = f32(f32(d[1] + d[2]) * 0.5);
    const lo = clipStart > f32(1e-6) ? f32(clipStart + quarter) : -1e9;
    const hi = clipEnd < f32(d[8] - f32(1e-3)) ? f32(clipEnd - quarter) : 1e9;
    if (centre >= lo && centre < hi) dets.push(d);
  }
  const order = argsort(dets.length, i => -dets[i][5]);
  const D = order.map(i => dets[i]);
  const area = D.map(d => f32(f32(d[2] - d[1]) * f32(d[4] - d[3])));
  const used = new Uint8Array(D.length), out = [];
  for (let i = 0; i < D.length; i++) {
    if (used[i]) continue;
    const a = D[i], members = [];
    for (let j = 0; j < D.length; j++) {
      if (j !== i && (used[j] || D[j][6] !== a[6])) continue;
      const b = D[j];
      const ti = Math.max(0, f32(Math.min(a[2], b[2]) - Math.max(a[1], b[1])));
      const fi = Math.max(0, f32(Math.min(a[4], b[4]) - Math.max(a[3], b[3])));
      const inter = f32(ti * fi), union = f32(f32(area[i] + area[j]) - inter);
      const iou = union > 0 ? f32(inter / Math.max(union, f32(1e-12))) : 0;
      if (j === i || iou >= WBF_IOU) members.push(j);
    }
    let total = 0;
    for (const j of members) { used[j] = 1; total = f32(total + D[j][5]); }
    const weight = j => (total > 0 ? D[j][5] : 1);
    if (!(total > 0)) total = members.length;
    const avg = k => {
      let s = 0;
      for (const j of members) s = f32(s + f32(weight(j) * D[j][k]));
      return f32(s / total);
    };
    out.push({ row: [avg(1), avg(2), avg(3), avg(4), a[5], a[6]], n: members.length });
  }
  return out;
}

/** 1-D IoU of two frequency bands, zero when they do not overlap. */
function freqIou(loA, hiA, loB, hiB) {
  const inter = f32(Math.min(hiA, hiB) - Math.max(loA, loB));
  const union = f32(f32(f32(hiA - loA) + f32(hiB - loB)) - Math.max(inter, 0));
  if (inter <= 0 || union <= 0) return 0;
  return f32(inter / Math.max(union, f32(1e-12)));
}

/**
 * Step 3, the link join (link_songs_tensor in onnx_export.py): two boxes of
 * one class are linked when their time gap is at most `gap` (negative: they
 * overlap by at least that much) and their frequency IoU is at least
 * `minIou`; every connected group becomes one box. This repeats on the groups
 * until none joins another. Groups [t0, t1, flo, fhi, sum, max, n, cls], in
 * order of their first box (by class, then start), then sorted by start.
 */
function joinBoxes(fused, gap, minIou) {
  const order = argsort(fused.length, i => fused[i].row[5] * 1e6 + fused[i].row[0]);
  let groups = order.map(i => {
    const [t0, t1, flo, fhi, score, cls] = fused[i].row, w = fused[i].n;
    return [t0, t1, flo, fhi, f32(score * w), score, w, cls];
  });
  for (;;) {
    const m = groups.length, parent = Array.from({ length: m }, (_, i) => i);
    const root = i => { while (parent[i] !== i) i = parent[i] = parent[parent[i]]; return i; };
    for (let a = 0; a < m; a++) {
      const A = groups[a];
      for (let b = a + 1; b < m; b++) {
        const B = groups[b];
        if (A[7] !== B[7] || !(f32(Math.max(A[0], B[0]) - Math.min(A[1], B[1])) <= gap)) continue;
        if (!(freqIou(A[2], A[3], B[2], B[3]) >= minIou)) continue;
        const ra = root(a), rb = root(b);
        if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
      }
    }
    const merged = [], slot = new Map();
    for (let i = 0; i < m; i++) {
      const r = root(i), g = groups[i];
      if (!slot.has(r)) { slot.set(r, merged.length); merged.push([...g]); continue; }
      const s = merged[slot.get(r)];
      s[0] = Math.min(s[0], g[0]); s[1] = Math.max(s[1], g[1]);
      s[2] = Math.min(s[2], g[2]); s[3] = Math.max(s[3], g[3]);
      s[4] = f32(s[4] + g[4]); s[5] = Math.max(s[5], g[5]); s[6] = f32(s[6] + g[6]);
    }
    if (merged.length === m) break;
    groups = merged;
  }
  return argsort(groups.length, i => groups[i][0]).map(i => groups[i]);
}

/** Step 2 (nested_keep_tensor): drop fused boxes at least NESTED_RATIO inside a larger box of the same class. */
function dropNested(fused) {
  const rows = fused.map(x => x.row);
  const area = rows.map(s => f32(f32(s[1] - s[0]) * f32(s[3] - s[2])));
  return fused.filter((_, i) => !rows.some((b, j) => {
    const a = rows[i];
    if (j === i || b[5] !== a[5]) return false;
    const larger = area[j] > area[i] || (area[j] === area[i] && (b[4] > a[4] || (b[4] === a[4] && j < i)));
    if (!larger) return false;
    const ti = Math.max(0, f32(Math.min(a[1], b[1]) - Math.max(a[0], b[0])));
    const fi = Math.max(0, f32(Math.min(a[3], b[3]) - Math.max(a[2], b[2])));
    return f32(ti * fi) >= f32(NESTED_RATIO * area[i]);
  }));
}

/**
 * Boxes of rows from..to-1, sorted by start. With `split`, null if a box of step 1 has
 * no area: step 2 drops such a box for a larger one anywhere, so it cannot be fused apart.
 */
function fuseRange(raw, from, to, conf, gap, split) {
  const windows = fuseWindows(raw, from, to, conf);
  if (split && windows.some(({ row: s }) => !(f32(f32(s[1] - s[0]) * f32(s[3] - s[2])) > 0))) return null;
  return joinBoxes(dropNested(windows), gap, JOIN_FREQ_IOU)
    .map(s => ({ t0: s[0], t1: s[1], flo: s[2], fhi: s[3], avg: f32(s[4] / s[6]), max: s[5], n: s[6], cls: s[7] }));
}

/**
 * Where rows from..to-1 can be fused apart: row indices `at` where every box above the
 * confidence in the rows before ends more than `sep` before every such box from `at` on
 * starts. `end` is where the boxes before `at` (from `from` on) end.
 */
function cuts(raw, from, to, conf, sep) {
  const c = f32(conf), starts = new Float64Array(to - from + 1);
  starts[to - from] = Infinity;
  for (let r = to - 1; r >= from; r--) {
    const o = r * COLS;
    starts[r - from] = raw[o + 5] > c ? Math.min(starts[r - from + 1], raw[o + 1]) : starts[r - from + 1];
  }
  const out = [];
  let end = -Infinity, boxes = false;
  for (let r = from; r < to - 1; r++) {
    const o = r * COLS;
    if (raw[o + 5] > c) { end = Math.max(end, raw[o + 2]); boxes = true; }
    const next = starts[r + 1 - from];
    if (boxes && next < Infinity && next - end > sep) {
      out.push({ at: r + 1, end });
      boxes = false;
    }
  }
  return out;
}

/** Whether the first n rows of a and b are the same. */
function sameRows(a, b, n) {
  for (let i = 0; i < n * COLS; i++) if (a[i] !== b[i]) return false;
  return true;
}

/**
 * Boxes of the BirdBox graph for one (confidence, phrase gap) setting.
 *
 * While detection runs, pass a `cache` (an empty object, kept from one call to the next)
 * and `later`, the earliest start of a box in rows still to come: the start of the next
 * window. The boxes of the parts no later row can reach are kept, and the next call only
 * fuses the rows after them. The cache holds while its rows stay the same (a recording
 * that becomes its file changes them) and so do the confidence and the phrase gap.
 *
 * @param {Float32Array} raw  raw_all rows, 9 columns each
 * @param {number} rows
 * @param {number} conf       confidence threshold (boxes must score above it)
 * @param {number} phraseGap  seconds; 0 keeps call-level boxes
 * @param {object} [cache]
 * @param {number} [later]    seconds; Infinity when no more rows come
 */
export function fuseBoxes(raw, rows, conf, phraseGap, cache = null, later = Infinity) {
  const gap = f32(phraseGap) > 0 ? f32(phraseGap) : JOIN_GAP;
  const sep = Math.max(gap, 0) + CUT_MARGIN, key = `${conf}|${phraseGap}`;
  let from = 0, songs = [];
  if (cache) {
    if (cache.key === key && cache.rows <= rows && sameRows(cache.raw, raw, cache.rows)) {
      from = cache.rows;
      songs = cache.songs.slice();
    } else Object.assign(cache, { key, rows: 0, songs: [], raw: new Float32Array(0) });
  }
  let a = from, kept = null;
  for (const cut of [...cuts(raw, from, rows, conf, sep), { at: rows, end: Infinity }]) {
    const part = fuseRange(raw, a, cut.at, conf, gap, true);
    if (!part) {
      // A box without area: all rows at once, and nothing kept
      if (cache) Object.assign(cache, { rows: 0, songs: [], raw: new Float32Array(0) });
      return fuseRange(raw, 0, rows, conf, gap, false);
    }
    for (const s of part) songs.push(s);
    a = cut.at;
    if (later - cut.end > sep) kept = { rows: cut.at, count: songs.length };
  }
  if (cache && kept) Object.assign(cache, { rows: kept.rows, songs: songs.slice(0, kept.count), raw: raw.slice(0, kept.rows * COLS) });
  return songs;
}
