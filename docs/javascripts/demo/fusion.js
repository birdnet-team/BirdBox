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
 *   2. Join: phrase gap > 0 joins same-band boxes (frequency IoU >= 0.3)
 *      closer than the gap; 0 keeps calls and joins only boxes that overlap
 *      by 0.05 s with frequency IoU >= 0.8.
 *   3. Boxes lying at least 80% inside a larger box of the same class go.
 */

const f32 = Math.fround;
const COLS = 9;
const CLIP = 3;
const WBF_IOU = f32(0.4);
const JOIN_GAP = f32(-0.05);
const JOIN_FREQ_IOU = f32(0.8);
const PHRASE_FREQ_IOU = f32(0.3);
const NESTED_RATIO = f32(0.8);

/** Stable argsort of 0..n-1 by key (ascending). */
const argsort = (n, key) => Array.from({ length: n }, (_, i) => i).sort((a, b) => key(a) - key(b) || a - b);

/** Step 1: rows [t0, t1, flo, fhi, score, cls] plus the raw boxes behind each. */
function fuseWindows(raw, rows, conf) {
  const c = f32(conf), quarter = CLIP / 4, dets = [];
  for (let r = 0; r < rows; r++) {
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

/** Step 2: songs [t0, t1, flo, fhi, sum, max, n, cls]; n counts raw boxes. */
function joinBoxes(fused, gap, minIou) {
  const order = argsort(fused.length, i => fused[i].row[5] * 1e6 + fused[i].row[0]);
  const songs = [];
  let active = [];
  for (const i of order) {
    const [t0, t1, flo, fhi, score, cls] = fused[i].row, w = fused[i].n;
    let best = -1, bestIou = -1;
    const kept = [];
    for (const s of active) {
      if (f32(t0 - s[1]) > gap || s[7] !== cls) { songs.push(s); continue; }
      const iou = freqIou(s[2], s[3], flo, fhi);
      if (iou >= minIou && iou > bestIou) { best = kept.length; bestIou = iou; }
      kept.push(s);
    }
    active = kept;
    if (best < 0) active.push([t0, t1, flo, fhi, f32(score * w), score, w, cls]);
    else {
      const s = active[best];
      s[1] = Math.max(s[1], t1); s[2] = Math.min(s[2], flo); s[3] = Math.max(s[3], fhi);
      s[4] = f32(s[4] + f32(score * w)); s[5] = Math.max(s[5], score); s[6] = f32(s[6] + w);
    }
  }
  songs.push(...active);
  return argsort(songs.length, i => songs[i][0]).map(i => songs[i]);
}

/** Step 3: drop boxes at least NESTED_RATIO inside a larger box of the same class. */
function dropNested(songs) {
  const area = songs.map(s => f32(f32(s[1] - s[0]) * f32(s[3] - s[2])));
  return songs.filter((a, i) => !songs.some((b, j) => {
    if (j === i || b[7] !== a[7]) return false;
    const larger = area[j] > area[i] || (area[j] === area[i] && (b[5] > a[5] || (b[5] === a[5] && j < i)));
    if (!larger) return false;
    const ti = Math.max(0, f32(Math.min(a[1], b[1]) - Math.max(a[0], b[0])));
    const fi = Math.max(0, f32(Math.min(a[3], b[3]) - Math.max(a[2], b[2])));
    return f32(ti * fi) >= f32(NESTED_RATIO * area[i]);
  }));
}

/**
 * Boxes of the BirdBox graph for one (confidence, phrase gap) setting.
 * @param {Float32Array} raw  raw_all rows, 9 columns each
 * @param {number} rows
 * @param {number} conf       confidence threshold (boxes must score above it)
 * @param {number} phraseGap  seconds; 0 keeps call-level boxes
 */
export function fuseBoxes(raw, rows, conf, phraseGap) {
  const phrase = f32(phraseGap) > 0;
  const fused = fuseWindows(raw, rows, conf);
  const songs = joinBoxes(fused, phrase ? f32(phraseGap) : JOIN_GAP, phrase ? PHRASE_FREQ_IOU : JOIN_FREQ_IOU);
  return dropNested(songs).map(s => ({ t0: s[0], t1: s[1], flo: s[2], fhi: s[3], avg: f32(s[4] / s[6]), max: s[5], n: s[6], cls: s[7] }));
}
