// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * BirdBox song reconstruction in JS, a float32 port of merge_raw_songs() with
 * merge_method "reconstruct" (reconstruct_songs_tensor) in
 * src/inference/onnx_export.py, the merge of the species models such as
 * All-In-One.
 *
 * Input are the per-window boxes of the graph (its raw_all tensor, 9 columns:
 * clip, t0, t1, f_lo, f_hi, score, class, clip start, file end), computed once
 * at the lowest confidence. As with fusion.js, keeping the rows above a
 * threshold equals a graph run at that threshold.
 *
 * Boxes of one class are sorted by start time; a box joins the open song while
 * it starts at most `songGap` seconds after the song's end, otherwise it opens
 * a new one. Frequencies are not compared.
 */

const f32 = Math.fround;
const COLS = 9;

/**
 * Boxes of the BirdBox graph for one (confidence, song gap) setting.
 * @param {Float32Array} raw  raw_all rows, 9 columns each
 * @param {number} rows
 * @param {number} conf       confidence threshold (boxes must score above it)
 * @param {number} songGap    seconds
 */
export function reconstructSongs(raw, rows, conf, songGap) {
  const c = f32(conf), gap = f32(songGap), dets = [];
  for (let r = 0; r < rows; r++) {
    const d = raw.subarray(r * COLS, r * COLS + COLS);
    if (d[5] > c) dets.push(d);
  }
  // Class, then start time; the graph sorts this key in float64
  const key = d => d[6] * 1e6 + d[1];
  dets.sort((a, b) => key(a) - key(b));
  const songs = [];
  let s = null;
  for (const d of dets) {
    if (s && d[6] === s.cls && f32(d[1] - s.t1) <= gap) {
      s.t1 = Math.max(s.t1, d[2]); s.flo = Math.min(s.flo, d[3]); s.fhi = Math.max(s.fhi, d[4]);
      s.sum = f32(s.sum + d[5]); s.max = Math.max(s.max, d[5]); s.n++;
    } else {
      s = { t0: d[1], t1: d[2], flo: d[3], fhi: d[4], sum: d[5], max: d[5], n: 1, cls: d[6] };
      songs.push(s);
    }
  }
  return songs
    .map(({ sum, ...b }) => ({ ...b, avg: f32(sum / b.n) }))
    .sort((a, b) => a.t0 - b.t0);
}
