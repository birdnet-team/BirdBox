// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Which boxes may still change while detection runs: the demo draws them dashed.
 *
 * Rows arrive window by window, and a row is final once it is there. The boxes merged
 * from them are not: the next windows start 0.75 s before the line (the time up to which
 * every window that owns a box has been seen), and their rows can still change a box
 * near it. Window fusion can fuse a box with theirs, hide it as nested, or join it into
 * a phrase up to the phrase gap away. A phrase that grows also widens its band, and can
 * then take in a box beside it that ended long before. So a box may change
 *
 *   - when it ends less than one window step (1.5 s) plus the gap before the line, or
 *   - when it is linked to a box that may change: of the same class, at most the gap
 *     apart in time and, for window fusion, with overlapping bands. Song reconstruction
 *     does not compare frequencies.
 *
 * tests/js/demo_dashed_boxes.mjs checks that no box this leaves solid changes later, in
 * files and while recording, for every model in the manifest and the slider's range.
 */

const SETTLED = 1.5;   // one window step

/**
 * @param {Array<{t0: number, t1: number, flo: number, fhi: number, cls: number}>} boxes
 *   merged boxes (fusion.js or songs.js)
 * @param {number|null} line  seconds up to which every window that owns a box has been
 *   seen; null when detection is done, and nothing may change
 * @param {number} gap  phrase gap (fusion) or song gap (reconstruct), seconds
 * @param {'fusion'|'reconstruct'} merge
 * @returns {Set<object>} the boxes that may still change
 */
export function pendingBoxes(boxes, line, gap, merge) {
  const pending = new Set();
  if (line == null) return pending;
  const todo = [];
  for (const b of boxes) if (b.t1 > line - SETTLED - gap) { pending.add(b); todo.push(b); }
  const bands = merge === 'fusion';
  while (todo.length) {
    const p = todo.pop();
    for (const b of boxes) {
      if (pending.has(b) || b.cls !== p.cls) continue;
      if (Math.max(b.t0, p.t0) - Math.min(b.t1, p.t1) > gap) continue;
      if (bands && (b.fhi <= p.flo || p.fhi <= b.flo)) continue;
      pending.add(b);
      todo.push(b);
    }
  }
  return pending;
}
