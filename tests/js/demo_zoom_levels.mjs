// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Zoomed out, the demo's spectrogram shows in each column the maximum of the
 * frames behind it (SpecView in docs/javascripts/demo/spec-view.js). Only every
 * fourth zoom level keeps its columns; the tiles of the levels between are
 * computed from the kept level below them. Does every tile show the maximum of
 * the frames loaded so far, also while frames keep arriving?
 *
 * Frames arrive as the demo gets them: a file in slices of 1024 frames, a file at
 * once, and a recording a few frames at a time, its tiles filling up instead of
 * being drawn anew. After every arrival, tiles of all zoom levels are "drawn" the
 * way the viewer draws them (cached, and repainted where the viewer marks them
 * changed), and every column must equal the maximum of its frames: a column left
 * at an earlier value fails.
 *
 * Usage (Node 18 or newer, no npm install needed):
 *   cd tests/js && node demo_zoom_levels.mjs [--minutes=50]
 */
const root = new URL('../../', import.meta.url);
const { SpecView } = await import(new URL('docs/javascripts/demo/spec-view.js', root));

const args = process.argv.slice(2);
const opt = (name, fallback) => args.find(a => a.startsWith(`--${name}=`))?.split('=')[1] ?? fallback;
const MINUTES = +opt('minutes', 50);
const TILE = 1024, FPS = 100;

let seed = 1;
const rand = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };

/** Frame f, band r: low noise, now and then loud calls, and stretches of silence. */
function value(f, r) {
  if (((f >> 9) % 11) === 0) return 0;
  let h = Math.imul(f, 0x9e3779b1) ^ Math.imul(r + 1, 0x85ebca77);
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
  h = (h ^ (h >>> 12)) & 0xff;
  if (((f >> 6) % 13) === 5 && r > 10 && r < 30) return 200 + (h & 55);
  return h < 250 ? h >> 2 : h;
}
const frames = (start, count, rows) => Uint8Array.from({ length: count * rows }, (_, i) => value(start + Math.floor(i / rows), i % rows));

/** A viewer without its page: the data side of SpecView only. */
function viewer(total, rows, live) {
  const v = Object.create(SpecView.prototype);
  Object.assign(v, { root: {}, opts: { axis: { type: 'linear', fMin: 0, fMax: 16000 } }, display: { cmap: 'inferno', scale: 'linear', brightness: 0, contrast: 0 },
    tiles: new Map(), pooled: new Map(), version: 0, render() {}, _syncControls() {} });
  v.setSource({ frames: total, rows, fps: FPS, duration: total / FPS });
  if (live) v.liveMode = { follow: true };   // tiles keep what they show and fill up, as while recording
  return v;
}

/**
 * The test's own zoom levels, all of them: column c of level k is the maximum of
 * columns 2c and 2c + 1 of level k − 1, as far as frames have arrived. Kept up to
 * date column by column as frames arrive.
 */
function reference(total, rows) {
  const levels = [new Uint8Array(total * rows)];
  for (let n = total; n > 1; n = Math.ceil(n / 2)) levels.push(new Uint8Array(Math.ceil(n / 2) * rows));
  let loaded = 0;
  return {
    levels,
    add(start, data) {
      levels[0].set(data, start * rows);
      const before = loaded;
      loaded = Math.max(loaded, start + data.length / rows);
      for (let k = 1; k < levels.length; k++) {
        const below = levels[k - 1], out = levels[k], belowBuilt = Math.ceil(loaded / 2 ** (k - 1));
        for (let c = Math.floor(Math.min(before, start) / 2 ** k); c < Math.ceil(loaded / 2 ** k); c++) {
          for (let r = 0; r < rows; r++) {
            const x = below[2 * c * rows + r], y = 2 * c + 1 < belowBuilt ? below[(2 * c + 1) * rows + r] : 0;
            out[c * rows + r] = x > y ? x : y;
          }
        }
      }
    },
  };
}

/** What _tile does with the columns, without a canvas: the columns a tile shows, kept per tile. */
const shown = new Map();
function draw(v, k, index) {
  const key = `${k}:${index}`, level = v._level(k), rows = v.rows, c0 = index * TILE, width = Math.min(TILE, level.built - c0);
  if (width <= 0) return 0;
  let tile = v.tiles.get(key);
  if (!tile || tile.painted < width) {
    const from = tile && v.liveMode ? tile.painted : 0;
    const src = level.data || v._pooled(k, index, width), first = level.data ? c0 + from : from;
    const cols = (from && shown.get(`${v.id}/${key}`)) || new Uint8Array(TILE * rows);
    cols.set(src.subarray(first * rows, (first + width - from) * rows), from * rows);
    shown.set(`${v.id}/${key}`, cols);
    tile = { canvas: null, version: v.version, painted: width };
    v.tiles.set(key, tile);
  }
  // Every column the tile shows must be the maximum of its frames loaded so far
  const cols = shown.get(`${v.id}/${key}`), want = v.reference.levels[k];
  for (let i = 0; i < width * rows; i++) {
    if (cols[i] !== want[c0 * rows + i]) {
      const c = Math.floor(i / rows), r = i % rows;
      throw new Error(`level ${k}, tile ${index}, column ${c}, band ${r}: shows ${cols[i]}, its frames' maximum is ${want[c0 * rows + i]} (${v.loadedFrames} frames loaded)`);
    }
  }
  return width;
}

/** Tiles to draw after an arrival: the newest of every level, and a few others. */
function drawSome(v) {
  let tiles = 0;
  const top = Math.ceil(Math.log2(Math.max(2, v.loadedFrames)));
  for (let k = 0; k <= top; k++) {
    const last = Math.max(0, Math.ceil(v.loadedFrames / 2 ** k / TILE) - 1);
    tiles += draw(v, k, last) > 0;
    if (rand() < 0.3) tiles += draw(v, k, Math.floor(rand() * (last + 1))) > 0;
  }
  return tiles;
}

const cases = [
  { id: 'file in slices', rows: 64, minutes: MINUTES, arrive: () => 1024 },
  { id: 'file at once', rows: 97, minutes: MINUTES / 5, arrive: (n, total) => total },
  { id: 'recording', rows: 64, minutes: MINUTES / 5, live: true, arrive: () => 1 + Math.floor(rand() * 40) },
  { id: 'recording, single frames', rows: 31, minutes: 1, live: true, arrive: () => 1 },
];
let failed = 0;
for (const c of cases) {
  const total = Math.round(c.minutes * 60 * FPS) + 37, v = viewer(total, c.rows, c.live);
  v.id = c.id;
  v.reference = reference(total, c.rows);
  const t0 = performance.now();
  let arrivals = 0, drawn = 0, keptBytes = 0, pooledBytes = 0;
  try {
    for (let n = 0; n < total;) {
      const count = Math.min(total - n, c.arrive(n, total)), data = frames(n, count, c.rows);
      if (c.live) v.addFrames(n, data); else v.addSlice(n, data);
      v.reference.add(n, data);
      n += count;
      arrivals++;
      drawn += drawSome(v);
      // A display change draws every tile anew
      if (rand() < 0.01) { v.tiles.clear(); v.version++; }
    }
    // The whole file once more, every tile of every level
    for (let k = 0; 2 ** k < 2 * total; k++) for (let i = 0; i * TILE * 2 ** k < total; i++) drawn += draw(v, k, i) > 0;
    keptBytes = v.levels.slice(1).reduce((s, l) => s + (l?.data?.length || 0), 0);
    pooledBytes = [...v.pooled.values()].reduce((s, p) => s + p.data.length, 0);
    const frameBytes = total * c.rows;
    if (keptBytes > frameBytes / 14) throw new Error(`the kept zoom levels hold ${keptBytes} bytes, more than 1/14 of the frames' ${frameBytes}`);
    console.log(`ok   ${c.id}: ${total} frames x ${c.rows} bands in ${arrivals} arrivals, ${drawn} tiles checked; zoom levels kept ${(keptBytes / frameBytes * 100).toFixed(1)}% of the frames, `
      + `pooled tiles ${(pooledBytes / 1e6).toFixed(1)} MB (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
  } catch (err) {
    failed++;
    console.log(`FAIL ${c.id}: ${err.message}`);
  }
}
if (failed) process.exit(1);
