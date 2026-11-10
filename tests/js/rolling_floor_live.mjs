// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Experimental rolling floor (BIRDBOX_FLOOR_WINDOW, onnx_export.py): is a window's
 * rows final as soon as a recording reaches the window's end?
 *
 * For every window, the graph's features run on the recording so far, which ends
 * at that window's end, with the segment of that window (`iter`). The window then
 * goes through the network alone, and its rows are compared with the rows of the
 * finished file (all columns but the clip index and the file end). With the 60 s
 * segment floor most windows differ, because their floor depends on later audio.
 *
 * Usage (Node 18 or newer):
 *   cd tests/js && npm install
 *   node rolling_floor_live.mjs model.onnx [--seconds=130] [--every=1]
 *
 * Audio: the first seconds of tests/test.wav (32 kHz). The last two windows of the
 * cut are left out: they see the end of the file.
 */
import { readFileSync } from 'node:fs';

const root = new URL('../../', import.meta.url);
const ort = await import(new URL('./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs', import.meta.url));
const { ChunkedGraph } = await import(new URL('docs/javascripts/demo/chunked-graph.js', root));

const args = process.argv.slice(2);
const opt = (name, fallback) => args.find(a => a.startsWith(`--${name}=`))?.split('=')[1] ?? fallback;
const model = args.find(a => a.endsWith('.onnx'));
if (!model) throw new Error('Usage: node rolling_floor_live.mjs model.onnx [--seconds=130] [--every=1]');
const SECONDS = +opt('seconds', 130), EVERY = +opt('every', 1);
const SR = 32000, CONF = 0.05, HOP = 1.5, CLIP = 3, SEGMENT = 60;

function readPcm16Wav(path) {
  const buf = readFileSync(path), dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  for (let p = 12; p + 8 <= buf.length;) {
    const id = buf.toString('ascii', p, p + 4), size = dv.getUint32(p + 4, true);
    if (id === 'data') {
      const out = new Float32Array(Math.floor(size / 2));
      for (let i = 0; i < out.length; i++) out[i] = dv.getInt16(p + 8 + i * 2, true) / 32768;
      return out;
    }
    p += 8 + size + (size & 1);
  }
  throw new Error(`${path}: no data chunk`);
}

ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';
const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
const audio = readPcm16Wav(new URL('tests/test.wav', root).pathname).subarray(0, SECONDS * SR);
const graph = await ChunkedGraph.create(ort, new Uint8Array(readFileSync(model)), options);
const key = (d, o) => [1, 2, 3, 4, 5, 6, 7].map(i => d[o + i].toFixed(5)).join(',');
const int64 = k => new ort.Tensor('int64', BigInt64Array.of(BigInt(k)), []);

// Rows of the finished file per window start (column 7)
const file = new Map();
for await (const { data, rows } of graph.run(audio, CONF, 8)) {
  for (let r = 0; r < rows; r++) {
    const start = data[r * 9 + 7].toFixed(2);
    if (!file.has(start)) file.set(start, []);
    file.get(start).push(key(data, r * 9));
  }
}

const windows = Math.floor((audio.length / SR - CLIP) / HOP) + 1;
let same = 0, differ = 0, rows = 0;
const ms = [];
for (let c = 0; c < windows - 2; c += EVERY) {
  const start = c * HOP, n = (start + CLIP) * SR;
  const piece = new ort.Tensor('float32', audio.subarray(0, n), [n]);
  const t0 = performance.now();
  const out = await graph.features.run({ audio: piece, audio_in: piece, iter: int64(Math.floor(start / SEGMENT)) });
  const clips = out[graph.clips.from], times = out[graph.times.from], [, mels, frames] = clips.dims;
  const slot = [...times.data].findIndex(t => Math.abs(t - start) < 1e-3);
  if (slot < 0) throw new Error(`window at ${start} s missing from the features`);
  const feeds = {
    [graph.clips.to]: new ort.Tensor('float32', clips.data.subarray(slot * mels * frames, (slot + 1) * mels * frames), [1, mels, frames]),
    [graph.times.to]: new ort.Tensor('float32', times.data.subarray(slot, slot + 1), [1]),
    conf_in: new ort.Tensor('float32', Float32Array.of(CONF), [1]),
  };
  for (const name of graph.shared) feeds[name] = out[name];
  const raw = (await graph.detect.run(feeds)).raw;
  ms.push(performance.now() - t0);
  const live = [];
  for (let r = 0; r < (raw.dims[0] || 0); r++) live.push(key(raw.data, r * 9));
  rows += live.length;
  const ref = file.get(start.toFixed(2)) ?? [];
  const ok = live.length === ref.length && live.slice().sort().join('|') === ref.slice().sort().join('|');
  if (ok) same++;
  else if (++differ <= 3) console.log(`  window at ${start} s: ${live.length} rows live, ${ref.length} in the file`);
}
ms.sort((a, b) => a - b);
console.log(`${model.split('/').pop()}, ${SECONDS} s: ${same} windows identical to the finished file, ${differ} differ ` +
  `(${rows} rows at conf ${CONF}); features and network per window ${(ms[ms.length >> 1] / 1000).toFixed(2)} s median`);
process.exitCode = differ ? 1 : 0;
