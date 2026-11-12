// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * The demo's live path (_liveLoop in demo.js), on a recording
 * that arrives in real time: are its rows those of the finished file, and how soon?
 *
 * Audio "arrives" on a clock that only moves while the graph runs, by the time each
 * run takes, so the result does not depend on how busy this machine is otherwise.
 * It goes to the stream (ChunkedGraph.live) as it arrives, and whenever windows are
 * ready (readyWindows), the stream runs all of them (at most LIVE_MAX_CLIPS), as the
 * demo does. At the end the rest
 * is a file run that continues from the last window done (graph.run with `from`), and
 * the rows, with the file end set, must equal one run of the whole file.
 *
 * Usage (Node 18 or newer):
 *   cd tests/js && npm install
 *   node rolling_floor_live.mjs model.onnx [--seconds=130,61.9]
 *
 * Audio: the first seconds of tests/test.wav (32 kHz).
 */
import { readFileSync } from 'node:fs';

const root = new URL('../../', import.meta.url);
const ort = await import(new URL('./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs', import.meta.url));
const { ChunkedGraph, fileEnd, LOOKAHEAD, readyWindows } = await import(new URL('docs/javascripts/demo/chunked-graph.js', root));

const args = process.argv.slice(2);
const opt = (name, fallback) => args.find(a => a.startsWith(`--${name}=`))?.split('=')[1] ?? fallback;
const model = args.find(a => a.endsWith('.onnx'));
if (!model) throw new Error('Usage: node rolling_floor_live.mjs model.onnx [--seconds=130,61.9]');
const SECONDS = opt('seconds', '130,61.9').split(',').map(Number);
const SR = 32000, CONF = 0.05, HOP = 1.5, CLIP = 3, LIVE_MAX_CLIPS = 8;

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

const concat = (a, b) => { const o = new Float32Array(a.length + b.length); o.set(a); o.set(b, a.length); return o; };
const median = x => x.slice().sort((a, b) => a - b)[x.length >> 1];

ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';
const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
const graph = await ChunkedGraph.create(ort, new Uint8Array(readFileSync(model)), options);
const pcm = readPcm16Wav(new URL('tests/test.wav', root).pathname);
let failed = false;

for (const seconds of SECONDS) {
  const audio = pcm.slice(0, Math.round(seconds * SR));
  // The finished file in one run
  let file = new Float32Array(0), fileRows = 0;
  for await (const { data, rows } of graph.run(audio, CONF, 8)) { file = concat(file, data); fileRows += rows; }

  // Live: the clock moves by the time each run takes
  // The clock counts samples, so waiting for a window lands exactly on it
  const live = graph.live();
  let clock = 0, next = { segment: 0, clip: 0, clips: 0 }, liveData = new Float32Array(0), liveRows = 0;
  const latency = [], runs = [];
  for (;;) {
    const n = Math.min(audio.length, clock);
    if (n > live.length) live.push(audio.subarray(live.length, n));
    const ready = readyWindows(n);
    if (ready <= next.clips) {
      // Wait for the next window to be ready, or for the end of the recording
      const need = (next.clips * HOP + CLIP) * SR + LOOKAHEAD;
      if (need > audio.length) break;
      clock = need;
      continue;
    }
    const count = Math.min(ready - next.clips, LIVE_MAX_CLIPS);
    const t0 = performance.now();
    const res = await live.windows(CONF, next, count);
    const ms = performance.now() - t0;
    runs.push(ms);
    clock += Math.round(ms / 1000 * SR);
    for (let w = next.clips; w < next.clips + count; w++) latency.push(clock / SR - (w * HOP + CLIP));
    liveData = concat(liveData, res.data);
    liveRows += res.rows;
    next = res.next;
  }
  // Stop: the file run goes on from the last window done
  const doneLive = next.clips;
  for (let r = 0; r < liveRows; r++) liveData[r * 9 + 8] = fileEnd(audio.length);
  for await (const { data, rows } of graph.run(audio, CONF, 8, next)) { liveData = concat(liveData, data); liveRows += rows; }

  let differ = 0;
  if (liveRows !== fileRows) differ = Math.abs(liveRows - fileRows) || 1;
  else for (let i = 0; i < liveData.length; i++) if (liveData[i] !== file[i]) differ++;
  failed ||= differ > 0;
  console.log(`${model.split('/').pop()}, ${seconds} s: ${doneLive} windows live, ${liveRows} rows; ` +
    (differ ? `${differ} values differ from the finished file` : 'identical to the finished file') +
    `; a run ${(median(runs) / 1000).toFixed(2)} s median (${runs.length} runs), boxes ${median(latency).toFixed(1)} s ` +
    `after a window is complete (median), ${Math.max(...latency).toFixed(1)} s at most`);
}
process.exitCode = failed ? 1 : 0;
