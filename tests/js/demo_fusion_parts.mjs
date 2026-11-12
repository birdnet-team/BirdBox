// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * While detection runs, the demo fuses the boxes of the parts no later window can
 * reach once and keeps them (fuseBoxes with a cache, docs/javascripts/demo/fusion.js),
 * so an update only fuses the newest rows. Are its boxes those of fusing all rows anew?
 *
 * Detection runs the way the demo runs it: a file one window at a time
 * (ChunkedGraph.run), and a recording, whose rows become those of its file at the
 * end. After every window, for every setting of the sliders, the boxes with the
 * cache must equal those without, field for field and in the same order. Without
 * a cache, fuseBoxes splits the rows where fusion cannot link boxes;
 * demo_merge_parity.mjs checks that against the graph's own merge.
 *
 * Usage (Node 18 or newer):
 *   cd tests/js && npm install
 *   node demo_fusion_parts.mjs [model.onnx ...] [--audio=a.wav] [--seconds=300]
 *
 * Audio: 16-bit PCM WAV at 32 kHz, default tests/test.wav. Models: default those
 * in docs/models/manifest.json that merge by window fusion.
 */
import { readFileSync } from 'node:fs';

const root = new URL('../../', import.meta.url);
const ort = await import(new URL('./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs', import.meta.url));
const { ChunkedGraph, readyWindows } = await import(new URL('docs/javascripts/demo/chunked-graph.js', root));
const { fuseBoxes } = await import(new URL('docs/javascripts/demo/fusion.js', root));

const args = process.argv.slice(2);
const opt = (name, fallback) => args.find(a => a.startsWith(`--${name}=`))?.split('=')[1] ?? fallback;
const manifest = JSON.parse(readFileSync(new URL('docs/models/manifest.json', root)));
const models = args.filter(a => a.endsWith('.onnx'));
if (!models.length) models.push(...manifest.models.filter(m => m.merge_method === 'fusion').map(m => new URL(`docs/models/${m.file}`, root).pathname));
const AUDIO = opt('audio', new URL('tests/test.wav', root).pathname);
const SECONDS = +opt('seconds', 300);
const SR = 32000, HOP = 1.5, RAW_CONF = 0.05;
const CHUNK_CLIPS = 1;        // detector.js: windows per run in a file
const CONFS = [0.05, 0.15, 0.3, 0.5];
const GAPS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1];   // the gap slider runs from 0 to 1 s
const FIELDS = ['t0', 't1', 'flo', 'fhi', 'avg', 'max', 'n', 'cls'];

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
const same = (a, b) => a.length === b.length && a.every((x, i) => FIELDS.every(f => Object.is(x[f], b[i][f])));

/**
 * The boxes with a cache after every step against those without, for every setting.
 * @param steps  [{ rows, later }]: rows so far and where the rows still to come start
 * @returns the share of rows the cache held before the last step, or -1 if a setting differs
 */
function check(label, data, steps) {
  let differ = 0, held = 0, updates = 0, cachedMs = 0, freshMs = 0;
  const examples = [];
  for (const conf of CONFS) {
    for (const gap of GAPS) {
      const cache = {};
      steps.forEach(({ rows, later }, i) => {
        const raw = data.slice(0, rows * 9);   // a new array for every step, as in the demo
        let t = performance.now();
        const got = fuseBoxes(raw, rows, conf, gap, cache, later);
        cachedMs += performance.now() - t;
        t = performance.now();
        const want = fuseBoxes(raw, rows, conf, gap);
        freshMs += performance.now() - t;
        updates++;
        if (!same(got, want)) {
          differ++;
          if (examples.length < 5) examples.push(`conf ${conf} gap ${gap}: after step ${i + 1} of ${steps.length}, ${got.length} boxes against ${want.length}`);
        }
        if (i === steps.length - 2) held += cache.rows / rows;
      });
    }
  }
  const settings = CONFS.length * GAPS.length;
  console.log(`  ${label}: ${steps.length} steps, ${settings} settings, ${updates} updates; the cache held ${(100 * held / settings).toFixed(0)}% of the rows before the last step; `
    + `fusion ${(freshMs / updates).toFixed(2)} ms an update without it, ${(cachedMs / updates).toFixed(2)} ms with it; differ: ${differ} ${differ ? 'FAILED' : 'ok'}`);
  for (const e of examples) console.log(`    ${e}`);
  return !differ;
}

ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';
const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
const pcm = readPcm16Wav(AUDIO).slice(0, Math.round(SECONDS * SR));
let failed = false;

for (const path of models) {
  const file = path.split('/').pop();
  const graph = await ChunkedGraph.create(ort, new Uint8Array(readFileSync(path)), options);
  console.log(`${file}, ${AUDIO.split('/').pop()}, ${(pcm.length / SR).toFixed(1)} s`);

  // A file: the demo's chunks; the rows of the next chunk start with its window
  let data = new Float32Array(0), rows = 0;
  const steps = [];
  for await (const chunk of graph.run(pcm.slice(), RAW_CONF, CHUNK_CLIPS)) {
    data = concat(data, chunk.data);
    rows += chunk.rows;
    steps.push({ rows, later: chunk.next.clips * HOP });
  }
  steps[steps.length - 1].later = Infinity;   // the run is done
  failed = !check('file', data, steps) || failed;

  // A recording: one window as soon as its audio is there, with no end of file yet. At
  // Stop, its rows get the file's end (the demo's _finalSoFar), and the cache must not hold.
  const live = graph.live();
  live.push(pcm);
  let liveData = new Float32Array(0), liveRows = 0, next = { segment: 0, clip: 0, clips: 0 };
  while (next.clips < readyWindows(pcm.length)) {
    const res = await live.windows(RAW_CONF, next, 1);
    liveData = concat(liveData, res.data);
    liveRows += res.rows;
    next = res.next;
  }
  let liveOk = true;
  for (const conf of CONFS) {
    for (const gap of GAPS) {
      const cache = {};
      fuseBoxes(liveData, liveRows, conf, gap, cache, next.clips * HOP);
      liveOk &&= cache.rows > 0 && same(fuseBoxes(data, rows, conf, gap, cache, Infinity), fuseBoxes(data, rows, conf, gap));
    }
  }
  console.log(`  a recording's cache, then the rows of its file: ${liveOk ? 'ok' : 'FAILED'}`);
  failed = !liveOk || failed;

  // A box without area, which fusion drops for a larger box anywhere: no parts, no cache.
  // In the middle of its window, which owns it, with the highest score, so it is fused alone.
  const flat = data.slice(), r = rows >> 1, o = r * 9;
  flat[o + 1] = flat[o + 2] = Math.fround(flat[o + 7] + 1.5);
  flat[o + 5] = 0.99;
  const flatSteps = steps.filter(s => s.rows > r);
  const flatCache = {};
  fuseBoxes(flat, rows, 0.15, 0.05, flatCache, steps[steps.length - 2].later);
  const fallback = flatCache.rows === 0;
  console.log(`  a box without area: the cache holds no rows: ${fallback ? 'ok' : 'FAILED'}`);
  failed = !fallback || !check('a box without area', flat, flatSteps) || failed;
}

if (failed) process.exit(1);
