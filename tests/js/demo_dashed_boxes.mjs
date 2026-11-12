// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * The demo draws a box dashed while it may still change (pendingBoxes in
 * docs/javascripts/demo/pending.js). Does every box it draws solid stay as it is?
 *
 * Detection runs the way the demo runs it: a file two windows at a time
 * (ChunkedGraph.run), and a recording, one window at a time as its audio arrives
 * (ChunkedGraph.live). After every step, the boxes are merged for each setting of
 * the sliders, and every box the rule leaves solid must be among the boxes of the
 * finished file, unchanged. For comparison, the test also
 * counts the boxes that would change under the earlier rule, which dashed only boxes
 * reaching past the line.
 *
 * Usage (Node 18 or newer):
 *   cd tests/js && npm install
 *   node demo_dashed_boxes.mjs [model.onnx ...] [--audio=a.wav] [--seconds=300] [--only=file,live]
 *
 * Audio: 16-bit PCM WAV at 32 kHz, default tests/test.wav. Models: default all in
 * docs/models/manifest.json.
 */
import { readFileSync } from 'node:fs';

const root = new URL('../../', import.meta.url);
const ort = await import(new URL('./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs', import.meta.url));
const { ChunkedGraph, NO_END, readyWindows } = await import(new URL('docs/javascripts/demo/chunked-graph.js', root));
const { fuseBoxes } = await import(new URL('docs/javascripts/demo/fusion.js', root));
const { reconstructSongs } = await import(new URL('docs/javascripts/demo/songs.js', root));
const { pendingBoxes } = await import(new URL('docs/javascripts/demo/pending.js', root));

const args = process.argv.slice(2);
const opt = (name, fallback) => args.find(a => a.startsWith(`--${name}=`))?.split('=')[1] ?? fallback;
const manifest = JSON.parse(readFileSync(new URL('docs/models/manifest.json', root)));
const models = args.filter(a => a.endsWith('.onnx'));
if (!models.length) models.push(...manifest.models.map(m => new URL(`docs/models/${m.file}`, root).pathname));
const AUDIO = opt('audio', new URL('tests/test.wav', root).pathname);
const SECONDS = +opt('seconds', 300);
const ONLY = new Set(opt('only', 'file,live').split(','));
const SR = 32000, HOP = 1.5, RAW_CONF = 0.05;
const CHUNK_CLIPS = 2;        // detector.js: windows per run in a file
const LIVE_OWNED = 2.25;      // demo.js: a window owns its boxes up to here
const MERGE = { fusion: fuseBoxes, reconstruct: reconstructSongs };
const CONFS = [0.05, 0.15, 0.3, 0.5];
const GAPS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1];   // the gap slider runs from 0 to 1 s

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
const key = b => `${b.t0}|${b.t1}|${b.flo}|${b.fhi}|${b.cls}`;

/**
 * Boxes the rule draws solid that are not among the final boxes, for every setting.
 * @param steps  [{ rows, line }]: rows so far and the detection line after each step
 */
function check(label, method, data, finalData, finalRows, steps) {
  const merge = MERGE[method];
  let misses = 0, earlier = 0, views = 0, dashed = 0;
  const examples = [];
  for (const conf of CONFS) {
    for (const gap of GAPS) {
      const finalKeys = new Set(merge(finalData, finalRows, conf, gap).map(key));
      const missed = new Set(), missedEarlier = new Set();
      for (const { rows, line } of steps) {
        const boxes = merge(data.subarray(0, rows * 9), rows, conf, gap), pending = pendingBoxes(boxes, line, gap, method);
        views += boxes.length;
        dashed += pending.size;
        for (const b of boxes) {
          if (finalKeys.has(key(b))) continue;
          if (!pending.has(b) && !missed.has(key(b))) {
            missed.add(key(b));
            if (examples.length < 5) examples.push(`conf ${conf} gap ${gap}: ${b.t0.toFixed(2)}-${b.t1.toFixed(2)} s, ${(line - b.t1).toFixed(2)} s before the line`);
          }
          if (b.t1 <= line) missedEarlier.add(key(b));
        }
      }
      misses += missed.size;
      earlier += missedEarlier.size;
    }
  }
  const ok = misses === 0;
  console.log(`  ${label}: ${steps.length} steps, ${CONFS.length * GAPS.length} settings, ${(100 * dashed / views).toFixed(1)}% of box views dashed; `
    + `boxes drawn solid that change later: ${misses} (the earlier rule: ${earlier}) ${ok ? 'ok' : 'FAILED'}`);
  for (const e of examples) console.log(`    ${e}`);
  return ok;
}

ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';
const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
const pcm = readPcm16Wav(AUDIO).slice(0, Math.round(SECONDS * SR));
let failed = false;

for (const path of models) {
  const file = path.split('/').pop(), entry = manifest.models.find(m => m.file === file);
  if (!entry) throw new Error(`${file} is not in docs/models/manifest.json`);
  const method = entry.merge_method;
  if (!MERGE[method]) { console.log(`${file}: merge ${method} has no port, skipped`); continue; }
  const graph = await ChunkedGraph.create(ort, new Uint8Array(readFileSync(path)), options);
  console.log(`${file} (${method}), ${AUDIO.split('/').pop()}, ${(pcm.length / SR).toFixed(1)} s`);

  // A file: the demo's chunks, the line after each (analyzedTo)
  let data = new Float32Array(0), rows = 0;
  const steps = [];
  for await (const chunk of graph.run(pcm.slice(), RAW_CONF, CHUNK_CLIPS)) {
    data = concat(data, chunk.data);
    rows += chunk.rows;
    steps.push({ rows, line: chunk.analyzedTo });
  }
  if (ONLY.has('file')) failed = !check('file', method, data, data, rows, steps) || failed;

  // A recording: one window as soon as its audio is there, with no end of file yet (_liveRolling)
  if (ONLY.has('live')) {
    const live = graph.live();
    live.push(pcm);
    let liveData = new Float32Array(0), liveRows = 0, next = { segment: 0, clip: 0, clips: 0 };
    const liveSteps = [];
    while (next.clips < readyWindows(pcm.length)) {
      const res = await live.windows(RAW_CONF, next, 1);
      for (let r = 0; r < res.rows; r++) if (res.data[r * 9 + 8] !== NO_END) throw new Error('live rows must not know the end of the file');
      liveData = concat(liveData, res.data);
      liveRows += res.rows;
      next = res.next;
      liveSteps.push({ rows: liveRows, line: (next.clips - 1) * HOP + LIVE_OWNED });
    }
    // After Stop the boxes become those of the saved file (rolling_floor_live.mjs)
    failed = !check('recording', method, liveData, data, rows, liveSteps) || failed;
  }
}
process.exit(failed ? 1 : 0);
