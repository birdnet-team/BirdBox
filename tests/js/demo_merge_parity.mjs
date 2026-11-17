// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Parity of the Demo tab's merges with the merge inside the ONNX graph:
 * window fusion (docs/javascripts/demo/fusion.js) and song reconstruction
 * (docs/javascripts/demo/songs.js).
 *
 * The demo runs a model of docs/models/manifest.json once per recording, asks
 * for the per-window boxes (raw_all) and merges them in JavaScript, so the
 * sliders need no new run. This script runs the same ONNX Runtime Web bundle
 * as the demo (WASM, single thread) and compares every box of every (conf, gap)
 * setting with the graph's own detections.
 *
 * The graph applies the confidence before raw_all, and its merge reads only
 * raw_all and the gap. So the graph's detections for a setting come from its
 * merge alone (mergeGraph in onnx-graph.js), fed with the rows above the
 * confidence: one run of the network, at the lowest confidence, serves all
 * settings. Runs of the full graph at FULL_CONFS check that this equals what it
 * computes itself.
 *
 * Rerun it after changing the merges in src/inference/onnx_export.py (or the
 * FUSION_* constants), or after exporting a new model.
 *
 * Usage (Node 18 or newer):
 *   cd tests/js && npm install
 *   node demo_merge_parity.mjs [model.onnx] [--merge=fusion|reconstruct] [--conf=0.05,0.15,0.3,0.5] [--one-run] [audio.wav ...]
 *
 * Model: default docs/models/Just-Bird_fp16.onnx, the default model of the
 * demo. The merge comes from the model's manifest entry (by file name), or
 * from --merge for models not in the manifest.
 *
 * Audio: 16-bit PCM or float WAV at 32 kHz; default tests/test.wav (5 minutes
 * of audio).
 */
import { readFileSync } from 'node:fs';

const root = new URL('../../', import.meta.url);
const ort = await import(new URL('./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs', import.meta.url));
const { fuseBoxes } = await import(new URL('docs/javascripts/demo/fusion.js', root));
const { reconstructSongs } = await import(new URL('docs/javascripts/demo/songs.js', root));
const { addGraphOutput, mergeGraph, FLOAT } = await import(new URL('docs/javascripts/demo/onnx-graph.js', root));

const args = process.argv.slice(2);
const MODEL = args.find(a => a.endsWith('.onnx')) || new URL('docs/models/Just-Bird_fp16.onnx', root).pathname;
const manifest = JSON.parse(readFileSync(new URL('docs/models/manifest.json', root)));
const METHOD = args.find(a => a.startsWith('--merge='))?.slice(8)
  || manifest.models.find(m => m.file === MODEL.split('/').pop())?.merge_method;
const MERGE = { fusion: fuseBoxes, reconstruct: reconstructSongs }[METHOD];
if (!MERGE) throw new Error(`${MODEL}: not in docs/models/manifest.json, pass --merge=fusion or --merge=reconstruct`);
const GAPS = { fusion: [0, 0.05, 0.3, 0.5, 0.55, 1], reconstruct: [0, 0.1, 0.3, 1] }[METHOD];
const CONFS = (args.find(a => a.startsWith('--conf='))?.slice(7) ?? '0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5').split(',').map(Number);
const SETTINGS = CONFS.flatMap(conf => GAPS.map(gap => [conf, gap]));
// Runs of the full graph: the lowest confidence, which also gives raw_all, and one above
// it; --one-run leaves out the second (run_parallel.mjs does for all models but one)
const FULL_CONFS = args.includes('--one-run') ? [Math.min(...CONFS)] : [Math.min(...CONFS), 0.3];
const COLS = ['t0', 't1', 'flo', 'fhi', 'avg', 'max', 'n', 'cls'];
const TOLERANCE = 1e-5;   // relative; the port is meant to be exact

/** Mono samples of a 32 kHz WAV file (16-bit PCM or 32-bit float). */
function readWav(path) {
  const buf = readFileSync(path), dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE') throw new Error(`${path}: not a WAV file`);
  let fmt = null;
  for (let p = 12; p + 8 <= buf.length;) {
    const id = buf.toString('ascii', p, p + 4), size = dv.getUint32(p + 4, true);
    if (id === 'fmt ') fmt = { format: dv.getUint16(p + 8, true), channels: dv.getUint16(p + 10, true), rate: dv.getUint32(p + 12, true), bits: dv.getUint16(p + 22, true) };
    if (id === 'data') {
      if (!fmt || fmt.rate !== 32000) throw new Error(`${path}: needs 32 kHz audio`);
      const { channels, bits } = fmt, step = bits / 8 * channels, n = Math.floor(size / step), out = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let c = 0; c < channels; c++) {
          const o = p + 8 + i * step + c * bits / 8;
          sum += bits === 16 ? dv.getInt16(o, true) / 32768 : bits === 32 && fmt.format === 3 ? dv.getFloat32(o, true) : NaN;
        }
        out[i] = sum / channels;
      }
      if (Number.isNaN(out[0])) throw new Error(`${path}: needs 16-bit PCM or 32-bit float samples`);
      return out;
    }
    p += 8 + size + (size & 1);
  }
  throw new Error(`${path}: no data chunk`);
}

ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';
const bytes = new Uint8Array(readFileSync(MODEL));
const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
const session = await ort.InferenceSession.create(addGraphOutput(bytes, 'raw_all', FLOAT, ['num_raw', 9]), options);
const merge = await ort.InferenceSession.create(mergeGraph(bytes), options);
const entry = manifest.models.find(m => m.file === MODEL.split('/').pop());
const FULL_GAP = entry?.phrase_gap ?? entry?.song_gap ?? GAPS[1];

/** The largest relative deviation of two box lists (Infinity if their lengths differ). */
function deviation(a, b) {
  if (a.dims[0] !== b.dims[0]) return Infinity;
  let dev = 0;
  a.data.forEach((v, i) => { dev = Math.max(dev, Math.abs(v - b.data[i]) / Math.max(1, Math.abs(b.data[i]))); });
  return dev;
}

const files = args.filter(a => !a.endsWith('.onnx') && !a.startsWith('--'));
if (!files.length) files.push(new URL('tests/test.wav', root).pathname);
let failed = 0, total = 0;
const report = (ok, text) => {
  total++;
  if (!ok) failed++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'} ${text}`);
};
for (const file of files) {
  const audio = readWav(file);
  const feeds = (conf, gap) => ({
    audio: new ort.Tensor('float32', audio.slice(), [audio.length]),
    conf: new ort.Tensor('float32', Float32Array.of(conf), [1]),
    song_gap: new ort.Tensor('float32', Float32Array.of(gap), [1]),
  });
  // The graph's merge alone, on the rows above the confidence
  const merged = async (raw, conf, gap) => {
    const c = Math.fround(conf), rows = [];
    for (let r = 0; r < raw.dims[0]; r++) if (raw.data[r * 9 + 5] > c) rows.push(r);
    const kept = new Float32Array(rows.length * 9);
    rows.forEach((r, i) => kept.set(raw.data.subarray(r * 9, r * 9 + 9), i * 9));
    return (await merge.run({
      raw_all: new ort.Tensor('float32', kept, [rows.length, 9]),
      song_gap: new ort.Tensor('float32', Float32Array.of(gap), [1]),
    })).detections;
  };

  // As in the demo: raw_all at the lowest slider confidence
  const first = await session.run(feeds(FULL_CONFS[0], FULL_GAP), ['raw_all', 'detections']);
  const raw = first.raw_all;
  console.log(`${file.split('/').pop()}: ${(audio.length / 32000).toFixed(1)} s, ${raw.dims[0]} window boxes, ${METHOD}`);
  // The merge alone against the full graph
  for (const conf of FULL_CONFS) {
    const full = conf === FULL_CONFS[0] ? first.detections : (await session.run(feeds(conf, FULL_GAP), ['detections'])).detections;
    const dev = deviation(await merged(raw, conf, FULL_GAP), full);
    report(dev === 0, `full graph at conf ${conf.toFixed(2)} gap ${FULL_GAP.toFixed(2)}: ${full.dims[0]} boxes, the same from its merge alone`
      + (dev ? `: max rel deviation ${dev.toExponential(1)}` : ''));
  }
  for (const [conf, gap] of SETTINGS) {
    const ref = await merged(raw, conf, gap);
    const js = MERGE(raw.data, raw.dims[0], conf, gap);
    let dev = js.length === ref.dims[0] ? 0 : Infinity;
    if (dev === 0) js.forEach((b, i) => COLS.forEach((k, j) => {
      const r = ref.data[i * 8 + j];
      dev = Math.max(dev, Math.abs(b[k] - r) / Math.max(1, Math.abs(r)));
    }));
    report(dev <= TOLERANCE, `conf ${conf.toFixed(2)} gap ${gap.toFixed(2)}: graph ${ref.dims[0]} boxes, JS ${js.length}`
      + (Number.isFinite(dev) ? `, max rel deviation ${dev.toExponential(1)}` : ''));
  }
}
console.log(`${total - failed}/${total} settings identical`);
process.exit(failed ? 1 : 0);
