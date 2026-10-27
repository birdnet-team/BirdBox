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
 * as the demo (WASM, single thread), once for raw_all and once per (conf, gap)
 * setting for the graph's own detections, and compares every box.
 *
 * Rerun it after changing the merges in src/inference/onnx_export.py (or the
 * FUSION_* / PHRASE_* constants), or after exporting a new model.
 *
 * Usage (Node 18 or newer):
 *   cd tests/js && npm install
 *   node demo_merge_parity.mjs [model.onnx] [--merge=fusion|reconstruct] [audio.wav ...]
 *
 * Model: default docs/models/Just-Bird_fp16.onnx, the default model of the
 * demo. The merge comes from the model's manifest entry (by file name), or
 * from --merge for models not in the manifest.
 *
 * Audio: 16-bit PCM or float WAV at 32 kHz; default tests/test.wav (5 minutes
 * of audio, about 8 minutes to run on a laptop).
 */
import { readFileSync } from 'node:fs';

const root = new URL('../../', import.meta.url);
const ort = await import(new URL('./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs', import.meta.url));
const { fuseBoxes } = await import(new URL('docs/javascripts/demo/fusion.js', root));
const { reconstructSongs } = await import(new URL('docs/javascripts/demo/songs.js', root));
const { addGraphOutput, FLOAT } = await import(new URL('docs/javascripts/demo/onnx-graph.js', root));

const args = process.argv.slice(2);
const MODEL = args.find(a => a.endsWith('.onnx')) || new URL('docs/models/Just-Bird_fp16.onnx', root).pathname;
const manifest = JSON.parse(readFileSync(new URL('docs/models/manifest.json', root)));
const METHOD = args.find(a => a.startsWith('--merge='))?.slice(8)
  || manifest.models.find(m => m.file === MODEL.split('/').pop())?.merge_method;
const MERGE = { fusion: fuseBoxes, reconstruct: reconstructSongs }[METHOD];
if (!MERGE) throw new Error(`${MODEL}: not in docs/models/manifest.json, pass --merge=fusion or --merge=reconstruct`);
const GAPS = { fusion: [0, 0.05, 0.3, 1], reconstruct: [0, 0.1, 0.3, 1] }[METHOD];
const SETTINGS = [0.05, 0.15, 0.3, 0.5].flatMap(conf => GAPS.map(gap => [conf, gap]));
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
const model = addGraphOutput(new Uint8Array(readFileSync(MODEL)), 'raw_all', FLOAT, ['num_raw', 9]);
const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });

const files = args.filter(a => !a.endsWith('.onnx') && !a.startsWith('--'));
if (!files.length) files.push(new URL('tests/test.wav', root).pathname);
let failed = 0, total = 0;
for (const file of files) {
  const audio = readWav(file);
  const feeds = (conf, gap) => ({
    audio: new ort.Tensor('float32', audio.slice(), [audio.length]),
    conf: new ort.Tensor('float32', Float32Array.of(conf), [1]),
    song_gap: new ort.Tensor('float32', Float32Array.of(gap), [1]),
  });
  // As in the demo: raw_all at the lowest slider confidence
  const raw = (await session.run(feeds(0.05, 0), ['raw_all'])).raw_all;
  console.log(`${file.split('/').pop()}: ${(audio.length / 32000).toFixed(1)} s, ${raw.dims[0]} window boxes, ${METHOD}`);
  for (const [conf, gap] of SETTINGS) {
    const ref = (await session.run(feeds(conf, gap), ['detections'])).detections;
    const js = MERGE(raw.data, raw.dims[0], conf, gap);
    let dev = js.length === ref.dims[0] ? 0 : Infinity;
    if (dev === 0) js.forEach((b, i) => COLS.forEach((k, j) => {
      const r = ref.data[i * 8 + j];
      dev = Math.max(dev, Math.abs(b[k] - r) / Math.max(1, Math.abs(r)));
    }));
    total++;
    const ok = dev <= TOLERANCE;
    if (!ok) failed++;
    console.log(`  ${ok ? 'PASS' : 'FAIL'} conf ${conf.toFixed(2)} gap ${gap.toFixed(2)}: graph ${ref.dims[0]} boxes, JS ${js.length}` + (Number.isFinite(dev) ? `, max rel deviation ${dev.toExponential(1)}` : ''));
  }
}
console.log(`${total - failed}/${total} settings identical`);
process.exit(failed ? 1 : 0);
