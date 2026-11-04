// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Parity of the demo's chunked detection (docs/javascripts/demo/chunked-graph.js)
 * with one run of the whole graph: the raw_all rows of the chunks, concatenated,
 * must equal the rows of the full run, on the same ONNX Runtime Web bundle. The
 * same holds for a run that stops and continues in a fresh graph, as the demo's
 * Stop and Continue Detecting do.
 *
 * Rerun it after exporting a new model or changing the segment body in
 * src/inference/onnx_export.py.
 *
 * Usage (Node 18 or newer):
 *   cd tests/js && npm install
 *   node demo_chunk_parity.mjs [model.onnx ...] [--chunk=2] [--seconds=3,60,61,100]
 *
 * Audio: cuts of tests/test.wav (32 kHz). Default models: all in docs/models/manifest.json.
 */
import { readFileSync } from 'node:fs';

const root = new URL('../../', import.meta.url);
const ort = await import(new URL('./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs', import.meta.url));
const { ChunkedGraph } = await import(new URL('docs/javascripts/demo/chunked-graph.js', root));
const { addGraphOutput, FLOAT } = await import(new URL('docs/javascripts/demo/onnx-graph.js', root));

const args = process.argv.slice(2);
const opt = (name, fallback) => args.find(a => a.startsWith(`--${name}=`))?.split('=')[1] ?? fallback;
const manifest = JSON.parse(readFileSync(new URL('docs/models/manifest.json', root)));
const models = args.filter(a => a.endsWith('.onnx'));
if (!models.length) models.push(...manifest.models.map(m => new URL(`docs/models/${m.file}`, root).pathname));
const CHUNK = +opt('chunk', 2);
const SECONDS = opt('seconds', '3,5.8,60,61,100').split(',').map(Number);
const CONF = 0.05;

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
const pcm = readPcm16Wav(new URL('tests/test.wav', root).pathname);
let failed = false;

for (const path of models) {
  const bytes = new Uint8Array(readFileSync(path));
  const full = await ort.InferenceSession.create(addGraphOutput(bytes, 'raw_all', FLOAT, ['num_raw', 9]), options);
  const chunked = await ChunkedGraph.create(ort, bytes, options);
  console.log(path.split('/').pop());
  for (const seconds of SECONDS) {
    const audio = pcm.slice(0, Math.round(seconds * 32000));
    const t0 = performance.now();
    const expected = (await full.run({
      audio: new ort.Tensor('float32', audio, [audio.length]),
      conf: new ort.Tensor('float32', Float32Array.of(CONF), [1]),
      song_gap: new ort.Tensor('float32', Float32Array.of(0), [1]),
    }, ['raw_all'])).raw_all;
    const t1 = performance.now();
    const parts = [];
    let rows = 0, chunks = 0, firstAt = null;
    for await (const part of chunked.run(audio, CONF, CHUNK)) {
      firstAt ??= performance.now() - t1;
      parts.push(part.data);
      rows += part.rows;
      chunks++;
    }
    const t2 = performance.now();
    const got = new Float32Array(rows * 9);
    let at = 0;
    for (const p of parts) { got.set(p, at); at += p.length; }
    const compare = values => {
      let differing = values.length === expected.data.length ? 0 : -1;
      if (!differing) for (let i = 0; i < values.length; i++) if (values[i] !== expected.data[i]) differing++;
      if (differing) failed = true;
      return differing;
    };
    const differing = compare(got);

    // Stop twice, after a third and after two thirds of the chunks, each time continuing in a new graph
    const stops = [Math.ceil(chunks / 3), Math.ceil(2 * chunks / 3)], resumedParts = [];
    let from, runs = 0;
    for (const limit of [stops[0], stops[1] - stops[0], Infinity]) {
      const graph = runs++ ? await ChunkedGraph.create(ort, bytes, options) : chunked;
      let taken = 0;
      if (limit > 0) for await (const part of graph.run(audio, CONF, CHUNK, from)) {
        resumedParts.push(part.data);
        from = part.next;
        if (++taken === limit) break;
      }
    }
    const resumed = new Float32Array(resumedParts.reduce((n, p) => n + p.length, 0));
    at = 0;
    for (const p of resumedParts) { resumed.set(p, at); at += p.length; }
    const differingResumed = compare(resumed);

    console.log(`  ${seconds} s: ${rows} rows (full ${expected.dims[0]}), ${differing} differing values, ${differingResumed} after stop and continue, `
      + `${chunks} chunks, full ${((t1 - t0) / 1000).toFixed(1)} s, chunked ${((t2 - t1) / 1000).toFixed(1)} s, first boxes after ${(firstAt / 1000).toFixed(1)} s`);
  }
}
if (failed) { console.error('FAILED: chunked detection differs from the full run'); process.exit(1); }
console.log('All rows equal.');
