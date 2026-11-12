// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * The demo computes each window's noise floor itself (RollingFloor in
 * docs/javascripts/demo/chunked-graph.js) and runs StreamDetect cut at the floor
 * (floorCut in onnx-graph.js), so the network gets only the windows' own frames.
 * Are the rows those of StreamDetect computing the floor?
 *
 * Both run the demo's way, one window per run (ChunkedGraph.run), on:
 *   - test.wav;
 *   - 65 s of digital silence, then test.wav: floor windows without a frame of sound,
 *     where the floor comes from all frames, and the move from there to sound;
 *   - test.wav, then 70 s of digital silence.
 * The rows must be equal, value for value. demo_chunk_parity.mjs compares the demo's
 * path, with the floor computed here, against one run of the whole graph.
 *
 * Usage (Node 18 or newer):
 *   cd tests/js && npm install
 *   node demo_floor.mjs [model.onnx ...] [--seconds=120]
 *
 * Audio: the first seconds of tests/test.wav (32 kHz). Models: default all in
 * docs/models/manifest.json.
 */
import { readFileSync } from 'node:fs';

const root = new URL('../../', import.meta.url);
const ort = await import(new URL('./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs', import.meta.url));
const { ChunkedGraph } = await import(new URL('docs/javascripts/demo/chunked-graph.js', root));
const { streamGraphs } = await import(new URL('docs/javascripts/demo/onnx-graph.js', root));

const args = process.argv.slice(2);
const opt = (name, fallback) => args.find(a => a.startsWith(`--${name}=`))?.split('=')[1] ?? fallback;
const manifest = JSON.parse(readFileSync(new URL('docs/models/manifest.json', root)));
const models = args.filter(a => a.endsWith('.onnx'));
if (!models.length) models.push(...manifest.models.map(m => new URL(`docs/models/${m.file}`, root).pathname));
const SECONDS = +opt('seconds', 120);
const SR = 32000, CONF = 0.05;

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

const join = (...parts) => {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
};

/** Rows of a run, one window at a time. */
async function rows(graph, audio) {
  const parts = [];
  for await (const c of graph.run(audio, CONF, 1)) parts.push(c.data);
  return { data: join(...parts) };
}

ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';
const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
const sound = readPcm16Wav(new URL('tests/test.wav', root).pathname).slice(0, Math.round(SECONDS * SR));
const cases = [
  ['test.wav', sound],
  ['65 s of silence, then test.wav', join(new Float32Array(65 * SR), sound)],
  ['test.wav, then 70 s of silence', join(sound, new Float32Array(70 * SR))],
];
let failed = false;

for (const path of models) {
  const bytes = new Uint8Array(readFileSync(path));
  const here = await ChunkedGraph.create(ort, bytes, options);
  if (!here.floorInput) {
    console.log(`${path.split('/').pop()}: FAILED, StreamDetect could not be cut at the floor (floorCut)`);
    failed = true;
    continue;
  }
  // The same graph with StreamDetect as exported, computing the floor itself
  const graph = await ChunkedGraph.create(ort, bytes, options);
  graph.detect = await ort.InferenceSession.create(streamGraphs(bytes).detect, options);
  graph.floorInput = false;
  console.log(path.split('/').pop());
  for (const [label, audio] of cases) {
    const want = await rows(graph, audio), got = await rows(here, audio);
    let differing = got.data.length === want.data.length ? 0 : -1;
    if (!differing) for (let i = 0; i < got.data.length; i++) if (!Object.is(got.data[i], want.data[i])) differing++;
    if (differing) failed = true;
    console.log(`  ${label}: ${want.data.length / 9} rows, ${differing} differing values ${differing ? 'FAILED' : 'ok'}`);
  }
}
if (failed) { console.error('FAILED: the floor computed here gives other rows'); process.exit(1); }
console.log('All rows equal.');
