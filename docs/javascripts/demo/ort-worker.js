// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Inference worker for the demo: one single-threaded ONNX Runtime Web (WASM)
 * session with the BirdBox graph. Running it here keeps the page responsive
 * and lets the page stop a run by terminating the worker.
 *
 * Messages in:
 *   { type: 'init', model: ArrayBuffer, output }
 *   { type: 'detect', id, audio: Float32Array, conf }
 * Messages out: { type: 'ready', ms } | { type: 'result', id, ms, rows, data } | { type: 'error', id, error }
 */
import { addGraphOutput, FLOAT } from './onnx-graph.js';

// WASM-only bundle on purpose: from 1.21.0 on, ort.min.js loads the JSEP wasm
// even for the CPU backend, and WebKit (Safari) then needs 5-7 GB for 60 s of
// audio instead of about 1 GB. 1.20.1 matches Python ONNX Runtime box for box.
const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

let ort = null;
let session = null;
let output = null;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

async function init(msg) {
  const t0 = performance.now();
  ort = await import(`${ORT}ort.wasm.min.mjs`);
  ort.env.wasm.wasmPaths = ORT;
  // Threads need cross-origin isolation, which GitHub Pages cannot provide
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.logLevel = 'error';
  output = msg.output;
  // raw_all rows: clip, t0, t1, f_lo, f_hi, score, class, clip start, file end
  const model = addGraphOutput(new Uint8Array(msg.model), output, FLOAT, ['num_raw', 9]);
  session = await ort.InferenceSession.create(model, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  post({ type: 'ready', ms: performance.now() - t0 });
}

async function detect(msg) {
  const feeds = {
    audio: new ort.Tensor('float32', msg.audio, [msg.audio.length]),
    conf: new ort.Tensor('float32', Float32Array.of(msg.conf), [1]),
    // Only feeds the graph's own fusion, which this run skips
    song_gap: new ort.Tensor('float32', Float32Array.of(0), [1]),
  };
  const out = await session.run(feeds, [output]);
  const tensor = out[output];
  const data = tensor.data.slice();
  return [{ data, rows: tensor.dims[0] || 0 }, [data.buffer]];
}

self.onmessage = async ({ data: msg }) => {
  if (msg.type === 'init') {
    try {
      await init(msg);
    } catch (err) {
      post({ type: 'error', error: String(err?.message || err) });
    }
    return;
  }
  try {
    const t0 = performance.now();
    const [result, transfer] = await detect(msg);
    post({ type: 'result', id: msg.id, ms: performance.now() - t0, ...result }, transfer);
  } catch (err) {
    post({ type: 'error', id: msg.id, error: String(err?.message || err) });
  }
};
