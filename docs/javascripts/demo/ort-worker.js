// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Inference worker for the demo: one single-threaded ONNX Runtime Web (WASM)
 * session with the BirdBox graph. Running it here keeps the page responsive
 * and lets the page stop a run by terminating the worker.
 *
 * Messages in:
 *   { type: 'init', model: ArrayBuffer }
 *   { type: 'detect', id, audio: Float32Array, conf, chunkClips }
 * Messages out:
 *   { type: 'ready', ms }
 *   { type: 'chunk', id, data, rows, clips, analyzedTo }   after every chunkClips clips
 *   { type: 'result', id, ms }                             when the recording is done
 *   { type: 'error', id, error }
 */
import { ChunkedGraph } from './chunked-graph.js';

// WASM-only bundle on purpose: from 1.21.0 on, ort.min.js loads the JSEP wasm
// even for the CPU backend, and WebKit (Safari) then needs 5-7 GB for 60 s of
// audio instead of about 1 GB. 1.20.1 matches Python ONNX Runtime box for box.
const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

let ort = null;
let graph = null;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

async function init(msg) {
  const t0 = performance.now();
  ort = await import(`${ORT}ort.wasm.min.mjs`);
  ort.env.wasm.wasmPaths = ORT;
  // Threads need cross-origin isolation, which GitHub Pages cannot provide
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.logLevel = 'error';
  graph = await ChunkedGraph.create(ort, new Uint8Array(msg.model), {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  post({ type: 'ready', ms: performance.now() - t0 });
}

/** raw_all rows of each chunk: clip, t0, t1, f_lo, f_hi, score, class, clip start, file end */
async function detect(msg) {
  for await (const chunk of graph.run(msg.audio, msg.conf, msg.chunkClips)) {
    post({ type: 'chunk', id: msg.id, ...chunk }, [chunk.data.buffer]);
  }
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
    await detect(msg);
    post({ type: 'result', id: msg.id, ms: performance.now() - t0 });
  } catch (err) {
    post({ type: 'error', id: msg.id, error: String(err?.message || err) });
  }
};
