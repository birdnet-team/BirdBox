// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Inference worker for the demo: one single-threaded ONNX Runtime Web (WASM)
 * session with the BirdBox graph. Running it here keeps the page responsive
 * and lets the page stop a run by terminating the worker.
 *
 * Messages in:
 *   { type: 'init', model: ArrayBuffer }
 *   { type: 'detect', id, audio: Float32Array, conf, chunkClips, from? }
 *   { type: 'tail', id, audio: Float32Array, offset, conf, clips }          a recording in progress
 *   { type: 'segment', id, audio?: Float32Array, from, conf, chunkClips }   the next clips of a settled segment of it;
 *                                                                           audio starts the segment, without it, it goes on
 *   { type: 'segment-end' }                                                 that segment is not needed any more
 *   { type: 'windows', id, audio: Float32Array, from, count, conf }        rolling floor: the next windows of a recording
 *                                                                           in progress, final (ChunkedGraph.windows)
 * Messages out:
 *   { type: 'ready', ms, rolling }                                         rolling: the graph's floor window in seconds, or 0
 *   { type: 'chunk', id, data, rows, next, analyzedTo }    after every chunkClips clips
 *   { type: 'result', id, ms, data?, rows?, next? }        when the recording, tail or clips are done
 *   { type: 'error', id, error }
 */
import { ChunkedGraph } from './chunked-graph.js';

// WASM-only bundle on purpose: from 1.21.0 on, ort.min.js loads the JSEP wasm
// even for the CPU backend, and WebKit (Safari) then needs 5-7 GB for 60 s of
// audio instead of about 1 GB. 1.20.1 matches Python ONNX Runtime box for box.
const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

let ort = null;
let graph = null;
let segment = null;    // the settled segment in progress, with its spectrogram

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
  post({ type: 'ready', ms: performance.now() - t0, rolling: graph.rolling });
}

/** raw_all rows of each chunk: clip, t0, t1, f_lo, f_hi, score, class, clip start, file end */
async function detect(msg) {
  segment = null;
  for await (const chunk of graph.run(msg.audio, msg.conf, msg.chunkClips, msg.from)) {
    post({ type: 'chunk', id: msg.id, ...chunk }, [chunk.data.buffer]);
  }
}

/** A few clips at a time, so the live boxes of a recording in progress can go between them. */
async function segmentClips(msg) {
  if (msg.audio) segment = graph.run(msg.audio, msg.conf, msg.chunkClips, msg.from, msg.from.segment + 1);
  return (await segment.next()).value;
}

const JOBS = {
  tail: msg => graph.detectTail(msg.audio, msg.offset, msg.conf, msg.clips),
  windows: msg => graph.windows(msg.audio, msg.conf, msg.from, msg.count),
  segment: segmentClips,
  detect,
};

self.onmessage = async ({ data: msg }) => {
  if (msg.type === 'segment-end') {
    segment = null;
    return;
  }
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
    const result = await JOBS[msg.type](msg);
    post({ type: 'result', id: msg.id, ms: performance.now() - t0, ...result }, result ? [result.data.buffer] : []);
  } catch (err) {
    post({ type: 'error', id: msg.id, error: String(err?.message || err) });
  }
};
