// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Inference worker for the demo: one single-threaded ONNX Runtime Web (WASM)
 * session with the BirdBox graph. Running it here keeps the page responsive
 * and lets the page stop a run by terminating the worker.
 *
 * Messages in:
 *   { type: 'init', model: ArrayBuffer }
 *   { type: 'detect', id, length, conf, chunkClips, from? }   a recording of `length` samples, which
 *                                                          the page sends a block at a time when asked
 *   { type: 'audio', id, from, data: Float32Array }        such a block
 *   { type: 'live-start' }                                 a recording starts (ChunkedGraph.live)
 *   { type: 'live-audio', pcm: Float32Array }              its next samples
 *   { type: 'live-end' }                                   it ended: its frames are not needed any more
 *   { type: 'windows', id, from, count, conf }             its next windows, final
 * Messages out:
 *   { type: 'ready', ms }
 *   { type: 'audio-want', id, from, to }                   samples from..to-1 of the recording, please
 *   { type: 'chunk', id, data, rows, next, analyzedTo }    after every chunkClips windows
 *   { type: 'result', id, ms, data?, rows?, next? }        when the recording or the windows are done
 *   { type: 'error', id, error }
 */
import { BlockSource, ChunkedGraph } from './chunked-graph.js';

// WASM-only bundle on purpose: from 1.21.0 on, ort.min.js loads the JSEP wasm
// even for the CPU backend, and WebKit (Safari) then needs 5-7 GB for 60 s of
// audio instead of about 1 GB. 1.20.1 matches Python ONNX Runtime box for box.
const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

let ort = null;
let graph = null;
let live = null;       // the recording in progress, with its frames
let feed = null;       // the audio of the recording being detected, from the page (BlockSource)

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
  feed = new BlockSource(msg.length, (from, to) => post({ type: 'audio-want', id: msg.id, from, to }));
  feed.id = msg.id;
  try {
    for await (const chunk of graph.run(feed, msg.conf, msg.chunkClips, msg.from)) {
      post({ type: 'chunk', id: msg.id, ...chunk }, [chunk.data.buffer]);
    }
  } finally {
    feed = null;
  }
}

const JOBS = {
  windows: msg => live.windows(msg.conf, msg.from, msg.count),
  detect,
};

self.onmessage = async ({ data: msg }) => {
  if (msg.type === 'audio') { if (feed?.id === msg.id) feed.receive(msg); return; }
  if (msg.type === 'live-end') { live = null; return; }
  // Between the awaits of a run: the run reads the samples there are when it starts
  if (msg.type === 'live-start') { live = graph.live(); return; }
  if (msg.type === 'live-audio') { live?.push(msg.pcm); return; }
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
