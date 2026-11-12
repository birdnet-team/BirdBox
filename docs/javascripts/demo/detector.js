// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * The BirdBox graph in a worker (ort-worker.js): loads a model listed in
 * docs/models/manifest.json and runs it once per recording.
 *
 * A run reports the graph's per-window boxes (raw_all) at the lowest slider
 * confidence, a few seconds of audio at a time (chunked-graph.js). fusion.js or
 * songs.js turns them into the boxes for any slider setting, depending on the
 * model's merge.
 */

const MANIFEST = new URL('../../models/manifest.json', import.meta.url);
export const RAW_CONF = 0.05;   // lowest value of the confidence slider
const CHUNK_CLIPS = 1;          // on the 1.5 s window grid: new boxes every 1.5 s of audio, as fast as two per run
/** Merges the demo repeats in JS: fusion.js and songs.js */
export const MERGES = ['fusion', 'reconstruct'];

export class CancelledError extends Error {
  constructor() { super('cancelled'); this.cancelled = true; }
}

let manifest = null;

/** The manifest's models with a merge the demo can repeat; the first is the default. */
export async function models() {
  if (!manifest) {
    manifest = (async () => {
      const res = await fetch(MANIFEST, { cache: 'no-cache' });
      if (!res.ok) throw new Error(`Could not load the model list (HTTP ${res.status})`);
      const entries = ((await res.json()).models || []).filter(m => MERGES.includes(m.merge_method));
      if (!entries.length) throw new Error('docs/models/manifest.json lists no model with a fusion or reconstruct merge');
      return entries.map(m => ({ ...m, url: new URL(m.file, MANIFEST).href }));
    })();
    manifest.catch(() => { manifest = null; });
  }
  return manifest;
}

async function download(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download the model (HTTP ${res.status})`);
  // Content-Length is the compressed size when the server gzips the file
  const total = res.headers.get('Content-Encoding') ? 0 : +res.headers.get('Content-Length') || 0;
  if (!res.body?.getReader) return res.arrayBuffer();
  const reader = res.body.getReader(), chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.({ loaded, total });
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}

class Detector {
  constructor() { this.ready = null; this.worker = null; this.job = null; this.nextId = 1; }

  get loaded() { return !!this.worker && !this.starting; }

  /**
   * Download the model of `entry` and start the worker; onProgress gets
   * { loaded, total } while downloading. Another entry replaces the current model.
   */
  ensure(entry, onProgress) {
    if (this.entry !== entry) {
      this.reset();
      this.entry = entry;
      this.model = null;
    }
    if (!this.ready) {
      this.starting = true;
      const ready = this.ready = (async () => {
        // Kept, so a stopped run restarts the worker without a new download
        const bytes = await (this.model ?? download(entry.url, onProgress));
        if (this.ready !== ready) throw new CancelledError();   // another model was picked meanwhile
        this.model = bytes;
        const model = bytes.slice(0);
        onProgress?.({ starting: true });
        const worker = new Worker(new URL('./ort-worker.js', import.meta.url), { type: 'module', name: 'birdbox' });
        await new Promise((resolve, reject) => {
          worker.onmessage = ({ data }) => {
            if (data.type === 'ready') resolve();
            else if (data.type === 'error') reject(new Error(data.error));
          };
          worker.onerror = e => { e.preventDefault?.(); reject(new Error(e.message || 'The detector could not start')); };
          worker.postMessage({ type: 'init', model }, [model]);
        }).catch(err => { worker.terminate(); throw err; });
        if (this.ready !== ready) { worker.terminate(); throw new CancelledError(); }
        worker.onmessage = ({ data }) => this._done(data);
        worker.onerror = e => {
          e.preventDefault?.();
          this._fail(new Error(e.message || 'The detector stopped (out of memory?)'));
        };
        this.worker = worker;
        this.starting = false;
        return entry;
      })();
      ready.catch(() => {
        if (this.ready !== ready) return;
        this.ready = null;
        this.starting = false;
      });
    }
    return this.ready;
  }

  /**
   * Detect over the whole recording. onChunk({ data, rows, next, analyzedTo }) gets
   * the next rows of raw_all as soon as they are ready. A stopped run continues
   * with `from` set to the `next` of its last chunk. The worker asks for the audio a
   * block at a time, so `audio` is not copied and must not change during the run.
   * @returns {Promise<{ms: number}>}
   */
  detect(audio, onChunk, from) {
    if (this.job) return Promise.reject(new Error('A detection is already running'));
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.job = { id, resolve, reject, onChunk, audio };
      this.worker.postMessage({ type: 'detect', id, length: audio.length, conf: RAW_CONF, chunkClips: CHUNK_CLIPS, from });
    });
  }

  /** A recording starts, its audio follows with liveAudio. */
  liveStart() {
    this.worker.postMessage({ type: 'live-start' });
  }

  /** Its next samples (a copy is sent). */
  liveAudio(pcm) {
    const copy = pcm.slice();
    this.worker.postMessage({ type: 'live-audio', pcm: copy }, [copy.buffer]);
  }

  /**
   * Final rows of the next `count` windows of the recording; `from` is a `next` of an
   * earlier call. The windows must be ready in the audio sent so far (readyWindows).
   * @returns {Promise<{data: Float32Array, rows: number, next: object, ms: number}>}
   */
  liveWindows(from, count) {
    if (this.job) return Promise.reject(new Error('A detection is already running'));
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.job = { id, resolve, reject };
      this.worker.postMessage({ type: 'windows', id, from, count, conf: RAW_CONF });
    });
  }

  /** The recording ended: the worker can let go of its frames. */
  liveEnd() {
    this.worker?.postMessage({ type: 'live-end' });
  }

  _done(msg) {
    const job = this.job;
    if (!job || msg.id !== job.id) return;
    if (msg.type === 'chunk') { job.onChunk(msg); return; }
    if (msg.type === 'audio-want') {
      const data = job.audio.slice(msg.from, msg.to);
      this.worker.postMessage({ type: 'audio', id: job.id, from: msg.from, data }, [data.buffer]);
      return;
    }
    this.job = null;
    if (msg.type === 'error') job.reject(new Error(msg.error));
    else job.resolve(msg);
  }

  _fail(err) {
    const job = this.job;
    this.reset();
    job?.reject(err);
  }

  /** Stop a run in progress (ONNX Runtime cannot interrupt one); the next run starts a fresh worker. */
  abort() {
    if (!this.job) return;
    const job = this.job;
    this.reset();
    job.reject(new CancelledError());
  }

  reset() {
    this.worker?.terminate();
    this.worker = null;
    this.job = null;
    this.ready = null;
    this.starting = false;
  }
}

export const detector = new Detector();
