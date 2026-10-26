// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * The BirdBox graph in a worker (ort-worker.js): loads the model listed in
 * docs/models/manifest.json and runs it once per recording.
 *
 * A run returns the graph's per-window boxes (raw_all) at the lowest slider
 * confidence; fusion.js turns them into the boxes for any slider setting.
 */

const MANIFEST = new URL('../../models/manifest.json', import.meta.url);
export const RAW_OUTPUT = 'raw_all';
export const RAW_CONF = 0.05;   // lowest value of the confidence slider

export class CancelledError extends Error {
  constructor() { super('cancelled'); this.cancelled = true; }
}

let manifest = null;

/** The first window-fusion model of the manifest (Just-Bird). */
export async function modelEntry() {
  if (!manifest) {
    manifest = (async () => {
      const res = await fetch(MANIFEST, { cache: 'no-cache' });
      if (!res.ok) throw new Error(`Could not load the model list (HTTP ${res.status})`);
      const entry = ((await res.json()).models || []).find(m => m.merge_method === 'fusion');
      if (!entry) throw new Error('docs/models/manifest.json lists no window-fusion model');
      return { ...entry, url: new URL(entry.file, MANIFEST).href };
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

  /** Download the model and start the worker; onProgress gets { loaded, total } while downloading. */
  ensure(onProgress) {
    if (!this.ready) {
      this.starting = true;
      this.ready = (async () => {
        const entry = await modelEntry();
        // Kept, so a stopped run restarts the worker without a new download
        this.model ??= await download(entry.url, onProgress);
        const model = this.model.slice(0);
        onProgress?.({ starting: true });
        const worker = new Worker(new URL('./ort-worker.js', import.meta.url), { type: 'module', name: 'birdbox' });
        await new Promise((resolve, reject) => {
          worker.onmessage = ({ data }) => {
            if (data.type === 'ready') resolve();
            else if (data.type === 'error') reject(new Error(data.error));
          };
          worker.onerror = e => { e.preventDefault?.(); reject(new Error(e.message || 'The detector could not start')); };
          worker.postMessage({ type: 'init', model, output: RAW_OUTPUT }, [model]);
        }).catch(err => { worker.terminate(); throw err; });
        worker.onmessage = ({ data }) => this._done(data);
        worker.onerror = e => {
          e.preventDefault?.();
          this._fail(new Error(e.message || 'The detector stopped (out of memory?)'));
        };
        this.worker = worker;
        this.starting = false;
        return entry;
      })();
      this.ready.catch(() => { this.ready = null; this.starting = false; });
    }
    return this.ready;
  }

  /**
   * One graph run over the whole recording (the graph walks it in segments).
   * @returns {Promise<{data: Float32Array, rows: number, ms: number}>}
   */
  detect(audio) {
    if (this.job) return Promise.reject(new Error('A detection is already running'));
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.job = { id, resolve, reject };
      this.worker.postMessage({ type: 'detect', id, audio, conf: RAW_CONF }, [audio.buffer]);
    });
  }

  _done(msg) {
    const job = this.job;
    if (!job || msg.id !== job.id) return;
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
  }
}

export const detector = new Detector();
