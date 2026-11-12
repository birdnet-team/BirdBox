// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Spectrogram worker for the demo viewer.
 *
 * 'stft': log-power spectrogram for display (Hann, 1024-point FFT, 10 ms hop,
 *         0 Hz up to fMax, 12 kHz by default), with each band's noise floor
 *         subtracted so calls stand out.
 * 'logfloor': the log-floor mel spectrogram BirdBox detects on (flattop
 *         2048-point FFT, hop 375, 256 HTK mel bands from 50 Hz to 15 kHz, dB
 *         minus each band's noise floor over the 60 s before each window's end).
 *         Its rows are mel bands, not linear bins.
 *
 * 'live-start', then 'live-audio' for each piece of a recording in progress:
 *         the same two spectrograms, one frame at a time. Each band's noise floor
 *         comes from the last seconds (a running estimate).
 *         Frames come back as { type: 'live-frames', id, start, data }.
 *
 * Results stream back in slices of frames (Uint8 levels, frame-major):
 *   { type: 'meta', id, frames, rows, fps, floor, rowScale?, fMin?, fMax? }  then  { type: 'slice', id, start, data }  then  { type: 'done', id }
 */

const fftCache = new Map();

function fftTables(n) {
  let t = fftCache.get(n);
  if (t) return t;
  const levels = Math.log2(n);
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let x = i, r = 0;
    for (let b = 0; b < levels; b++) { r = (r << 1) | (x & 1); x >>= 1; }
    rev[i] = r;
  }
  const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = -Math.sin(2 * Math.PI * i / n); }
  t = { rev, cos, sin };
  fftCache.set(n, t);
  return t;
}

/** In-place iterative radix-2 FFT. */
function fft(re, im) {
  const n = re.length, { rev, cos, sin } = fftTables(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) {
      let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
      tmp = im[i]; im[i] = im[j]; im[j] = tmp;
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1, step = n / size;
    for (let start = 0; start < n; start += size) {
      for (let k = 0, w = 0; k < half; k++, w += step) {
        const a = start + k, b = a + half;
        const tr = re[b] * cos[w] - im[b] * sin[w];
        const ti = re[b] * sin[w] + im[b] * cos[w];
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
      }
    }
  }
}

function percentile(sorted, q) {
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))];
}

function stft(id, pcm, sampleRate, fMax = 12000) {
  const N = 1024, HOP = Math.round(sampleRate / 100), ROWS = Math.round(Math.min(fMax, sampleRate / 2) / (sampleRate / N));
  const frames = Math.max(1, Math.floor((pcm.length - N) / HOP) + 1);
  const win = new Float64Array(N);
  for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N);
  const re = new Float64Array(N), im = new Float64Array(N);
  const frameDb = (f, out) => {
    const off = f * HOP;
    for (let i = 0; i < N; i++) { re[i] = (pcm[off + i] || 0) * win[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k < ROWS; k++) out[k] = 10 * Math.log10(re[k] * re[k] + im[k] * im[k] + 1e-12);
  };
  // Display contrast: subtract each band's noise floor (median level over a
  // sample of frames), so calls stand out over wind and traffic rumble.
  const sampleCount = Math.min(frames, 600), tmp = new Float32Array(ROWS);
  const byRow = Array.from({ length: ROWS }, () => new Float32Array(sampleCount));
  for (let s = 0; s < sampleCount; s++) {
    frameDb(Math.floor((s + 0.5) * frames / sampleCount), tmp);
    for (let k = 0; k < ROWS; k++) byRow[k][s] = tmp[k];
  }
  const noise = new Float32Array(ROWS), residual = new Float32Array(sampleCount * ROWS);
  for (let k = 0; k < ROWS; k++) {
    const sorted = byRow[k].slice().sort();
    noise[k] = percentile(sorted, 0.5);
    for (let s = 0; s < sampleCount; s++) residual[s * ROWS + k] = byRow[k][s] - noise[k];
  }
  residual.sort();
  // Quantize -6..hi dB above the noise floor; the viewer's default display
  // starts at +7 dB and the brightness control can reach below that.
  const lo = -6, hi = Math.max(24, percentile(residual, 0.9995));
  postMessage({ type: 'meta', id, frames, rows: ROWS, fps: sampleRate / HOP, hzPerRow: sampleRate / N, floor: (7 - lo) / (hi - lo) });
  const SLICE = 1024;
  for (let start = 0; start < frames; start += SLICE) {
    const end = Math.min(frames, start + SLICE), data = new Uint8Array((end - start) * ROWS);
    for (let f = start; f < end; f++) {
      frameDb(f, tmp);
      const o = (f - start) * ROWS;
      for (let k = 0; k < ROWS; k++) {
        const v = (tmp[k] - noise[k] - lo) / (hi - lo);
        data[o + k] = v <= 0 ? 0 : v >= 1 ? 255 : Math.round(255 * v);
      }
    }
    postMessage({ type: 'slice', id, start, data }, [data.buffer]);
  }
  postMessage({ type: 'done', id });
}

// Same values as onnx_export.py
const MEL = { nFft: 2048, hop: 375, mels: 256, fMin: 50, fMax: 15000, leftPad: 0.5, floorWindow: 60, percentile: 0.2, lo: -3, hi: 45, clipSeconds: 3, clipFrames: 252, windowFrames: 128 };

const hzToMel = hz => 2595 * Math.log10(1 + hz / 700);
const melToHz = mel => 700 * (Math.pow(10, mel / 2595) - 1);

/** HTK mel filters as librosa.filters.mel builds them (Slaney area norm), as [bin, weight, ...] per band. */
function melBands(sampleRate) {
  const { nFft, mels, fMin, fMax } = MEL, nFreqs = nFft / 2 + 1;
  const m0 = hzToMel(fMin), m1 = hzToMel(fMax);
  const hz = Array.from({ length: mels + 2 }, (_, i) => melToHz(m0 + i * (m1 - m0) / (mels + 1)));
  return Array.from({ length: mels }, (_, m) => {
    const entries = [], norm = 2 / (hz[m + 2] - hz[m]);
    for (let k = 0; k < nFreqs; k++) {
      const f = k * sampleRate / nFft;
      const w = Math.max(0, Math.min((f - hz[m]) / (hz[m + 1] - hz[m]), (hz[m + 2] - f) / (hz[m + 2] - hz[m + 1])));
      if (w > 0) entries.push(k, w * norm);
    }
    return Float32Array.from(entries);
  });
}

/** numpy.percentile (linear) of the first n values of a sorted array. */
function percentileLinear(sorted, n, q) {
  const pos = (n - 1) * q, below = Math.floor(pos), above = Math.min(below + 1, n - 1);
  return sorted[below] + (sorted[above] - sorted[below]) * (pos - below);
}

/** percentileLinear of the first n values of an unsorted array, which it reorders (quickselect). */
function percentileSelect(a, n, q) {
  const pos = (n - 1) * q, k = Math.floor(pos);
  let lo = 0, hi = n - 1;
  while (lo < hi) {
    const pivot = a[(lo + hi) >> 1];
    let i = lo, j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) { const t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else break;
  }
  let next = a[k];
  if (k + 1 < n) {
    next = Infinity;
    for (let i = k + 1; i < n; i++) if (a[i] < next) next = a[i];
  }
  return a[k] + (next - a[k]) * (pos - k);
}

/**
 * Log-floor mel spectrogram, as BirdBox computes it: log power in dB minus each band's
 * noise floor (its 20th percentile over the 60 s before a window's end, digital
 * silence left out), -3..45 dB mapped to the colour range, on the model's frame grid.
 * Window w is frames 128 w to 128 w + 251; frames after window w-1's end show window
 * w's floor, so they are the pixels the model sees in the second half of window w.
 */
function logfloor(id, pcm, sampleRate) {
  const { nFft, hop, mels, lo, hi, clipFrames, windowFrames } = MEL;
  const padLen = Math.floor(MEL.leftPad * sampleRate);
  const padded = new Float32Array(Math.max(nFft, padLen + pcm.length));
  // Int32 scale; the pad repeats the start of the recording
  for (let i = 0; i < Math.min(padLen, pcm.length); i++) padded[i] = pcm[i] * 2147483648;
  for (let i = 0; i < pcm.length; i++) padded[padLen + i] = pcm[i] * 2147483648;
  const frames = 1 + Math.floor((padded.length - nFft) / hop), padFrames = Math.floor(padLen / hop);
  const kept = frames - padFrames;
  if (kept <= 0) throw new Error('audio too short for a spectrogram');
  // Grid frame m starts at sample m hop + origin of the recording (FRAME_ORIGIN in onnx_export.py)
  const origin = padFrames * hop - padLen, window = Math.round(MEL.floorWindow * sampleRate);
  const win = new Float64Array(nFft), a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368];
  for (let i = 0; i < nFft; i++) {
    const k = 2 * Math.PI * i / nFft;
    win[i] = a[0] - a[1] * Math.cos(k) + a[2] * Math.cos(2 * k) - a[3] * Math.cos(3 * k) + a[4] * Math.cos(4 * k);
  }
  const bands = melBands(sampleRate), re = new Float64Array(nFft), im = new Float64Array(nFft);
  // Band-major: band m of ring slot r at m RING + r, so a band's floor window is contiguous
  const RING = 8192, db = new Float32Array(RING * mels), sound = new Uint8Array(RING);
  const melFrame = (f, r) => {
    const off = f * hop;
    for (let i = 0; i < nFft; i++) { re[i] = padded[off + i] * win[i]; im[i] = 0; }
    fft(re, im);
    let total = 0;
    for (let m = 0; m < mels; m++) {
      const band = bands[m];
      let sum = 0;
      for (let k = 0; k < band.length; k += 2) { const bin = band[k]; sum += band[k + 1] * (re[bin] * re[bin] + im[bin] * im[bin]); }
      db[m * RING + r] = 10 * Math.log10(Math.max(sum, 1e-10));
      total += sum;
    }
    return total > 0;
  };
  postMessage({ type: 'meta', id, frames: kept, rows: mels, fps: sampleRate / hop, floor: 0, rowScale: 'mel', fMin: MEL.fMin, fMax: MEL.fMax });
  const column = new Float32Array(RING), floor = new Float32Array(mels);
  let computed = 0, shown = 0;
  for (let w = 0; shown < kept; w++) {
    const last = Math.min(kept - 1, w * windowFrames + clipFrames - 1);
    for (; computed <= last; computed++) sound[computed % RING] = melFrame(padFrames + computed, computed % RING) ? 1 : 0;
    // The frames that end by the window's end and start in the 60 s before it (pcen_inference.rolling_floor)
    const end = w * windowFrames * hop + MEL.clipSeconds * sampleRate;   // the window's end, in samples
    const first = Math.max(0, Math.ceil((end - window - origin) / hop));
    let anySound = false;
    for (let j = first; j <= last && !anySound; j++) anySound = sound[j % RING] === 1;
    for (let m = 0; m < mels; m++) {
      let c = 0;
      const band = db.subarray(m * RING, (m + 1) * RING);
      for (let j = first; j <= last; j++) { const r = j % RING; if (!anySound || sound[r]) column[c++] = band[r]; }
      floor[m] = percentileSelect(column, c, MEL.percentile);
    }
    const n = last + 1 - shown, data = new Uint8Array(n * mels);
    for (let j = 0; j < n; j++) {
      const r = (shown + j) % RING;
      for (let m = 0; m < mels; m++) {
        const v = (db[m * RING + r] - floor[m] - lo) / (hi - lo);
        data[j * mels + m] = v <= 0 ? 0 : v >= 1 ? 255 : Math.round(255 * v);
      }
    }
    postMessage({ type: 'slice', id, start: shown, data }, [data.buffer]);
    shown = last + 1;
  }
  postMessage({ type: 'done', id });
}

/**
 * A spectrogram of audio that is still arriving. Frames line up with the finished
 * spectrogram's. Bands are levelled by a running estimate of their noise floor.
 * Frames already sent stay as they are.
 */
class LiveSpectrogram {
  constructor(id, kind, sampleRate, fMax = 12000) {
    this.id = id;
    this.mel = kind === 'logfloor';
    this.buffer = new Float32Array(1 << 18);
    this.length = 0;
    this.position = 0;
    this.sent = 0;
    this.held = [];
    if (this.mel) {
      const { nFft, hop, mels, lo, hi, percentile: q } = MEL;
      this.n = nFft; this.hop = hop; this.rows = mels; this.lo = lo; this.hi = hi; this.q = q;
      // The file spectrogram's first kept frame starts this far before the audio
      const padLen = Math.floor(MEL.leftPad * sampleRate);
      this.position = Math.floor(padLen / hop) * hop - padLen;
      this.bands = melBands(sampleRate);
      const a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368];
      this.win = Float64Array.from({ length: nFft }, (_, i) => {
        const k = 2 * Math.PI * i / nFft;
        return a[0] - a[1] * Math.cos(k) + a[2] * Math.cos(2 * k) - a[3] * Math.cos(3 * k) + a[4] * Math.cos(4 * k);
      });
      postMessage({ type: 'meta', id, rows: mels, fps: sampleRate / hop, floor: 0, rowScale: 'mel', fMin: MEL.fMin, fMax: MEL.fMax });
    } else {
      const N = 1024;
      this.n = N; this.hop = Math.round(sampleRate / 100); this.q = 0.5;
      this.rows = Math.round(Math.min(fMax, sampleRate / 2) / (sampleRate / N));
      // The finished spectrogram's range: -6 dB up to its loudest calls, usually about here
      this.lo = -6; this.hi = 28;
      this.win = Float64Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N));
      postMessage({ type: 'meta', id, rows: this.rows, fps: sampleRate / this.hop, hzPerRow: sampleRate / N, floor: (7 - this.lo) / (this.hi - this.lo) });
    }
    this.re = new Float64Array(this.n);
    this.im = new Float64Array(this.n);
    this.fps = sampleRate / this.hop;
    // History for the floor: one frame in every EVERY over the last seconds (60 s for log-floor, as the model)
    const seconds = this.mel ? MEL.floorWindow : LIVE_FLOOR_SECONDS;
    this.every = Math.max(1, Math.round(this.fps * seconds / LIVE_FLOOR_FRAMES));
    this.history = new Float32Array(LIVE_FLOOR_FRAMES * this.rows);
    this.stored = 0;
    this.frames = 0;
    this.floor = new Float32Array(this.rows);
    this.column = new Float32Array(LIVE_FLOOR_FRAMES);
    this.current = new Float32Array(this.rows);
  }

  _frame(offset) {
    const { n, re, im, win, buffer, rows, current } = this;
    const scale = this.mel ? 2147483648 : 1;
    for (let i = 0, p = offset; i < n; i++, p++) { re[i] = (p >= 0 ? buffer[p] : 0) * scale * win[i]; im[i] = 0; }
    fft(re, im);
    if (!this.mel) {
      for (let k = 0; k < rows; k++) current[k] = 10 * Math.log10(re[k] * re[k] + im[k] * im[k] + 1e-12);
      return;
    }
    for (let m = 0; m < rows; m++) {
      const band = this.bands[m];
      let sum = 0;
      for (let k = 0; k < band.length; k += 2) { const bin = band[k]; sum += band[k + 1] * (re[bin] * re[bin] + im[bin] * im[bin]); }
      current[m] = 10 * Math.log10(Math.max(sum, 1e-10));
    }
  }

  _updateFloor() {
    const { rows, history, column, floor } = this, count = Math.min(this.stored, LIVE_FLOOR_FRAMES);
    for (let k = 0; k < rows; k++) {
      for (let s = 0; s < count; s++) column[s] = history[s * rows + k];
      const sorted = column.subarray(0, count).sort();
      floor[k] = percentileLinear(sorted, count, this.q);
    }
  }

  _level(db, k) {
    const v = (db - this.floor[k] - this.lo) / (this.hi - this.lo);
    return v <= 0 ? 0 : v >= 1 ? 255 : Math.round(255 * v);
  }

  push(pcm) {
    if (this.length + pcm.length > this.buffer.length) {
      const keep = this.position > 0 ? this.position : 0;
      this.buffer.copyWithin(0, keep, this.length);
      this.length -= keep;
      this.position -= keep;
      if (this.length + pcm.length > this.buffer.length) {
        const grown = new Float32Array(2 * (this.length + pcm.length));
        grown.set(this.buffer.subarray(0, this.length));
        this.buffer = grown;
      }
    }
    this.buffer.set(pcm, this.length);
    this.length += pcm.length;
    const out = [];
    const emit = levels => out.push(levels);
    while (this.position + this.n <= this.length) {
      this._frame(this.position);
      this.position += this.hop;
      if (this.frames % this.every === 0) {
        this.history.set(this.current, (this.stored % LIVE_FLOOR_FRAMES) * this.rows);
        this.stored++;
      }
      this.frames++;
      // The floor needs a little history; it then follows the sounds
      if (this.frames < LIVE_FIRST_FRAMES) {
        this.held.push(this.current.slice());
        continue;
      }
      if (this.held.length || this.frames % LIVE_FLOOR_REFRESH === 0) {
        if (this.held.length) {
          // Very first estimate, from all the frames held back
          const n = this.held.length, sorted = new Float32Array(n);
          for (let k = 0; k < this.rows; k++) {
            for (let j = 0; j < n; j++) sorted[j] = this.held[j][k];
            sorted.sort();
            this.floor[k] = percentileLinear(sorted, n, this.q);
          }
        } else {
          this._updateFloor();
        }
      }
      for (const held of this.held) {
        const levels = new Uint8Array(this.rows);
        for (let k = 0; k < this.rows; k++) levels[k] = this._level(held[k], k);
        emit(levels);
      }
      this.held.length = 0;
      const levels = new Uint8Array(this.rows);
      for (let k = 0; k < this.rows; k++) levels[k] = this._level(this.current[k], k);
      emit(levels);
    }
    if (!out.length) return;
    const data = new Uint8Array(out.length * this.rows);
    out.forEach((levels, j) => data.set(levels, j * this.rows));
    postMessage({ type: 'live-frames', id: this.id, start: this.sent, data }, [data.buffer]);
    this.sent += out.length;
  }
}

const LIVE_FLOOR_SECONDS = 20;  // STFT: seconds of history for the floor
const LIVE_FLOOR_FRAMES = 300;
const LIVE_FLOOR_REFRESH = 25;  // frames between updates of the floor
const LIVE_FIRST_FRAMES = 12;   // about 0.1 s until the first frames can be levelled
let live = null;

self.onmessage = ({ data: msg }) => {
  try {
    if (msg.type === 'live-start') live = new LiveSpectrogram(msg.id, msg.kind, msg.sampleRate, msg.fMax);
    else if (msg.type === 'live-audio') live?.push(msg.pcm);
    else if (msg.type === 'stft') stft(msg.id, msg.pcm, msg.sampleRate, msg.fMax);
    else if (msg.type === 'logfloor') logfloor(msg.id, msg.pcm, msg.sampleRate);
  } catch (err) {
    postMessage({ type: 'error', id: msg.id, error: String(err?.message || err) });
  }
};
