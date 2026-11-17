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
 *         comes from the last seconds (a running estimate); the STFT's first frames
 *         wait until the microphone's input has settled.
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

const LIVE_FLOOR_SECONDS = 20;  // audio behind the noise floor of a recording in progress
const LIVE_FLOOR_FRAMES = 300;  // frames kept from it
const LIVE_FLOOR_REFRESH = 25;  // frames between updates of the floor
const LIVE_SETTLE = 30;         // frames of settled input the first floor is taken from (0.3 s)
const LIVE_HOLD = 100;          // at most this many frames of input are held back for it (1 s)
const LIVE_RISE_DB = 3;         // input has settled when its level rose less than this over the last 0.15 s
const LIVE_JUMP_DB = 10;        // a frame this far above the floor in most bands: the input itself got louder
const LIVE_WAKE = 300;          // ... which starts the floor anew only in the first 3 s of sound (a microphone waking up late)
const LIVE_POOL = 8;            // bands on each side a floor from few frames also takes in
const LIVE_POOLED_UNTIL = 60;   // ... until it rests on this many frames
const SILENT_DB = -115;         // frames below this in every band are digital silence (1e-12 power is -120 dB)
// The brightest colour, as in the finished spectrogram: the 99.95th percentile of the levels above
// the floor, at least 24 dB. The colours start at +7 dB, which stays at the same place in the scale.
const HI_MIN = 24, HI_QUANTILE = 0.9995, CUT = (7 + 6) / (HI_MIN + 6);
const RES_MIN = -60, RES_STEP = 0.1, RES_BINS = 1500;   // histogram of the levels above the floor, -60..90 dB

/** The value at quantile q of the first n values of a, which it reorders. */
function select(a, n, q) {
  const sorted = a.subarray(0, n).sort();
  return percentile(sorted, q);
}

/**
 * The spectrogram of audio that is still arriving. Frames line up with the
 * finished spectrogram's. Bands are levelled by a running estimate of their
 * noise floor, and frames already sent stay as they are.
 *
 * A microphone's first moments are often a fade-in or near silence. So frames are
 * held back until the input has settled (its level stops rising, at least 0.3 s
 * and at most 1 s of sound), the floor is taken from the settled frames, and the
 * held frames are then levelled with it. While a floor rests on few frames, each
 * band also takes in its neighbours: noise changes smoothly with frequency. When,
 * in the first 3 s of sound, the input itself gets much louder in most bands for
 * 0.3 s (a microphone that comes alive late), the floor is taken anew the same way.
 * The brightest colour follows the loudest levels so far, as in the finished
 * spectrogram.
 */
class LiveSpectrogram {
  constructor(id, sampleRate, fMax = 12000) {
    const N = 1024;
    this.id = id;
    this.n = N;
    this.hop = Math.round(sampleRate / 100);
    this.rows = Math.round(Math.min(fMax, sampleRate / 2) / (sampleRate / N));
    this.residuals = new Float64Array(RES_BINS);
    this.counted = 0;
    this._setRange(HI_MIN);
    this.win = Float64Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N));
    this.re = new Float64Array(N);
    this.im = new Float64Array(N);
    this.buffer = new Float32Array(1 << 18);
    this.length = 0;     // samples in the buffer
    this.position = 0;   // where the next frame starts in it
    this.frames = 0;
    this.sent = 0;
    this.held = [];      // frames not sent yet: { db, sound, index, level }
    this.settled = false;
    this.firstSound = -1;  // the first frame that is not digital silence
    // History for the floor: one frame in every `every` over the last LIVE_FLOOR_SECONDS
    const fps = sampleRate / this.hop;
    this.every = Math.max(1, Math.round(fps * LIVE_FLOOR_SECONDS / LIVE_FLOOR_FRAMES));
    this.history = new Float32Array(LIVE_FLOOR_FRAMES * this.rows);
    this.stored = 0;
    this.floor = new Float32Array(this.rows);
    this.pool = new Float32Array(LIVE_FLOOR_FRAMES * (2 * LIVE_POOL + 1));
    postMessage({ type: 'meta', id, rows: this.rows, fps, hzPerRow: sampleRate / N, floor: (7 - this.lo) / (this.hi - this.lo) });
  }

  _frame(offset) {
    const { n, re, im, win, buffer, rows } = this, db = new Float32Array(rows);
    for (let i = 0; i < n; i++) { re[i] = buffer[offset + i] * win[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k < rows; k++) db[k] = 10 * Math.log10(re[k] * re[k] + im[k] * im[k] + 1e-12);
    return db;
  }

  /**
   * Each band's median over `count` frames (frame(j) gives frame j's dB), taking in
   * `reach` bands on each side.
   */
  _setFloor(count, frame, reach) {
    const { rows, pool, floor } = this;
    for (let k = 0; k < rows; k++) {
      let c = 0;
      for (let b = Math.max(0, k - reach); b <= Math.min(rows - 1, k + reach); b++) for (let j = 0; j < count; j++) pool[c++] = frame(j)[b];
      floor[k] = select(pool, c, 0.5);
    }
  }

  /** The floor from the history, taking in neighbouring bands while it is short. */
  _historyFloor() {
    const count = Math.min(this.stored, LIVE_FLOOR_FRAMES), rows = this.rows, history = this.history;
    if (!count) return;   // nothing but digital silence so far
    const reach = Math.round(LIVE_POOL * Math.max(0, 1 - count / LIVE_POOLED_UNTIL));
    const views = Array.from({ length: count }, (_, j) => history.subarray(j * rows, (j + 1) * rows));
    this._setFloor(count, j => views[j], reach);
  }

  _remember(db) {
    if (this.frames % this.every === 0) {
      this.history.set(db, (this.stored % LIVE_FLOOR_FRAMES) * this.rows);
      this.stored++;
    }
  }

  /** Levels from lo to hi dB above the floor, +7 dB at the same place for any hi (CUT). */
  _setRange(hi) {
    this.hi = hi;
    this.lo = (7 - CUT * hi) / (1 - CUT);
  }

  /** Count a frame's levels above the floor. */
  _count(db) {
    const { rows, floor, residuals } = this;
    for (let k = 0; k < rows; k++) {
      const bin = Math.floor((db[k] - floor[k] - RES_MIN) / RES_STEP);
      residuals[bin < 0 ? 0 : bin >= RES_BINS ? RES_BINS - 1 : bin]++;
    }
    this.counted += rows;
  }

  /** hi from the levels counted so far. */
  _updateRange() {
    let above = this.counted * (1 - HI_QUANTILE), bin = RES_BINS - 1;
    for (; bin > 0 && above > this.residuals[bin]; bin--) above -= this.residuals[bin];
    this._setRange(Math.max(HI_MIN, RES_MIN + (bin + 0.5) * RES_STEP));
  }

  /** Start the floor anew from the last LIVE_SETTLE frames of sound held back. */
  _settle() {
    // Digital silence only (a muted microphone): a floor from it shows it dark
    let sound = this.held.filter(h => h.sound).slice(-LIVE_SETTLE);
    if (!sound.length) sound = this.held.slice(-LIVE_SETTLE);
    this._setFloor(sound.length, j => sound[j].db, LIVE_POOL);
    this.stored = 0;
    for (const h of sound) if (h.index % this.every === 0) {
      this.history.set(h.db, (this.stored % LIVE_FLOOR_FRAMES) * this.rows);
      this.stored++;
    }
    for (const h of sound) this._count(h.db);
    this._updateRange();
    this.settled = true;
  }

  /** Has the input settled: enough sound held back, and its level no longer rising? */
  _hasSettled() {
    const sound = this.held.filter(h => h.sound);
    if (sound.length >= LIVE_HOLD || (!sound.length && this.held.length >= LIVE_HOLD)) return true;
    if (sound.length < LIVE_SETTLE) return false;
    const recent = sound.slice(-LIVE_SETTLE), half = LIVE_SETTLE >> 1;
    const level = list => { const x = Float32Array.from(list, h => h.level); return select(x, x.length, 0.5); };
    return level(recent.slice(half)) - level(recent.slice(0, half)) < LIVE_RISE_DB;
  }

  /** How far a frame lies above the floor in most bands. */
  _above(db) {
    const d = new Float32Array(this.rows);
    for (let k = 0; k < this.rows; k++) d[k] = db[k] - this.floor[k];
    return select(d, d.length, 0.5);
  }

  _levels(db) {
    const levels = new Uint8Array(this.rows), { lo, hi, floor } = this;
    for (let k = 0; k < this.rows; k++) {
      const v = (db[k] - floor[k] - lo) / (hi - lo);
      levels[k] = v <= 0 ? 0 : v >= 1 ? 255 : Math.round(255 * v);
    }
    return levels;
  }

  push(pcm) {
    if (this.length + pcm.length > this.buffer.length) {
      // Samples before the next frame are done with
      this.buffer.copyWithin(0, this.position, this.length);
      this.length -= this.position;
      this.position = 0;
      if (this.length + pcm.length > this.buffer.length) {
        const grown = new Float32Array(2 * (this.length + pcm.length));
        grown.set(this.buffer.subarray(0, this.length));
        this.buffer = grown;
      }
    }
    this.buffer.set(pcm, this.length);
    this.length += pcm.length;
    const out = [];
    const release = () => { for (const h of this.held) out.push(this._levels(h.db)); this.held.length = 0; };
    while (this.position + this.n <= this.length) {
      const db = this._frame(this.position);
      this.position += this.hop;
      const index = this.frames++;
      let max = -Infinity;
      for (let k = 0; k < this.rows; k++) if (db[k] > max) max = db[k];
      const sound = max > SILENT_DB;
      if (sound && this.firstSound < 0) this.firstSound = index;
      if (!this.settled) {
        // Hold the frames back until the input has settled; digital silence does not count
        this.held.push({ db, sound, index, level: sound ? select(db.slice(), db.length, 0.5) : 0 });
        if (this._hasSettled()) { this._settle(); release(); }
        continue;
      }
      if (sound && index < this.firstSound + LIVE_WAKE && this._above(db) > LIVE_JUMP_DB) {
        // Much louder in most bands: hold back; if it lasts, the input itself changed
        this.held.push({ db, sound, index, level: 0 });
        if (this.held.length >= LIVE_SETTLE) { this._settle(); release(); }
        continue;
      }
      release();
      if (sound) { this._remember(db); this._count(db); }
      if (this.frames % LIVE_FLOOR_REFRESH === 0) { this._historyFloor(); this._updateRange(); }
      out.push(this._levels(db));
    }
    if (!out.length) return;
    const data = new Uint8Array(out.length * this.rows);
    out.forEach((levels, j) => data.set(levels, j * this.rows));
    postMessage({ type: 'live-frames', id: this.id, start: this.sent, data }, [data.buffer]);
    this.sent += out.length;
  }
}

/**
 * The log-floor spectrogram of audio that is still arriving, as BirdBox sees it.
 * Frames line up with the finished spectrogram's. Bands are levelled by a running
 * estimate of their floor over the last 60 s. Frames already sent stay as they are.
 */
class LiveLogFloor {
  constructor(id, sampleRate) {
    this.id = id;
    this.buffer = new Float32Array(1 << 18);
    this.length = 0;
    this.position = 0;
    this.sent = 0;
    this.held = [];
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
    this.re = new Float64Array(this.n);
    this.im = new Float64Array(this.n);
    this.fps = sampleRate / this.hop;
    // History for the floor: one frame in every `every` over the last 60 s, as the model's
    this.every = Math.max(1, Math.round(this.fps * MEL.floorWindow / LIVE_FLOOR_FRAMES));
    this.history = new Float32Array(LIVE_FLOOR_FRAMES * this.rows);
    this.stored = 0;
    this.frames = 0;
    this.floor = new Float32Array(this.rows);
    this.column = new Float32Array(LIVE_FLOOR_FRAMES);
    this.current = new Float32Array(this.rows);
  }

  _frame(offset) {
    const { n, re, im, win, buffer, rows, current } = this;
    for (let i = 0, p = offset; i < n; i++, p++) { re[i] = (p >= 0 ? buffer[p] : 0) * 2147483648 * win[i]; im[i] = 0; }
    fft(re, im);
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

const LIVE_FIRST_FRAMES = 12;   // log-floor: about 0.1 s until the first frames can be levelled
let live = null;

self.onmessage = ({ data: msg }) => {
  try {
    if (msg.type === 'live-start') live = msg.kind === 'logfloor' ? new LiveLogFloor(msg.id, msg.sampleRate) : new LiveSpectrogram(msg.id, msg.sampleRate, msg.fMax);
    else if (msg.type === 'live-audio') live?.push(msg.pcm);
    else if (msg.type === 'stft') stft(msg.id, msg.pcm, msg.sampleRate, msg.fMax);
    else if (msg.type === 'logfloor') logfloor(msg.id, msg.pcm, msg.sampleRate);
  } catch (err) {
    postMessage({ type: 'error', id: msg.id, error: String(err?.message || err) });
  }
};
