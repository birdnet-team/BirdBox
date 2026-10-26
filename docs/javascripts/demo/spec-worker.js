// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Spectrogram worker for the demo viewer.
 *
 * 'stft': log-power spectrogram for display (Hann, 1024-point FFT, 10 ms hop,
 *         0 Hz up to fMax, 12 kHz by default), with each band's noise floor
 *         subtracted so calls stand out.
 * 'pcen': the PCEN mel spectrogram BirdBox detects on (flattop 2048-point FFT,
 *         hop 375, 256 HTK mel bands from 50 Hz to 15 kHz, PCEN over the whole
 *         recording).
 *         Its rows are mel bands, not linear bins.
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
const PCEN = { nFft: 2048, hop: 375, mels: 256, fMin: 50, fMax: 15000, leftPad: 0.5, warmup: 100, gain: 0.75, bias: 1, power: 0.35, timeConstant: 1, eps: 1e-6, vmax: 100 };

const hzToMel = hz => 2595 * Math.log10(1 + hz / 700);
const melToHz = mel => 700 * (Math.pow(10, mel / 2595) - 1);

/** Triangular HTK mel filters as [bin, weight, bin, weight, ...] per band. */
function melBands(sampleRate) {
  const { nFft, mels, fMin, fMax } = PCEN, nFreqs = nFft / 2 + 1;
  const m0 = hzToMel(fMin), m1 = hzToMel(fMax);
  const bins = Array.from({ length: mels + 2 }, (_, i) => (nFft + 1) * melToHz(m0 + i / (mels + 1) * (m1 - m0)) / sampleRate);
  return Array.from({ length: mels }, (_, m) => {
    const [left, center, right] = [bins[m], bins[m + 1], bins[m + 2]], entries = [];
    for (let f = Math.max(0, Math.floor(left)); f <= Math.min(nFreqs - 1, Math.ceil(right)); f++) {
      let w = 0;
      if (f >= left && f <= center && center > left) w = (f - left) / (center - left);
      else if (f > center && f <= right && right > center) w = (right - f) / (right - center);
      if (w > 0) entries.push(f, w);
    }
    return Float32Array.from(entries);
  });
}

/**
 * PCEN over the whole recording, streamed: the smoother is warmed up on the
 * first frames, then runs over the 0.5 s left-padded recording; the padding
 * frames are dropped.
 */
function pcen(id, pcm, sampleRate) {
  const { nFft, hop, mels, gain, bias, power, eps, vmax } = PCEN;
  const padLen = Math.floor(PCEN.leftPad * sampleRate);
  const padded = new Float32Array(Math.max(nFft, padLen + pcm.length));
  // Int32 scale; the pad repeats the start of the recording
  for (let i = 0; i < Math.min(padLen, pcm.length); i++) padded[i] = pcm[i] * 2147483648;
  for (let i = 0; i < pcm.length; i++) padded[padLen + i] = pcm[i] * 2147483648;
  const frames = 1 + Math.floor((padded.length - nFft) / hop), padFrames = Math.floor(padLen / hop);
  const kept = frames - padFrames;
  if (kept <= 0) throw new Error('audio too short for a spectrogram');
  const win = new Float64Array(nFft), a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368];
  for (let i = 0; i < nFft; i++) {
    const k = 2 * Math.PI * i / nFft;
    win[i] = a[0] - a[1] * Math.cos(k) + a[2] * Math.cos(2 * k) - a[3] * Math.cos(3 * k) + a[4] * Math.cos(4 * k);
  }
  const bands = melBands(sampleRate), re = new Float64Array(nFft), im = new Float64Array(nFft);
  const melFrame = (f, out) => {
    const off = f * hop;
    for (let i = 0; i < nFft; i++) { re[i] = padded[off + i] * win[i]; im[i] = 0; }
    fft(re, im);
    for (let m = 0; m < mels; m++) {
      const band = bands[m];
      let sum = 0;
      for (let k = 0; k < band.length; k += 2) { const bin = band[k]; sum += band[k + 1] * (re[bin] * re[bin] + im[bin] * im[bin]); }
      out[m] = sum;
    }
  };
  const ratio = PCEN.timeConstant * sampleRate / hop;
  const coef = Math.min(1, (Math.sqrt(1 + 4 * ratio * ratio) - 1) / (2 * ratio * ratio)), decay = 1 - coef;
  const smooth = new Float64Array(mels), mel = new Float32Array(mels), offset = Math.pow(bias, power);
  const loop = Math.min(PCEN.warmup, Math.floor(frames / 4));
  for (let f = 0; f < loop; f++) {
    melFrame(f, mel);
    for (let m = 0; m < mels; m++) smooth[m] = f === 0 ? coef * mel[m] : coef * mel[m] + decay * smooth[m];
  }
  postMessage({ type: 'meta', id, frames: kept, rows: mels, fps: sampleRate / hop, floor: 0, rowScale: 'mel', fMin: PCEN.fMin, fMax: PCEN.fMax });
  const SLICE = 1024;
  let data = null, start = 0;
  for (let f = 0; f < frames; f++) {
    melFrame(f, mel);
    for (let m = 0; m < mels; m++) smooth[m] = loop === 0 && f === 0 ? coef * mel[m] : coef * mel[m] + decay * smooth[m];
    const k = f - padFrames;
    if (k < 0) continue;
    if (!data) { start = k; data = new Uint8Array(Math.min(SLICE, kept - k) * mels); }
    const o = (k - start) * mels;
    for (let m = 0; m < mels; m++) {
      const v = (Math.pow(mel[m] / Math.pow(smooth[m] + eps, gain) + bias, power) - offset) / vmax;
      data[o + m] = v <= 0 ? 0 : v >= 1 ? 255 : Math.round(255 * v);
    }
    if (k - start + 1 === data.length / mels) {
      postMessage({ type: 'slice', id, start, data }, [data.buffer]);
      data = null;
    }
  }
  postMessage({ type: 'done', id });
}

self.onmessage = ({ data: msg }) => {
  try {
    if (msg.type === 'stft') stft(msg.id, msg.pcm, msg.sampleRate, msg.fMax);
    else if (msg.type === 'pcen') pcen(msg.id, msg.pcm, msg.sampleRate);
  } catch (err) {
    postMessage({ type: 'error', id: msg.id, error: String(err?.message || err) });
  }
};
