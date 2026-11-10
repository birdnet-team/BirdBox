// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Audio decoding and recording for the demo.
 *
 * Audio is decoded straight to 32 kHz (the BirdBox input rate) and
 * multi-channel files are averaged to mono, like librosa.load(sr=32000,
 * mono=True) in detect_birds.py. Files that are already 32 kHz decode
 * without any resampling, so their samples match the CLI bit for bit.
 */

export const SAMPLE_RATE = 32000;

function mixToMono(buffer) {
  const n = buffer.numberOfChannels;
  if (n === 1) return buffer.getChannelData(0).slice();
  const out = new Float32Array(buffer.length);
  for (let c = 0; c < n; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < out.length; i++) out[i] += data[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= n;
  return out;
}

/**
 * Native sample rate from a WAV, FLAC or MP3 header (null if unknown). The
 * decoder resamples straight to 32 kHz, so this is only for display.
 */
function sniffSampleRate(buffer) {
  const b = new Uint8Array(buffer), dv = new DataView(buffer);
  const tag = String.fromCharCode(...b.subarray(0, 4));
  if (tag === 'RIFF') {
    for (let p = 12; p + 16 <= b.length;) {
      const size = dv.getUint32(p + 4, true);
      if (String.fromCharCode(...b.subarray(p, p + 4)) === 'fmt ') return dv.getUint32(p + 12, true);
      p += 8 + size + (size & 1);
    }
    return null;
  }
  if (tag === 'fLaC' && b.length > 21) return (b[18] << 12) | (b[19] << 4) | (b[20] >> 4);
  let p = 0;
  if (tag.startsWith('ID3') && b.length > 10) p = 10 + (((b[6] & 0x7f) << 21) | ((b[7] & 0x7f) << 14) | ((b[8] & 0x7f) << 7) | (b[9] & 0x7f));
  for (let i = p; i < Math.min(b.length - 3, p + 65536); i++) {
    if (b[i] !== 0xff || (b[i + 1] & 0xe0) !== 0xe0) continue;
    const version = (b[i + 1] >> 3) & 3, layer = (b[i + 1] >> 1) & 3, index = (b[i + 2] >> 2) & 3;
    if (version === 1 || layer === 0 || index === 3) continue;
    const base = [44100, 48000, 32000][index];
    return version === 3 ? base : version === 2 ? base / 2 : base / 4;
  }
  return null;
}

async function decodeAt(arrayBuffer, rate) {
  const ctx = new OfflineAudioContext(1, 1, rate);
  return ctx.decodeAudioData(arrayBuffer);
}

/**
 * Decode an encoded audio file to mono Float32 PCM at 32 kHz.
 * @returns {Promise<{pcm: Float32Array, duration: number, sourceRate: number, channels: number}>}
 */
export async function decodeAudio(arrayBuffer) {
  let decoded;
  try {
    decoded = await decodeAt(arrayBuffer.slice(0), SAMPLE_RATE);
  } catch (err) {
    // Some engines refuse a 32 kHz context; decode natively and resample.
    const native = await new (window.AudioContext || window.webkitAudioContext)().decodeAudioData(arrayBuffer.slice(0));
    const length = Math.ceil(native.duration * SAMPLE_RATE);
    const ctx = new OfflineAudioContext(native.numberOfChannels, length, SAMPLE_RATE);
    const src = ctx.createBufferSource();
    src.buffer = native;
    src.connect(ctx.destination);
    src.start();
    decoded = await ctx.startRendering();
    decoded.sourceRate = native.sampleRate;
  }
  const pcm = mixToMono(decoded);
  if (!pcm.length) throw new Error('empty');
  return {
    pcm,
    duration: pcm.length / SAMPLE_RATE,
    sourceRate: decoded.sourceRate || sniffSampleRate(arrayBuffer),
    channels: decoded.numberOfChannels,
  };
}

/** Modified Bessel function of the first kind, order 0 (for the Kaiser window). */
function bessel0(x) {
  let sum = 1, term = 1;
  for (let k = 1; term > 1e-10 * sum; k++) {
    term *= (x / (2 * k)) ** 2;
    sum += term;
  }
  return sum;
}

/**
 * Resamples a stream to 32 kHz with a Kaiser-windowed sinc filter, for what the
 * demo shows and detects while recording. Sample n of the output is the signal
 * at time n / 32000, whatever the input rate. A recording is saved as this output,
 * so the boxes found while recording are the boxes of its file.
 */
export class StreamResampler {
  constructor(inRate, outRate = SAMPLE_RATE) {
    this.step = inRate / outRate;
    if (inRate === outRate) return;
    const HALF = 32, PHASES = 256, BETA = 9;
    // Keeps BirdBox's band (up to 15 kHz) and removes what would alias
    const cutoff = Math.min(inRate, outRate) / 2 * 0.97 / inRate;
    this.half = HALF;
    this.phases = PHASES;
    this.table = new Float32Array((PHASES + 1) * 2 * HALF);
    for (let p = 0; p <= PHASES; p++) {
      let sum = 0;
      const row = p * 2 * HALF;
      for (let t = 0; t < 2 * HALF; t++) {
        const x = t - HALF + 1 - p / PHASES, w = x / HALF;
        const kaiser = Math.abs(w) >= 1 ? 0 : bessel0(BETA * Math.sqrt(1 - w * w)) / bessel0(BETA);
        const sinc = x === 0 ? 1 : Math.sin(2 * Math.PI * cutoff * x) / (2 * Math.PI * cutoff * x);
        this.table[row + t] = kaiser * sinc;
        sum += kaiser * sinc;
      }
      for (let t = 0; t < 2 * HALF; t++) this.table[row + t] /= sum;
    }
    this.buffer = new Float32Array(2 * HALF);   // silence before the first sample
    this.length = HALF;
    this.position = HALF;                       // the first output sample is the first input sample's time
  }

  /** @param {Float32Array} input  @returns {Float32Array} the 32 kHz samples this input completes */
  process(input) {
    if (this.step === 1) return input.slice();
    const { half, phases, table, step } = this;
    if (this.length + input.length > this.buffer.length) {
      const grown = new Float32Array(Math.max(2 * this.buffer.length, this.length + input.length));
      grown.set(this.buffer.subarray(0, this.length));
      this.buffer = grown;
    }
    this.buffer.set(input, this.length);
    this.length += input.length;
    const buffer = this.buffer, end = this.length - half;
    const out = new Float32Array(Math.max(0, Math.ceil((end - this.position) / step)) + 1);
    let n = 0, position = this.position;
    while (position < end) {
      const index = Math.floor(position), frac = (position - index) * phases, p = Math.floor(frac), mix = frac - p;
      const a = p * 2 * half, b = a + 2 * half, start = index - half + 1;
      let sum = 0;
      for (let t = 0; t < 2 * half; t++) sum += buffer[start + t] * (table[a + t] + mix * (table[b + t] - table[a + t]));
      out[n++] = sum;
      position += step;
    }
    // Keep what the next output samples still need
    const keep = Math.max(0, Math.floor(position) - half + 1);
    buffer.copyWithin(0, keep, this.length);
    this.length -= keep;
    this.position = position - keep;
    return out.subarray(0, n).slice();
  }
}

/** 16-bit PCM WAV of Int16Array chunks. */
function wavBytes(chunks, rate) {
  const samples = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const bytes = new ArrayBuffer(44 + samples * 2), view = new DataView(bytes);
  const text = (at, value) => { for (let i = 0; i < value.length; i++) view.setUint8(at + i, value.charCodeAt(i)); };
  text(0, 'RIFF'); view.setUint32(4, 36 + samples * 2, true); text(8, 'WAVEfmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, 'data'); view.setUint32(40, samples * 2, true);
  let o = 44;
  for (const chunk of chunks) {
    for (let i = 0; i < chunk.length; i++, o += 2) view.setInt16(o, chunk[i], true);
  }
  return bytes;
}

let levels = null;

/**
 * What this browser's decoder makes of each 16-bit value, index value + 32768. Decoders
 * disagree on the scale above zero (Chromium and Firefox reach 1, WebKit stays below it),
 * and a recording in progress has to hold the very samples its file will decode to.
 */
function decoderLevels() {
  if (!levels) {
    const all = Int16Array.from({ length: 65536 }, (_, i) => i - 32768);
    const plain = Float32Array.from(all, v => v / 32768);
    levels = decodeAudio(wavBytes([all], SAMPLE_RATE))
      .then(({ pcm }) => (pcm.length === all.length ? pcm : plain))
      .catch(() => plain);
  }
  return levels;
}

/**
 * Peak level below which the microphone counts as silent (-90 dBFS). A live
 * microphone always picks up some room noise above this; a muted one delivers
 * zeros.
 */
const SILENT_PEAK = 10 ** (-90 / 20);
/** Milliseconds without any signal before the recording counts as silent. */
const SILENT_AFTER_MS = 1500;

/**
 * Microphone recorder: raw samples with the browser's processing turned off, kept
 * as a 32 kHz 16-bit WAV, so the recording reaches BirdBox without compression.
 * onAudio gets its samples as they arrive, exactly as the file will decode to them.
 * stop() returns the file.
 */
export class Recorder {
  constructor({ maxSeconds = 300, onTick, onAudio, onLimit } = {}) {
    this.maxSeconds = maxSeconds;
    this.onTick = onTick;
    this.onAudio = onAudio;
    this.onLimit = onLimit;
    this.chunks = [];
    this.samples = 0;     // at the device's rate
    this.rate = 0;
    this.stream = null;
    this.timer = null;
    this.level = null;
    this.peak = 0;
    this.lastSignalAt = 0;
    this.stopped = false;
  }

  /** Seconds recorded so far. */
  get elapsed() { return this.rate ? this.samples / this.rate : 0; }

  /** True while no signal has come in for a moment, or the browser reports the track as muted. */
  get silent() {
    if (this.stream?.getAudioTracks()[0]?.muted) return true;
    return performance.now() - this.lastSignalAt >= SILENT_AFTER_MS;
  }

  async start() {
    // Created before the permission prompt, while the click still counts as a
    // user gesture; a context created later may stay suspended and read zeros.
    this.audioContext = new AudioContext();
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      await this.audioContext.audioWorklet.addModule(new URL('./recorder-worklet.js', import.meta.url))
        .catch(err => { err.recorderSetup = true; throw err; });
      this.levels = await decoderLevels();
    } catch (err) {
      this.stream?.getTracks().forEach(track => track.stop());
      this.audioContext.close();
      throw err;
    }
    this.rate = this.audioContext.sampleRate;
    this.resampler = new StreamResampler(this.rate);
    this.lastSignalAt = performance.now();
    this.source = this.audioContext.createMediaStreamSource(this.stream);
    this.tap = new AudioWorkletNode(this.audioContext, 'bbd-recorder-tap', { numberOfOutputs: 1, outputChannelCount: [1] });
    // Connected, so every browser keeps pulling the tap; it writes silence
    this.source.connect(this.tap).connect(this.audioContext.destination);
    this.tap.port.onmessage = ({ data }) => this._onChunk(data);
    this.stream.getAudioTracks()[0]?.addEventListener('ended', () => this.onLimit?.());
    this.audioContext.resume().catch(() => {});
    this.timer = setInterval(() => {
      this.level = this.peak;
      this.peak = 0;
      this.onTick?.(this.elapsed);
    }, 250);
  }

  _onChunk(chunk) {
    if (this.stopped) return;
    if (this.samples + chunk.length > this.maxSeconds * this.rate) {
      chunk = chunk.subarray(0, Math.max(0, Math.round(this.maxSeconds * this.rate) - this.samples));
      if (!this.limited) {
        this.limited = true;
        queueMicrotask(() => this.onLimit?.());
      }
    }
    if (!chunk.length) return;
    this.samples += chunk.length;
    let peak = this.peak;
    for (let i = 0; i < chunk.length; i++) { const v = Math.abs(chunk[i]); if (v > peak) peak = v; }
    this.peak = peak;
    if (peak >= SILENT_PEAK) this.lastSignalAt = performance.now();
    const pcm = this.resampler.process(chunk), values = new Int16Array(pcm.length), levels = this.levels;
    for (let i = 0; i < pcm.length; i++) {
      const v = Math.max(-32768, Math.min(32767, Math.round(pcm[i] * 32768)));
      values[i] = v;
      pcm[i] = levels[v + 32768];
    }
    this.chunks.push(values);
    this.onAudio?.(pcm);
  }

  _release() {
    this.stopped = true;
    clearInterval(this.timer);
    this.source?.disconnect();
    this.tap?.disconnect();
    this.stream?.getTracks().forEach(track => track.stop());
    this.audioContext?.close().catch(() => {});
  }

  /** Stop and return the recording as a WAV file (null if nothing arrived). */
  stop() {
    this._release();
    if (!this.samples) return null;
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    return new File([wavBytes(this.chunks, SAMPLE_RATE)], `recording_${stamp}.wav`, { type: 'audio/wav' });
  }

  cancel() {
    this._release();
    this.chunks = [];
  }
}
