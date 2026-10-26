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

/** Best container the browser can record into. */
function recorderMime() {
  const options = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'];
  return options.find(m => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || '';
}

/** Microphone recorder producing a File; raw processing disabled for clean bird audio. */
export class Recorder {
  constructor({ maxSeconds = 300, onTick } = {}) {
    this.maxSeconds = maxSeconds;
    this.onTick = onTick;
    this.chunks = [];
    this.recorder = null;
    this.stream = null;
    this.timer = null;
    this.startedAt = 0;
  }

  get elapsed() { return this.startedAt ? (performance.now() - this.startedAt) / 1000 : 0; }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    const mimeType = recorderMime();
    this.recorder = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined);
    this.chunks = [];
    this.recorder.ondataavailable = e => { if (e.data.size) this.chunks.push(e.data); };
    this.recorder.start(1000);
    this.startedAt = performance.now();
    this.timer = setInterval(() => {
      this.onTick?.(this.elapsed);
      if (this.elapsed >= this.maxSeconds) this.stop();
    }, 250);
    this.done = new Promise(resolve => {
      this.recorder.onstop = () => {
        clearInterval(this.timer);
        this.stream.getTracks().forEach(track => track.stop());
        const type = this.recorder.mimeType || mimeType || 'audio/webm';
        const ext = type.includes('mp4') ? 'm4a' : type.includes('ogg') ? 'ogg' : 'webm';
        const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        resolve(!this.cancelled && this.chunks.length ? new File(this.chunks, `recording_${stamp}.${ext}`, { type }) : null);
      };
    });
  }

  /** Stop and resolve with the recorded File. */
  stop() {
    if (this.recorder && this.recorder.state !== 'inactive') this.recorder.stop();
    return this.done;
  }

  cancel() {
    this.cancelled = true;
    return this.stop();
  }
}
