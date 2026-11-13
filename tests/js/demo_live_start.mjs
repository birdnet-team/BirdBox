// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * While recording, the demo draws a live spectrogram (LiveSpectrogram in
 * docs/javascripts/demo/spec-worker.js); after Stop, the finished one replaces it.
 * A microphone's first moments are often a fade-in, near silence or digital zeros.
 * Does the live spectrogram look like the finished one from the start, without a
 * bright bump, and does it keep up?
 *
 * Audio arrives as from the recorder, in pieces of varying size, with different
 * starts of a microphone. Both spectrograms are compared as the viewer shows them
 * (through its colour scale at default brightness), per 0.25 s:
 *   - in the first 3 s, no quarter is brighter than the finished one by more than
 *     BUMP (out of 255);
 *   - from 3 s on, where either shows something, they differ by at most DIFFER on
 *     average;
 *   - every frame is sent, none later than about 1 s after its audio arrived.
 *
 * Usage (Node 18 or newer, no npm install needed):
 *   cd tests/js && node demo_live_start.mjs [--audio=a.wav] [--seconds=30]
 *
 * Audio: 16-bit PCM WAV at 32 kHz, default tests/test.wav.
 */
import { readFileSync } from 'node:fs';

const root = new URL('../../', import.meta.url);
const args = process.argv.slice(2);
const opt = (name, fallback) => args.find(a => a.startsWith(`--${name}=`))?.split('=')[1] ?? fallback;
const AUDIO = opt('audio', new URL('tests/test.wav', root).pathname);
const SECONDS = +opt('seconds', 30);
const SR = 32000, FMAX = 15000, FPS = 100;
const BUMP = 12, DIFFER = 25, WAIT = 1.2;   // WAIT: 1 s held back, plus one piece of audio

// The worker in this process: postMessage goes to whoever listens
let sink = null;
globalThis.self = globalThis;
globalThis.postMessage = m => sink?.(m);
await import(new URL('docs/javascripts/demo/spec-worker.js', root));
const worker = self.onmessage;

function readPcm16Wav(path) {
  const buf = readFileSync(path), dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  for (let p = 12; p + 8 <= buf.length;) {
    const id = buf.toString('ascii', p, p + 4), size = dv.getUint32(p + 4, true);
    if (id === 'data') {
      const out = new Float32Array(Math.floor(size / 2));
      for (let i = 0; i < out.length; i++) out[i] = dv.getInt16(p + 8 + i * 2, true) / 32768;
      return out;
    }
    p += 8 + size + (size & 1);
  }
  throw new Error(`${path}: no data chunk`);
}

let seed = 7;
const rand = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296 - 0.5; };

/** What the viewer shows of levels with display floor F at default brightness and contrast, 0..255. */
const shown = (frames, F) => frames.map(fr => Uint8Array.from(fr, v => { const x = (v / 255 - F) / Math.max(0.05, 1 - F); return x <= 0 ? 0 : x >= 1 ? 255 : Math.round(255 * x); }));

/** The live spectrogram, audio in pieces of 128 to 5120 samples; for each frame, the frames of audio there were when it was sent. */
function live(pcm) {
  let rows = 0, floor = 0, arrived = 0;
  const frames = [], sentAt = [];
  sink = m => {
    if (m.type === 'meta') { rows = m.rows; floor = m.floor; }
    else if (m.type === 'live-frames') for (let j = 0; j < m.data.length / rows; j++) { frames[m.start + j] = m.data.subarray(j * rows, (j + 1) * rows); sentAt[m.start + j] = arrived; }
  };
  worker({ data: { type: 'live-start', id: 1, kind: 'stft', sampleRate: SR, fMax: FMAX } });
  for (let i = 0; i < pcm.length;) {
    const n = 128 * (1 + Math.floor((rand() + 0.5) * 40));
    arrived = Math.max(0, Math.floor((Math.min(pcm.length, i + n) - 1024) / (SR / FPS)) + 1);
    worker({ data: { type: 'live-audio', pcm: pcm.slice(i, i + n) } });
    i += n;
  }
  return { frames: shown(frames, floor), sentAt };
}

function finished(pcm) {
  let rows = 0, floor = 0;
  const frames = [];
  sink = m => {
    if (m.type === 'meta') { rows = m.rows; floor = m.floor; }
    else if (m.type === 'slice') for (let j = 0; j < m.data.length / rows; j++) frames[m.start + j] = m.data.subarray(j * rows, (j + 1) * rows);
  };
  worker({ data: { type: 'stft', id: 1, pcm, sampleRate: SR, fMax: FMAX } });
  return shown(frames, floor);
}

const mean = (frames, a, b) => {
  let s = 0, n = 0;
  for (let f = a; f < Math.min(b, frames.length); f++) for (const v of frames[f]) { s += v; n++; }
  return n ? s / n : 0;
};
/** Mean difference from frame a on, where either shows something. */
function differ(x, y, a) {
  let s = 0, n = 0;
  for (let f = a; f < x.length; f++) for (let r = 0; r < x[f].length; r++) if (x[f][r] || y[f][r]) { s += Math.abs(x[f][r] - y[f][r]); n++; }
  return n ? s / n : 0;
}

const sound = readPcm16Wav(AUDIO).slice(0, SECONDS * SR);
const shift = (pcm, lead) => { const x = new Float32Array(pcm.length); x.set(pcm.subarray(0, pcm.length - lead), lead); return x; };
const quiet = seconds => { const x = shift(sound, seconds * SR); for (let i = 0; i < seconds * SR; i++) x[i] = rand() * 2e-5; return x; };
const starts = {
  'as recorded': sound,
  '40 ms of digital zeros (Chrome)': shift(sound, 1280),
  '0.3 s fade-in': sound.map((v, i) => v * Math.min(1, i / (0.3 * SR))),
  '0.5 s of near silence': quiet(0.5),
  '1.5 s of near silence': quiet(1.5),
  '2 s of digital zeros': shift(sound, 2 * SR),
  'digital zeros only': new Float32Array(10 * SR),
  'a loud call at the start': sound.map((v, i) => { const t = i / SR; return t < 0.4 ? v + 0.3 * Math.sin(2 * Math.PI * (2000 * t + 5000 * t * t)) * Math.sin(Math.PI * t / 0.4) : v; }),
  'a knock at 5 s': sound.map((v, i) => (i >= 5 * SR && i < 5.1 * SR ? v + rand() * 0.6 : v)),
  'broadband noise from 5 to 7 s': sound.map((v, i) => (i >= 5 * SR && i < 7 * SR ? v + rand() * 0.2 : v)),
};

let failed = 0;
for (const [name, pcm] of Object.entries(starts)) {
  const done = finished(pcm), { frames, sentAt } = live(pcm), problems = [];
  let bump = -Infinity;
  for (let q = 0; q < 12; q++) bump = Math.max(bump, mean(frames, q * 25, q * 25 + 25) - mean(done, q * 25, q * 25 + 25));
  const later = differ(frames, done, 3 * FPS);
  let wait = 0;
  sentAt.forEach((s, j) => { wait = Math.max(wait, (s - j) / FPS); });
  if (frames.length !== done.length || frames.some(f => !f)) problems.push(`${frames.filter(Boolean).length} of ${done.length} frames sent`);
  if (bump > BUMP) problems.push(`a quarter of the first 3 s is ${bump.toFixed(0)} brighter than the finished one (at most ${BUMP})`);
  if (later > DIFFER) problems.push(`from 3 s on it differs by ${later.toFixed(1)} (at most ${DIFFER})`);
  if (wait > WAIT) problems.push(`a frame waited ${wait.toFixed(2)} s (at most ${WAIT})`);
  if (problems.length) failed++;
  console.log(`${problems.length ? 'FAIL' : 'ok  '} ${name}: brightest excess in the first 3 s ${bump.toFixed(0)}, difference from 3 s ${later.toFixed(1)}, longest wait ${wait.toFixed(2)} s${problems.length ? ' - ' + problems.join('; ') : ''}`);
}
if (failed) process.exit(1);
