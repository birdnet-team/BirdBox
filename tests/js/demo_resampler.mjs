// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * The recorder's resampler (StreamResampler in docs/javascripts/demo/audio.js): sines
 * through it at the common microphone rates, and what it does to boxes. A recording goes
 * up to 48 or 44.1 kHz and back through it, and its boxes must agree with the
 * original's (MIN_RESAMPLED_F1). Whether boxes found while recording equal those of the
 * saved file is rolling_floor_live.mjs.
 *
 * Usage (Node 18 or newer):
 *   cd tests/js && npm install
 *   node demo_resampler.mjs [model.onnx ...] [--audio=a.wav,b.wav] [--only=sines,resampled]
 *
 * --only runs some parts (run_parallel.mjs spreads them over processes): the sine checks
 * (no model), the boxes through the resampler.
 *
 * Audio: 16-bit PCM WAV at 32 kHz, default tests/test.wav (5 minutes of audio).
 * Models: default all in docs/models/manifest.json.
 */
import { readFileSync } from 'node:fs';

const root = new URL('../../', import.meta.url);
const ort = await import(new URL('./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs', import.meta.url));
const { ChunkedGraph } = await import(new URL('docs/javascripts/demo/chunked-graph.js', root));
const { fuseBoxes } = await import(new URL('docs/javascripts/demo/fusion.js', root));
const { reconstructSongs } = await import(new URL('docs/javascripts/demo/songs.js', root));
const { StreamResampler } = await import(new URL('docs/javascripts/demo/audio.js', root));

const args = process.argv.slice(2);
const opt = (name, fallback) => args.find(a => a.startsWith(`--${name}=`))?.split('=')[1] ?? fallback;
const manifest = JSON.parse(readFileSync(new URL('docs/models/manifest.json', root)));
const models = args.filter(a => a.endsWith('.onnx'));
if (!models.length) models.push(...manifest.models.map(m => new URL(`docs/models/${m.file}`, root).pathname));
const AUDIO = opt('audio', new URL('tests/test.wav', root).pathname).split(',');
const ONLY = new Set(opt('only', 'sines,resampled').split(','));
const MIN_RESAMPLED_F1 = 0.95;  // boxes of audio that went through the resampler against the original's
const SR = 32000, RAW_CONF = 0.05;
const MERGE = { fusion: fuseBoxes, reconstruct: reconstructSongs };
let failed = false;

function concat(parts) {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}

/* ---------------- resampler ---------------- */

/** Amplitude of a sine of `hz` in `signal`, by correlation. */
function amplitudeAt(signal, hz) {
  let c = 0, s = 0;
  for (let i = 0; i < signal.length; i++) {
    c += signal[i] * Math.cos(2 * Math.PI * hz * i / SR);
    s += signal[i] * Math.sin(2 * Math.PI * hz * i / SR);
  }
  return 2 * Math.hypot(c, s) / signal.length;
}

for (const rate of ONLY.has('sines') ? [48000, 44100, 96000, 16000, 32000] : []) {
  const inBand = Math.min(10000, rate * 0.3), alias = rate > 40000 ? 21000 : null;
  const seconds = 3, total = seconds * rate;
  const sine = (hz, i) => Math.sin(2 * Math.PI * hz * i / rate);
  /** The resampler's output for a signal, fed in uneven pieces, as a recording arrives. */
  const resample = signal => {
    const resampler = new StreamResampler(rate), parts = [];
    for (let i = 0, step = 777; i < total; i += step, step = 777 + (i % 1500)) {
      const piece = new Float32Array(Math.min(step, total - i));
      for (let j = 0; j < piece.length; j++) piece[j] = signal(i + j);
      parts.push(resampler.process(piece));
    }
    return concat(parts);
  };
  const clean = resample(i => 0.5 * sine(inBand, i));
  const passDb = 20 * Math.log10(amplitudeAt(clean.subarray(SR / 2, 2 * SR), inBand) / 0.5);
  // Sample n is the signal at time n / 32000: the tone's phase must fit
  let phaseError = 0;
  for (let n = SR; n < SR + 2000; n++) phaseError = Math.max(phaseError, Math.abs(clean[n] - 0.5 * sine(inBand, n * rate / SR)));
  const aliasDb = alias ? 20 * Math.log10(amplitudeAt(resample(i => 0.5 * sine(alias, i)).subarray(SR / 2, 2 * SR), SR - alias) / 0.5) : -Infinity;
  const good = Math.abs(passDb) < 0.1 && aliasDb < -60 && Math.abs(clean.length - seconds * SR) < 100 && phaseError < 0.01;
  if (!good) failed = true;
  console.log(`Resampler ${rate} Hz: ${clean.length} samples, ${inBand} Hz at ${passDb.toFixed(2)} dB, phase error ${phaseError.toFixed(4)}`
    + `${alias ? `, ${alias} Hz alias at ${aliasDb.toFixed(0)} dB` : ''} ${good ? 'ok' : 'FAILED'}`);
}

/* ---------------- boxes through the resampler ---------------- */

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

const iou = (a, b) => {
  const t = Math.min(a.t1, b.t1) - Math.max(a.t0, b.t0), f = Math.min(a.fhi, b.fhi) - Math.max(a.flo, b.flo);
  if (t <= 0 || f <= 0) return 0;
  const inter = t * f;
  return inter / ((a.t1 - a.t0) * (a.fhi - a.flo) + (b.t1 - b.t0) * (b.fhi - b.flo) - inter);
};

/** Precision and recall of greedily matched boxes. */
function agreement(live, final, threshold) {
  const pairs = [];
  for (const l of live) for (const f of final) { const v = iou(l, f); if (v >= threshold && l.cls === f.cls) pairs.push([v, l, f]); }
  pairs.sort((a, b) => b[0] - a[0]);
  const usedL = new Set(), usedF = new Set();
  for (const [, l, f] of pairs) if (!usedL.has(l) && !usedF.has(f)) { usedL.add(l); usedF.add(f); }
  const tp = usedL.size;
  return { precision: tp / Math.max(1, live.length), recall: tp / Math.max(1, final.length), f1: 2 * tp / Math.max(1, live.length + final.length) };
}

ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';
const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };

/** Raw rows of a whole run. */
async function fullRun(graph, pcm) {
  const parts = [];
  let rows = 0;
  for await (const part of graph.run(pcm.slice(), RAW_CONF, 4)) { parts.push(part.data); rows += part.rows; }
  return { data: concat(parts), rows };
}

for (const path of ONLY.has('resampled') ? models : []) {
  const file = path.split('/').pop(), entry = manifest.models.find(m => m.file === file);
  if (!entry) throw new Error(`${file} is not in docs/models/manifest.json`);
  const merge = MERGE[entry.merge_method], gap = entry.merge_method === 'fusion' ? entry.phrase_gap : entry.song_gap;
  const graph = await ChunkedGraph.create(ort, new Uint8Array(readFileSync(path)), options);
  console.log(`\n${file} (confidence ${entry.conf}, gap ${gap})`);
  for (const audioPath of AUDIO) {
    const pcm = readPcm16Wav(audioPath), duration = pcm.length / SR;
    const whole = await fullRun(graph, pcm);
    const final = merge(whole.data, whole.rows, entry.conf, gap);
    console.log(`  ${audioPath.split('/').pop()}: ${duration.toFixed(0)} s, ${final.length} final boxes`);

    // What the recorder's resampler does to boxes: up to 48 and 44.1 kHz, then back through it
    for (const rate of [48000, 44100]) {
      const up = new StreamResampler(SR, rate), down = new StreamResampler(rate), parts = [];
      for (let i = 0; i < pcm.length; i += 4096) parts.push(down.process(up.process(pcm.subarray(i, i + 4096))));
      const back = await fullRun(graph, concat(parts));
      const a = agreement(merge(back.data, back.rows, entry.conf, gap), final, 0.5);
      if (a.f1 < MIN_RESAMPLED_F1) failed = true;
      console.log(`    through the resampler at ${rate} Hz: F1 ${a.f1.toFixed(3)} against the original ${a.f1 < MIN_RESAMPLED_F1 ? 'BELOW MINIMUM' : 'ok'}`);
    }
  }
}
if (failed) { console.error('FAILED'); process.exit(1); }
console.log('\nAll checks passed.');
