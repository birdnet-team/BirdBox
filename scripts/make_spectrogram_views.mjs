// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * The spectrogram figures of docs/how-it-works/demo-spectrogram.md: the demo's STFT and log-floor
 * views of ten seconds of the sample soundscape, the model's input image for one
 * window, the level statistics of one band, and ten seconds from --start both in the
 * long file and as the start of a recording that begins there. The views are drawn as the viewer
 * draws them at its defaults (inferno, brightness and contrast 0), from the demo's
 * own worker (docs/javascripts/demo/spec-worker.js). The boxes come from the
 * Just-Bird graph at the demo's defaults. docs/javascripts/spectrogram-views.js
 * draws the axes, boxes and charts over them.
 *
 * Usage (Node 18 or newer, ffmpeg on the PATH):
 *   cd tests/js && npm install && cd ../..
 *   node scripts/make_spectrogram_views.mjs [--from=40] [--to=50] [--window=43.5] [--band=4000] [--start=75]
 *
 * Writes docs/img/how-it-works/spectrogram_views_{stft,logfloor,window}.png,
 * spectrogram_start_{long,cut}.png and spectrogram_views.json. It checks that the log-floor view shows the window's
 * second half with the model's own values and fails otherwise.
 */
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';
import vm from 'node:vm';

const root = new URL('../', import.meta.url);
const path = p => new URL(p, root).pathname;
const args = process.argv.slice(2);
const opt = (name, fallback) => +(args.find(a => a.startsWith(`--${name}=`))?.split('=')[1] ?? fallback);
const T0 = opt('from', 40), T1 = opt('to', 50), WIN = opt('window', 43.5), BAND_HZ = opt('band', 4000), START = opt('start', 75);
const SR = 32000, OUT = path('docs/img/how-it-works/');
const DEMO = path('docs/javascripts/demo/');

// The sample as the browser decodes it: 16-bit FLAC at 32 kHz, samples / 32768
const raw = execFileSync('ffmpeg', ['-loglevel', 'error', '-i', path('docs/audio/BirdNET_soundscape_32kHz.flac'), '-f', 'f32le', '-ac', '1', '-ar', String(SR), '-'], { maxBuffer: 1 << 30 });
const pcm = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4).slice();

// ---------- boxes: Just-Bird at the demo's defaults (docs/models/manifest.json) ----------
const manifest = JSON.parse(readFileSync(path('docs/models/manifest.json')));
const model = manifest.models.find(m => m.file === 'Just-Bird_fp16.onnx');
const ort = await import(path('tests/js/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs'));
ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';
const session = await ort.InferenceSession.create(new Uint8Array(readFileSync(path(`docs/models/${model.file}`))), { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
/** The graph's boxes for `audio`, with times shifted by `offset` seconds. */
async function detect(audio, offset = 0) {
  const { detections } = await session.run({
    audio: new ort.Tensor('float32', audio, [audio.length]),
    conf: new ort.Tensor('float32', Float32Array.of(model.conf), [1]),
    song_gap: new ort.Tensor('float32', Float32Array.of(model.phrase_gap), [1]),
  }, ['detections']);
  const out = [];
  for (let i = 0; i < detections.dims[0]; i++) {
    const r = detections.data.subarray(i * detections.dims[1], (i + 1) * detections.dims[1]);
    out.push({ t0: +(r[0] + offset).toFixed(3), t1: +(r[1] + offset).toFixed(3), f0: Math.round(r[2]), f1: Math.round(r[3]), conf: +r[5].toFixed(3) });
  }
  return out;
}
const allBoxes = await detect(pcm);
const boxes = allBoxes.filter(b => b.t1 > T0 && b.t0 < T1);

// ---------- the worker's spectrograms ----------
const INFERNO = vm.runInNewContext(readFileSync(DEMO + 'colormaps.js', 'utf8').replace(/export /g, '') + '; COLORMAPS.inferno.lut');
const messages = [];
const ctx = vm.createContext({ self: {}, postMessage: m => messages.push(m), pcm });
vm.runInContext(readFileSync(DEMO + 'spec-worker.js', 'utf8'), ctx);
function collect(code) {
  messages.length = 0;
  vm.runInContext(code, ctx);
  const meta = messages.find(m => m.type === 'meta'), data = new Uint8Array(meta.frames * meta.rows);
  for (const m of messages) if (m.type === 'slice') data.set(m.data, m.start * meta.rows);
  return { ...meta, data };
}
const stft = collect(`stft(1, pcm, ${SR}, 15000)`);
const logfloor = collect(`logfloor(2, pcm, ${SR})`);

// ---------- PNG ----------
const crcTable = Array.from({ length: 256 }, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
function crc(buf) { let c = 0xffffffff; for (const b of buf) c = crcTable[(c ^ b) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
function chunk(type, data) {
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, 'ascii');
  data.copy(out, 8);
  out.writeUInt32BE(crc(out.subarray(4, 8 + data.length)), 8 + data.length);
  return out;
}
function writePng(file, w, h, rgb) {
  const rows = Buffer.alloc(h * (w * 3 + 1));
  for (let y = 0; y < h; y++) Buffer.from(rgb.buffer, rgb.byteOffset + y * w * 3, w * 3).copy(rows, y * (w * 3 + 1) + 1);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2;
  writeFileSync(OUT + file, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(rows, { level: 9 })), chunk('IEND', Buffer.alloc(0))]));
}

/** The viewer's colour table at its defaults (spec-view.js, _buildLut). */
function viewerLut(floor) {
  const lut = new Uint8Array(768);
  for (let i = 0; i < 256; i++) {
    let x = (i / 255 - floor) / Math.max(0.05, 1 - floor);
    x = x <= 0 ? 0 : x >= 1 ? 1 : x;
    lut.set(INFERNO.subarray(Math.round(x * 255) * 3, Math.round(x * 255) * 3 + 3), i * 3);
  }
  return lut;
}
/**
 * The stretch from `from` to `to` s of a view: one column per frame, one pixel row per data
 * row, high frequencies on top. `offset`: the time in the sample of the view's first frame.
 */
function drawStretch(spec, file, from = T0, to = T1, offset = 0) {
  const f0 = Math.round((from - offset) * spec.fps), w = Math.round((to - offset) * spec.fps) - f0, h = spec.rows, lut = viewerLut(spec.floor);
  const rgb = new Uint8Array(w * h * 3);
  for (let x = 0; x < w; x++) for (let r = 0; r < h; r++) {
    const v = spec.data[(f0 + x) * h + r];
    rgb.set(lut.subarray(v * 3, v * 3 + 3), ((h - 1 - r) * w + x) * 3);
  }
  writePng(file, w, h, rgb);
}
drawStretch(stft, 'spectrogram_views_stft.png');
drawStretch(logfloor, 'spectrogram_views_logfloor.png');

// ---------- the start of a file: START to START + 10 s in the long file and as a recording's start ----------
const S0 = START, S1 = START + 10;
const cut = pcm.slice(Math.round(S0 * SR));
ctx.cut = cut;
drawStretch(logfloor, 'spectrogram_start_long.png', S0, S1);
drawStretch(collect(`logfloor(3, cut, ${SR})`), 'spectrogram_start_cut.png', S0, S1, S0);
const iou = (a, b) => {
  const t = Math.max(0, Math.min(a.t1, b.t1) - Math.max(a.t0, b.t0)), f = Math.max(0, Math.min(a.f1, b.f1) - Math.max(a.f0, b.f0));
  const area = x => (x.t1 - x.t0) * (x.f1 - x.f0);
  return t * f / (area(a) + area(b) - t * f);
};
// Boxes in the stretch; `only`: no box of the other run at IoU 0.5 or more (greedy, best pairs first)
const inStart = b => b.t1 > S0 && b.t0 < S1, startLong = allBoxes.filter(inStart), startCut = (await detect(cut, S0)).filter(inStart);
const pairs = [];
startLong.forEach((a, i) => startCut.forEach((b, j) => { const v = iou(a, b); if (v >= 0.5) pairs.push([v, i, j]); }));
const usedLong = new Set(), usedCut = new Set();
for (const [, i, j] of pairs.sort((p, q) => q[0] - p[0])) if (!usedLong.has(i) && !usedCut.has(j)) { usedLong.add(i); usedCut.add(j); }
const start = {
  from: S0, to: S1,
  long: startLong.map((b, i) => ({ ...b, only: !usedLong.has(i) })),
  cut: startCut.map((b, j) => ({ ...b, only: !usedCut.has(j) })),
};
console.log(`start of a file at ${S0} s: ${startLong.length} boxes in the long file, ${startCut.length} as the start, ${usedLong.size} in both`);

// ---------- the model's input for the window that starts at WIN (pcen_inference.py, onnx_export.py render) ----------
vm.runInContext(`
var modelWindow = function (w) {
  const { nFft, hop, mels, lo, hi, clipFrames, windowFrames } = MEL, sampleRate = ${SR};
  const padLen = Math.floor(MEL.leftPad * sampleRate), padFrames = Math.floor(padLen / hop), origin = padFrames * hop - padLen;
  const padded = new Float32Array(Math.max(nFft, padLen + pcm.length));
  for (let i = 0; i < Math.min(padLen, pcm.length); i++) padded[i] = pcm[i] * 2147483648;
  for (let i = 0; i < pcm.length; i++) padded[padLen + i] = pcm[i] * 2147483648;
  const a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368], win = new Float64Array(nFft);
  for (let i = 0; i < nFft; i++) { const k = 2 * Math.PI * i / nFft; win[i] = a[0] - a[1] * Math.cos(k) + a[2] * Math.cos(2 * k) - a[3] * Math.cos(3 * k) + a[4] * Math.cos(4 * k); }
  const bands = melBands(sampleRate), re = new Float64Array(nFft), im = new Float64Array(nFft);
  // The frames that end by the window's end and start in the 60 s before it
  const end = w * windowFrames * hop + MEL.clipSeconds * sampleRate;
  const first = Math.max(0, Math.ceil((end - Math.round(MEL.floorWindow * sampleRate) - origin) / hop)), last = w * windowFrames + clipFrames - 1;
  const db = [], sound = [];
  for (let j = first; j <= last; j++) {
    const off = (padFrames + j) * hop;
    for (let i = 0; i < nFft; i++) { re[i] = padded[off + i] * win[i]; im[i] = 0; }
    fft(re, im);
    const col = new Float32Array(mels);
    let total = 0;
    for (let m = 0; m < mels; m++) {
      const band = bands[m];
      let s = 0;
      for (let k = 0; k < band.length; k += 2) { const b = band[k]; s += band[k + 1] * (re[b] * re[b] + im[b] * im[b]); }
      col[m] = 10 * Math.log10(Math.max(s, 1e-10));
      total += s;
    }
    db.push(col);
    sound.push(total > 0);
  }
  const any = sound.some(Boolean), floor = new Float32Array(mels), column = new Float32Array(db.length);
  for (let m = 0; m < mels; m++) {
    let c = 0;
    for (let j = 0; j < db.length; j++) if (!any || sound[j]) column[c++] = db[j][m];
    floor[m] = percentileSelect(column, c, MEL.percentile);
  }
  // The clip on the 0..100 scale of the model's features
  const clip = db.slice(db.length - clipFrames).map(col => col.map((v, m) => (v - floor[m] - lo) * 100 / (hi - lo)));
  return { clip, db, sound, anySound: any, floor };
};`, ctx);
const w = Math.round(WIN / 1.5);
const win = vm.runInContext(`modelWindow(${w})`, ctx);
{
  // Colormap levels and the nearest-neighbour stretch from 252 frames to 256 columns (onnx_export.py render)
  const S = 256, rgb = new Uint8Array(S * S * 3);
  for (let x = 0; x < S; x++) {
    const col = win.clip[Math.floor((x + 0.5) * 252 / S)];
    for (let m = 0; m < 256; m++) {
      const level = Math.min(255, Math.max(0, Math.floor(col[m] / 100 * 256)));
      rgb.set(INFERNO.subarray(level * 3, level * 3 + 3), ((255 - m) * S + x) * 3);
    }
  }
  writePng('spectrogram_views_window.png', S, S, rgb);
  // The log-floor view shows the window's second half with the window's own floor
  let differ = 0;
  for (let j = 128; j < 252; j++) for (let m = 0; m < 256; m++) {
    const v = win.clip[j][m] / 100, byte = v <= 0 ? 0 : v >= 1 ? 255 : Math.round(255 * v);
    if (byte !== logfloor.data[(w * 128 + j) * 256 + m]) differ++;
  }
  console.log(`window ${(w * 1.5).toFixed(1)}–${(w * 1.5 + 3).toFixed(1)} s: ${differ} values of its second half differ from the log-floor view`);
  if (differ) process.exitCode = 1;
}

// ---------- levels of one band, in dB above its median ----------
// `median`: the median the STFT view levels by (of its sample of frames); `floor`: the model's
// floor; `blackBelow`: the level (in dB, not above the median) below which the picture is black
function levels(values, { median, floor, blackBelow }) {
  const s = Float64Array.from(values).sort();
  median ??= s[Math.round(0.5 * (s.length - 1))];
  const hist = new Array(70).fill(0);   // -10 to +60 dB, 1 dB bins
  for (const v of s) { const b = Math.floor(v - median + 10); if (b >= 0 && b < 70) hist[b]++; }
  return {
    ...(floor == null ? {} : { floor: +(floor - median).toFixed(3) }),
    black: +(s.filter(v => v < blackBelow).length / s.length).toFixed(4),
    hist: hist.map(n => +(n / s.length).toFixed(5)),
  };
}
// The STFT view's median of a bin: over 600 frames spread over the file (spec-worker.js stft)
function viewMedian(values) {
  const count = Math.min(values.length, 600);
  const sample = Float64Array.from({ length: count }, (_, s) => values[Math.floor((s + 0.5) * values.length / count)]).sort();
  return sample[Math.round(0.5 * (count - 1))];
}
// The STFT view's floor is taken over the whole file: its bin at BAND_HZ over the whole file
const stftBand = vm.runInContext(`(() => {
  const N = 1024, HOP = ${Math.round(SR / 100)}, k = Math.round(${BAND_HZ} / (${SR} / N)), out = [];
  const win = Float64Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N));
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let f = 0; f + N <= pcm.length; f += HOP) {
    for (let i = 0; i < N; i++) { re[i] = pcm[f + i] * win[i]; im[i] = 0; }
    fft(re, im);
    out.push(10 * Math.log10(re[k] * re[k] + im[k] * im[k] + 1e-12));
  }
  return out;
})()`, ctx);
const stftMedian = viewMedian(stftBand);
// The model's floor is taken over the 60 s before the window's end: its mel band at BAND_HZ there
const hzToMel = hz => 2595 * Math.log10(1 + hz / 700), m0 = hzToMel(50), m1 = hzToMel(15000);
const melBand = Math.round((hzToMel(BAND_HZ) - m0) / (m1 - m0) * 257) - 1;
const stops = Array.from({ length: 11 }, (_, i) => { const j = Math.round(i * 25.5) * 3; return `rgb(${INFERNO[j]},${INFERNO[j + 1]},${INFERNO[j + 2]})`; });
writeFileSync(OUT + 'spectrogram_views.json', JSON.stringify({
  from: T0, to: T1, window: w * 1.5, hi: +(-6 + 13 / stft.floor).toFixed(2), boxes, stops, start,
  band: {
    hz: BAND_HZ,
    // the frames of the window's floor: the 60 s up to its end, fewer at the start of the file
    floorSeconds: +Math.min(60, w * 1.5 + 3).toFixed(1),
    // the frames the floor is taken over, without digital silence, as the model's floor
    mel: levels(win.db.filter((c, j) => win.anySound ? win.sound[j] : true).map(c => c[melBand]),
      { floor: win.floor[melBand], blackBelow: win.floor[melBand] - 3 }),
    stft: levels(stftBand, { median: stftMedian, blackBelow: stftMedian + 7 }),
  },
}) + '\n');
console.log(`${boxes.length} of ${allBoxes.length} boxes in ${T0}–${T1} s; STFT white at ${(-6 + 13 / stft.floor).toFixed(1)} dB above the median`);
