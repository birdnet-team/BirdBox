// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Runs a BirdBox graph a few clips at a time, so boxes appear while detection
 * runs, with the same rows as one run of the whole graph.
 *
 * The graph walks a recording in 60 s segments, and a segment's spectrogram
 * has to be computed as a whole: the noise floor of log-floor (and PCEN's
 * start state) depends on the whole segment, so cutting the audio into shorter
 * pieces would change the boxes. The network sees each clip on its own, though.
 * So the features are computed once per segment, and the segment's clips then
 * go through the network in small batches.
 *
 * A segment also reads 2 s of audio on each side, and nothing beyond that. So a
 * recording in progress can run a segment for good as soon as the audio is 2 s past
 * its end (settledAt): its rows are the ones a run of the finished file gives.
 *
 * A graph with a rolling floor (exported with BIRDBOX_FLOOR_WINDOW, `rolling` here)
 * takes each clip's floor from the 60 s before the clip's end, on one frame grid for
 * the whole file, so nothing depends on where a minute starts. ChunkedGraph.create
 * returns a StreamGraph for it: the spectrogram is computed once, a few seconds of
 * audio at a time, kept as frames, and each window goes through the network as soon
 * as its frames are there. A window's rows are final as soon as the audio reaches
 * its end (readyWindows), in a file and in a recording in progress (live()).
 */
import { modelMetadata, splitSegmentGraph, streamGraphs } from './onnx-graph.js';

const SAMPLE_RATE = 32000;
const CLIP_SECONDS = 3;
const OWNED_UNTIL = 0.75 * CLIP_SECONDS;   // a window owns its boxes up to here (fusion.js)
const scalar = k => BigInt64Array.of(BigInt(k));
const SEGMENT_SECONDS = 60, CONTEXT_SECONDS = 2;   // SEGMENT_SAMPLES and CONTEXT_SAMPLES in onnx_export.py
const CLIP_SAMPLES = CLIP_SECONDS * SAMPLE_RATE, HOP_SAMPLES = CLIP_SAMPLES / 2;
export const NO_END = 1e9;                   // file end of rows while the recording is still growing
const SEGMENT_CLIPS = SEGMENT_SECONDS / (CLIP_SECONDS / 2);   // 40 windows start in each segment
/**
 * Samples after a window's end that its last spectrogram frame reads, on the rolling floor's
 * frame grid (FRAME_ORIGIN -250, CLIP_FRAMES 252, HOP_LENGTH 375, N_FFT 2048 in onnx_export.py).
 * That frame ends 77 samples before the window does, so this is 0.
 */
export const LOOKAHEAD = Math.max(0, -250 + (252 - 1) * 375 + 2048 - CLIP_SAMPLES);

/** Windows whose rows are final in a recording of n samples, with a rolling floor. */
export const readyWindows = n => Math.max(0, Math.floor((n - LOOKAHEAD - CLIP_SAMPLES) / HOP_SAMPLES) + 1);

/** Samples a recording needs before segment k no longer depends on what comes next. */
export const settledAt = k => ((k + 1) * SEGMENT_SECONDS + CONTEXT_SECONDS) * SAMPLE_RATE;

/** File end of raw_all (column 8) for n samples: where the last window ends (_pad_to_clip_length in onnx_export.py). */
export const fileEnd = n => Math.fround((CLIP_SAMPLES + Math.ceil(Math.max(0, n - CLIP_SAMPLES) / HOP_SAMPLES) * HOP_SAMPLES) / SAMPLE_RATE);

export class ChunkedGraph {
  /**
   * @param ort  the ONNX Runtime Web module
   * @param {Uint8Array} model  ModelProto bytes of a BirdBox graph
   * @param {object} options  session options
   */
  static async create(ort, model, options) {
    const meta = modelMetadata(model), rolling = +(meta.floor_window_seconds || 0);
    const stream = rolling ? streamGraphs(model) : null;
    if (stream) return StreamGraph.create(ort, stream, options, rolling, +meta.frame_origin);
    const split = splitSegmentGraph(model);
    const features = await ort.InferenceSession.create(split.features, options);
    // The graph declares no types for the tensors at the cut: a short run shows them
    const audio = new ort.Tensor('float32', new Float32Array(CLIP_SECONDS * SAMPLE_RATE), [CLIP_SECONDS * SAMPLE_RATE]);
    const probe = await features.run({ audio, audio_in: audio, iter: new ort.Tensor('int64', scalar(0), []) });
    const types = {};
    for (const { from, to } of split.takes) types[to] = probe[from].type;
    for (const name of split.shared) types[name] = probe[name].type;
    const detect = await ort.InferenceSession.create(split.detect(types), options);
    // Clips have three dimensions (clips, mels, frames), their start times one
    const clips = split.takes.find(t => probe[t.from].dims.length === 3);
    const times = split.takes.find(t => probe[t.from].dims.length === 1);
    if (!clips || !times) throw new Error('The ONNX graph does not have the expected clip tensors');
    return Object.assign(new ChunkedGraph(), { ort, features, detect, clips, times, shared: split.shared, rolling: 0 });
  }

  /**
   * Boxes of the last `clips` windows of a piece of audio that is still growing
   * (a recording in progress). The spectrogram covers just this piece, so its noise
   * floor is an estimate: the boxes are provisional. samples must end on the window
   * grid, offset (seconds) is where it starts in the recording. Rows come back in
   * recording time, with no end of file in sight (column 8), so that no window owns
   * the time after its middle half until the next one has been seen.
   *
   * @param {Float32Array} samples  mono audio at 32 kHz, at most one segment long
   * @param {number} offset  start of samples in the recording, on the window grid
   * @param {number} conf  confidence threshold
   * @param {number} clips  how many of the last windows to run
   * @returns {Promise<{data: Float32Array, rows: number}>}
   */
  async detectTail(samples, offset, conf, clips) {
    const { ort, features, detect, clips: clipCut, times, shared } = this;
    const audio = new ort.Tensor('float32', samples, [samples.length]);
    const out = await features.run({ audio, audio_in: audio, iter: new ort.Tensor('int64', scalar(0), []) });
    const clipTensor = out[clipCut.from], timeTensor = out[times.from];
    const [slots, mels, frames] = clipTensor.dims, size = mels * frames, available = slots - 1;
    const n = Math.min(clips, available), first = available - n;
    if (n <= 0) return { data: new Float32Array(0), rows: 0 };
    const feeds = {
      [clipCut.to]: new ort.Tensor('float32', clipTensor.data.subarray(first * size, available * size), [n, mels, frames]),
      [times.to]: new ort.Tensor('float32', timeTensor.data.subarray(first, available), [n]),
      conf_in: new ort.Tensor('float32', Float32Array.of(conf), [1]),
    };
    for (const name of shared) feeds[name] = out[name];
    const raw = (await detect.run(feeds)).raw, rows = raw.dims[0] || 0, data = raw.data.slice();
    for (let r = 0; r < rows; r++) {
      const o = r * 9;
      data[o + 1] += offset;
      data[o + 2] += offset;
      data[o + 7] += offset;
      data[o + 8] = NO_END;
    }
    return { data, rows };
  }

  /**
   * Yields { data, rows, next, analyzedTo } after every chunkClips clips. data holds
   * raw_all rows (9 columns); concatenated, they equal raw_all of a full run. next is
   * the position to continue from, { segment, clip, clips }, where clips counts the
   * clips done so far. analyzedTo is the time up to which every window that owns a box
   * has been seen.
   *
   * @param {Float32Array} samples  mono audio at 32 kHz
   * @param {number} conf  confidence threshold
   * @param {number} chunkClips  clips per network run
   * @param {{segment: number, clip: number, clips: number}} [from]  a `next` to continue from
   * @param {number} [until]  segment to stop before: samples may be a settled piece of the recording
   */
  async* run(samples, conf, chunkClips, from = { segment: 0, clip: 0, clips: 0 }, until = Infinity) {
    const { ort, features, detect, clips, times, shared } = this;
    const audio = new ort.Tensor('float32', samples, [samples.length]);
    const confTensor = new ort.Tensor('float32', Float32Array.of(conf), [1]);
    let segments = from.segment + 1, done = from.clips;
    for (let k = from.segment; k < Math.min(segments, until); k++) {
      const out = await features.run({ audio, audio_in: audio, iter: new ort.Tensor('int64', scalar(k), []) });
      segments = Number(out.trip_count.data[0]);
      const clipTensor = out[clips.from], timeTensor = out[times.from];
      const [slots, mels, frames] = clipTensor.dims, size = mels * frames;
      // The last slot is the graph's empty clip for segments without clips
      for (let c = k === from.segment ? from.clip : 0; c < slots - 1; c += chunkClips) {
        const n = Math.min(chunkClips, slots - 1 - c);
        const feeds = {
          [clips.to]: new ort.Tensor('float32', clipTensor.data.subarray(c * size, (c + n) * size), [n, mels, frames]),
          [times.to]: new ort.Tensor('float32', timeTensor.data.subarray(c, c + n), [n]),
          conf_in: confTensor,
        };
        for (const name of shared) feeds[name] = out[name];
        const raw = (await detect.run(feeds)).raw;
        const rows = raw.dims[0] || 0, data = raw.data.slice();
        // Column 0 counts the clips of the segment, as in a full run
        for (let r = 0; r < rows; r++) data[r * 9] += c;
        done += n;
        // At the end of a segment, the next one can start without this one's spectrogram (the graph has none after the last)
        const next = c + n < slots - 1 || k + 1 >= segments ? { segment: k, clip: c + n, clips: done } : { segment: k + 1, clip: 0, clips: done };
        yield { data, rows, next, analyzedTo: timeTensor.data[c + n - 1] + OWNED_UNTIL };
      }
    }
  }
}

/* ---------------- rolling floor: streaming ---------------- */

const N_MELS = 256, HOP_LENGTH = 375, N_FFT = 2048, CLIP_FRAMES = 252;
const WINDOW_FRAMES = HOP_SAMPLES / HOP_LENGTH;    // 128 frames from one window to the next
const LEFT_PAD = 16000;                            // the segment code's 0.5 s left pad (PCEN_LEFT_PAD_SECONDS)
const FILE_BLOCK_FRAMES = 512;                     // spectrogram frames per run in a file: 6 s of audio
const f32 = Math.fround;

/** Samples _pad_to_clip_length (onnx_export.py) pads n samples to: the end of the last window. */
const paddedLength = n => CLIP_SAMPLES + Math.ceil(Math.max(0, n - CLIP_SAMPLES) / HOP_SAMPLES) * HOP_SAMPLES;

/** `next` of window w, as ChunkedGraph.run reports it: at the end of the file it stays in the last segment. */
const nextOf = (w, total) => w < total
  ? { segment: Math.floor(w / SEGMENT_CLIPS), clip: w % SEGMENT_CLIPS, clips: w }
  : { segment: Math.floor((w - 1) / SEGMENT_CLIPS), clip: (w - 1) % SEGMENT_CLIPS + 1, clips: w };

/** Spectrogram frames from frame `base` on, band by band (n_mels, frames) as the graph wants them. */
class Frames {
  constructor() { this.cap = 0; this.db = new Float32Array(0); this.sound = new Uint8Array(0); this.base = 0; this.count = 0; }

  get end() { return this.base + this.count; }

  /** Forget the frames before `frame`. */
  drop(frame) {
    const k = Math.min(Math.max(0, frame - this.base), this.count);
    if (!k) return;
    for (let b = 0; b < N_MELS; b++) this.db.copyWithin(b * this.cap, b * this.cap + k, b * this.cap + this.count);
    this.sound.copyWithin(0, k, this.count);
    this.base += k;
    this.count -= k;
  }

  /** Append n frames of a (n_mels, n) block. */
  append(db, sound, n) {
    if (this.count + n > this.cap) {
      const cap = Math.max(2 * this.cap, this.count + n, 4096), grown = new Float32Array(N_MELS * cap);
      for (let b = 0; b < N_MELS; b++) grown.set(this.db.subarray(b * this.cap, b * this.cap + this.count), b * cap);
      const s = new Uint8Array(cap);
      s.set(this.sound.subarray(0, this.count));
      Object.assign(this, { cap, db: grown, sound: s });
    }
    for (let b = 0; b < N_MELS; b++) this.db.set(db.subarray(b * n, (b + 1) * n), b * this.cap + this.count);
    this.sound.set(sound.subarray(0, n), this.count);
    this.count += n;
  }

  /** Frames from..to-1, as new arrays. */
  slice(from, to) {
    const n = to - from, o = from - this.base, db = new Float32Array(N_MELS * n);
    if (o < 0 || o + n > this.count) throw new Error(`frames ${from} to ${to - 1} are not computed (${this.base} to ${this.end - 1})`);
    for (let b = 0; b < N_MELS; b++) db.set(this.db.subarray(b * this.cap + o, b * this.cap + o + n), b * n);
    return { db, sound: this.sound.slice(o, o + n), n };
  }
}

/**
 * A rolling-floor graph run as a stream (StreamSpectrogram and StreamDetect in
 * onnx_export.py): frame m of a file covers samples m HOP_LENGTH + origin to
 * m HOP_LENGTH + origin + N_FFT, window w is frames 128 w to 128 w + 251, and its floor
 * comes from the frames in the floor window before its end. Rows equal the segment
 * graph's, bit for bit (tests/js/demo_chunk_parity.mjs, rolling_floor_live.mjs).
 */
export class StreamGraph {
  static async create(ort, graphs, options, rolling, origin) {
    const spectrogram = await ort.InferenceSession.create(graphs.spectrogram, options);
    const detect = await ort.InferenceSession.create(graphs.detect, options);
    return Object.assign(new StreamGraph(), { ort, spectrogram, detect, rolling, origin, window: rolling * SAMPLE_RATE });
  }

  /** First frame of window w's floor window. */
  _first(w) {
    return Math.max(0, Math.ceil((w * HOP_SAMPLES + CLIP_SAMPLES - this.window - this.origin) / HOP_LENGTH));
  }

  /** Last frame window w needs. */
  _last(w) { return w * WINDOW_FRAMES + CLIP_FRAMES - 1; }

  /** Compute frames up to `upTo` from n samples of audio, `block` frames at most per run when audio allows. */
  async _frames(frames, audio, n, upTo, block) {
    const { ort, origin } = this, available = Math.floor((n - origin - N_FFT) / HOP_LENGTH);
    if (upTo > available) throw new Error(`frame ${upTo} needs more than ${n} samples`);
    while (frames.end <= upTo) {
      const m0 = frames.end, m1 = Math.min(available, Math.max(upTo, m0 + block - 1));
      const s0 = m0 * HOP_LENGTH + origin, s1 = m1 * HOP_LENGTH + origin + N_FFT;
      // Before the file start: the samples the segment code's left pad puts there
      let piece = audio.subarray(Math.max(0, s0), s1);
      if (s0 < 0) {
        const pre = new Float32Array(s1 - s0);
        pre.set(audio.subarray(LEFT_PAD + s0, LEFT_PAD));
        pre.set(piece, -s0);
        piece = pre;
      }
      const out = await this.spectrogram.run({ audio: new ort.Tensor('float32', piece, [piece.length]) });
      const count = out.db.dims[1];
      if (count !== m1 - m0 + 1) throw new Error(`${count} frames from the spectrogram, ${m1 - m0 + 1} expected`);
      frames.append(out.db.data, out.sound.data, count);
    }
  }

  /** Rows of windows w0 .. w0+count-1 from the frames, column 0 counting the clips of each segment. */
  async _detect(frames, w0, count, conf) {
    const { ort } = this, first = this._first(w0);
    frames.drop(first);   // the next windows start their floor windows later
    const { db, sound, n } = frames.slice(first, this._last(w0 + count - 1) + 1);
    const times = new Float32Array(count);
    for (let i = 0; i < count; i++) times[i] = (w0 + i) * HOP_SAMPLES / SAMPLE_RATE;
    const raw = (await this.detect.run({
      db: new ort.Tensor('float32', db, [N_MELS, n]),
      sound: new ort.Tensor('bool', sound, [n]),
      first_frame: new ort.Tensor('int64', BigInt64Array.of(BigInt(first)), []),
      times: new ort.Tensor('float32', times, [count]),
      conf: new ort.Tensor('float32', Float32Array.of(conf), [1]),
    })).raw;
    const rows = raw.dims[0] || 0, data = raw.data.slice();
    for (let r = 0; r < rows; r++) data[r * 9] = Math.round(data[r * 9 + 7] / (HOP_SAMPLES / SAMPLE_RATE)) % SEGMENT_CLIPS;
    return { data, rows };
  }

  /**
   * ChunkedGraph.run for a file: yields { data, rows, next, analyzedTo } after every
   * chunkClips windows, with the same rows, `next` and continuation from it.
   */
  async* run(samples, conf, chunkClips, from = { segment: 0, clip: 0, clips: 0 }, until = Infinity) {
    const n = samples.length, target = paddedLength(n);
    let audio = samples;
    if (target > n) {
      audio = new Float32Array(target);
      audio.set(samples);
    }
    const total = (target - CLIP_SAMPLES) / HOP_SAMPLES + 1, stop = Math.min(total, until * SEGMENT_CLIPS);
    // _limit_raw_to_content in float32: after the file's last sample only silence was added
    const content = f32(f32(n) / SAMPLE_RATE), duration = f32(f32(target) / SAMPLE_RATE);
    const padded = duration > f32(content + f32(0.5 / SAMPLE_RATE));
    const frames = new Frames();
    for (let w = from.clips; w < stop;) {
      const k = Math.min(chunkClips, stop - w);
      await this._frames(frames, audio, target, this._last(w + k - 1), FILE_BLOCK_FRAMES);
      let { data, rows } = await this._detect(frames, w, k, conf);
      let kept = 0;
      for (let r = 0; r < rows; r++) {
        const o = r * 9, t0 = data[o + 1], t1 = padded ? Math.min(data[o + 2], content) : data[o + 2];
        if (padded && !(t0 < content && t1 > t0)) continue;
        if (kept !== r) data.copyWithin(kept * 9, o, o + 9);
        data[kept * 9 + 2] = t1;
        data[kept * 9 + 8] = duration;
        kept++;
      }
      w += k;
      yield { data: data.slice(0, kept * 9), rows: kept, next: nextOf(w, total), analyzedTo: (w - 1) * HOP_SAMPLES / SAMPLE_RATE + OWNED_UNTIL };
    }
  }

  /**
   * A recording in progress: push() its audio as it arrives, windows() gives the final
   * rows of the next windows once they are ready (readyWindows of the samples pushed),
   * with the file end (column 8) NO_END. Only new audio is turned into frames.
   */
  live() {
    const graph = this, frames = new Frames();
    let audio = new Float32Array(60 * SAMPLE_RATE), n = 0;
    return {
      get length() { return n; },
      push(pcm) {
        if (n + pcm.length > audio.length) {
          const grown = new Float32Array(Math.max(2 * audio.length, n + pcm.length));
          grown.set(audio.subarray(0, n));
          audio = grown;
        }
        audio.set(pcm, n);
        n += pcm.length;
      },
      async windows(conf, from, count) {
        if (from.clips + count > readyWindows(n)) throw new Error(`windows ${from.clips} to ${from.clips + count - 1} are not ready in ${n} samples`);
        await graph._frames(frames, audio, n, graph._last(from.clips + count - 1), WINDOW_FRAMES);
        const { data, rows } = await graph._detect(frames, from.clips, count, conf);
        for (let r = 0; r < rows; r++) data[r * 9 + 8] = NO_END;
        const w = from.clips + count;
        return { data, rows, next: { segment: Math.floor(w / SEGMENT_CLIPS), clip: w % SEGMENT_CLIPS, clips: w } };
      },
    };
  }
}

