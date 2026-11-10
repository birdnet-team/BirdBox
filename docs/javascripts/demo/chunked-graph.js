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
 */
import { splitSegmentGraph } from './onnx-graph.js';

const SAMPLE_RATE = 32000;
const CLIP_SECONDS = 3;
const OWNED_UNTIL = 0.75 * CLIP_SECONDS;   // a window owns its boxes up to here (fusion.js)
const scalar = k => BigInt64Array.of(BigInt(k));
const SEGMENT_SECONDS = 60, CONTEXT_SECONDS = 2;   // SEGMENT_SAMPLES and CONTEXT_SAMPLES in onnx_export.py
const CLIP_SAMPLES = CLIP_SECONDS * SAMPLE_RATE, HOP_SAMPLES = CLIP_SAMPLES / 2;
export const NO_END = 1e9;                   // file end of rows while the recording is still growing

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
    return Object.assign(new ChunkedGraph(), { ort, features, detect, clips, times, shared: split.shared });
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
