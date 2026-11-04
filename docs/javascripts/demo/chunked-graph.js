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
 */
import { splitSegmentGraph } from './onnx-graph.js';

const SAMPLE_RATE = 32000;
const CLIP_SECONDS = 3;
const OWNED_UNTIL = 0.75 * CLIP_SECONDS;   // a window owns its boxes up to here (fusion.js)
const scalar = k => BigInt64Array.of(BigInt(k));

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
   */
  async* run(samples, conf, chunkClips, from = { segment: 0, clip: 0, clips: 0 }) {
    const { ort, features, detect, clips, times, shared } = this;
    const audio = new ort.Tensor('float32', samples, [samples.length]);
    const confTensor = new ort.Tensor('float32', Float32Array.of(conf), [1]);
    let segments = from.segment + 1, done = from.clips;
    for (let k = from.segment; k < segments; k++) {
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
        yield { data, rows, next: { segment: k, clip: c + n, clips: done }, analyzedTo: timeTensor.data[c + n - 1] + OWNED_UNTIL };
      }
    }
  }
}
