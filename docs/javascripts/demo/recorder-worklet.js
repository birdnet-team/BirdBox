// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Audio worklet of the demo's recorder: hands the microphone's samples to the
 * page, about 20 times per second, exactly as they arrive (mixed to mono).
 */
const CHUNK = 2048;

class RecorderTap extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunk = new Float32Array(CHUNK);
    this.filled = 0;
  }

  process(inputs) {
    const channels = inputs[0];
    if (!channels?.length) return true;
    const frames = channels[0].length, gain = 1 / channels.length;
    for (let i = 0; i < frames; i++) {
      let sum = 0;
      for (const channel of channels) sum += channel[i];
      this.chunk[this.filled++] = sum * gain;
      if (this.filled === CHUNK) {
        this.port.postMessage(this.chunk, [this.chunk.buffer]);
        this.chunk = new Float32Array(CHUNK);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor('bbd-recorder-tap', RecorderTap);
