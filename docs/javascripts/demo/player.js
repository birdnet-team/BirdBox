// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * One audio element for the demo. Emits 'time' while playing and 'state'
 * on play/pause; play(from, to) plays a segment.
 */
class Player extends EventTarget {
  constructor() {
    super();
    this.audio = new Audio();
    this.audio.preload = 'auto';
    this.stopAt = null;
    this.raf = 0;
    const tick = () => {
      if (this.stopAt != null && this.audio.currentTime >= this.stopAt) {
        this.audio.pause();
        this.audio.currentTime = this.stopAt;
        this.stopAt = null;
      }
      this.dispatchEvent(new CustomEvent('time', { detail: this.audio.currentTime }));
      if (!this.audio.paused) this.raf = requestAnimationFrame(tick);
    };
    this.audio.addEventListener('play', () => {
      cancelAnimationFrame(this.raf);
      this.raf = requestAnimationFrame(tick);
      this.dispatchEvent(new Event('state'));
    });
    this.audio.addEventListener('pause', () => { this.dispatchEvent(new Event('state')); tick(); });
    this.audio.addEventListener('ended', () => { this.stopAt = null; this.dispatchEvent(new Event('state')); });
    this.audio.addEventListener('seeked', () => this.dispatchEvent(new CustomEvent('time', { detail: this.audio.currentTime })));
  }

  get time() { return this.audio.currentTime || 0; }
  get playing() { return !this.audio.paused; }
  get duration() { return this.audio.duration || 0; }

  setSource(url) {
    this.pause();
    this.stopAt = null;
    this.audio.src = url || '';
    if (url) this.audio.load();
  }

  async play(from, to) {
    if (!this.audio.src) return;
    if (from != null) this.audio.currentTime = Math.max(0, from);
    this.stopAt = to != null ? to : null;
    try { await this.audio.play(); } catch { /* autoplay blocked until a user gesture */ }
  }

  pause() { this.audio.pause(); }

  toggle() { this.playing ? this.pause() : this.play(); }

  seek(time) {
    if (!this.audio.src) return;
    this.stopAt = null;
    this.audio.currentTime = Math.max(0, time);
  }
}

export const player = new Player();
