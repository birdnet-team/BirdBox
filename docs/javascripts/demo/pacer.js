// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * Shows the windows of a file evenly spaced while detection runs.
 *
 * The worker finishes windows unevenly: now and then a few take up to twice as long.
 * Shown as they come, the line would stall and jump. So a few finished windows are
 * held in hand (about HAND_MS of them, at least two), and the windows show one
 * interval apart:
 *
 *   - the interval is the worker's mean time per window, a mean that follows it
 *     slowly (FOLLOW), so the windows in hand cover a slow stretch;
 *   - with fewer windows in hand the pace slows a little, with more it quickens a
 *     little (CORRECT per window); with none, the next window shows as soon as it is
 *     done;
 *   - the first window waits for a full hand, which the gentle pace would take long
 *     to fill.
 *
 * When a redraw costs more than its share of the time (SHARE: a long recording with
 * many boxes) or windows come faster than MIN_STEP_MS, each step shows the same number
 * of windows. tests/js/demo_pacer.mjs checks the spacing on simulated workers.
 */

export const HAND_MS = 600;   // windows held back: about this much of the worker's time
const FOLLOW = 0.05;          // how fast the mean follows the worker's time per window
const CORRECT = 0.03;         // pace change per window in hand too few or too many
const MIN_STEP_MS = 50;       // redraw at most this often,
const SHARE = 0.25;           // and spend at most this share of the time on it

export class Pacer {
  /**
   * @param {(item: object) => void} show  draws the last item of a step
   * @param {object} [options]  for tests: a clock in ms (now), setTimeout and clearTimeout
   *   (later, cancel), and other settings (tune: { handMs, follow, correct })
   */
  constructor(show, { now = () => performance.now(), later = (fn, ms) => setTimeout(fn, ms), cancel = t => clearTimeout(t), tune = {} } = {}) {
    Object.assign(this, { show, now, later, cancel });
    this.handMs = tune.handMs ?? HAND_MS;
    this.follow = tune.follow ?? FOLLOW;
    this.correct = tune.correct ?? CORRECT;
    this.queue = [];
    this.mean = 0;            // ms per window
    this.lastIn = this.now(); // the run starts now
    this.lastGap = 0;
    this.count = 0;           // windows pushed
    this.minStep = MIN_STEP_MS;
    this.timer = null;
    this.idle = true;
    this.started = false;
    this.ending = false;      // the worker is done: the rest shows, also in a smaller step
    this.finished = null;
  }

  /** Windows held back. Before the mean is known: what the first windows took, at least two. */
  get inHand() { return Math.max(2, Math.round(this.handMs / (this.mean || this.lastGap || this.handMs))); }

  /** Windows per step: a redraw at most every minStep ms. */
  get size() { return Math.max(1, Math.ceil(this.minStep / (this.mean || this.minStep))); }

  /** A window is done; `item` is what show() gets. The first one does not count towards the mean: it also waits for the first frames. */
  push(item) {
    const now = this.now(), gap = now - this.lastIn;
    if (this.count++) this.mean = this.mean ? this.mean + this.follow * (gap - this.mean) : gap;
    this.lastIn = now;
    this.lastGap = gap;
    this.queue.push(item);
    // A step waits for its windows, and the first one for a full hand besides
    if (this.idle && this.queue.length >= this.size + (this.started ? 0 : this.inHand)) {
      this.started = true;
      this._step();
    }
  }

  /** Resolves once the last window is shown. */
  finish() {
    this.ending = true;
    if (this.idle && this.queue.length) {   // the rest: fewer windows than a hand or a step
      this.started = true;
      this._step();
    }
    if (this.idle) return Promise.resolve();
    return new Promise(resolve => { this.finished = resolve; });
  }

  /** Drop what is in hand: the run was stopped or replaced. */
  stop() {
    this.cancel(this.timer);
    this.queue = [];
    this.idle = true;
    this.finished = null;
  }

  _step() {
    this.timer = null;
    const size = this.size;
    if (!this.queue.length || (this.queue.length < size && !this.ending)) {
      // Nothing to show, or not a whole step: the next window shows (or completes the step) as soon as it is done
      this.idle = true;
      if (!this.queue.length) {
        this.finished?.();
        this.finished = null;
      }
      return;
    }
    this.idle = false;
    const t0 = this.now(), inHand = this.inHand;
    // Far behind (a hidden page, whose timers are slowed): catch up in larger steps
    const n = Math.min(this.queue.length, Math.max(size, Math.ceil((this.queue.length - inHand - 2) / 3)));
    const items = this.queue.splice(0, n);
    this.show(items[items.length - 1]);
    const now = this.now();
    this.minStep = Math.max(MIN_STEP_MS, (now - t0) / SHARE);
    if (this.finished && !this.queue.length) {
      this.idle = true;
      this.finished();
      this.finished = null;
      return;
    }
    const q = this.queue.length;
    const pace = Math.min(1.2, Math.max(0.6, 1 + this.correct * (inHand - q)));
    // The next step comes after the time of its own windows (after the first redraw, steps may grow)
    this.timer = this.later(() => this._step(), Math.max(0, t0 + this.size * (this.mean || this.minStep) * pace - now));
  }
}
