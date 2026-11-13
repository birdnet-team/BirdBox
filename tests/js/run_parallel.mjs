// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * The demo tests of this folder in parallel processes (`npm test`).
 *
 * Every test keeps ONNX Runtime Web on one wasm thread, as the demo runs it (GitHub
 * Pages cannot provide the cross-origin isolation that threads need), so each process
 * computes exactly what the browser does. Only independent parts run side by side:
 * per model, confidence and audio length.
 * Each job's output is printed when it ends, and the run fails if any job fails.
 *
 * Usage (Node 18 or newer):
 *   cd tests/js && npm install
 *   node run_parallel.mjs [--jobs=N]
 *
 * N defaults to the cores but one, and to what the free memory allows at about 2 GB
 * a job. `npm run test:serial` runs the tests one after another, as before.
 */
import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import os from 'node:os';

const here = new URL('./', import.meta.url).pathname;
const root = new URL('../../', import.meta.url);
const arg = process.argv.slice(2).find(a => a.startsWith('--jobs='));
const MEMORY_PER_JOB = 2 * 2 ** 30;
const available = typeof process.availableMemory === 'function' ? process.availableMemory() : os.freemem();
const JOBS = arg ? +arg.split('=')[1]
  : Math.max(1, Math.min((os.availableParallelism?.() ?? os.cpus().length) - 1, Math.floor(available / MEMORY_PER_JOB)));

const manifest = JSON.parse(readFileSync(new URL('docs/models/manifest.json', root)));
const models = manifest.models.map(m => new URL(`docs/models/${m.file}`, root).pathname);
const name = path => path.split('/').pop();

const jobs = ['0.05', '0.15', '0.3', '0.5'].map(c => ['demo_merge_parity.mjs', `--conf=${c}`]);
for (const m of models) for (const s of ['3,5.8', '60', '61', '100']) jobs.push(['demo_chunk_parity.mjs', m, `--seconds=${s}`]);
// A recording's boxes while it arrives, and the recorder's resampler
for (const m of models) {
  for (const s of ['130', '61.9']) jobs.push(['rolling_floor_live.mjs', m, `--seconds=${s}`]);
  jobs.push(['demo_resampler.mjs', m, '--only=resampled']);
}
jobs.push(['demo_resampler.mjs', '--only=sines']);
// Which boxes are dashed, in files and while recording
for (const m of models) jobs.push(['demo_dashed_boxes.mjs', m]);
// Even spacing of a file's windows, on simulated workers
jobs.push(['demo_pacer.mjs']);
// Fusion that keeps the boxes later windows cannot reach, against fusing anew
jobs.push(['demo_fusion_parts.mjs']);
// The noise floor computed in the page, against the graph computing it
for (const m of models) jobs.push(['demo_floor.mjs', m]);
// The spectrogram's zoomed-out columns, against the maxima of their frames
jobs.push(['demo_zoom_levels.mjs']);
// A recording's live spectrogram, against the finished one, for different starts of a microphone
jobs.push(['demo_live_start.mjs']);
// The longest jobs first, so the last ones to finish are short
const cost = job => (job[0] === 'demo_merge_parity.mjs' ? 4 : job[0] === 'demo_dashed_boxes.mjs' ? 2 : 1)
  * (name(job[1] ?? '').startsWith('Birds-With-Labels') ? 2 : 1);
jobs.sort((a, b) => cost(b) - cost(a));

/** One job: its output and exit code. */
const run = job => new Promise(resolve => {
  const t0 = performance.now(), child = spawn(process.execPath, job, { cwd: here });
  let out = '';
  child.stdout.on('data', d => { out += d; });
  child.stderr.on('data', d => { out += d; });
  child.on('close', code => resolve({ job, code, out, s: (performance.now() - t0) / 1000 }));
});

console.log(`${jobs.length} jobs, ${JOBS} at a time`);
const t0 = performance.now(), failed = [];
let next = 0;
const worker = async () => {
  while (next < jobs.length) {
    const { job, code, out, s } = await run(jobs[next++]);
    const label = [job[0], ...job.slice(1).map(a => a.startsWith('/') ? name(a) : a)].join(' ');
    console.log(`\n=== ${label} (${s.toFixed(0)} s)${code ? ` FAILED, exit ${code}` : ''}\n${out.trimEnd()}`);
    if (code) failed.push(label);
  }
};
await Promise.all(Array.from({ length: Math.min(JOBS, jobs.length) }, worker));
const minutes = (performance.now() - t0) / 60000;
if (failed.length) {
  console.error(`\n${failed.length} of ${jobs.length} jobs FAILED in ${minutes.toFixed(1)} min:\n  ${failed.join('\n  ')}`);
  process.exit(1);
}
console.log(`\nAll ${jobs.length} jobs passed in ${minutes.toFixed(1)} min.`);
