// SPDX-FileCopyrightText: 2026 BirdNET-Team
// SPDX-License-Identifier: MIT
/**
 * The demo tests of this folder in parallel processes (`npm test`).
 *
 * Every test keeps ONNX Runtime Web on one wasm thread, as the demo runs it (GitHub
 * Pages cannot provide the cross-origin isolation that threads need), so each process
 * computes exactly what the browser does. Only independent parts run side by side:
 * per model, check and audio length.
 * Each job's output is printed when it ends, and the run fails if any job fails.
 *
 * The models run the same JavaScript except for their merge: window fusion
 * (Just-Bird) or song reconstruction (Birds-With-Labels). So every check runs on the
 * first model of docs/models/manifest.json, the demo's default, and the others run
 * the checks of their merge and one chunk parity check of their graph. --all runs
 * every check on every model, as after exporting a new model.
 *
 * Usage (Node 18 or newer):
 *   cd tests/js && npm install
 *   node run_parallel.mjs [--all] [--jobs=N] [--shard=K/M]
 *
 * N defaults to the cores but one, and to what the free memory allows at about 2 GB
 * a job. --shard splits the jobs into M parts of about equal time and runs part K
 * (1 to M), so that M machines can share them (the demo tests workflow).
 * `npm run test:serial` runs the tests one after another.
 */
import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import os from 'node:os';

const here = new URL('./', import.meta.url).pathname;
const root = new URL('../../', import.meta.url);
const opt = key => process.argv.slice(2).find(a => a.startsWith(`--${key}=`))?.split('=')[1];
const MEMORY_PER_JOB = 2 * 2 ** 30;
const available = typeof process.availableMemory === 'function' ? process.availableMemory() : os.freemem();
const JOBS = opt('jobs') ? +opt('jobs')
  : Math.max(1, Math.min((os.availableParallelism?.() ?? os.cpus().length) - 1, Math.floor(available / MEMORY_PER_JOB)));
const [SHARD, SHARDS] = (opt('shard') ?? '1/1').split('/').map(Number);
if (!(SHARD >= 1 && SHARD <= SHARDS)) throw new Error('--shard=K/M needs 1 <= K <= M');

const manifest = JSON.parse(readFileSync(new URL('docs/models/manifest.json', root)));
const models = manifest.models.map(m => new URL(`docs/models/${m.file}`, root).pathname);
const name = path => path.split('/').pop();
const ALL = process.argv.includes('--all');
const each = ALL ? models : models.slice(0, 1);

// The merges against the graph's, and which boxes are dashed, in files and while
// recording. The other models: one run of the full graph, and half the audio for the
// dashed boxes.
const first = m => m === models[0] || ALL;
const jobs = models.map(m => ['demo_merge_parity.mjs', m, ...first(m) ? [] : ['--one-run']]);
for (const m of models) {
  for (const only of ['file', 'live']) jobs.push(['demo_dashed_boxes.mjs', m, `--only=${only}`, ...first(m) ? [] : ['--seconds=150']]);
}
// Detection in chunks against one run of the graph
for (const m of models) {
  for (const s of first(m) ? ['3,5.8', '60', '61', '100'] : ['3,5.8', '61']) {
    jobs.push(['demo_chunk_parity.mjs', m, `--seconds=${s}`]);
  }
}
// A recording's boxes while it arrives, and the recorder's resampler
for (const m of each) {
  for (const s of ['130', '61.9']) jobs.push(['rolling_floor_live.mjs', m, `--seconds=${s}`]);
  jobs.push(['demo_resampler.mjs', m, '--only=resampled']);
}
jobs.push(['demo_resampler.mjs', '--only=sines']);
// Even spacing of a file's windows, on simulated workers
jobs.push(['demo_pacer.mjs']);
// Fusion that keeps the boxes later windows cannot reach, against fusing anew
jobs.push(['demo_fusion_parts.mjs']);
// The noise floor computed in the page, against the graph computing it
for (const m of each) jobs.push(['demo_floor.mjs', m]);
// The spectrogram's zoomed-out columns, against the maxima of their frames
jobs.push(['demo_zoom_levels.mjs']);
// A recording's live spectrogram, against the finished one, for different starts of a microphone
jobs.push(['demo_live_start.mjs']);
// About the seconds a job takes on Just-Bird with 4 jobs on a GitHub runner (three
// times a desktop's), per second of audio where the job sets it. Birds-With-Labels
// takes about twice as long.
const SECONDS = {
  'demo_merge_parity.mjs': 240, 'demo_resampler.mjs': 290, 'demo_floor.mjs': 340,
  'demo_fusion_parts.mjs': 210, 'demo_live_start.mjs': 25, 'demo_zoom_levels.mjs': 15,
  'demo_dashed_boxes.mjs --only=file': 100, 'demo_dashed_boxes.mjs --only=live': 190,
};
const PER_AUDIO_SECOND = {
  'demo_chunk_parity.mjs': 1.75, 'rolling_floor_live.mjs': 0.67, 'demo_dashed_boxes.mjs': 1 / 300,
};
const cost = job => {
  if (job.includes('--only=sines')) return 0;
  const audio = job.find(a => a.startsWith('--seconds='))?.split('=')[1].split(',').map(Number);
  const only = job.find(a => a.startsWith('--only=')), base = SECONDS[only ? `${job[0]} ${only}` : job[0]];
  let s = audio && PER_AUDIO_SECOND[job[0]] ? PER_AUDIO_SECOND[job[0]] * Math.max(...audio) * (base ?? 1) : base ?? 0;
  if (job.includes('--one-run')) s /= 2;
  return s * (name(job[1] ?? '').startsWith('Birds-With-Labels') ? 2 : 1);
};
// The longest jobs first, so the last ones to finish are short. A shard takes each job
// in turn if it is the part with the least time so far.
jobs.sort((a, b) => cost(b) - cost(a));
const load = new Array(SHARDS).fill(0);
const mine = jobs.filter(job => {
  const k = load.indexOf(Math.min(...load));
  load[k] += cost(job);
  return k === SHARD - 1;
});
jobs.splice(0, jobs.length, ...mine);

/** One job: its output and exit code. */
const run = job => new Promise(resolve => {
  const t0 = performance.now(), child = spawn(process.execPath, job, { cwd: here });
  let out = '';
  child.stdout.on('data', d => { out += d; });
  child.stderr.on('data', d => { out += d; });
  child.on('close', code => resolve({ job, code, out, s: (performance.now() - t0) / 1000 }));
});

console.log(`${jobs.length} jobs${SHARDS > 1 ? ` (part ${SHARD} of ${SHARDS})` : ''}, ${JOBS} at a time`);
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
