# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""Where should the BirdNET V3.0 window sit to name the species of a box?

For every annotated box of a soundscape dataset (file, start, end, eBird code), BirdNET V3.0
names a species from 3 s windows placed in different ways. The annotated boxes stand in for
the boxes BirdBox finds. As in the BirdNET website demo: birds only, each species' highest
score over the box's windows, no location filter.

  centred   one window centred on the box; a box over 3 s: windows from its start, every
            1.5 s (the demo until November 2026)
  grid15    BirdBox's own windows, one every 1.5 s. A box up to 1.5 s takes the window that
            owns its centre (the middle 1.5 s of a window, as in the window fusion); a longer
            box the windows from the one around its first 1.5 s to the one around its last
            (the demo since November 2026: scored alongside the detection)
  grid075   the same rule on windows every 0.75 s (twice the BirdNET V3.0 work)
  inside15  grid15, and a box up to 1.5 s also takes a neighbouring window it lies wholly in
  fit15     grid15, but a box up to 3 s takes only the window that owns its centre when it
            lies wholly inside it
  near15    grid15, but a box up to 3 s always takes only the window that owns its centre

Writes one JSON with, per box and scheme: [top species, its score, the true species' score,
the true species' rank]. summarize.py turns it into the tables of
docs/models-and-metrics/just-bird-species-from-birdnet.md.

The BirdNET V3.0 files are those of the BirdNET Live app (birdnet-team/birdnet-live-app,
assets/models): the FP16 model, its labels CSV and taxonomy.csv, which gives the eBird codes.

Usage:
  python scripts/birdnet_labels/label_windows.py MODEL.onnx LABELS.csv taxonomy.csv DATASET_DIR OUT.json
      [--files=N] [--threads=T] [--mid]

  DATASET_DIR holds annotations.csv (Filename, Start Time (s), End Time (s), ..., Species eBird
  Code) and soundscape_data_wav/ with the recordings at 32 kHz. --files=N takes N recordings
  spread evenly over the sorted file names. --mid keeps only boxes over 1.5 s and up to 3 s.
"""
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

SR, CLIP, HOP = 32000, 3.0, 1.5
LEN = int(CLIP * SR)
BATCH = 64


def js_round(x):
    """Math.round of JavaScript, which the demo uses."""
    return int(math.floor(x + 0.5))


def last_start(n, hop):
    """Index of the last window on a grid every `hop` s: the first one that reaches the end."""
    return math.ceil(max(0, n - LEN) / int(hop * SR))


def owner(time, n, hop):
    """The window that owns `time`: windows own their middle `hop` seconds."""
    return min(last_start(n, hop), max(0, math.floor((time - (CLIP - hop) / 2) / hop)))


def centred(t0, t1, n):
    """suggestionWindows() of the demo until November 2026: window starts in samples."""
    max_start = max(0.0, n / SR - CLIP)
    if t1 - t0 <= CLIP:
        return [js_round(min(max_start, max(0.0, (t0 + t1) / 2 - CLIP / 2)) * SR)]
    out, s = [], t0
    while True:
        start = min(max_start, s)
        out.append(js_round(start * SR))
        if start + CLIP >= t1 or start >= max_start:
            break
        s += 1.5
    return out


def grid(t0, t1, n, hop=HOP):
    """_windowsOf() of the demo since November 2026, on windows every `hop` s: starts in samples."""
    step = int(hop * SR)
    if t1 - t0 <= CLIP - hop:
        return [owner((t0 + t1) / 2, n, hop) * step]
    first, last = owner(t0 + (CLIP - hop) / 2, n, hop), owner(t1 - (CLIP - hop) / 2, n, hop)
    return [k * step for k in range(first, last + 1)]


def inside(t0, t1, n):
    """grid15, and a box up to 1.5 s also takes a neighbouring window it lies wholly in."""
    starts = grid(t0, t1, n)
    if len(starts) > 1:
        return starts
    step = int(HOP * SR)
    for s in (starts[0] - step, starts[0] + step):
        if 0 <= s <= last_start(n, HOP) * step and s / SR <= t0 and s / SR + CLIP >= t1:
            starts.append(s)
    return starts


def single(t0, t1, n, only_if_inside):
    """A box up to 3 s takes only the window that owns its centre (fit15: if it lies wholly in it)."""
    if t1 - t0 > CLIP:
        return grid(t0, t1, n)
    k = owner((t0 + t1) / 2, n, HOP)
    if not only_if_inside or (k * HOP <= t0 and t1 <= k * HOP + CLIP):
        return [k * int(HOP * SR)]
    return grid(t0, t1, n)


SCHEMES = {
    'centred': centred,
    'grid15': grid,
    'grid075': lambda t0, t1, n: grid(t0, t1, n, 0.75),
    'inside15': inside,
    'fit15': lambda t0, t1, n: single(t0, t1, n, True),
    'near15': lambda t0, t1, n: single(t0, t1, n, False),
}


def read_species(labels_csv, taxonomy_csv):
    """Per output of the model: is it a bird, and its eBird code ('' if the taxonomy has none)."""
    with open(taxonomy_csv, newline='', encoding='utf-8') as f:
        ebird = {r['scientific_name']: r['ebird_code'] for r in csv.DictReader(f)}
    with open(labels_csv, newline='', encoding='utf-8') as f:
        labels = list(csv.DictReader(f, delimiter=';'))
    for i, row in enumerate(labels):
        if int(row['idx']) != i:
            raise ValueError(f'{labels_csv}: idx {row["idx"]} in row {i}')
    is_bird = np.array([row['class'] == 'Aves' for row in labels])
    codes = [ebird.get(row['sci_name'], '') for row in labels]
    return is_bird, codes


def main():
    args = sys.argv[1:]
    opt = lambda name, default: next((int(a.split('=')[1]) for a in args if a.startswith(f'--{name}=')), default)
    positional = [a for a in args if not a.startswith('--')]
    if len(positional) != 5:
        sys.exit(__doc__)
    model, labels_csv, taxonomy_csv, dataset, out_path = positional
    max_files, threads, mid_only = opt('files', 0), opt('threads', 8), '--mid' in args

    is_bird, codes = read_species(labels_csv, taxonomy_csv)
    index_of = {c: i for i, c in enumerate(codes) if c and is_bird[i]}
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(model, options, providers=['CPUExecutionProvider'])
    input_name, output_name = session.get_inputs()[0].name, session.get_outputs()[0].name

    rows = defaultdict(list)
    with open(Path(dataset) / 'annotations.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows[r['Filename']].append((float(r['Start Time (s)']), float(r['End Time (s)']), r['Species eBird Code']))
    files = sorted(rows)
    if max_files and len(files) > max_files:
        files = [files[int(i * len(files) / max_files)] for i in range(max_files)]

    results, skipped, windows, t_start = [], defaultdict(int), defaultdict(int), time.time()
    audio_dir = Path(dataset) / 'soundscape_data_wav'
    for fi, name in enumerate(files):
        path = audio_dir / (Path(name).stem + '.wav')
        pcm, sr = sf.read(path if path.exists() else audio_dir / name, dtype='float32', always_2d=True)
        if sr != SR:
            raise ValueError(f'{name}: {sr} Hz, the script expects {SR} Hz')
        pcm = pcm.mean(axis=1) if pcm.shape[1] > 1 else pcm[:, 0]
        n = len(pcm)
        boxes = []
        for t0, t1, code in rows[name]:
            if code not in index_of:
                skipped[code] += 1
                continue
            t0, t1 = max(0.0, t0), min(n / SR, t1)
            if t1 <= t0 or (mid_only and not HOP < t1 - t0 <= CLIP):
                continue
            boxes.append((t0, t1, index_of[code], {k: fn(t0, t1, n) for k, fn in SCHEMES.items()}))
        for k in SCHEMES:
            windows[k] += len({s for b in boxes for s in b[3][k]})
        starts = sorted({s for b in boxes for ws in b[3].values() for s in ws})
        scores = {}
        for i in range(0, len(starts), BATCH):
            part = starts[i:i + BATCH]
            batch = np.zeros((len(part), LEN), np.float32)   # silence after the end, as in the demo
            for j, s in enumerate(part):
                piece = pcm[s:s + LEN]
                batch[j, :len(piece)] = piece
            out = session.run([output_name], {input_name: batch})[0].astype(np.float32)
            out[:, ~is_bird] = 0
            scores.update(zip(part, out))
        for t0, t1, truth, wins in boxes:
            entry = {'file': name, 't0': t0, 't1': t1, 'truth': codes[truth]}
            for k, ws in wins.items():
                pooled = np.max([scores[s] for s in ws], axis=0)
                top = int(np.argmax(pooled))
                entry[k] = [codes[top], round(float(pooled[top]), 4), round(float(pooled[truth]), 4), int((pooled > pooled[truth]).sum()) + 1]
            results.append(entry)
        print(f'{fi + 1}/{len(files)} {name}: {len(boxes)} boxes, {len(starts)} windows, {time.time() - t_start:.0f} s', flush=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'dataset': Path(dataset).name, 'files': len(files), 'mid_only': mid_only,
                   'skipped': dict(skipped), 'windows': dict(windows), 'boxes': results}, f)
    top = sorted(skipped.items(), key=lambda kv: -kv[1])[:8]
    print(f'{len(results)} boxes; skipped {sum(skipped.values())} whose species is not a BirdNET V3.0 bird: {dict(top)}')


if __name__ == '__main__':
    main()
