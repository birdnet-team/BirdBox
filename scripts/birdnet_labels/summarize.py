# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""Tables from label_windows.py: how often each window placement names the annotated species.

  top-1        the species BirdNET V3.0 scores highest is the annotated one
  top-5        the annotated species is among the five highest (the demo's species menu)
  named right / named wrong / unnamed
               the demo names a species only from a score of 0.35 on

Differences to the centred window come with 95% intervals of a paired bootstrap over
recordings (2,000 resamples).

Usage: python scripts/birdnet_labels/summarize.py OUT.json [OUT.json ...]
"""
import json
import random
import sys
from collections import defaultdict

SPECIES_MIN = 0.35
LENGTHS = [('up to 1.5 s', 0, 1.5), ('1.5 to 3 s', 1.5, 3), ('over 3 s', 3, 1e9)]


def top1(boxes, k):
    return sum(b[k][0] == b['truth'] for b in boxes) / len(boxes)


def interval(boxes, k, rng):
    """Paired bootstrap over recordings of top-1(k) minus top-1(centred)."""
    by_file = defaultdict(list)
    for b in boxes:
        by_file[b['file']].append(b)
    files = sorted(by_file)
    diffs = sorted(top1(s, k) - top1(s, 'centred') for s in (
        [b for f in rng.choices(files, k=len(files)) for b in by_file[f]] for _ in range(2000)))
    return diffs[50], diffs[1949]


def summarize(name, boxes, schemes, rng):
    recordings = len({b['file'] for b in boxes})
    print(f'\n== {name}: {len(boxes)} boxes in {recordings} recordings')
    print(f'{"":10}{"top-1":>7}{"top-5":>7}{"named right":>13}{"named wrong":>13}{"unnamed":>9}   top-1 minus centred (95% interval)')
    for k in schemes:
        right = sum(b[k][0] == b['truth'] and b[k][1] >= SPECIES_MIN for b in boxes) / len(boxes)
        wrong = sum(b[k][0] != b['truth'] and b[k][1] >= SPECIES_MIN for b in boxes) / len(boxes)
        top5 = sum(b[k][3] <= 5 for b in boxes) / len(boxes)
        line = f'{k:10}{top1(boxes, k):7.3f}{top5:7.3f}{right:13.3f}{wrong:13.3f}{1 - right - wrong:9.3f}'
        if k != 'centred':
            lo, hi = interval(boxes, k, rng)
            line += f'   {top1(boxes, k) - top1(boxes, "centred"):+.3f} ({lo:+.3f} to {hi:+.3f})'
        print(line)
    for label, lo, hi in LENGTHS:
        part = [b for b in boxes if lo < b['t1'] - b['t0'] <= hi]
        if part:
            print(f'  boxes {label:12}{len(part):6}: ' + '  '.join(f'{k} {top1(part, k):.3f}' for k in schemes))
    changed = [b for b in boxes if b['grid15'][0] != b['centred'][0]]
    better = sum(b['grid15'][0] == b['truth'] for b in changed)
    worse = sum(b['centred'][0] == b['truth'] for b in changed)
    print(f'  grid15 names another species than centred for {len(changed)} boxes ({len(changed) / len(boxes):.1%}): '
          f'right only with grid15 {better}, right only centred {worse}, wrong both ways {len(changed) - better - worse}')


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rng = random.Random(1)
    everything = []
    for path in sys.argv[1:]:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        schemes = [k for k in ('centred', 'grid15', 'grid075', 'inside15', 'fit15', 'near15') if k in d['windows']]
        summarize(d['dataset'] + (' (boxes 1.5 to 3 s)' if d.get('mid_only') else ''), d['boxes'], schemes, rng)
        print('  windows scored: ' + ', '.join(f'{k} {v}' for k, v in d['windows'].items())
              + f' | skipped boxes: {sum(d["skipped"].values())}')
        everything += [dict(b, file=f'{d["dataset"]}/{b["file"]}') for b in d['boxes']]
    if len(sys.argv) > 2:
        summarize('all files', everything, schemes, rng)


if __name__ == '__main__':
    main()
