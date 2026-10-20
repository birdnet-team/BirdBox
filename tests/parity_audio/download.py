#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""
Download the soundscapes used by tests/model_format_parity.py.

The recordings are listed in recordings.csv next to this script: 12 xeno-canto
soundscapes from Europe and Africa (1 to 5 minutes each, quality A), chosen to
cover different habitats, including one with insects only. They are not stored
in the repository because of their size and their licences (CC BY-NC, some
-ND); this script fetches them from xeno-canto, which needs no API key.

    python tests/parity_audio/download.py           # skips files already present
    python tests/parity_audio/download.py --force   # download everything again
"""

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORDINGS = HERE / "recordings.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Download files that already exist again.")
    args = parser.parse_args()

    with open(RECORDINGS, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    failed = 0
    for row in rows:
        target = HERE / row["file"]
        if target.exists() and not args.force:
            print(f"present     {target.name}")
            continue
        url = f"https://xeno-canto.org/{row['xc_id']}/download"
        try:
            partial = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(url, partial)
            partial.rename(target)
            print(f"downloaded  {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
        except OSError as exc:
            failed += 1
            print(f"FAILED      {target.name}: {exc}", file=sys.stderr)

    print(f"\nRecordings by {', '.join(sorted({r['recordist'] for r in rows}))} via xeno-canto.org "
          "(licences in recordings.csv).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
