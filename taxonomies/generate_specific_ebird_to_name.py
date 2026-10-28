# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""
Generate the `ebird_to_name` block of a species mapping in `src/config.py`.

Use this when adding a new model (species mapping) to BirdBox. Every entry in
`SPECIES_MAPPING` in `src/config.py` needs two dictionaries:

- `id_to_ebird`: model class index -> eBird species code, e.g. `0: "hawama"`.
  This comes from the training setup and defines the model's label space.
- `ebird_to_name`: eBird species code -> "Genus species_Common Name", e.g.
  `"hawama": "Chlorodrepanis virens_Hawaii Amakihi"`. The detection outputs
  use it to show readable species names.

Writing `ebird_to_name` by hand for dozens of species is tedious and error-prone.
This script builds it from `id_to_ebird` instead:

1) Read the eBird codes from `SPECIES_MAPPING[<name>]["id_to_ebird"]`.
2) Look up each code in the master list `taxonomies/eBird_code_to_name.txt`.
3) Write a ready-to-paste `'ebird_to_name': {...}` block to
   `<MappingName>_ebird_to_name.txt` at the repository root (gitignored).

Requirement:
    `taxonomies/eBird_code_to_name.txt` is not part of the repository (it is
    gitignored). It is a JSON object that maps each eBird code to
    "Genus species_Common Name" and back. A copy lives in the BirdBox-Train
    repository.

Workflow:
    1) Add `id_to_ebird` for the new mapping to `src/config.py`.
    2) Run this script with the mapping name.
    3) Paste the generated block into that mapping in `src/config.py`, then
       delete the generated file.
    4) If the script reports "Missing codes", fix those codes in `id_to_ebird`
       or add them to the master list, and run it again.

Usage:
    python taxonomies/generate_specific_ebird_to_name.py Birds-With-Labels
"""

import json
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """Find repo root by locating src/config.py."""
    for candidate in (start, *start.parents):
        if (candidate / "src" / "config.py").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find repository root containing src/config.py "
        f"starting from {start}"
    )


script_dir = Path(__file__).resolve().parent
repo_root = find_repo_root(script_dir)
sys.path.append(str(repo_root / "src"))

from config import SPECIES_MAPPING  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print(
            f"Usage: {Path(__file__).name} <species_mapping_name>\n\n"
            "This generates a ready-to-paste `'ebird_to_name': {...}` snippet for the given\n"
            "mapping, based on the mapping's `id_to_ebird` in `src/config.py`.\n\n"
            f"Available mappings: {', '.join(sorted(SPECIES_MAPPING.keys()))}",
            file=sys.stderr,
        )
        sys.exit(1)

    mapping_name = sys.argv[1]
    if mapping_name not in SPECIES_MAPPING:
        print(
            f"Unknown species mapping '{mapping_name}'. "
            f"Available mappings: {', '.join(sorted(SPECIES_MAPPING.keys()))}",
            file=sys.stderr,
        )
        sys.exit(1)

    code_map_path = repo_root / "taxonomies" / "eBird_code_to_name.txt"
    with code_map_path.open() as f:
        raw = json.load(f)

    # raw is bidirectional; keep only entries where the key is the eBird code
    code_to_name = {
        code: full_name
        for code, full_name in raw.items()
        if "_" in full_name and not "_" in code  # crude but works: code -> "Genus species_Common"
    }

    id_to_ebird = SPECIES_MAPPING[mapping_name]["id_to_ebird"]
    codes = set(id_to_ebird.values())

    missing = [c for c in sorted(codes) if c not in code_to_name]
    if missing:
        print(
            f"Missing codes in eBird_code_to_name.txt for '{mapping_name}': {missing}",
            file=sys.stderr,
        )

    output_lines = []
    output_lines.append("        'ebird_to_name': {")
    for code in sorted(codes):
        if code in code_to_name:
            output_lines.append(f'            "{code}": "{code_to_name[code]}",')
    output_lines.append("        },")

    safe_name = mapping_name.replace(" ", "_").replace("/", "_")
    output_path = repo_root / f"{safe_name}_ebird_to_name.txt"
    with output_path.open("w") as f:
        f.write("\n".join(output_lines))

    print(f"Wrote '{mapping_name}' ebird_to_name mapping to {output_path}")


if __name__ == "__main__":
    main()