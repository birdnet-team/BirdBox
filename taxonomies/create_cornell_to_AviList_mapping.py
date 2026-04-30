#!/usr/bin/env python3
"""
Build a species lookup JSON from an AviList Excel sheet.

With that one can convert species ID's from Cornell Lab of Ornithology to AviList species ID's.
The reason for this is that BirdNET related datasets like https://zenodo.org/records/7079124 or https://zenodo.org/records/7078499
are using Cornell Taxonomies (Clements/eBird) whereas Xeno-Canto relies on IOC Taxonomies.
With AviList it is possible to convert between the two taxonomies.
https://www.avilist.org/checklist/v2025/

This script reads the AviList workbook and applies the following pipeline:
1) Keep only rows where Taxon_rank == "species" (case-insensitive).
2) Remove rows where Species_code_Cornell_Lab is empty/null.
3) Keep only these columns:
   - Species_code_Cornell_Lab
   - Scientific_name
   - English_name_AviList
4) Normalize Species_code_Cornell_Lab to lowercase.
5) Deduplicate by Species_code_Cornell_Lab (first row wins).
6) For each row, if English_name_AviList is empty, use Scientific_name as fallback.
7) Export to JSON using Species_code_Cornell_Lab as the top-level key.

Output JSON format:
{
  "barswa": {
    "scientific_name": "Hirundo rustica",
    "english_name": "Barn Swallow"
  }
}

Usage:
    python taxonomies/create_cornell_to_AviList_mapping.py
    python taxonomies/create_cornell_to_AviList_mapping.py --input "taxonomies/AviList-v2025-11Jun-extended.xlsx" --output "taxonomies/Cornell-to-AviList-mapping.json"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REQUIRED_COLUMNS = [
    "Taxon_rank",
    "Species_code_Cornell_Lab",
    "Scientific_name",
    "English_name_AviList",
]
OUTPUT_COLUMNS = [
    "Species_code_Cornell_Lab",
    "Scientific_name",
    "English_name_AviList",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for input workbook and output JSON path."""
    parser = argparse.ArgumentParser(
        description=(
            'Create a JSON map keyed by "Species_code_Cornell_Lab" for rows where '
            'Taxon_rank equals "species".'
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "AviList-v2025-11Jun-extended.xlsx",
        help="Path to the input .xlsx file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "Cornell-to-AviList-mapping.json",
        help="Path to the output .json file.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    """Normalize output values: convert NaN to empty string and strip whitespace."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def warn_on_taxonomic_ambiguities(reduced_df: pd.DataFrame) -> None:
    """
    Print warnings for ambiguous mappings caused by splits/merges.

    Notes:
    - A "split" (for our purposes here) means the same Cornell species code appears on multiple
      rows that disagree on the AviList scientific/english name. The script will otherwise keep
      the first row it encounters.
    - A "merge" / synonym situation (again for our purposes here) means multiple Cornell codes
      map to the same AviList scientific name. This is not an error, but it can be surprising.
    """
    # --- Split detection: one Cornell code -> multiple distinct AviList taxa (by names) ---
    # (We treat name pairs as the "taxon identity" for warning purposes.)
    split_groups = (
        reduced_df.assign(
            _taxon_pair=list(
                zip(
                    reduced_df["Scientific_name"].astype(str),
                    reduced_df["English_name_AviList"].astype(str),
                )
            )
        )
        .groupby("Species_code_Cornell_Lab")["_taxon_pair"]
        .nunique(dropna=False)
    )
    split_codes = split_groups[split_groups > 1].index.tolist()
    if split_codes:
        print(
            f"WARNING: {len(split_codes)} Cornell code(s) map to multiple AviList taxa "
            f"(possible splits). The script will keep the first row found for each code.",
            file=sys.stderr,
        )
        # Show a small sample to keep output readable.
        for code in split_codes[:30]:
            rows = reduced_df[reduced_df["Species_code_Cornell_Lab"] == code][
                ["Scientific_name", "English_name_AviList"]
            ].drop_duplicates()
            variants = [
                f"{clean_text(r.Scientific_name)} | {clean_text(r.English_name_AviList)}"
                for r in rows.itertuples(index=False)
            ]
            print(f"  - {code}: {variants}", file=sys.stderr)
        if len(split_codes) > 30:
            print(
                f"  ... and {len(split_codes) - 30} more. (Increase the limit in the script if needed.)",
                file=sys.stderr,
            )

    # --- Merge/synonym detection: multiple Cornell codes -> same AviList scientific name ---
    # We use Scientific_name as the merge key because that's typically the stable identifier.
    sci_to_codes = reduced_df.groupby("Scientific_name")["Species_code_Cornell_Lab"].nunique()
    merged_scis = sci_to_codes[sci_to_codes > 1].index.tolist()
    if merged_scis:
        print(
            f"NOTE: {len(merged_scis)} AviList scientific name(s) are referenced by multiple "
            f"Cornell codes (possible merges/synonyms).",
            file=sys.stderr,
        )
        for sci in merged_scis[:30]:
            codes = (
                reduced_df[reduced_df["Scientific_name"] == sci]["Species_code_Cornell_Lab"]
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
            print(f"  - {clean_text(sci)}: {codes}", file=sys.stderr)
        if len(merged_scis) > 30:
            print(
                f"  ... and {len(merged_scis) - 30} more. (Increase the limit in the script if needed.)",
                file=sys.stderr,
            )


def main() -> None:
    """Run the full Excel -> filtered/deduplicated -> JSON conversion."""
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    last_error: PermissionError | None = None
    for _ in range(5):
        try:
            df = pd.read_excel(args.input)
            break
        except PermissionError as exc:
            last_error = exc
            time.sleep(1)
    else:
        raise PermissionError(
            f'Cannot read "{args.input}" because the file is locked or inaccessible.\n'
            "Close the workbook in Excel (and any preview/other app), wait for OneDrive "
            "sync to settle, then run the script again."
        ) from last_error

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    species_df = df[df["Taxon_rank"].astype(str).str.strip().str.lower() == "species"].copy()
    species_count = len(species_df)

    species_df = species_df[species_df["Species_code_Cornell_Lab"].notna()].copy()
    species_df["Species_code_Cornell_Lab"] = (
        species_df["Species_code_Cornell_Lab"].astype(str).str.strip().str.lower()
    )
    species_df = species_df[species_df["Species_code_Cornell_Lab"] != ""].copy()
    non_empty_code_count = len(species_df)

    reduced_df = species_df[OUTPUT_COLUMNS].copy()
    # Emit warnings BEFORE deduplication so we don't silently hide split/merge situations.
    warn_on_taxonomic_ambiguities(reduced_df)
    deduped_df = reduced_df.drop_duplicates(
        subset=["Species_code_Cornell_Lab"],
        keep="first",
    )

    species_map: dict[str, dict[str, str]] = {}
    for _, row in deduped_df.iterrows():
        code = row["Species_code_Cornell_Lab"]
        scientific_name = clean_text(row["Scientific_name"])
        english_name = clean_text(row["English_name_AviList"])
        if not english_name:
            english_name = scientific_name

        species_map[code] = {
            "scientific_name": scientific_name,
            "english_name": english_name,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(species_map, f, indent=2)

    print(
        f'Saved {len(deduped_df)} unique species codes to "{args.output}". '
        f"Source rows: {len(df)} total, {species_count} with Taxon_rank=species, "
        f"{non_empty_code_count} with non-empty Species_code_Cornell_Lab."
    )


if __name__ == "__main__":
    main()
