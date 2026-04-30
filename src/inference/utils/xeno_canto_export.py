"""
Helpers for exporting BirdBox detections as Xeno-canto Annota-JSON.
"""

import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional


EXPORT_ONLY_FIELDS = {
    "original_set_metadata",
    "annotation_xc_id",
    "set_import_date",
    "signal_noise_ratio",
}


@lru_cache(maxsize=1)
def _load_cornell_to_avilist_mapping() -> Dict:
    """Load the Cornell/eBird code to AviList species mapping."""
    mapping_path = (
        Path(__file__).resolve().parents[3]
        / "taxonomies"
        / "Cornell-to-AviList-mapping.json"
    )
    if not mapping_path.exists():
        return {}

    with mapping_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _scientific_name_for_detection(detection: Dict, species_mappings: Optional[Dict]) -> str:
    """Return the AviList scientific name for a Cornell/eBird-labeled detection."""
    species_code = str(detection.get("species", "")).strip()
    avilist_match = _load_cornell_to_avilist_mapping().get(species_code.lower(), {})
    if avilist_match.get("scientific_name"):
        return str(avilist_match["scientific_name"])

    if detection.get("scientific_name"):
        return str(detection["scientific_name"])

    ebird_to_name = (species_mappings or {}).get("ebird_to_name", {})
    full_name = ebird_to_name.get(species_code)

    if full_name and "_" in full_name:
        return full_name.split("_", 1)[0]

    return full_name or species_code


def _sound_file_for_detection(detection: Dict, audio_path: Optional[str]) -> str:
    """Return the source sound filename for single-file or multi-file detections."""
    if detection.get("filename"):
        return str(detection["filename"])
    if detection.get("file_path"):
        return Path(str(detection["file_path"])).name
    if audio_path:
        return Path(str(audio_path)).name
    return ""


def _extract_xc_nr(detection: Dict, sound_file: str) -> str:
    """Extract an XC recording number from metadata or filenames such as XC123456.wav."""
    if detection.get("xc_nr"):
        return str(detection["xc_nr"])

    match = re.search(r"(?:^|[^A-Za-z0-9])XC[_-]?(\d+)|^XC[_-]?(\d+)", sound_file, re.IGNORECASE)
    if match:
        return next(group for group in match.groups() if group)

    return ""


def _format_float(value, digits: int = 3):
    """Convert numeric values to stable JSON-friendly floats."""
    if value is None:
        return ""
    return round(float(value), digits)


def _drop_export_only_fields(obj):
    """Recursively remove fields marked as XC export-only."""
    if isinstance(obj, dict):
        return {
            key: _drop_export_only_fields(value)
            for key, value in obj.items()
            if key not in EXPORT_ONLY_FIELDS
        }
    if isinstance(obj, list):
        return [_drop_export_only_fields(item) for item in obj]
    return obj


def build_xeno_canto_json(
    detections: List[Dict],
    audio_path: Optional[str] = None,
    species_mappings: Optional[Dict] = None,
    set_name: str = "BirdBox detection results",
) -> Dict:
    """
    Build upload-focused Xeno-canto Annota-JSON FINAL Fv0 for BirdBox detections.

    This intentionally emits a lean payload:
    - required fields
    - relevant "should-have" fields we can infer reliably
    - no XC export-oriented structures such as original_set_metadata
    """
    annotations = []
    identification_date = date.today().isoformat()

    for index, detection in enumerate(detections, start=1):
        sound_file = _sound_file_for_detection(detection, audio_path)
        scientific_name = _scientific_name_for_detection(detection, species_mappings)
        confidence = detection.get("avg_confidence", detection.get("confidence"))

        remarks = f"Detected by BirdBox; eBird code: {detection.get('species', '')}"
        if confidence is not None:
            remarks += f"; confidence: {float(confidence):.3f}"

        annotations.append(
            {
                "annotation_source_id": f"birdbox-{index:06d}",
                "sound_file": sound_file,
                "xc_nr": _extract_xc_nr(detection, sound_file),
                "annotator": "BirdBox",
                "annotator_xc_id": "",
                "frequency_high": _format_float(detection.get("freq_high_hz")),
                "frequency_low": _format_float(detection.get("freq_low_hz")),
                "start_time": _format_float(detection.get("time_start")),
                "end_time": _format_float(detection.get("time_end")),
                "scientific_name": scientific_name,
                "sound_type": "call",
                "date_identified": identification_date,
                "annotation_remarks": remarks,
            }
        )

    taxon_coverage = ", ".join(
        sorted(
            {
                annotation["scientific_name"]
                for annotation in annotations
                if annotation["scientific_name"]
            }
        )
    )

    output = {
        "set_source": "BirdBox detection results",
        "set_uri": "",
        "set_name": set_name,
        "annotation_software_name_and_version": "BirdBox",
        "set_creator": "BirdBox",
        "set_creator_id": "",
        "set_owner": "",
        "set_license": "",
        "project_uri": "",
        "project_name": "BirdBox",
        "funding": "",
        "set_remarks": "Generated from BirdBox bird call detections.",
        "scope": [
            {
                "taxon_coverage": taxon_coverage or "birds",
                "completeness": "part",
            }
        ],
        "annotations": annotations,
    }
    return _drop_export_only_fields(output)
