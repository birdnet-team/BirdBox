"""Output filenames and results-directory handling for detection exports."""

from pathlib import Path
from typing import Dict, Iterable, Optional

RAW_DETECTIONS_JSON = 'raw_detections.json'
RESULTS_ROOT = 'results'
ACTIVE_RUN_MARKER = '.active_run'

OUTPUT_FORMAT_FILENAMES: Dict[str, str] = {
    'json-with-algorithm-metadata': 'with_algorithm_metadata.json',
    'simplified-csv': 'simplified.csv',
    'xeno-canto-annota-json': 'xeno-canto-annota.json',
    'raven-selection-table': 'raven_selection_table.txt',
}

DEFAULT_RESULTS_DIR = RESULTS_ROOT

_PIPELINE_ARTIFACTS = frozenset(
    {RAW_DETECTIONS_JSON, *OUTPUT_FORMAT_FILENAMES.values()}
)


def is_default_results_path(path: Optional[str]) -> bool:
    """True when the path is the canonical default results directory."""
    if path is None:
        return True
    normalized = Path(path).as_posix().rstrip('/')
    return normalized == RESULTS_ROOT


def _dir_has_pipeline_outputs(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    for entry in directory.iterdir():
        if entry.name.startswith('run_') and entry.is_dir():
            continue
        if entry.name == ACTIVE_RUN_MARKER:
            continue
        if entry.is_file() and entry.name in _PIPELINE_ARTIFACTS:
            return True
    return False


def _next_run_directory(results_root: Path) -> Path:
    run_index = 2
    while True:
        candidate = results_root / f'run_{run_index}'
        if not candidate.exists() or not _dir_has_pipeline_outputs(candidate):
            return candidate
        run_index += 1


def _write_active_run_marker(results_root: Path, run_dir: Path) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / ACTIVE_RUN_MARKER).write_text(run_dir.as_posix() + '\n', encoding='utf-8')


def prepare_results_directory(explicit: Optional[str] = None) -> str:
    """
    Choose the output directory for a new inference run.

    Uses ``results/`` by default. If that folder already holds pipeline outputs,
    creates ``results/run_2``, ``results/run_3``, … and records the active path
    in ``results/.active_run`` so later evaluation steps stay aligned.

    Custom paths (e.g. ``results/Hawaii``) are returned unchanged.
    """
    if explicit is not None and not is_default_results_path(explicit):
        return explicit

    results_root = Path(RESULTS_ROOT)
    results_root.mkdir(parents=True, exist_ok=True)

    if _dir_has_pipeline_outputs(results_root):
        run_dir = _next_run_directory(results_root)
        print(f"Note: {RESULTS_ROOT}/ already contains outputs; writing to {run_dir.as_posix()}/")
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_active_run_marker(results_root, run_dir)
        return run_dir.as_posix()

    _write_active_run_marker(results_root, results_root)
    return RESULTS_ROOT


def resolve_results_directory(explicit: Optional[str] = None) -> str:
    """
    Active results directory for evaluation steps (follows the latest detect run).

    Reads ``results/.active_run`` when using the default ``results`` path.
    """
    if explicit is not None and not is_default_results_path(explicit):
        return explicit

    marker = Path(RESULTS_ROOT) / ACTIVE_RUN_MARKER
    if marker.is_file():
        return marker.read_text(encoding='utf-8').strip()
    return RESULTS_ROOT


def algorithm_metadata_json_path(output_dir: str, *, raw: bool) -> Path:
    """Path for merged or raw (--no-merge) algorithm-metadata JSON."""
    name = RAW_DETECTIONS_JSON if raw else OUTPUT_FORMAT_FILENAMES['json-with-algorithm-metadata']
    return Path(output_dir) / name


def format_output_path(output_dir: str, format_key: str) -> Path:
    """Return the full path for a format file inside the output directory."""
    if format_key not in OUTPUT_FORMAT_FILENAMES:
        raise ValueError(f"Unknown output format: {format_key}")
    return Path(output_dir) / OUTPUT_FORMAT_FILENAMES[format_key]


def resolve_raw_detections_json(path: str) -> Path:
    """
    Resolve a CLI path to the raw detections JSON from detect_birds --no-merge.

    Accepts either the results directory (e.g. ``results`` or ``results/run_2``)
    or an explicit file path.
    """
    p = Path(path)

    if p.is_dir():
        return p / RAW_DETECTIONS_JSON
    if p.is_file():
        return p
    if p.name == RAW_DETECTIONS_JSON:
        return p
    if p.suffix == '':
        return p / RAW_DETECTIONS_JSON
    return p


def resolve_format_path(path: str, format_key: str) -> Path:
    """
    Resolve a CLI path to a concrete output file.

    Accepts either an output directory or an explicit file path.
    For raw detections JSON, use ``resolve_raw_detections_json()`` instead.
    """
    p = Path(path)
    expected_name = OUTPUT_FORMAT_FILENAMES[format_key]

    if p.is_dir():
        return format_output_path(p, format_key)
    if p.is_file():
        return p
    if p.name == expected_name:
        return p
    if p.suffix == '':
        return format_output_path(p, format_key)
    return p


def ensure_output_directory(output_path: str) -> bool:
    """Ensure the output directory exists, creating it if needed."""
    if not output_path:
        return True

    output_dir = Path(output_path)
    if output_dir.exists():
        return True

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}")
        return True
    except Exception as e:
        print(f"✗ Error creating directory: {e}")
        return False
