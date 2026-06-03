
## Audio Inputs

`detect_birds.py` accepts:

- single audio file path
- or a directory (searched recursively)

Supported extensions:

- `.wav`
- `.flac`
- `.ogg`
- `.mp3`

!!! warning "Lossy Audio Formats"
    MP3 and OGG files are supported but can reduce recall for faint or high-frequency calls, because the model was trained on lossless WAV files. Use WAV or FLAC whenever possible.

- Stereo content is collapsed to mono automatically.

## Model Files

Supported model artifacts are passed to `--model`:

- `.pt`
- `.onnx`
- `.engine`
- other YOLO-compatible formats

!!! danger "Mapping Must Match Model"
    The selected `--species-mapping` must match the label space the model was trained with. A mismatch produces silently incorrect species labels in the output without raising an error.

## Species Mapping Source

Mappings are loaded from `src/config.py` via:

- `get_species_mapping_for_model(model_path)` (app-side model-name inference)
- `get_species_mapping(species_mapping_name)` (explicit mapping retrieval)

The mapping object provides:

- class id -> eBird code
- eBird code -> display name
- class id -> color
- fixed clip/image defaults used in inference