# BirdBox licensing notice

Copyright (c) 2025-2026 BirdNET-Team

This file explains which license applies to which part of BirdBox. The license
texts themselves ([LICENSE](LICENSE), [LICENSE-AGPL-3.0](LICENSE-AGPL-3.0)) are
authoritative.

## Summary

- **BirdBox's own code is MIT.** Source code, scripts, documentation and images
  written by BirdNET-Team are licensed under the [MIT License](LICENSE), unless a
  file or the table below says otherwise.
- **The project as a whole is subject to AGPL-3.0.** BirdBox is built on
  [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (the
  `ultralytics` Python package), which is licensed under the
  [GNU Affero General Public License v3.0](LICENSE-AGPL-3.0). Ultralytics also
  treats models trained or fine-tuned with it as AGPL-3.0, so the BirdBox model
  weights are AGPL-3.0 in every format (`.pt`, `.onnx`, `.tflite`, `.engine`).
  Running BirdBox together with Ultralytics or the BirdBox weights, distributing
  it in that form, or offering it to users over a network (for example the
  Streamlit app) must therefore comply with AGPL-3.0. In particular, if you
  modify that combined work and let users interact with it over a network, you
  must offer them its corresponding source code (AGPL-3.0, section 13). The
  weights stay AGPL-3.0 even when they run without the `ultralytics` package,
  for example as an ONNX graph in ONNX Runtime.
- **Model weights.** The BirdBox model weights are Copyright (c) 2025-2026
  BirdNET-Team and licensed under the GNU Affero General Public License v3.0.
- **The MIT code can be reused on its own.** Files authored by BirdNET-Team can
  be copied and reused under MIT terms in projects that use neither Ultralytics
  nor the BirdBox weights.

Revisions of BirdBox published before this notice were distributed under
AGPL-3.0-only. Copies obtained under those terms may continue to be used under
them.

## License by path

| Path | License | Notes |
|---|---|---|
| Everything authored by BirdNET-Team not listed below (`src/`, `tests/`, `taxonomies/*.py`, `install.py`, `run_*.sh`/`.ps1`/`.bat`, `docs/`, configuration) | MIT | Source files carry `SPDX-License-Identifier: MIT` headers. |
| `mel_filterbank()` in `src/inference/onnx_export.py` | ISC | Adapted from `librosa.filters.mel` (librosa 0.11.0), marked with `SPDX-Snippet` tags. Rest of the file is MIT. Notice [below](#librosa). |
| Inferno colormap table in `docs/javascripts/onnx_spectrogram.js` | CC0-1.0 | 8-bit samples of the inferno colormap by Nathaniel J. Smith and Stéfan van der Walt, released under CC0 and shipped with matplotlib. |
| `docs/models/Just-Bird_fp16.onnx` | AGPL-3.0 | YOLO11 weights trained with Ultralytics, exported to ONNX. |
| Model weights distributed outside this repository ([TUC-Cloud](https://tuc.cloud/index.php/s/ET4KE4LdSaysSSL), default model of the Streamlit app) | AGPL-3.0 | YOLO11 weights trained with Ultralytics. |
| `tests/test.wav`, `docs/audio/sample_15s.wav` | CC-BY-4.0 | Soundscape `UHH_001_S01_20161121_150000` from the BirdNET Hawaii soundscape collection. Attribution [below](#audio-samples). |
| `taxonomies/Cornell-to-AviList-mapping.json` | Not MIT | Data derived from the [AviList](https://www.avilist.org/) v2025 checklist. Subject to AviList's terms of use. |
| Soundscapes fetched by `tests/parity_audio/download.py` | Not MIT | Not part of the repository. xeno-canto recordings under the Creative Commons licenses listed in `tests/parity_audio/recordings.csv`. |

## Third-party dependencies

BirdBox installs or loads third-party packages that are not part of this
repository and keep their own licenses: the Python packages in
`requirements.txt`, `model_format_requirements/` and `install.py` (notably
`ultralytics`, AGPL-3.0), and the scripts the documentation site loads from
CDNs (see `extra_javascript` in `mkdocs.yml`).

## Audio samples

`tests/test.wav` and `docs/audio/sample_15s.wav` are taken from:

> Amanda Navine, Stefan Kahl, Ann Tanimoto-Johnson, Holger Klinck, Patrick Hart
> (2022). *A collection of fully-annotated soundscape recordings from the Island
> of Hawai'i* (Version 1) [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.7078499

Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Changes: `tests/test.wav` is recording `UHH_001_S01_20161121_150000`, converted
from FLAC to 16-bit mono WAV at 32 kHz. The samples are identical to the original.
`docs/audio/sample_15s.wav` is the first 15 seconds of `tests/test.wav`.

## librosa

`mel_filterbank()` in `src/inference/onnx_export.py` is adapted from librosa,
which is distributed under the following license:

```text
ISC License

Copyright (c) 2013--2023, librosa development team.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
```
