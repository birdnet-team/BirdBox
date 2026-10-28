# BirdBox licensing notice

Copyright (c) 2025-2026 BirdNET-Team

This file explains which license applies to which part of BirdBox. The license
texts themselves ([LICENSE](LICENSE), [LICENSE-AGPL-3.0](LICENSE-AGPL-3.0)) are
authoritative.

## Summary

- **BirdBox's own code is MIT.** Source code, scripts, documentation and images
  written by BirdNET-Team and contributors are licensed under the
  [MIT License](LICENSE), unless a file or the table below says otherwise.
- **The project as a whole is subject to AGPL-3.0.** BirdBox is built on
  [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (the
  `ultralytics` Python package), which is licensed under the
  [GNU Affero General Public License v3.0](LICENSE-AGPL-3.0). Ultralytics also
  treats models trained or fine-tuned with it as AGPL-3.0, so the BirdBox model
  weights are AGPL-3.0 in every format (`.pt`, `.onnx`, `.tflite`, `.engine`).
  Running BirdBox together with Ultralytics or the BirdBox weights, distributing
  it in that form, or offering it to users over a network must therefore
  comply with AGPL-3.0. In particular, if you
  modify that combined work and let users interact with it over a network, you
  must offer them its corresponding source code (AGPL-3.0, section 13). The
  weights stay AGPL-3.0 even when they run without the `ultralytics` package,
  for example as an ONNX graph in ONNX Runtime.
- **Model weights.** The BirdBox model weights are Copyright (c) 2025-2026
  BirdNET-Team and licensed under the GNU Affero General Public License v3.0.
- **The MIT code can be reused on its own.** Files authored by BirdNET-Team and
  contributors can be copied and reused under MIT terms in projects that use
  neither Ultralytics nor the BirdBox weights.

Revisions of BirdBox published before this notice were distributed under
AGPL-3.0-only. Copies obtained under those terms may continue to be used under
them.

## License by path

| Path | License | Notes |
|---|---|---|
| Everything authored by BirdNET-Team and contributors not listed below (`src/`, `tests/`, `taxonomies/*.py`, `install.py`, `run_*.sh`/`.ps1`/`.bat`, `docs/`, configuration) | MIT | Source files carry `SPDX-License-Identifier: MIT` headers. |
| `mel_filterbank()` in `src/inference/onnx_export.py` | ISC | Adapted from `librosa.filters.mel` (librosa 0.11.0), marked with `SPDX-Snippet` tags. Rest of the file is MIT. Notice [below](#librosa). |
| Inferno, magma and viridis colormap tables in `docs/javascripts/demo/colormaps.js` | CC0-1.0 | 8-bit samples of the matplotlib colormaps by Nathaniel J. Smith and Stéfan van der Walt, released under CC0. |
| `docs/models/Just-Bird_fp16.onnx`, `docs/models/Birds-With-Labels_fp16.onnx` | AGPL-3.0 | YOLO11 weights trained with Ultralytics, exported to ONNX. |
| Model weights distributed outside this repository ([TUC-Cloud](https://tuc.cloud/index.php/s/ET4KE4LdSaysSSL)) | AGPL-3.0 | YOLO11 weights trained with Ultralytics. |
| `tests/test.wav` and the spectrogram image `docs/img/example_viz.png` | CC-BY-4.0 | Soundscape `UHH_001_S01_20161121_150000` from the BirdNET Hawaii soundscape collection, and a spectrogram of it. Attribution [below](#audio-samples). |
| `docs/audio/BirdNET_soundscape_32kHz.flac`, the screenshots `docs/img/demo_screenshot_light.png`, `docs/img/demo_screenshot_dark.png` and the figures `docs/img/how-it-works/window_fusion.png`, `docs/img/how-it-works/window_fusion_dark.png` | MIT | Example soundscape `birdnet_analyzer/example/soundscape.wav` of [BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer) (Copyright (c) 2024 birdnet-team, MIT License), resampled from 48 kHz to 32 kHz. The same file is the sample of the BirdNET website demo. The screenshots show it in the Demo tab; the figures show a spectrogram of it with BirdBox detections. |
| `docs/img/metrics/just_bird_label_example.png`, `docs/img/metrics/just_bird_label_example_dark.png` | CC-BY-4.0 | Spectrogram of a training clip from the Northeastern United States soundscape collection, with annotation boxes. Attribution [below](#training-clips-in-the-documentation-figures). |
| `docs/img/metrics/just_bird_refine_cases.png`, `docs/img/metrics/just_bird_refine_cases_dark.png` | CC-BY-NC-SA-4.0 | Spectrograms of six training clips with annotation boxes. Three are adapted from xeno-canto recordings under CC BY-NC-SA 4.0, so the figure as a whole is shared under that license; the other three come from CC BY 4.0 soundscape collections. Attribution [below](#training-clips-in-the-documentation-figures). |
| `taxonomies/Cornell-to-AviList-mapping.json` | CC-BY-4.0 | Data derived from the AviList v2025 checklist. Attribution [below](#avilist). |
| Soundscapes fetched by `tests/parity_audio/download.py` | Not MIT | Not part of the repository. xeno-canto recordings under the Creative Commons licenses listed in `tests/parity_audio/recordings.csv`. |

## Third-party dependencies

BirdBox installs or loads third-party packages that are not part of this
repository and keep their own licenses: the Python packages in
`requirements.txt`, `model_format_requirements/` and `install.py` (notably
`ultralytics`, AGPL-3.0), and the scripts the documentation site loads from
CDNs (see `extra_javascript` in `mkdocs.yml`, and ONNX Runtime Web in
`docs/javascripts/demo/ort-worker.js` for the Demo tab).

## Audio samples

`tests/test.wav` and the spectrogram images listed above are taken from or made
from:

> Amanda Navine, Stefan Kahl, Ann Tanimoto-Johnson, Holger Klinck, Patrick Hart
> (2022). *A collection of fully-annotated soundscape recordings from the Island
> of Hawai'i* (Version 1) [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.7078499

Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Changes: `tests/test.wav` is recording `UHH_001_S01_20161121_150000`, converted
from FLAC to 16-bit mono WAV at 32 kHz. The samples are identical to the original.
`docs/img/example_viz.png` shows a spectrogram of `tests/test.wav` with BirdBox
detections drawn on top.

## Training clips in the documentation figures

`docs/img/metrics/just_bird_label_example*.png` and
`docs/img/metrics/just_bird_refine_cases*.png` show 3 s training clips of the
Just-Bird model. They are taken from:

> Stefan Kahl, Russell Charif, Holger Klinck (2022). *A collection of
> fully-annotated soundscape recordings from the Northeastern United States*
> (Version 2) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7079380
> (clip `SSW_101_9_12.0s_to_15.0s`, in both figures)

> Stefan Kahl, Connor M. Wood, Philip Chaon, M. Zachariah Peery, Holger Klinck
> (2022). *A collection of fully-annotated soundscape recordings from the Western
> United States* (Version 1) [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.7050014 (clip `SNE_022_19_1.5s_to_4.5s`)

> W. Alexander Hopping, Stefan Kahl, Holger Klinck (2022). *A collection of
> fully-annotated soundscape recordings from the Southwestern Amazon Basin*
> (Version 1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7079124
> (clip `PER_018_54_36.0s_to_39.0s`)

These three collections are licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

The other three clips in `just_bird_refine_cases*.png` come from xeno-canto
soundscapes of the Bird Sounds Global, Lifeplan and SystemShift projects:

- [XC1123613](https://xeno-canto.org/1123613), recorded by SystemShift (Dry Chaco, Argentina), clip 3–6 s
- [XC1125444](https://xeno-canto.org/1125444), recorded by BSG – site Municipio Asunción Ixtaltepec (Nizanda, Mexico), clip 9–12 s
- [XC1132774](https://xeno-canto.org/1132774), recorded by BSG – Lifeplan Nordic (Skommarhyttan, Sweden), clip 43.5–46.5 s

These recordings are licensed under
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Changes: each clip is a 3 s excerpt, resampled to 32 kHz where needed and
converted to a PCEN mel spectrogram. The human annotation boxes and the boxes of
the BirdBox label refinement are drawn on top.

## AviList

`taxonomies/Cornell-to-AviList-mapping.json` is derived from:

> AviList Core Team. 2025. *AviList: The Global Avian Checklist*, v2025.
> https://doi.org/10.2173/avilist.v2025

AviList: The Global Avian Checklist © 2025 by AviList Core Team is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Changes: generated from `AviList-v2025-11Jun-extended.xlsx` by
`taxonomies/create_cornell_to_AviList_mapping.py`. Only species-rank rows with a
Cornell Lab species code are kept, reduced to the columns
`Species_code_Cornell_Lab`, `Scientific_name` and `English_name_AviList`. Species
codes are lowercased and deduplicated (first row wins), and a missing English
name is replaced by the scientific name.

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
