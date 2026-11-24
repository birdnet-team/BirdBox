# Contributing to BirdBox

Thank you for helping improve BirdBox! Bug reports, fixes, and documentation
improvements are all welcome via issues and pull requests.

## Licensing of contributions

By submitting a contribution (code, documentation, or other files you wrote) to
this repository, you agree that it is licensed under the [MIT License](LICENSE),
the license of BirdBox's own code. You keep the copyright to your contribution.

BirdBox as a whole is subject to the AGPL-3.0 because it builds on Ultralytics
YOLO11 and ships YOLO11-trained weights. See [NOTICE.md](NOTICE.md) for details.

Please only contribute material you have the right to submit:

- **Your own work** is licensed under MIT, as described above.
- **Third-party code, data, or audio** must keep its original license. Say where
  it comes from and under which license in the pull request, and add an entry to
  [NOTICE.md](NOTICE.md). The license must allow redistribution, e.g. a
  permissive or Creative Commons license.
- **Model weights** trained with Ultralytics are AGPL-3.0.

## Tests

The unit tests in `tests/unit` check the merging of the windows' boxes, the
spectrogram features, the output formats, the label matching of the evaluation
tools and the species mappings. They load no model and run in a few seconds:

```bash
pip install -r requirements-test.txt
pytest
```

`requirements-test.txt` installs only the libraries the tested code imports, at
the versions of `requirements.txt`, so it works in a fresh environment as well
as in one made by `install.py`. The `tests` workflow runs the unit tests and
`mkdocs build --strict` on every pull request. A new function that does not
need a model should come with a test there.

The demo's checks in `tests/js` compare its JavaScript with its ONNX graphs in
`docs/models`. They take about 5 minutes on 4 cores:

```bash
cd tests/js
npm ci
npm test
```

`npm test` runs every check on the demo's default model and, on the other models,
the checks of their merge. After exporting a new model, `npm run test:all` runs
every check on every model.

The `demo tests` workflow runs them on pull requests that change
`docs/javascripts/demo`, `docs/models` or `tests/js`, split over two runners
(`npm test -- --shard=K/2`). It runs every check on every model when
`docs/models` changes. It can also be started by hand from the Actions tab.

Two scripts compare the models with the Python pipeline. They need the release's
model files and are run by hand when a model or an export changes:

- `tests/model_format_parity.py`: the `.onnx` and `.tflite` exports
  against the `.pt` model.
- `tests/onnx_graph_parity.py`: the demo's self-contained ONNX graphs against
  `detect_birds.py`.

Both run on `tests/test.wav` and the soundscapes of `tests/parity_audio`
(`python tests/parity_audio/download.py` fetches them).

## License headers

New source files start with an SPDX header (after the shebang line, if any):

```python
# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
```

Use the comment syntax of the file type (`//` for JavaScript and Java, `/* */`
for CSS, `REM` after `@echo off` for batch files). If you are not a member of
the BirdNET-Team, put your own name as the copyright holder.

The `license-check` workflow fails a pull request if a source file has no SPDX
header. To check locally before pushing, run
`python .github/scripts/check_license_headers.py`.

## Citing sources in the documentation

Sources are numbered alphabetically across the whole documentation, and the
Bibliography page (under Home) lists them all. To cite a paper on a page, write `[[0]](#ref-<key>)`
in the text and add its entry under `## References` at the end of the page:

```markdown
<a id="ref-<key>"></a>
**[0]** Surname, A., & Other, B. (2024). "Title." Venue.
```

Then run `python scripts/update_bibliography.py`. It numbers all citations,
sorts each page's references and rewrites `docs/bibliography.md`.
