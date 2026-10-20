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
