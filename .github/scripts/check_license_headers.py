#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""
Check that every tracked source file starts with an SPDX license header.

Any license identifier is accepted, so third-party files keep their own license.
See CONTRIBUTING.md for the header format.

Usage:
    python .github/scripts/check_license_headers.py
"""

import subprocess
import sys

SOURCE_PATTERNS = ["*.py", "*.js", "*.java", "*.sh", "*.ps1", "*.bat", "*.css"]
# The header follows a shebang, `@echo off`, or `#Requires` line, so look a little further than line 2.
HEADER_LINES = 10


def main() -> int:
    files = subprocess.run(
        ["git", "ls-files", *SOURCE_PATTERNS], capture_output=True, text=True, check=True
    ).stdout.split()

    missing = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as handle:
            head = [handle.readline() for _ in range(HEADER_LINES)]
        if not any("SPDX-License-Identifier:" in line for line in head):
            missing.append(path)

    for path in missing:
        # GitHub Actions shows this as an annotation on the file.
        print(f"::error file={path},line=1::Missing SPDX license header (see CONTRIBUTING.md)")

    print(f"Checked {len(files)} source files, {len(missing)} without an SPDX license header.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
