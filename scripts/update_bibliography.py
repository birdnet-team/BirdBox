# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""
Number the references of the documentation and write the Bibliography page.

Every source has one number across the whole site, given by its alphabetical
position (first author's surname, then year, then title). Each page keeps a
"## References" section at its end with the sources it cites, and
docs/bibliography.md lists all of them with the pages that cite them.

To add a source to a page:

1. Cite it in the text as ``[[0]](#ref-<key>)``. The number does not matter.
2. Add its entry under ``## References`` at the end of the page (create the
   section if needed)::

       <a id="ref-<key>"></a>
       **[0]** Surname, A., & Other, B. (2024). "Title." Venue.

   A source cited on several pages needs the same entry, with the same text,
   on each of them.
3. Run ``python scripts/update_bibliography.py`` from anywhere in the repository.

The script renumbers every citation and entry, sorts adjacent citations
("[[5]](#ref-a), [[2]](#ref-b)"), rebuilds each page's reference list in
number order, and rewrites docs/bibliography.md. Running it again changes
nothing.
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BIBLIOGRAPHY = DOCS / "bibliography.md"
ENTRY = re.compile(r'<a id="ref-([a-z0-9]+)"></a>\n\**\[(\d+)\]\**\s*(.+)')
CITE = re.compile(r"\[\[(\d+)\]\]\(#ref-([a-z0-9]+)\)")
GROUP = re.compile(r"\[\[\d+\]\]\(#ref-[a-z0-9]+\)(?:, \[\[\d+\]\]\(#ref-[a-z0-9]+\))+")


def nav_labels():
    """docs-relative path -> "Section › Page" from the nav in mkdocs.yml."""
    labels, stack = {}, []
    lines = (ROOT / "mkdocs.yml").read_text().split("\nnav:\n", 1)[1].splitlines()
    for line in lines:
        if not line.strip():
            continue
        if not line.startswith(" "):
            break
        depth = (len(line) - len(line.lstrip()) - 2) // 4  # 4 spaces per nav level
        item = line.strip()[2:]
        if item.startswith('"'):  # quoted label, may contain a colon ("In: Audio and Model")
            label, _, target = item[1:].partition('":')
        else:
            label, _, target = item.partition(":")
        target, label = target.strip(), label.strip()
        stack = stack[:depth] + [label]
        if target.endswith(".md"):
            labels[target] = " › ".join(stack)
        elif item.endswith(".md"):  # bare entry such as "- index.md"
            labels[item] = " › ".join(stack[:-1] + ["Home"])
    return labels


def main() -> int:
    pages = {md: md.read_text() for md in sorted(DOCS.rglob("*.md")) if md != BIBLIOGRAPHY}
    texts, cited_in = {}, {}
    for md, text in pages.items():
        for key, _, entry_text in ENTRY.findall(text):
            if texts.setdefault(key, entry_text.strip()) != entry_text.strip():
                print(f"Error: ref-{key} has different text in {md.relative_to(ROOT)}", file=sys.stderr)
                return 1
        for _, key in CITE.findall(text):
            cited_in.setdefault(key, [])
            if md not in cited_in[key]:
                cited_in[key].append(md)
    for key in set(cited_in) - set(texts):
        print(f"Error: ref-{key} is cited but has no entry", file=sys.stderr)
        return 1
    for key in set(texts) - set(cited_in):
        print(f"Error: ref-{key} has an entry but is never cited", file=sys.stderr)
        return 1

    def sort_key(key):
        entry_text = texts[key]
        author = entry_text.split(",", 1)[0]
        surname = " ".join(w for w in author.split() if not w.islower())  # "van Rijsbergen" -> "Rijsbergen"
        year = re.search(r"\((\d{4})\)", entry_text)
        return surname.lower(), year.group(1) if year else "", entry_text.lower()

    order = sorted(texts, key=sort_key)
    number = {key: i + 1 for i, key in enumerate(order)}

    def entry(key):
        return f'<a id="ref-{key}"></a>\n**[{number[key]}]** {texts[key]}'

    def sort_group(match):
        cites = CITE.findall(match.group(0))
        return ", ".join(f"[[{n}]](#ref-{k})" for n, k in sorted(cites, key=lambda c: int(c[0])))

    for md, text in pages.items():
        if not CITE.search(text):
            continue
        text = CITE.sub(lambda m: f"[[{number[m.group(2)]}]](#ref-{m.group(2)})", text)
        text = GROUP.sub(sort_group, text)
        head, found, _ = text.partition("## References")
        if not found:
            print(f"Error: {md.relative_to(ROOT)} cites sources but has no '## References'", file=sys.stderr)
            return 1
        keys = sorted({k for _, k in CITE.findall(head)}, key=lambda k: number[k])
        md.write_text(head + "## References\n\n" + "\n\n".join(entry(k) for k in keys) + "\n")

    labels = nav_labels()
    position = {path: i for i, path in enumerate(labels)}
    out = ["# Bibliography", "",
           "All sources cited in this documentation, sorted alphabetically by first author. "
           "The numbers are the same on every page, and each page also lists its sources at the bottom.", ""]
    for key in order:
        rel = sorted((str(md.relative_to(DOCS)) for md in cited_in[key]), key=lambda p: position.get(p, len(position)))
        links = ", ".join(f"[{labels.get(p, Path(p).stem)}]({os.path.relpath(DOCS / p, DOCS)})" for p in rel)
        out += [entry(key), "", f"*Cited in:* {links}", ""]
    BIBLIOGRAPHY.write_text("\n".join(out).rstrip() + "\n")
    print(f"{len(order)} sources numbered; wrote {BIBLIOGRAPHY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
