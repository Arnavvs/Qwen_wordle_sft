"""make_demo.py — inline the data blob into the playable demo.

`template.html` carries a single `__DEMO_DATA__` token; this replaces it with
`demo_data.json` so the result is one self-contained file with no fetches. The
Artifact CSP blocks external requests anyway, and a single file is what makes
the page shareable.

    python phase12_demo/build_demo_data.py    # first, if the data changed
    python phase12_demo/make_demo.py
"""
import _paths  # noqa: F401

import json
import os

TEMPLATE = "phase12_demo/template.html"
DATA = "phase12_demo/demo_data.json"
OUT = "phase12_demo/index.html"
TOKEN = "__DEMO_DATA__"


def main():
    with open(TEMPLATE, encoding="utf-8") as fh:
        html = fh.read()
    with open(DATA, encoding="utf-8") as fh:
        blob = fh.read()

    if TOKEN not in html:
        raise SystemExit(f"{TEMPLATE} has no {TOKEN} placeholder")
    # `</script>` inside a JSON string would close the host <script> element.
    # None of this data can contain it, but escaping is free and the failure
    # mode is a silently broken page.
    blob = blob.replace("</", "<\\/")
    html = html.replace(TOKEN, blob, 1)

    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html)

    d = json.loads(open(DATA, encoding="utf-8").read())
    print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")
    print(f"  {d['n_games']} games, {len(d['solvers'])} solvers, "
          f"{d['n_legal']} legal words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
