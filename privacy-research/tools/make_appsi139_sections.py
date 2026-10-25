"""Freeze policy sections from APPSI-139 for the rewrite task.

Usage::

    uv run python tools/make_appsi139_sections.py ANNOTATION_JSONL

ANNOTATION_JSONL is annotation.jsonl at the root of the AirLLM workspace: every APPSI-139 line in
document order, Markdown headings included, which the sentence CSV in data/ leaves out. It must match
its recorded SHA-256.

Each policy is a run of consecutive IDs, so a section never crosses from one policy into the next. A
section is the lines after a heading (a Markdown heading, or a line that is bold from end to end) up to
the next heading. A section can be drawn when it has 3 to 12 lines and 80 to 350 words, and none of its
lines holds an image, a table, a link, or a URL. Bold markers and Markdown escapes are removed from the
text, and sections whose start repeats another's (boilerplate such as "Links to other sites") are
dropped. The tool draws 20 sections, at most one per policy, across three length bands, and writes,
refusing to overwrite:

- data/rewrite/passages/appsi139-sections-20.json
"""

import argparse
import json
from pathlib import Path
import random
import re

from privacy_bench.core.files import digest_bytes, unique_keys, write_json
from privacy_bench.core.paths import DATA_DIR

SOURCE_SHA256 = "c38dbe2f470c241f835cc28ba9ba2c678b4f79041ea096ac9a850e994a1083e8"
OUTPUT = DATA_DIR / "rewrite" / "passages" / "appsi139-sections-20.json"
HEADING = re.compile(r"\s*(#{1,6}\s|\*\*[^*]+\*\*\s*$)")
UNUSABLE = re.compile(r"!\[|\]\(|\|.*\||https?://|www\.")
MARKDOWN_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!])")
LINES, WORDS = (3, 12), (80, 350)
# (lowest words, highest words, sections drawn)
BANDS = ((80, 159, 7), (160, 239, 7), (240, 350, 6))
SEED = 42


def read_lines(path: Path) -> list:
    raw = path.read_bytes()
    if digest_bytes(raw) != SOURCE_SHA256:
        raise ValueError(f"{path} does not match its recorded SHA-256 {SOURCE_SHA256}")
    lines = []
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line, object_pairs_hook=unique_keys)
        if not isinstance(item, dict) or set(item) != {"id", "text", "label"} or type(item["id"]) is not int:
            raise ValueError(f"{path}: line {number}: expected an object with an integer id, text, and label")
        lines.append(item)
    if len({line["id"] for line in lines}) != len(lines):
        raise ValueError(f"{path} repeats an ID")
    return lines


def policies(lines: list) -> list:
    runs = [[lines[0]]]
    for previous, line in zip(lines, lines[1:]):
        if line["id"] == previous["id"] + 1:
            runs[-1].append(line)
        else:
            runs.append([line])
    return runs


def clean(text: str) -> str:
    text = re.sub(r"\[\]\([^)]*\)", "", text)  # empty anchor links, which some headings start with
    text = MARKDOWN_ESCAPE.sub(r"\1", text.replace("**", "").replace("__", ""))
    return re.sub(r"\s+", " ", text).strip()


def sections(runs: list) -> list:
    found = []
    for policy, run in enumerate(runs):
        current = None
        for line in run:
            if HEADING.match(line["text"]):
                current = {"policy": policy, "policy_first_id": run[0]["id"],
                           "heading": clean(line["text"].lstrip("# ")).strip("_ "), "lines": []}
                found.append(current)
            elif current is not None:
                current["lines"].append(line)
    return found


def eligible(section: dict) -> bool:
    text = " ".join(clean(line["text"]) for line in section["lines"])
    words = len(text.split())
    return (LINES[0] <= len(section["lines"]) <= LINES[1] and WORDS[0] <= words <= WORDS[1]
            and not any(UNUSABLE.search(line["text"]) for line in section["lines"]))


def passage(section: dict) -> dict:
    text = " ".join(clean(line["text"]) for line in section["lines"])
    ids = [line["id"] for line in section["lines"]]
    return {"id": f"{ids[0]}-{ids[-1]}", "policy_first_id": section["policy_first_id"],
            "heading": section["heading"], "sentence_ids": ids, "words": len(text.split()), "text": text}


def draw(candidates: list) -> list:
    seen, unique = set(), []
    for item in candidates:
        start = re.sub(r"[^a-z0-9]+", " ", item["text"].lower())[:60]
        if start not in seen:
            seen.add(start)
            unique.append(item)
    rng = random.Random(SEED)
    rng.shuffle(unique)
    used, drawn = set(), []
    for low, high, count in BANDS:
        band = [item for item in unique if low <= item["words"] <= high and item["policy_first_id"] not in used]
        picked = []
        for item in band:
            if len(picked) < count and item["policy_first_id"] not in used:
                picked.append(item)
                used.add(item["policy_first_id"])
        if len(picked) < count:
            raise ValueError(f"Only {len(picked)} sections from distinct policies have {low}-{high} words")
        drawn.extend(sorted(picked, key=lambda item: item["words"]))
    return drawn


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotation", type=Path, help="annotation.jsonl in the AirLLM workspace")
    args = parser.parse_args(argv)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    runs = policies(read_lines(args.annotation))
    found = sections(runs)
    candidates = [passage(section) for section in found if eligible(section)]
    drawn = draw(candidates)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT, {
        "name": OUTPUT.stem,
        "source": {"file": "annotation.jsonl (AirLLM workspace)", "sha256": SOURCE_SHA256,
                   "corpus": "APPSI-139", "policies": len(runs)},
        "selection": {"section_lines": list(LINES), "section_words": list(WORDS),
                      "bands": [list(band) for band in BANDS], "one_per_policy": True, "seed": SEED,
                      "sections": len(found), "eligible": len(candidates)},
        "passages": drawn,
    })
    print(f"{len(runs)} policies, {len(found)} sections, {len(candidates)} eligible; wrote {len(drawn)} to {OUTPUT}")


if __name__ == "__main__":
    main()
