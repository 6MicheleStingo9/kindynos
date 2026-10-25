"""The data of the Space (data.json): the subdomains of the MIT Domain Taxonomy and the examples, risks of
the Kindynos test set that Laya labels right on all three axes.

usage (from the project root): python space/make_data.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIT = ROOT / "files" / "data" / "processed" / "full.jsonl"
RISKS = ROOT / "evaluation" / "area_testset" / "risks.jsonl"
# Two languages and every value of each axis but pre-deployment, which no test risk has
EXAMPLES = [
    "18598c69-1.2-1",  # AI, unintentional, post-deployment
    "23d095a1-4.2-1",  # human, intentional, post-deployment
    "23d095a1-5.1-2",  # human, unintentional, post-deployment
    "18598c69-6.5-1",  # human, unintentional, other
    "83fe243b-7.3-1",  # Italian: AI, unintentional, post-deployment
    "40722168-6.6-1",  # Italian: other, unintentional, other
]


def _read(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    names = {r["subdomain_id"]: r["subdomain_name"] for r in _read(MIT) if r["subdomain_id"]}
    # The 24 subdomains: the codes ending in .0 stand for a whole domain
    codes = sorted((c for c in names if not c.endswith(".0")), key=lambda c: [int(x) for x in c.split(".")])
    risks = {r["id"]: r for r in _read(RISKS)}
    data = {
        "subdomains": {c: names[c] for c in codes},
        "examples": [{"subdomain": risks[i]["subdomain"], "title": risks[i]["title"], "text": risks[i]["text"]} for i in EXAMPLES],
    }
    out = Path(__file__).parent / "data.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{out}: {len(codes)} subdomains, {len(EXAMPLES)} examples")


if __name__ == "__main__":
    main()
