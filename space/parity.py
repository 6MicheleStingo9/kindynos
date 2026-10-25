"""The Space labels a risk as Kindynos does. On the 145 risks of the Kindynos test set, in one process and
on the same CPU:
- the text that Laya reads, and the questions, are those of Kindynos;
- the labels and the probabilities of the Space (the model published on Hugging Face) are those of the
  Kindynos causality step (agents/causality_analyzer/laya_classifier.py: the local checkpoint when there
  is one, else the same published model).
For reference, it also compares them with the Laya run in evaluation/results/, made on a GPU.

usage (from the project root, in an environment with the Space's requirements and gradio):
    python space/parity.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(ROOT))

import app  # noqa: E402  (loads the model from the Hub)
from agents.causality_analyzer import laya_classifier  # noqa: E402

RISKS = ROOT / "evaluation" / "area_testset" / "risks.jsonl"
RUN = ROOT / "evaluation" / "results" / "area_testset__laya-causal-v4__2026-10-12.json"
TOLERANCE = 1e-4  # same weights on the same CPU: the probabilities match


def _compare(space: list, other: list) -> tuple:
    """(labels in common, largest difference in probability)."""
    same, worst = 0, 0.0
    for a, b in zip(space, other):
        for axis in app.AXES:
            same += a[axis]["value"] == b[axis]["value"]
            worst = max(worst, *(abs(a[axis]["probabilities"][v] - p) for v, p in b[axis]["probabilities"].items()))
    return same, worst


def main() -> None:
    assert app.QUESTIONS == laya_classifier.QUESTIONS, "the questions of the published model differ from those of Kindynos"
    risks = [json.loads(line) for line in RISKS.read_text(encoding="utf-8").splitlines() if line.strip()]
    names = laya_classifier.subdomain_names()
    texts = sum(
        app.state(r["subdomain"], r["title"], r["text"])
        == laya_classifier.state(r["title"], r["text"], (r["subdomain"], names[r["subdomain"]]) if r["subdomain"] in names else None)
        for r in risks
    )
    space = [app.predict(r["subdomain"], r["title"], r["text"]) for r in risks]
    kindynos = laya_classifier.classify([(r["subdomain"], r["title"], r["text"]) for r in risks])
    run = json.loads(RUN.read_text(encoding="utf-8"))["predictions"]
    gpu = [{ax: {"value": run[r["id"]][ax]["choice"], "probabilities": run[r["id"]][ax]["probabilities"]} for ax in app.AXES} for r in risks]

    n = len(risks)
    same, worst = _compare(space, kindynos)
    print(f"same text as Kindynos: {texts}/{n}")
    print(f"against Kindynos ({laya_classifier.model_name()}, CPU): same label {same}/{3 * n}, largest difference in probability {worst:.6f}")
    same_gpu, worst_gpu = _compare(space, gpu)
    print(f"against the run of {RUN.name} (GPU): same label {same_gpu}/{3 * n}, largest difference in probability {worst_gpu:.4f}")
    if texts != n or same != 3 * n or worst > TOLERANCE:
        sys.exit("The Space does not label as Kindynos does")


if __name__ == "__main__":
    main()
