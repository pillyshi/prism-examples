"""Prism feature analysis for SMDIS dataset (all 8 symptom tags)."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import subprocess
from pathlib import Path

from prism import Prism
from prism.models import Axis, AxisLabels


REPO = "sociocom/JMED-LLM"
DATASET_SMALL = "datasets/smdis.csv"
DATASET_ALL = "datasets/all/SMDIS.csv"
RESULTS_DIR = Path("results")

AXES: dict[str, Axis] = {
    "influenza": Axis(
        name="influenza",
        question="Does this SNS post indicate the person or someone nearby had influenza within a day?",
        hypothesis="This SNS post indicates the person had influenza.",
    ),
    "diarrhea": Axis(
        name="diarrhea",
        question="Does this SNS post indicate the person or someone nearby had diarrhea within a day?",
        hypothesis="This SNS post indicates the person had diarrhea.",
    ),
    "hayfever": Axis(
        name="hayfever",
        question="Does this SNS post indicate the person or someone nearby had hay fever symptoms within a day?",
        hypothesis="This SNS post indicates the person had hay fever.",
    ),
    "cough": Axis(
        name="cough",
        question="Does this SNS post indicate the person or someone nearby had a cough or phlegm within a day?",
        hypothesis="This SNS post indicates the person had a cough.",
    ),
    "headache": Axis(
        name="headache",
        question="Does this SNS post indicate the person or someone nearby had a headache within a day?",
        hypothesis="This SNS post indicates the person had a headache.",
    ),
    "fever": Axis(
        name="fever",
        question="Does this SNS post indicate the person or someone nearby had a fever within a day?",
        hypothesis="This SNS post indicates the person had a fever.",
    ),
    "runnynose": Axis(
        name="runnynose",
        question="Does this SNS post indicate the person or someone nearby had a runny nose or nasal congestion within a day?",
        hypothesis="This SNS post indicates the person had a runny nose or nasal congestion.",
    ),
    "cold": Axis(
        name="cold",
        question="Does this SNS post indicate the person or someone nearby had a cold within a day?",
        hypothesis="This SNS post indicates the person had a cold.",
    ),
}


def download_if_missing(local_path: str, gh_path: str) -> None:
    if Path(local_path).exists():
        return
    print(f"Downloading {gh_path} ...")
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    sha = subprocess.check_output(
        ["gh", "api", f"repos/{REPO}/contents/{gh_path}", "--jq", ".sha"],
        text=True,
    ).strip()
    content_b64 = subprocess.check_output(
        ["gh", "api", f"repos/{REPO}/git/blobs/{sha}", "--jq", ".content"],
        text=True,
    )
    data = base64.b64decode(content_b64)
    Path(local_path).write_bytes(data)
    print(f"Saved to {local_path}")


def load_smdis(path: str, tag: str) -> tuple[list[str], list[float]]:
    texts, labels = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["tag"] != tag:
                continue
            post = row["question"].split("「")[1].split("」")[0]
            texts.append(post)
            labels.append(1.0 if row["answer"] == "A" else -1.0)
    return texts, labels


def run_tag(prism: Prism, tag: str, texts: list[str], labels: list[float]) -> dict:
    axis = AXES[tag]
    axis_labels = AxisLabels(axis=axis, labels=labels)

    print(f"  Generating features for {tag} ...")
    features_by_axis = prism.generate_features(texts, [axis], axes_labels=[axis_labels])

    print(f"  Scoring {len(texts)} texts with NLI ...")
    matrices = prism.score(texts, features_by_axis, axes_labels=[axis_labels])

    results, _ = prism.select(matrices)
    result = results[axis]
    features = features_by_axis[axis]

    return {
        "tag": tag,
        "n_texts": len(texts),
        "n_positive": labels.count(1.0),
        "n_negative": labels.count(-1.0),
        "cv_score": result.cv_score,
        "cv_scoring": result.cv_scoring,
        "features": [
            {"name": f.name, "question": f.question, "hypothesis": f.hypothesis}
            for f in features
        ],
        "selected": [
            {"name": f.name, "coef": c}
            for f, c in zip(result.selected_features, result.coef)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SMDIS feature analysis with Prism")
    parser.add_argument("--all", action="store_true", help="Use full dataset (datasets/all/SMDIS.csv)")
    args = parser.parse_args()

    if args.all:
        local_path = DATASET_ALL.replace("datasets/all/", "datasets/")
        gh_path = DATASET_ALL
    else:
        local_path = DATASET_SMALL
        gh_path = DATASET_SMALL

    download_if_missing(local_path, gh_path)

    prism = Prism(
        llm=os.environ.get("PRISM_LLM", "gpt-4o-mini"),
        nli_model="cross-encoder/nli-deberta-v3-large",
        mode="classification",
    )

    RESULTS_DIR.mkdir(exist_ok=True)

    for tag in AXES:
        texts, labels = load_smdis(local_path, tag)
        print(f"\n[{tag}] {len(texts)} texts (pos={labels.count(1)}, neg={labels.count(-1)})")

        output = run_tag(prism, tag, texts, labels)

        out_path = RESULTS_DIR / f"{tag}.json"
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))

        print(f"  Selected features:")
        for s in output["selected"]:
            print(f"    [{s['coef']:+.3f}] {s['name']}")
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
