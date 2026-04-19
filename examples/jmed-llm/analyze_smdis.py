"""Prism axis discovery and feature analysis for SMDIS dataset."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import subprocess
from pathlib import Path

from prism import Prism


REPO = "sociocom/JMED-LLM"
DATASET_SMALL_LOCAL = "datasets/smdis.csv"
DATASET_SMALL_GH = "datasets/smdis.csv"
DATASET_ALL_LOCAL = "datasets/SMDIS.csv"
DATASET_ALL_GH = "datasets/all/SMDIS.csv"
RESULTS_DIR = Path("results")


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


def load_texts(path: str) -> list[str]:
    seen: set[str] = set()
    texts: list[str] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            post = row["question"].split("「")[1].split("」")[0]
            if post not in seen:
                seen.add(post)
                texts.append(post)
    return texts


def run(prism: Prism, texts: list[str], n_axes: int, n_features: int) -> dict:
    print(f"Discovering {n_axes} axes ...")
    axes = prism.discover_axes(texts, n=n_axes, language="Japanese")
    print(f"Discovered {len(axes)} axes:")
    for ax in axes:
        print(f"  - {ax.name}")

    print("Labeling texts per axis (NLI) ...")
    axes_labels = prism.label_axes(texts, axes)

    print("Generating features ...")
    features_by_axis = prism.generate_features(texts, axes, n_features=n_features, axes_labels=axes_labels, language="Japanese")

    print("Scoring with NLI ...")
    matrices = prism.score(texts, features_by_axis, axes_labels=axes_labels)

    print("Selecting features ...")
    results, _ = prism.select(matrices)

    output_axes = []
    for axis in axes:
        result = results[axis]
        output_axes.append({
            "name": axis.name,
            "question": axis.question,
            "hypothesis": axis.hypothesis,
            "cv_score": result.cv_score,
            "cv_scoring": result.cv_scoring,
            "selected_features": [
                {"name": f.name, "coef": c}
                for f, c in zip(result.selected_features, result.coef)
            ],
        })

    return {"n_texts": len(texts), "axes": output_axes}


def main() -> None:
    parser = argparse.ArgumentParser(description="SMDIS axis discovery and feature analysis with Prism")
    parser.add_argument("--all", action="store_true", help="Use full dataset (datasets/all/SMDIS.csv)")
    parser.add_argument("--n-axes", type=int, default=10, help="Number of axes to discover")
    parser.add_argument("--n-features", type=int, default=10, help="Number of features per axis")
    args = parser.parse_args()

    if args.all:
        local_path, gh_path = DATASET_ALL_LOCAL, DATASET_ALL_GH
    else:
        local_path, gh_path = DATASET_SMALL_LOCAL, DATASET_SMALL_GH

    download_if_missing(local_path, gh_path)

    texts = load_texts(local_path)
    print(f"Loaded {len(texts)} unique texts")

    prism = Prism(
        llm=os.environ.get("PRISM_LLM", "gpt-4o-mini"),
        nli_model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        mode="classification",
    )

    out_dir = RESULTS_DIR / ("all" if args.all else "small")
    out_dir.mkdir(parents=True, exist_ok=True)

    output = run(prism, texts, args.n_axes, args.n_features)

    out_path = out_dir / "discover.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    print(f"\n=== Results ===")
    for ax in output["axes"]:
        print(f"[{ax['name']}] cv={ax['cv_score']:.3f}")
        for f in ax["selected_features"]:
            print(f"  [{f['coef']:+.3f}] {f['name']}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
