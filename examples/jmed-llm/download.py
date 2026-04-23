import subprocess
import base64
from pathlib import Path


REPO = "sociocom/JMED-LLM"


def download_if_missing(local_path: str, gh_path: str) -> None:
    if Path(local_path).exists():
        print(local_path)
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


if __name__ == "__main__":
    download_if_missing("datasets/SMDIS_SMALL.csv", "datasets/smdis.csv")
    download_if_missing("datasets/SMDIS.csv", "datasets/all/SMDIS.csv")
