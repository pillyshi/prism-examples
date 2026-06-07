import subprocess
import base64
from pathlib import Path

from dvcgen import out, stage

from jmed_llm_examples.download import download_if_missing


stage(
    name="smdis_small_download"
)

download_if_missing(out("datasets/smdis.csv"), "datasets/smdis.csv")
