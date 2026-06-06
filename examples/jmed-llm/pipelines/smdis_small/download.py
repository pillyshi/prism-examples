import subprocess
import base64
from pathlib import Path

from dvcgen import out, stage

from jmed_llm_examples.download import download_if_missing


stage(
    name="smdis_small_download"
)

PATH_DATASET = out("datasets/smdis.csv")

download_if_missing(PATH_DATASET, "datasets/smdis.csv")
