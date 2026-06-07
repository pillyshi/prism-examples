import json
import csv
from pathlib import Path


text2tags = {}
with open("datasets/smdis.csv") as f:
    for row in csv.DictReader(f):
        if row["answer"] != "A":
            continue
        post = row["question"].split("「")[1].split("」")[0]
        tags = text2tags.setdefault(post, list())
        tags.append(row["tag"])

PATH_ROWS = Path("artifacts/smdis_small/preprocess/rows.json").resolve()
PATH_ROWS.parent.mkdir(parents=True, exist_ok=True)
with open(PATH_ROWS, "w") as f:
    json.dump([{
        "text": text,
        "tags": tags
    } for text, tags in text2tags.items()], f)
