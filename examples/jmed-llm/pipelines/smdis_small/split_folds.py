import os
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


PATH_FOLDS = "artifacts/smdis_small/split_folds/folds"

with open("artifacts/smdis_small/preprocess/rows.json") as f:
    rows = json.load(f)

texts = []
labels = []
for row in rows:
    texts.append(row["text"])
    labels.append(row["tags"])
texts = np.array(texts)
labels = np.array(labels)

mlb = MultiLabelBinarizer()
labels_multi = mlb.fit_transform(labels)

mskf = MultilabelStratifiedKFold(5, shuffle=True, random_state=1)
for i, (itr, ite) in enumerate(mskf.split(texts, labels_multi)):
    Path(PATH_FOLDS).joinpath(f"fold_{i}").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        os.path.join(PATH_FOLDS, f"fold_{i}", "train.npz"),
        texts=texts[itr],
        labels=labels[itr],
        Y=labels_multi[itr]
    )
    np.savez_compressed(
        os.path.join(PATH_FOLDS, f"fold_{i}", "test.npz"),
        texts=texts[ite],
        labels=labels[ite],
        Y=labels_multi[ite]
    )

joblib.dump(mlb, "artifacts/smdis_small/split_folds/mlb.joblib")
