import json
from pathlib import Path
from argparse import ArgumentParser

import joblib
import numpy as np
from dvcgen import dep, param, out, stage
from semaxis import UnsupervisedTransformer


FOLDS = param("folds", [0, 1, 2, 3, 4])

stage(
    name="smdis_small_transform",
    foreach=FOLDS,
)

parser = ArgumentParser()
parser.add_argument("--fold", type=int, default=0)
args = parser.parse_args()
fold = args.fold

PATH_TRAIN = dep("artifacts/smdis_small/split_folds/folds/fold_${item}/train.npz")
PATH_TEST = dep("artifacts/smdis_small/split_folds/folds/fold_${item}/test.npz")
PATH_TRAIN = PATH_TRAIN.replace("${item}", f"{fold}")
PATH_TEST = PATH_TEST.replace("${item}", f"{fold}")

LLM = param("llm", "gpt-4o-mini")
NLI_MODEL = param("nli_model", "akiFQC/bert-base-japanese-v3_nli-jsnli-jnli-jsick")
N_FEATURES = param("n_features", 128)
SEED = param("seed", 1)
SAMPLE_METHOD = param("sample_method", "random")
EMBEDDING_MODEL = param("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")

PATH_TRANSFORMER = out("artifacts/smdis_small/transform/folds/fold_${item}/transformer.joblib")
PATH_TRANSFORMED_TRAIN = out("artifacts/smdis_small/transform/folds/fold_${item}/train.npz")
PATH_TRANSFORMED_TEST = out("artifacts/smdis_small/transform/folds/fold_${item}/test.npz")
PATH_TRANSFORMER = PATH_TRANSFORMER.replace("${item}", f"{fold}")
PATH_TRANSFORMED_TRAIN = PATH_TRANSFORMED_TRAIN.replace("${item}", f"{fold}")
PATH_TRANSFORMED_TEST = PATH_TRANSFORMED_TEST.replace("${item}", f"{fold}")

train = np.load(PATH_TRAIN)
test = np.load(PATH_TRAIN)
transformer = UnsupervisedTransformer(
    llm="gpt-4o-mini",
    nli_model=NLI_MODEL,
    n_features=N_FEATURES,
    language="Japanese",
    seed=SEED,
    sample_method=SAMPLE_METHOD,
    embedding_model=EMBEDDING_MODEL
)
transformer.fit(train["texts"])
Path(PATH_TRANSFORMED_TRAIN).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    PATH_TRANSFORMED_TRAIN,
    X=transformer.transform(train["texts"]),
    Y=train["Y"]
)
np.savez_compressed(
    PATH_TRANSFORMED_TEST,
    X=transformer.transform(test["texts"]),
    Y=test["Y"]
)
joblib.dump(transformer, PATH_TRANSFORMER)
