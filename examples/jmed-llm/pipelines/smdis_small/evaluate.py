from argparse import ArgumentParser

import joblib
import mlflow
import numpy as np
from sklearn.metrics import f1_score


parser = ArgumentParser()
parser.add_argument("--fold", type=int, default=0)
args = parser.parse_args()

with mlflow.start_run():
    fold = mlflow.log_param("fold", args.fold)
    test = np.load(f"artifacts/smdis_small/transform/folds/fold_{fold}/test.npz")
    pipeline = joblib.load(f"artifacts/smdis_small/train/folds/fold_{fold}/pipeline.joblib")
    Y_pred = pipeline.predict(test["X"])
    mlflow.log_metric("f1_micro", f1_score(test["Y"], Y_pred, average="micro"))
