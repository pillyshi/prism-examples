from argparse import ArgumentParser
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


parser = ArgumentParser()
parser.add_argument("--fold", type=int, default=0)
args = parser.parse_args()

with mlflow.start_run():
    fold = mlflow.log_param("fold", args.fold)
    seed = mlflow.log_param("seed", 1)
    n_splits = mlflow.log_param("n_splits", 5)
    train = np.load(f"artifacts/smdis_small/transform/folds/fold_{fold}/train.npz")

    gs = GridSearchCV(
        Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", OneVsRestClassifier(SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                random_state=seed,
                class_weight="balanced")))
        ]),
        param_grid={
            "classifier__estimator__alpha": np.logspace(-4, 1, 10),
            "classifier__estimator__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        cv=MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        scoring="f1_micro"
    ).fit(train["X"], train["Y"])

    out_dir = Path(f"artifacts/smdis_small/train/folds/fold_{fold}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(gs.best_estimator_, out_dir / "pipeline.joblib")
