import json
import tempfile
from pathlib import Path
from argparse import ArgumentParser

import joblib
import numpy as np
from llama_cpp import Llama
from semaxis import SupervisedTransformer, UnsupervisedTransformer, LlamaCppClient
import mlflow


parser = ArgumentParser()
parser.add_argument("--fold", type=int, default=0)
args = parser.parse_args()

with mlflow.start_run():
    fold = mlflow.log_param("fold", args.fold)

    PATH_TRAIN = f"artifacts/smdis_small/split_folds/folds/fold_{fold}/train.npz"
    PATH_TEST = f"artifacts/smdis_small/split_folds/folds/fold_{fold}/test.npz"

    PATH_TRANSFORMER = f"artifacts/smdis_small/transform/folds/fold_{fold}/transformer.joblib"
    PATH_TRANSFORMED_TRAIN = f"artifacts/smdis_small/transform/folds/fold_{fold}/train.npz"
    PATH_TRANSFORMED_TEST = f"artifacts/smdis_small/transform/folds/fold_{fold}/test.npz"

    train = np.load(PATH_TRAIN)
    test = np.load(PATH_TRAIN)

    # UnsupervisedTransformerの話
    # n_ctx=2^10なら、n_features=2^3が限界？
    # n_ctx=2^11の限界を探る。
    # n_features=2^4 -> OK
    # n_features=2^5 -> context不足で出力が途切れる
    # tokens_per_hypothesisを60にしてみる
    # tokens_per_hypothesis=60でも出力途切れ？
    # n_ctx=2^11, n_features=2^4 -> OK
    # n_ctx=2^12, n_features=2^5 -> OK
    # n_ctx=2^12, n_features=2^6 -> OK? ダメな時もある
    tokens_per_hypothesis = mlflow.log_param("tokens_per_hypothesis", 50)  # 50 ~ 150
    n_features = mlflow.log_param("n_features", 2 ** 5)
    output_budget = n_features * tokens_per_hypothesis
    n_ctx = mlflow.log_param("n_ctx", 2 ** 12)
    if n_ctx - output_budget < 0:
        raise ValueError(f"{n_ctx - output_budget}")

    llm = LlamaCppClient(Llama.from_pretrained(
        repo_id=mlflow.log_param("repo_id", "unsloth/gemma-4-E4B-it-qat-GGUF"),
        filename=mlflow.log_param("filename", "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"),
    	# repo_id=mlflow.log_param("repo_id", "unsloth/gemma-4-E2B-it-qat-GGUF"),
    	# filename=mlflow.log_param("filename", "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"),
        # repo_id=mlflow.log_param("repo_id", "unsloth/gemma-4-12B-it-qat-GGUF"),
        # filename=mlflow.log_param("filename", "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"),
        # repo_id=mlflow.log_param("repo_id", "unsloth/Qwen3.5-9B-GGUF"),
        # filename=mlflow.log_param("filename", "Qwen3.5-9B-UD-Q4_K_XL.gguf"),
        n_ctx=n_ctx,
        n_gpu_layers=-1,
        flash_attn=True
    ))

    transformer = SupervisedTransformer(
        llm=llm,
        nli_model=mlflow.log_param("nli_model", "akiFQC/bert-base-japanese-v3_nli-jsnli-jnli-jsick"),
        n_features=n_features,
        context_limit=n_ctx - output_budget,
        language="Japanese",
        seed=mlflow.log_param("seed", 1),
        sample_method=mlflow.log_param("sample_method", "random"),
        embedding_model=mlflow.log_param("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
    )
    # transformer = UnsupervisedTransformer(
    #     llm=llm,
    #     context_limit=n_ctx - output_budget,
    #     nli_model=mlflow.log_param("nli_model", "akiFQC/bert-base-japanese-v3_nli-jsnli-jnli-jsick"),
    #     n_features=n_features,
    #     language="Japanese",
    #     seed=mlflow.log_param("seed", 1),
    #     sample_method=mlflow.log_param("sample_method", "random"),
    #     embedding_model=mlflow.log_param("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
    # )
    transformer.fit(train["texts"], train["Y"])
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

    for feature in transformer.features_:
        print(feature)

    # mlflow.log_artifact(PATH_TRANSFORMED_TRAIN)
    # mlflow.log_artifact(PATH_TRANSFORMED_TEST)
    # mlflow.log_artifact(PATH_TRANSFORMER)
