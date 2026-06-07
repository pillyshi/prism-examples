# smdis_small


## 概要

`smdis_small` は、JMED-LLM の `datasets/smdis.csv`
を使った小規模なマルチラベル分類実験です。 Semaxis の
`SupervisedTransformer` でテキストを特徴量に変換し、その特徴量に対して
scikit-learn の線形分類器を学習します。

評価は 5-fold cross validation で行い、各foldの test split に対する
`f1_micro` を MLflow に記録しています。

## パイプライン

1.  `download.py`: `sociocom/JMED-LLM` から `datasets/smdis.csv`
    を取得する。
2.  `preprocess.py`: 正解回答 `answer == "A"`
    の質問文から投稿本文とタグを抽出する。
3.  `split_folds.py`: `MultilabelStratifiedKFold(n_splits=5)` で fold
    0-4 に分割する。
4.  `transform.py`: Semaxis の `SupervisedTransformer`
    でテキストを特徴量行列に変換する。
5.  `train.py`: `SGDClassifier` + `OneVsRestClassifier` を
    `GridSearchCV` で学習する。
6.  `evaluate.py`: 各foldの test split で `f1_micro`
    を算出し、MLflowに記録する。

## 実験設定

主な設定は以下です。`repo_id` と `filename` が、Semaxis
の特徴生成で使ったlocal LLMを表します。

- feature generator: `semaxis.SupervisedTransformer`
- LLM backend: `llama-cpp-python`
- NLI model: `akiFQC/bert-base-japanese-v3_nli-jsnli-jnli-jsick`
- embedding model: `paraphrase-multilingual-MiniLM-L12-v2`
- classifier:
  `OneVsRestClassifier(SGDClassifier(loss="log_loss", penalty="elasticnet", class_weight="balanced"))`
- folds: 5
- metric: `f1_micro`

## 評価結果

この表は `mlflow.db` に保存された evaluate run から生成しています。
対象は `f1_micro` を持つFINISHED runのうち、`fold` 0-4
がそろった5-fold評価セットです。
foldが欠けている実行や、過去に実行されたfold 0のみの evaluate run
は結果表から除外しています。 各foldの `repo_id`, `filename`,
`n_features` などは、evaluate run の直前に実行された同じfoldの
`train.py` と、そのtrainの直前に実行された同じfoldの `transform.py`
から取得しています。 `experiment_id`
は、これらの主要パラメータが同じ5foldを1つの実験セットとしてまとめるためのIDです。

| experiment_id | f1_micro_mean | f1_micro_std | f1_micro_min | f1_micro_max | repo_id | filename | n_features | n_ctx | tokens_per_hypothesis | nli_model | embedding_model | n_splits | seed |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 1 | 0.7006 | 0.0598 | 0.6207 | 0.7573 | unsloth/gemma-4-E4B-it-qat-GGUF | gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf | 16 | 4096 | 50 | akiFQC/bert-base-japanese-v3_nli-jsnli-jnli-jsick | paraphrase-multilingual-MiniLM-L12-v2 | 5 | 1 |
| 2 | 0.6708 | 0.1183 | 0.5414 | 0.8222 | unsloth/gemma-4-E4B-it-qat-GGUF | gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf | 8 | 4096 | 50 | akiFQC/bert-base-japanese-v3_nli-jsnli-jnli-jsick | paraphrase-multilingual-MiniLM-L12-v2 | 5 | 1 |

## 再生成

`readme.qmd` から `readme.md`
を生成するには、プロジェクトルートで以下を実行します。

``` bash
quarto render pipelines/smdis_small/readme.qmd
```

MLflowの結果を確認する場合は以下を使います。

``` bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
