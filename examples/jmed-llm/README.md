# jmed-llm 解析例

[JMED-LLM](https://github.com/sociocom/JMED-LLM) の `smdis.csv` を使った、日本語医療テキストのマルチラベル分類実験です。

特徴量生成には [Semaxis](https://github.com/pillyshi/semaxis) を利用します。Semaxis は旧 Prism から改名されたプロジェクトです。

## 概要

この例では、JMED-LLM の SMDIS データから投稿本文とタグを抽出し、Semaxis の `SupervisedTransformer` でテキストを特徴量に変換します。その特徴量に対して scikit-learn の線形分類器を学習し、5-fold cross validation の `f1_micro` を MLflow に記録します。

現在の主な構成は以下です。

- パイプライン: `pipelines/smdis_small`
- データ取得元: `sociocom/JMED-LLM` の `datasets/smdis.csv`
- 特徴量生成: `semaxis.SupervisedTransformer`
- LLM backend: `llama-cpp-python`
- 評価管理: MLflow
- 実行環境管理: `uv`

## セットアップ

Python 3.12 以上と `uv` が必要です。JMED-LLM からのデータ取得には GitHub CLI (`gh`) も使います。

```bash
cd examples/jmed-llm
uv sync
```

GPU 付きのリモート環境で `llama-cpp-python` を CUDA 有効でビルドして使う場合は、`.env` に `HOST` を設定したうえで以下を使います。

```bash
make setup
```

`make setup` はリモートで `uv sync` を実行し、続けて `llama-cpp-python` をインストールします。必要に応じて `CUDA_VERSION`, `GCC_VERSION`, `LLAMA_CPP_VERSION` は `Makefile` の変数またはコマンドライン引数で上書きしてください。

## 実行

`smdis_small` は次の順に実行します。

```bash
uv run python pipelines/smdis_small/download.py
uv run python pipelines/smdis_small/preprocess.py
uv run python pipelines/smdis_small/split_folds.py

uv run python pipelines/smdis_small/transform.py --fold 0
uv run python pipelines/smdis_small/train.py --fold 0
uv run python pipelines/smdis_small/evaluate.py --fold 0
```

全 fold を実行する場合は `--fold 0` から `--fold 4` まで繰り返します。

`transform.py` は Hugging Face Hub から GGUF モデルを取得して `llama-cpp-python` で実行します。モデル、コンテキスト長、特徴数などの実験設定は `pipelines/smdis_small/transform.py` 内で MLflow parameter として記録されます。

## 結果確認

MLflow UI は以下で起動できます。

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`pipelines/smdis_small/readme.md` には、`mlflow.db` から生成した実験概要と評価結果をまとめています。Quarto で再生成する場合は以下を実行します。

```bash
make readme
```

## リモート実行補助

`.env` に `HOST` を設定すると、`Makefile` のリモート補助コマンドを使えます。

```bash
make sync      # ローカルの作業ツリーをリモートへ同期
make ssh       # リモートへログイン
make install   # リモートで uv sync
make fetch     # リモートの mlflow.db と mlruns を取得
```

任意コマンドをリモートで実行する場合は以下です。

```bash
make run CMD='uv run python pipelines/smdis_small/transform.py --fold 0'
```

## ライセンス

この例のコードはリポジトリのライセンスに従います。JMED-LLM など外部データセットやモデルは、それぞれの配布元のライセンスと利用条件に従ってください。
