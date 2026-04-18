# SMDIS 解析メモ

データセット: `datasets/smdis.csv`（100行、8タグ × 約12投稿/タグ）  
実行日: 2026-04-18  
prism-text: 0.2.1（SGD + F1評価）

## ラベル分布

各タグとも正例約13件・負例約87件（正例率約13%）。
同一のSNS投稿が全8タグ分の行として収録されている。

## CV スコア（F1）

| Tag | F1 |
|---|---|
| fever | 0.606 |
| cold | 0.451 |
| runnynose | 0.448 |
| headache | 0.440 |
| influenza | 0.382 |
| diarrhea | 0.245 |
| cough | 0.040 |
| hayfever | 0.022 |

fever・cold・runnynose はある程度の予測力を確認。cough・hayfever はほぼゼロで、特徴量が機能していない。

## 前回（0.1.0）との主な変更点

- Lasso CV (accuracy) → SGDClassifier + GridSearchCV (f1) に変更
- `generate_features(axes_labels=...)` でラベルの直接注入が可能になった
- `SelectionResult` に `cv_score` / `cv_scoring` が追加された

## 気になる点

### 異常に大きい係数

複数タグで極端な係数が出ている：

```
cold:      [-192.889] Mentions unrelated health issues
diarrhea:  [-573.972] Absence of diarrhea mentions
headache:  [-180.515] Health condition without headache
```

SGDの収束不安定性、または正例数が少なすぎることによるオーバーフィットが原因と思われる。

### 直感に反する符号

```
diarrhea: [-2.270] Direct mention of diarrhea  ← 負（前回は +0.670 で正）
headache: [-152.831] Headache indication        ← 負
```

「下痢の直接言及」が負例側に寄るのは直感に反する。タグ間クロス汚染（同一投稿が全タグに登場）の影響が疑われる。

### cough / hayfever のF1≈0

特徴量が10件と少ない状況では、症状の言語的特徴が不十分な場合に選択が機能しない。

## 制約と次のステップ

### 小規模データの限界
正例13件では SGD の CV が信頼できない。`--all`（1,920件 × 8タグ）で再実行すると安定するか確認する価値がある。

### タグ間クロス汚染
同一投稿が複数タグに登録されているため、「他タグでの負例」が特定タグの正例特徴を打ち消している可能性がある。`datasets/all/SMDIS.csv` では投稿ごとに独立してラベルが付いており、構造が異なる可能性がある。

### 特徴量数
現在 `n_features=10`。増やすことで選択の幅が広がる可能性がある。

## 推奨次手順

1. `--all` でフルデータ（1,920件）を使って再実行
2. `n_features` を増やして試す（例: 20）
