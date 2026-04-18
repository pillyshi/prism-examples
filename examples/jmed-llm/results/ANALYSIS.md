# SMDIS 解析メモ

データセット: `datasets/smdis.csv`（100行、8タグ × 約12投稿/タグ）  
実行日: 2026-04-18

## ラベル分布

各タグとも正例約13件・負例約87件（正例率約13%）。
なお、smdis.csv では同一のSNS投稿が全8タグ分の行として収録されている。

## 結果の観察

### diarrhea — 最も明快なシグナル

```
[+0.670] Mentions diarrhea directly
[-0.497] Mentions non-diarrhea related symptoms
```

「下痢」を直接言及していれば正例、他の症状を言及していれば負例、という直感に合致した結果。
NLIスコアリングとLassoが正しく機能している場合のベースラインとして参考になる。

### cold — 行動特徴が支配的

```
[+0.558] Describes plans to see a doctor
[-0.504] Compares ailments
[+0.306] Discusses taking cold medicine
```

受診意図や薬の服用が正の予測因子、他の症状との比較が負。
症状そのものより「取った行動」が予測力を持つという興味深いパターン。

### fever / headache — 直感と逆の負係数

```
fever:    [-0.473] Describes fever-related discomfort
          [-0.415] Reports family illness impact
headache: [-0.552] Indicates multiple symptoms
          [-0.384] Mentions severe discomfort
```

症状を表す特徴量に強い**負**の係数がついており、直感に反する。考えられる原因は以下。

1. **クラス不均衡 × accuracy ベースのLasso CV**: 正例率13%では全件を負例予測しても87%のaccuracyを達成できる。LassoCVがaccuracyを最大化しようとすると、正例側の特徴量を積極的に抑制する方向に正則化される可能性がある。
2. **タグ間クロス汚染**: 同一投稿が全8タグに登場するため、「発熱を描写した投稿」が headache タグでは負例となる。モデルが「発熱描写 → headacheでない」という逆のパターンを学習している可能性がある。

### runnynose / hayfever — 係数が小さく解釈しにくい

```
runnynose: [+0.063] Describes daily suffering
hayfever:  [-0.189] Mentions specific symptoms of hay fever
```

係数が小さく、符号も直感と合わない特徴が多い。クラス不均衡の影響が大きいと考えられる。

## 課題・制約

### 1. クラス不均衡に非対応のLasso CV

現在のPrismは `LassoCV` の評価指標にaccuracyを使用している。正例率13%程度の不均衡データでは、全件を負例と予測するだけで高いaccuracyを得られるため、Lassoが正例側の特徴量を過剰に抑制する傾向がある。

**対応案（Prism改修待ち）:** LassoのCV評価指標を `balanced_accuracy` / `f1` / `roc_auc` に変更するか、`class_weight='balanced'` を設定する。

### 2. 事前ラベルをPrismに渡すAPIがない

現状は `prism._generator.generate()` を直接呼ぶことで回避しているが、内部APIの変更に対して脆弱。

**対応案（Prism改修待ち）:** `generate_features()` に `AxisLabels` を直接渡せるパラメータを追加する。

### 3. 小規模データセット（smdis.csv）

タグあたり正例が約12〜13件しかなく、LassoCVの信頼性が低い。

**推奨:** Prismのクラス不均衡対応が入った後、`--all` オプション（1,920件 × 8タグ）で再実行する。

## 次のステップ

- Prismのクラス不均衡対応を待つ
- 改修後に `--all` でフルデータを使って再実行
