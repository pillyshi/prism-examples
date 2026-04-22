# Analysis: SMDIS Axis Discovery (small dataset)

**Dataset:** SMDIS (small split) — 1,920 unique texts (30,720 rows after deduplication)  
**Model:** prism-text 0.4.0, LLM: gpt-4o-mini, NLI: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli  
**Mode:** classification (F1 scoring)  
**Axes discovered:** 10

## Results by Axis

| Axis | CV F1 | Selected Features |
|------|------:|:-----------------:|
| 頭痛について言及 | 0.923 | 10 |
| インフルエンザについて言及 | 0.879 | 9 |
| 自身の健康状態を報告 | 0.858 | 10 |
| 咳について言及 | 0.786 | 9 |
| 下痢について言及 | 0.769 | 9 |
| 花粉症について言及 | 0.658 | 10 |
| 風邪の症状について言及 | 0.433 | 9 |
| 薬に関する情報を含む | 0.400 | 4 |
| エチケットに関する内容 | 0.067 | 8 |
| 体調不良についての記述 | 0.000 | 0 |

## Notable Observations

### High-performing axes
- **頭痛 (F1=0.923)**: 最も予測精度が高い。「頭痛の症状に関する言及」の係数が469.95と異常に大きく、このfeatureだけでほぼ判別できている。モデルが強く依存しており、汎化性には注意が必要。
- **インフルエンザ (F1=0.879)**: ワクチン言及(+2.94)・社会的影響(+2.87)・症状(+2.39)の3featureが高く寄与。意味的に一貫した軸。
- **健康状態報告 (F1=0.858)**: 「薬と治療法の要求」係数29.91が突出。診断・治療を求めるQA的な投稿を捉えている。

### Low-performing axes
- **体調不良 (F1=0.000)**: featureが1件も選択されなかった。「体調不良」は他のすべての軸と意味的に重複する汎用的な仮説で、識別軸として機能しなかった。
- **エチケット (F1=0.067)**: 小データセットではエチケット言及の投稿が希少で、クラス不均衡が原因と考えられる。

### Negative features (what the axis is *not*)
いくつかの軸で他軸の典型的featureが強い負の係数を持つ：
- 下痢軸: 「咳・痰の症状」が -2.68（下痢と咳は排他的）
- 咳軸: 「ウイルス感染の症状」が -4.00（インフルエンザとの区別）
- エチケット軸: 「感染症の感染リスク」が -4.18

## Issues & Next Steps

1. **係数の異常値**: 頭痛軸の+469.95、健康状態軸の+29.91はL1正則化が不十分な可能性。`--all` データセットで再実行し安定性を確認する。
2. **軸の品質**: 「体調不良についての記述」は汎用的すぎる。`discover_axes` のプロンプトに多様性制約を加えるか、軸をpost-filterすることを検討。
3. **エチケット軸**: 全データセット (`--all`) では改善の余地あり。
4. **sklearnの警告**: `penalty` deprecation (1.10で削除予定) と `use_legacy_attributes` 警告が出ているため、prism側での対応が必要。
