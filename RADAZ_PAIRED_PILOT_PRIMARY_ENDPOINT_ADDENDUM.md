# D/P paired pilot: primary endpoint addendum（2026-09-08 深夜固定）

本文書は `RADAZ_PAIRED_PILOT_PROTOCOL.md`
（bundle 記録 SHA256 `9315b0336b1ac202aa3a63022419ed012d0d369b1da22217f7a7fe9519cf4930`）への追補である。
凍結済み protocol・bundle・code は一切変更しない。追補は bundle ハッシュ対象外の新規ファイルとして置き、
本ファイル自身の SHA256 を `ICL_reserch_memo.md` に記録して外部タイムスタンプ（git commit）を付す。

## 固定時点の状態（開封前であることの根拠）

- queue は利用者指示により `paused_by_user`（2026-09-08 21:48 JST）。D は 32/60 epoch 完了、P は未開始。
- `workdirs/2D_RadAz/radaz_paired_pilot_v2/evaluation/` は存在せず、`results.json` は未生成。
  D/P いずれの物理評価値・輸送 skill も未計算・未閲覧である。
- ここまでに閲覧したのは D の train/validation の場 loss 曲線のみ（2026-09-08 21:46 JST の進捗確認）。
  これは開示事項として記録する。場の loss は主要 endpoint（輸送 skill）と同一ではないが、
  無情報とも言えないため、本追補は「完全 blind」ではなく「主要 endpoint 開封前の固定」と呼ぶ。

## 主要 endpoint（1 個）

`evaluate_radaz_paired_pilot.py`
（bundle 記録 SHA256 `ae1ef69b5f922f9819c96598f53d46945d180d8bed260cce7fa5c6909f47a47c`）が出力する
`results.json` の `paired_comparison.source_test.<case>.gamma.P_skill_vs_D` を用い、

**主要統計量 = source_test 6 条件にわたる `P_skill_vs_D`（観測量 gamma =時刻別 modal Γ）の中央値。**

- skill 定義は `radaz_metrics_v3.skill`：`1 − SSE(P)/SSE(D)`。**正 = P（物理 loss）が D（field MSE）より良い。**
- 事前登録する方向仮説：P は modal Γ の再現を改善する（中央値 > 0）。逆符号はそのまま反証として報告する。
- checkpoint は凍結どおり両者 last（epoch index 59）。best・snapshot への事後変更、split・観測量・
  要約統計（中央値）の事後変更を禁止する。

## 併記する副次量（主要判定には使わない）

1. 同統計量の最悪条件値（min）と、正となった条件数（/6）。
2. `flux_full`（全 mode 流束）について同じ中央値・min・正条件数。
3. source_validation split の同量（副次。下記 AR 注意を付す）。
4. 各 cell の copy / AR 比較（探索的文脈として）。

## 判定規則と noise floor

seed 42 一対のみでは、`P_skill_vs_D` の大きさが loss 介入の効果か run-to-run 変動かを
区別できない（`deterministic=False`、単一 seed、さらに D には後述の中断再開がある）。従って：

1. **今回の D/P 評価単独では「P が D に優る/劣る」という主張を行わない。**
   結果は方向と大きさの記録として報告する（protocol の既定方針を具体化）。
2. 量的な主張は、bundle に登録済みの後続候補 **seed 43 の D 再学習**（同一 config・同一データ順・
   無中断）を noise floor として得た後にのみ行う。その際の規則を先に固定する：
   - noise 尺度 N = source_test 6 条件の `skill(D_seed43, truth, D_seed42)` の絶対値の中央値。
   - 「効果あり」と主張できるのは |主要統計量| > N かつ 6 条件中 5 条件以上で符号が一致する場合のみ。
   - それ以外は「run-to-run 変動と区別できない」と記録する。
3. seed 43 D の学習は利用者の GPU 使用と D/P queue の完了後に別 queue として実行し、
   本 pilot の凍結物には触れない。

## 事前開示する交絡・注意（結果を見る前に固定）

1. **D の中断再開**：D は利用者指示により epoch 32 完了時点で停止・checkpoint 保全済みであり、
   再開後のミニバッチ順序 RNG が無中断実行と同一である保証はない。P が無中断で走った場合、
   D/P ペアリングには再開非対称が入る。この非対称が結論に効く疑いが出た場合、
   無中断の seed 43 D がその対照となる。P 側にも中断があれば同様に記録する。
2. **AR の λ は source_validation で選択済み**：source_validation split 上の AR 比較は AR に有利な
   楽観であり、意味を持つ AR 比較は source_test 側である。
3. **val_loss 曲線の非可換性**：P の logged `val_loss` は補助項を含み、D と最適化対象が異なる。
   両者の validation 曲線を同一指標として比較・図示しない。best.ckpt の意味も両者で異なる
   （主要 checkpoint が last であることの理由の一つ）。
4. **config の同一性**：pilot P の config は factorial cell A（`SimVP_gSTA_radaz_v3_A_60ep.py`）と
   ヘッダコメント以外同一である。P の完了 run は factorial cell A seed 42 を兼ねるものとして台帳に
   記録し、同一学習を再実行・二重集計しない。
5. source_test は開発に使用済みであり、本評価は探索的比較である（protocol どおり）。
   独立確認には未使用条件・独立 PIC realization が必要である。

## 変更履歴

- 2026-09-08 深夜：初版。D 32/60 epoch・P 未開始・評価未実行の時点で固定。
