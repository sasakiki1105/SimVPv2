# A 固定・損失対照 pilot（2026-09-08）

利用者が承認した「残差・時間尺度・簡単な baseline → A 固定で data-only と物理 loss の比較」に沿う探索実験。
先に用意した A/B/C/D×3 seed の factorial bundle は変更しない。本 pilot は別の計画として記録する。

## 比較

| 条件 | 構造 | 正規化 | loss |
|---|---|---|---|
| D | reduced A: hid_S=64, hid_T=256, N_T=4, 方位角 latent 64 | GroupNorm 8 groups | field MSE |
| P | D と同一 | D と同一 | field MSE + local-product power/cross-spectrum |

両者とも training seed=42、60 epoch、batch size=1、学習率 0.001、OneCycle。
初期 model state が同じであることを確認する。loss の有効化以外の設定は揃える。
P の重みは既に source-train 24 窓で計算した値を使用し、新たに test で調整しない。
drop_path_schedule=zero_to_max、drop_path=0。主 checkpoint は last.ckpt、epoch index=59。
best と snapshot は診断用であり、今回の主 checkpoint の代わりに選ばない。

source 6 条件のみを含む manifest を作り、既存 source-train 共通正規化を保持する。
train は0–1599、validation は1600–1799、source-test は1800–1999を評価に使う。
validation の全重複窓は学習ログ、物理評価は両 split とも10個の非重複10→10窓と明記する。
source-test は既に開発に使用済みであり独立な確認用テストとは呼ばない。
古い condition holdout の場は今回の pilot に含めない。
既存ローダーが3 splitを構築するためsource各caseにtrain/val/testの所属を明示する。
ローダーのtest segmentは1800–2000の201 frameを読み込むが、train_onlyではそのloaderを反復せず、
固定物理評価は従来通り1800–1999の200 frameを使う。test統計による正規化・重み更新はしない。

## baseline と残差解析

`analyze_radaz_nextstep_baselines.py` は source-train の後半1200–1599を baseline fitting/ACF に使う。
copy、入力10時点の平均、同train区間の平均、入力履歴の線形外挿、AR(10)を比較する。
AR は条件別・各観測成分別の線形予測器であり、未見条件へのzero-shot手法ではない。
正則化候補は1e-6,1e-4,1e-2,1,100、選択は6 source-validation条件のNRMSE²の平均。
Gamma(mode) と全mode和のfluxはそれぞれ一つのlambdaを選び、その後source-testへ固定適用する。
入力・出力窓は全手法で同じ。モデルはtrain全体を使い、ARのfitは後半400frameという差も記録する。

主に時間別の径方向band流束とmodal fluxのcopy/ARに対するskillを見る。
field誤差・複素係数・符号付きO・振幅・径方向/モード/lead別誤差も併記する。
平均biasと時間変動の誤差を分離し、amplitude/organization/interactionの交差項による相殺を記録する。
ACFの1/e crossingは探索的な時間尺度であり、普遍的予測限界や独立標本数とは扱わない。
150nsの既存endpointを維持し、長いhorizonは別計画として追加する。

Pの物理lossは窓内平均スペクトルを拘束する。時刻別輸送を直接拘束するlossではない。
改善すれば物理lossという介入の効果として記録するが、機構や粗視化との共通性を自動的に証明したとはしない。
差がなければ時間loss・直接flux読み出し・履歴などの候補を残差に基づいて一つずつ検討する。

## 実行・再現性・主張

まずD→Pの2本を逐次実行し、両方完了後に同じ評価器でlastを評価する。
途中結果を見てPの設定を変更しない。seed43/44は後続の両条件反復候補として記録し、今回のqueueには含めない。
旧Aの閾値からのPASS/STOPは使用しない。seed42だけで一般性・統計的有意性を主張しない。
実行前にconfig/model/loss/dataloader/evaluator/文書/manifestをhashで固定する。
queueは排他的lock、checkpoint確認、進捗JSONとログ、失敗時停止を備える。
CPU/VRAM/所要時間は実測する。旧Aは約24.7時間であり、今回も長時間の学習を想定する。

独立PICの準備は別作業。E20_B20/E10_B30の独立seedを初期再現確認に選び、初期粒子seed・restart親・出力cadenceを記録する。
新しい条件への汎化確認には、別途まだ使用していない運転条件が必要。ここでは新PICの生成・HPCへの投入は行わない。

## 起動修正記録

`radaz_paired_pilot_v1`はsource caseの`test`所属欠落によりデータ読込みで停止し、
モデル初期化・optimizer stepには到達しなかった。bundle・実行source・エラーログを保存した。
`radaz_paired_pilot_v2`はこの所属指定のみを補い、同一configで開始する。
