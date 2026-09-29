# 輸送出力と自律フィードバックへの追補（2026-09-29）

利用者が「将来的にロールアウト的な自律予測もしたいので出力に加えて」と指定したため、
輸送を入出力の状態に追加する。旧 `radaz_transport_channels_20260929` は凍結したまま残す。
新しい本学習の本命は本追補の4入力4出力。今回の作業は実装・技術確認で、長時間学習は別。

## 入出力と学習

- none：3物理場→3物理場の基準。
- raw：ne, ni, phi, γ_raw→同じ4物理場。
- corr：ne, ni, phi, γ_corr→同じ4物理場。
- 3案共通で既知E/Bを表す条件2チャネルを与える（予測対象には含めない）。
- 6 source条件・TRAIN/VAL/TEST範囲・過去10/未来10・15ns cadence・field affine・
  TRAIN γ RMS・AR基準・T10 maskは親版をそのまま継承。再fit・TEST開封をしない。
- 共有3場の初期予測を親版・案間で一致させる。γ入力の追加重みとγ出力readoutは0初期化。
- γ教師値は未来の真値3場から導出し、入力と同じTRAIN RMSで正規化する。
  過去入力と未来ラベルは別々に構成し、未来ラベルを入力へ戻さない。

損失は `L = MSE_fields + (1/3) MSE_gamma`。
MSE_fieldsは3場の平均で親版と同じ係数を保持。γは257×256有効格子だけで評価する。
各fieldのMSEへの係数が1/3なので、γにも1/3を置く。尺度は異なり損失の大きさが等しい
保証ではない。係数のTEST調整はしない。total/field/gammaを分けてログへ残す。
noneにはγ損失がないため、total lossの大小を案間の精度比較に使わない。
整合損失や真値Γへの投影は今回追加しない。

60epoch、seed42/43、同じB構造・GroupNorm・E/B FiLM・Adam/OneCycle・終端checkpoint。
none42→raw42→corr42→none43→raw43→corr43を逐次実行。共有TRAIN/VALはメモリで再利用。
epoch保存・optimizer/scheduler/RNG復元、独立generatorのデータ順序は親版と同じ。
3入力3出力と4入力4出力の差は入力と教師信号の両方を含む。
入力だけの寄与を切り分ける場合は、旧版の4入力3出力比較を別に実行する。

## 自律予測

初期の正解10frameだけから開始。未来10frameの予測4場をそのまま次の入力へ渡し、
同じ既知E/Bを付けて繰り返す。**予測γを密度・電位から再計算して置き換えない。**
有効格子の状態にclip・真値置換・再正規化・投影はしない。
無効な径paddingだけは、各予測の最後の有効行を複製して次の入力へ渡す。
rollout関数は未来の真値を引数に取らず、予測が終わってから真値を読み採点する。
非有限値は検出して停止し、安定な予測として記録しない。

まず1ブロックの教師あり学習のまま、自律推論の経路を用意する。
4出力化だけで長時間安定性が得られたとは言えず、複数ブロックの学習損失等は次の検討。
CLI rolloutはsource内の指定split先頭から開始し、split境界を越える要求を拒否する。
単一初期履歴の結果は記述的診断で、全位相プールの短時間予測評価とは区別する。

## 評価

raw/corrの主出力は、直接予測したγを4径帯域で平均したΓ_fullの未来10frame平均。
noneは予測3場から算出した同じΓ_fullを主評価する。
全案で場由来Γ_full、場由来Γ_T10、場のMSEを併記する。
直接γの局所誤差、場から再計算したγの局所誤差、直接γと再計算γのずれを分けて保存。
直接輸送だけ良い・場との不整合が大きい場合を成功として一括しない。

**直接γだけからΓ_T10は取り出せない。** 密度と電場のcross-spectrumのmode選択であり、
γ画像のFFTの同じmodeを足す操作とは違う。直接Γはfullだけ、T10は常に3場から計算する。
raw/corrは局所教師値が違うが方位平均の物理的ターゲットは同じ。
既存の持続・TRAIN平均・直接AR10・E/B共通AR10を両出力経路に適用する。
TESTでのモデル/係数選択や区間の追加はせず、既読source探索として扱う。

E/B転移の評価経路も引き継ぐ。明示的な転移manifestに記録した対象だけを評価し、
sourceの尺度・重み・共通ARを固定、ターゲットで局所ARをfitしない。
今回そのデータは開かない。転移rolloutの自動評価は未追加（関数自体は既知条件で利用可能）。

## 実行

作業ディレクトリSimVPv2、OpenSTL環境のPythonを使用する。

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python -m unittest discover -s tests -p test_radaz_transport_output.py -v
python run_radaz_transport_output.py prepare
python run_radaz_transport_output.py smoke
python run_radaz_transport_output.py seal
python run_radaz_transport_output.py status
# 本学習とその後の評価
python run_radaz_transport_output.py train --arm all
python run_radaz_transport_output.py evaluate --arm corr --seed 42 --split test
python run_radaz_transport_output.py rollout --arm corr --seed 42 --case E10_B20 --split test --blocks 4
```

4blocksは40未来frame=600nsの自律推論。長さは`--blocks`で指定する。
評価・rolloutには新しい4出力版の対応する終端checkpointが必要。
旧入力専用checkpoint・smoke checkpointは受け付けない。
成果物は `workdirs/2D_RadAz/radaz_transport_output_20260929/`。
親bundleとその全assets、新コード・プロトコル・contract・smokeを新bundleで照合する。
本学習を始める前にこの版をローカルcommitし、hashと確認結果をICLメモへ追記する。
