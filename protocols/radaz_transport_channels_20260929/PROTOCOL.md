# 輸送の第4入力チャネル：実装と探索比較（2026-09-29）

状態：実装・技術確認。3入力／局所積追加／揺動積追加を新規に同条件で比較する。
旧B/C・E8・R4の凍結定義は変更しない。学習は明示的なCLIコマンドで開始する。

## 問いと対照

同じ保存場に含まれる輸送相関を明示すると、未来の場から再構成する輸送が改善するか。
3場SimVPが4場と同等なら、明示入力の必要性は支持されない。条件別AR10や
source共通AR10＋E/Bが同等以上なら、この比較から空間モデルの優位を主張しない。

- `none`: ne, ni, phi → ne, ni, phi。
- `raw`: ne, ni, phi, −ne Ey/B → ne, ni, phi。
- `corr`: ne, ni, phi, −δne δEy/B → ne, ni, phi。

Eyは保存phiの周期中心差分。δは各時刻・各径位置での方位平均との差。
対象は保存場の積という輸送proxyで、各stepの積の平均や実粒子束とは区別する。
導出は実際のfloat32入力を物理単位へ戻した値から行う。同じ解像度・時刻を使い、
future教師やfine積からの別の相関情報を加えない。257×256の有効領域で計算後、
径方向末尾を260行へ複製する。γはTRAINのみの各案別global RMSで割り、符号を保つ。

## データと漏洩防止

既存source manifest `workdirs/2D_RadAz/radaz_paired_pilot_v2/source_manifest.json`。
E10/E20/E30/E40_B20、E10_B10/B30の6条件。各1 PIC実現、全て過去に検討済み。
TRAIN=[0,1600)、VAL=[1600,1800)、TEST=[1800,2000)。最終frame2000は使用しない。
過去10・未来10、15 ns間隔。splitを跨ぐ窓なし。全開始位相を使用し、
TRAIN 1581窓×6=9486、VAL/TEST 181窓×6=1086。

正規化は既存source TRAINのaffineを継承しclipしない。追加γのRMSもTRAIN0–1599のみ。
キャリブレーションはTRAIN/VALを読み、学習時はTESTを読み込まない。
既読TESTを未使用・盲検とは呼ばない。R4や新しいPIC実現のデータは本実装では開かない。

## モデル・学習

B構造（hid_S64, hid_T256, N_S4, N_T4、方位縮小率2）の新規学習。
全案でcondition_film=Trueとし、同じsource正規化のlog_vE/log_n0を与える。
旧B/CのD-onlyはこの条件情報を予測器で使わなかったため、旧checkpointは
今回の公平な3入力対照の代わりにしない。

全案の共有重みはseedごとに同一。追加encoder入力の重みだけゼロ初期化。
初期予測は同一で、追加重みは学習可能。構造差は最初の畳み込みの1入力分。
seed42/43、順序none42→raw42→corr42→none43→raw43→corr43。
各epochの順序は独立generatorのseed+1000003×epochで固定し、モデル構築と切り離す。
同じseed番号だけを根拠にPIC実現の反復とは扱わない。

60 epochs、batch1、float32、Adam lr上限0.001、weight_decay0、OneCycleLR pct_start0.1。
DropPathは0、translator GroupNorm8、3場の正規化MSEのみ（既存と同じ径paddingを含む）。
評価時はpaddingを除外する。追加損失・直接γ出力・新しい物理量は加えない。
主checkpointは60epoch終端、VALでepoch選択しない。各epoch終端でoptimizer/schedulerも保存。
再開は完了epochから行い、中断epochは先頭から再実行する。PAUSE_AFTER_EPOCHで停止可能。
この新しいtrainerの性能を、旧trainerや過去のcheckpointの直接的な再現結果とは呼ばない。

## 評価量・基準モデル

主：未来lead1–10（15–150 ns）の窓平均Γ_full、4径帯域等重みSSE。
副：同じ窓平均Γ_T10。T10はfewmode_stepAのsource VAL共通の固定集合をそのまま継承。
γ画像平均はΓ_fullに一致し、Γ_T10とは一般に異なる。主・副を結果で入れ替えない。
splitの開始点から最終窓まで1frame刻み、全位相プール。条件別と6条件中央値を併記。

null：最後の入力Γの持続、条件別TRAINの未来窓平均、条件別直接AR10、source共通AR10＋E/B。
ARは40履歴成分→未来窓平均4値を直接fit。共通ARには同じ条件2値を追加。
X/Y scalerはTRAIN、ridge係数はTRAIN fit、alpha={1e−4,…,1e2}はVALだけで選ぶ。
旧瞬時ターゲットで選んだARを流用しない。skill=1−SSE_model/SSE_null、分母0は未定義。
平均bias・径帯域別RMSE/skill・3場MSEを保存し、予測と真値の全mode1–64配列も残す。
必要な振幅・位相・符号解析はその配列と既存診断で追跡できるようにする。
この段階は記述比較。CIや有意性を新設せず、重複窓を独立反復と数えない。
追加情報としての電子モーメントG+E、小型非線形モデル、γ単独や帯域Γ枝は次段階の対照。
本実装だけで全ての強い基準モデルを上回ったとは主張できない。

## E/B汎化と以後の段階

入力生成は任意の正のE/Bに対してsourceの固定尺度・同じ式を使う。
本番のholdout評価は学習後にデータの既読状態を確認して別に実行し、
軸上補間・未学習の同時E/B組合せ・外挿・別PIC実現を区別する。
`evaluate --transfer-manifest <JSON>`は明示した未学習E/Bだけを読み、
sourceの尺度・重み・共通ARを固定して用いる。各caseに
`evaluation_authorized: true`、`exposure`、`transfer_kind`が必要。
自動で既存holdoutを探索・開封しない。転移時は局所ARを再fitせず、
平均nullもsource全条件TRAINの平均を使い、条件内評価の局所平均と区別する。
正解履歴を毎回与える本比較は自律rolloutやnative coarse liftingの実証ではない。

## コマンド（SimVPv2を作業ディレクトリとする）

Pythonは `C:\Users\astro\anaconda3\envs\OpenSTL\python.exe`。

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python -m unittest discover -s tests -p test_radaz_transport_channels.py -v
python prepare_radaz_transport_channels.py calibrate
python run_radaz_transport_channels.py smoke
python prepare_radaz_transport_channels.py seal
python run_radaz_transport_channels.py status
# 以降は長時間の本学習。上4行の技術確認とは別。
python run_radaz_transport_channels.py train --arm all
python run_radaz_transport_channels.py evaluate --arm corr --seed 42 --split test
```

成果物は `workdirs/2D_RadAz/radaz_transport_channels_20260929/`。
`contract.json`に定義・split・尺度、`baselines.json`にAR、`bundle.json`にコード・成果物hash。
`smoke.json`はTRAIN1窓の数step動作確認で予測性能ではない。
学習の進捗は`jobs/<arm><seed>/status.json`、各epochは`epochs.jsonl`、終端は`last.pt`。
保存途中の破損を避けるためJSONとcheckpointは一時ファイルから置換する。
コードhashの不一致は停止する。古いbundleの書き換えではなく新しい版を作る。
実装・技術確認・長時間学習・TEST評価を区別してICLメモに追記する。
