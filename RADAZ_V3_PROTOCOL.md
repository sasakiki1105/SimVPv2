# RadAz 修正版 v3：評価と次の対照実験（2026-09-08）

この文書は既存データを使う探索実験の計画であり、未使用データによる事前登録検証ではない。旧 A の A2 STOP は当時の実装に対する記録として保存する。正規化、輸送量、O 集約の定義を変えた v3 に旧 A1/A2 閾値を流用せず、旧 STOP から容量・解像度仮説の採否を決めない。

## 既存 A の再評価

- `evaluate_radaz_corrected_A.py` を標準入口とする。best（epoch index 29）と last（59）を両方評価する。結果を見て一方を主結果に昇格させない。
- 推論規約は保存済み BN 統計、各入力窓の統計、source-train 入力だけで均等校正した保存統計の 3 通りを全て報告する。校正入力は各 source の starts=1200,1300,1400,1500、各 10 フレーム。重み更新・target 使用・DropPath 有効化は行わない。
- 各入力窓での正規化はサンプル独立で、推論バッチの条件構成・順序で変化しない。これは元 checkpoint の推論方法を変える探索的介入であり、独立な再学習による再現ではない。
- source-validation は 1600–1799、source-test は 1800–1999。両方とも開始点を 20 フレーム刻みに固定し、10 入力→10 出力の 10 窓を使う。条件は 6 点。時間窓・モード・径方向 band を独立標本として数えない。
- persistence は最後の入力フレームを 10 回繰り返す。各モデルと同じ入力・target・集計式で評価する。
- 旧 v2 損失重みの調整は `calibrate_radaz_physics_loss_v2.py` で source-test の 1800–1819 を読んでいた。旧モデルにはこの選択経路があるため、source-test も完全未使用評価とは呼ばない。旧 holdout 4 条件も以前の開発で使用済みである。

## 修正版の物理量

`radaz_metrics_v3.py` が独立 NumPy 参照実装。学習 loss の `radial_reduction='local_product'` と合成データで一致を確認する。

1. 各径方向ノードで方位角 FFT を取り、局所の `|ne_n|²`、`|Ey_n|²`、`ne_n conj(Ey_n)` を作ってから 4 band 内で平均する。場を平均してから積を作る旧定義は mean-field proxy として歴史的再現にのみ使用する。
2. `Ey=-d(phi)/dy` は周期中心差分。片側 Fourier 係数の非零正波数には負波数側の寄与を含める係数 2、Nyquist には 1 を使う。符号は既存の `-Re(ne_n conj(Ey_n))/B` 規約。これは E×B 輸送 proxy であり、粒子速度からの実粒子流束ではない。
3. n=1..64 の modal flux と全 Fourier mode の総流束を区別し、省略した高波数成分の残差も出す。全 mode の和が実空間の径方向平均 `-ne*Ey/B` と一致することを検証する。
4. 平均 modal cospectrum の誤差は定常統計の診断として残す。予測能力は平均する前の `[window, lead, band, mode]` の誤差、全流束の時間誤差、ne/Ey/phi の局所複素係数誤差で判定する。lead 別・window 別も保存する。
5. Skill=`1-SSE_model/SSE_copy`。0 を上回れば同じ対象で copy より良い。copy の SSE=0 は未定義とし、epsilon で人工的な合格を作らない。条件別の値と平均・中央値・最悪値・有効条件数を全て出す。
6. O は平均 cospectrum に基づく符号付き `Re(C)/sqrt(Pn Pe)`。truth の `|O|>0.05` かつ全 truth Gamma² の 0.001 を超える bin に限定する。O 比は truth Gamma² による逆 CDF 加重中央値とする。正確に累積重み 0.5 なら小さい側を採る。符号付き O 誤差、O 比の 1 からの誤差、符号反転の輸送重み、振幅誤差、mask coverage を併記する。負の O 比を「改善」と解釈しない。
7. n0–O の相関は補助的記述量。6 条件の全 720 ラベル順列による p 値も出すが、固定された運転条件での記述的検査であり、独立 PIC 反復に基づく一般的推論ではない。欠測を落として合格にしない。

## 次の学習：正規化を揃えた 2×2

| cell | 方位角 latent 幅 | hid_S | hid_T | N_T |
|---|---:|---:|---:|---:|
| A | 64 | 64 | 256 | 4 |
| B | 128 | 64 | 256 | 4 |
| C | 64 | 128 | 256 | 4 |
| D | 128 | 128 | 256 | 4 |

全 cell は translator を GroupNorm（8 groups）に統一し、train/eval の BN 統計差を除く。既存 BN-A と新 GroupNorm-B の差を解像度効果とは呼ばない。径方向 latent は 65 で固定し、loss・optimizer・データ・seed・epoch 数も揃える。

実機 forward/backward 検証で、旧 MidMetaNet は `drop_path=0` でも最初の block を 0.01 から補間することを確認した。v3 は `drop_path_schedule='zero_to_max'` を明示し、0 指定時は全 block を確実に無効化する。旧 checkpoint の再現は `legacy` schedule を維持する。この学習前の修正は `bundle.pre_smoke.json` を残して新 bundle に記録する。既存 A 再評価で実行したコードは `executed_source` に保存し、実行時の hash と照合する。

B と C は encoder latent の総要素数が等しいが、translator の内部要素数・パラメータ数・演算量・受容野・decoder の処理は一致しない。「完全な capacity 対照」ではない。2×2 により channel 幅ごとの解像度効果 B−A と D−C、および交互作用を報告する。学習特徴への literal Nyquist 定理の適用や、一つの差が出ないことによる全解像度仮説の棄却はしない。必要なら次段階で parameter/FLOP/受容野を別に揃えた対照を設計する。

- 学習 seed は 42,43,44 を固定する。同じ seed を cell 間で対応付ける。これは ML 初期化の反復であり、PIC realization の反復ではない。
- 主 checkpoint は全 cell・全 seed で **60 epoch 完了時の last.ckpt（格納 index 59）**。validation 最小や test の O が良い epoch を後から選ばない。保存済み best と固定 snapshot は二次的な学習軌跡として全て提示する。
- snapshot の指定は完了 epoch 5,10,15,20,30,40,50,60、ファイル名は内部 index 4,9,14,19,29,39,49,59。sanity validation では保存しない。
- 全 source-validation 窓について、条件×channel 別の正規化 MSE・物理相対 RMSE・copy skill・要素数、条件平均・中央値・最悪条件を毎 epoch 記録する。重複する全 validation 窓の学習ログと、固定 10 窓の物理診断はサンプル集合を明記して区別する。両者の差だけから selector 効果を断定しない。
- `prepare_radaz_v3.py` は **source-train のみ**の 24 窓で新しい局所積 loss の重みを校正し、4 config と 12 本の学習コマンドを作る。0.05 Gaussian 摂動で測った各 loss の「出力勾配ノルム比」の中央値を 5% とする規約であり、実際の parameter 勾配比や学習中の寄与率を 5% と主張しない。
- 主な比較は各条件の時間 modal-flux skill と ne/Ey/phi 複素係数 skill。各条件で seed を対応付けた差を保存し、seed 平均と全 seed 値を出す。6 条件平均で一条件の悪化を隠さず、最悪条件を併記する。小標本で有意差・機構同定を主張しない。
- B/C/D を旧 gate の閾値変更で許可する流れは廃止する。この新計画は固定された探索的 factorial。学習を開始する時点で全 cell を同じ規約で比較し、不都合な seed/cell を除外しない。数値破綻・資源上限の場合は欠測と理由を記録し、結果を見た一部再設計は別版とする。

## 物理的 branch とサンプリングの訂正

`f=n vE/Ly` は構造が電子ドリフト速度で剛体移流するという追加仮定の下での周波数であり、一般の ECDI/MTSI 波の固有周波数ではない。15 ns 出力の Nyquist は 33.33 MHz だが、この移流仮定でさえ B30 の n=1 は 26.04 MHz、B25 は 31.25 MHz と下回る。観測 22.9 MHz と仮定した 156.25 MHz の折り返しが近いことから真の周波数は一意に決まらない。

Hann 窓を掛けた径方向スペクトルの重心が 491 rad/m 未満でも、純方位角固有 mode の証明にはならない。空間尺度の違いは記述可能だが、ECDI/MTSI 同定と alias branch の確定は未完了。低 cadence の共動座標変換だけでも一意な折り返し次数は回復できない。新しい PIC では、対象周波数上限を先に定め、十分速い cadence の複素 mode 係数・局所流束を時間診断として出し、複数 cadence の整合と径方向固有関数・分散関係を確認する。全場を極端に高頻度で保存する必要はない。

## 確認実験と再現性

独立 PIC realization と、まだ開発に使用していない運転条件または将来時間区間を別に準備する。realization ごとに初期乱数・粒子 seed・restart 親・時間区間を記録し、restart の同一系列を独立標本に数えない。既存 realization での探索が終わった後に、正規化・checkpoint 規約・指標・成功条件を固定して確認用データを開く。今回のコード修正・再評価は、この新データによる再学習・確認実験の代わりにはならない。

`bundle.json` は文書だけでなく model/loss/evaluation/dataloader/callback/config と manifest の SHA256 を固定する。評価では checkpoint SHA256、データの path/size/mtime/shape/axes と実際に使った正規化入力・target の hash も保存する。H5 全体の hash と部分 fingerprint は区別する。コード変更後は新しい版として凍結し直す。

根拠：BN の train/eval 統計仕様は [PyTorch 2.4](https://docs.pytorch.org/docs/2.4/generated/torch.nn.BatchNorm2d.html)、小標本相関は [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)、評価集合での選択バイアスは [Cawley & Talbot](https://www.jmlr.org/papers/v11/cawley10a.html)、電子ドリフトと ECDI 周波数の関係は [ECDI の一次文献](https://arxiv.org/html/2503.03015v1)を参照。
