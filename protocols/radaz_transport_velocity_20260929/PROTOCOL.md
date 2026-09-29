# 速度と輸送を含む5チャネルを先行する実験（2026-09-29）

利用者が5チャネルを先に学習し、どの入力が効いたかは後で調べると指定した。
この版では組合せ1案だけを本学習する。既存3/4チャネル凍結版は保持し、
入力ablationや教師信号を揃えた対照は自動投入しない。

## 状態と正規化

- 入出力物理場は順に ne, ni, phi, gamma_corr, u_ez の5量。
- gamma_corr=-delta(nbar_e)*delta(Ebar_y)/Bx。保存フレーム内のy平均を除く。
  電場は同じfloat32入力を物理単位に戻したphiの周期中央差分。
  corrは帯域平均に寄与しない平均密度×電場を除くため事前に選択し、
  rawより良いという結果に基づく選択ではない。
- u_ezはH5 fields/electron_wd。3場と同じフレーム・格子・保存窓を使う。
  速度の平均を粒子束と呼ばず、目的変数を実粒子束へ変更しない。
- 3場affine、gamma TRAIN RMS、E/B条件尺度は4出力版から継承。
  速度だけsource6条件TRAIN全有効格子のglobal RMSを計算し、
  ゼロoffsetで割る。条件別・VAL/TESTでの再fitやclipはしない。
- 有効格子257x256。周期重複y端点を除き、径方向は260へ最終行複製。
  既知条件log_vE/log_n0は共通2チャネル＋FiLM。実入力7、出力5。

## データ・学習と優先順位

親manifest: workdirs/2D_RadAz/radaz_paired_pilot_v2/source_manifest.json。
sourceはE10_B20,E20_B20,E30_B20,E40_B20,E10_B10,E10_B30、各1 PIC realization。
TRAIN [0,1600)、VAL [1600,1800)、TEST [1800,2000)。保存間隔15ns、
過去10/未来10。境界を跨がない全起点を使用し、TRAIN9486、VAL1086窓。
source TESTは既読の探索データであり、新しく独立な確認データではない。
prepare/smoke/trainはTESTや未学習E/Bを開かない。

既存B構造（hid_S64/hid_T256/NS4/NT4/gSTA/GroupNorm8/方位downsample2、
DropPath0）、Adam lr0.001・weight_decay0、OneCycle pct_start0.1、
batch1、float32、60epoch、seed42→43を逐次実行。学習は最初から。
各epochの順序は親の独立generatorを継承。追加入力重みと出力行はゼロ初期化し、
共有場の初期予測を4出力版と一致させる。

L=MSE_fields + MSE_gamma/3 + MSE_velocity/3。
field係数は親と同じ、gamma/velocityは有効格子だけ。全項を別に記録する。
損失重みをTESTで選ばず、他構成とtotal lossを比較しない。
速度入力と速度教師信号を同時に増やすため、この版だけから入力別寄与を主張しない。
終端epoch60を使用しVALでcheckpointを選ばない。
epochごとにmodel/optimizer/scheduler/RNGをatomic保存。
PAUSE_AFTER_EPOCHで安全に止め、再開時はそれらを復元する。
epoch途中の強制終了ではそのepochを先頭から再実行する。

## 評価と自律予測

主指標は直接予測gammaの4径帯域Gamma_fullを未来10frameで平均した誤差。
持続・source TRAIN平均・条件別AR10・共通AR10+E/Bを親から継承し比較。
条件を同重みにしたskill中央値、全条件個別値、seed別値を記録する。
副指標は場由来Gamma_full/T10、密度・電位・速度の誤差、
直接gammaと場から再計算したgammaの不整合。スペクトルpn/pe/crossも保存し、
積が合うだけで振幅・位相の再現を達成したとはしない。
直接gamma画像のFFTからT10を作らない。T10は密度–電場cross-spectrumからのみ計算。
合格閾値を新設しない。2学習seedを独立PIC realizationと数えない。
今回の新しい学習候補は1構成x2seed。過去の探索履歴は消えない。

自律推論は初期真値10frameだけを使い、予測した5量すべてを次入力へ返す。
gamma再計算による置換、真の未来速度、clip、再正規化は使わない。
無効paddingだけ複製し、E/Bは固定。非有限値は失敗として停止。
学習は1ブロックのままなので、5出力化は長時間安定性の証明ではない。
rollout CLIはsourceの指定split内、初期履歴1個からの記述的診断。
評価は明示的なevaluate/rolloutコマンドで行い、学習完了時のTEST自動開封はしない。
未学習E/Bは明示manifest・固定source尺度で評価できるが今回投入しない。

## 再現コマンド

SimVPv2内、OpenSTL Python、KMP_DUPLICATE_LIB_OK=TRUE。

    python -m unittest discover -s tests -p test_radaz_transport_velocity.py -v
    python run_radaz_transport_velocity.py prepare
    python run_radaz_transport_velocity.py smoke
    python run_radaz_transport_velocity.py seal
    python run_radaz_transport_velocity.py train
    python run_radaz_transport_velocity.py status
    python run_radaz_transport_velocity.py evaluate --arm corr_uez --seed 42 --split test
    python run_radaz_transport_velocity.py rollout --arm corr_uez --seed 42 --split test --blocks 4

成果物: workdirs/2D_RadAz/radaz_transport_velocity_20260929/。
学習開始前にローカルcommitとbundle SHA256を記録する。これは版・バイト同一性の
記録であり、外部タイムスタンプや独立確認を意味しない。remote操作なし。
