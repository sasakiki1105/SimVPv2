# 入力H5化の実装・切替記録（2026-09-30）

5場 `[ne,ni,phi,gamma_corr,u_ez]` を事前計算し、条件別・split別のH5に保存した。
新ランナーは起動時にH5をRAMへ読み込み、各窓では切り出しとE/B条件連結だけを行う。
既存の科学定義、元コード31資産、モデル、損失、Adam、OneCycle、seed順序は維持した。

- 全TRAIN/VAL 10,800フレームで独立なsingleton再計算とbit一致。
  さらに84窓で旧10+10窓の入力・教師とbit一致。TESTは開いていない。
- 入力H5は12ファイル、約13.4 GiB。再開のたびにgammaを計算し直さない。
- 既存8件＋新規5件、13/13テストPASS。旧→新のepoch境界切替を含む。
- 元PID29176は10:13:33 JSTにepoch8のVAL・checkpointを保存して正常にpause。
  累計75,888更新を保持し、元checkpointを`transition/seed42_epoch8_original.pt`へ保存。
  SHA256: `cf02bc569153924510240b007e9a9e789d015eaab233652c2120eb91ca20787a`。
- 新PID50716、10:18:14 JST起動。epoch9の更新を確認。
  seed42の残り→seed43各60epochという元の順番を維持。
- 元bundle: `3036b7b6ff24cbeff5fbedeaf6fbdc617bad8303c5ec1d70bc00c61c66fc7ad2`。
  新runtime: `964c4df920bd0237d34b1f3c655aab07abe9a6fcb447648956e6203a722078c9`。

## GPU検証で判明した限界

通常GPU設定では、同じ旧方式を同じcheckpoint・RNGから二度実行しても、
3更新後の重みはbit一致しなかった（最大絶対差7.78e-5）。旧→新の差は4.66e-5。
初回forwardの各損失は完全一致し、以後のbackward/updateから差が生じた。
したがって本学習の最終重みが、旧ランナーを無中断実行した仮想結果とbit一致するとは主張しない。

検証専用プロセスで決定論的GPU演算を有効にすると、新旧の入力・教師・3 Adam更新後の
model/optimizer/scheduler/CPU・CUDA RNGがすべてbit一致した。
本学習のGPU演算設定は旧方式から変更していない。
これはデータ経路の等価性を調べる工学的検証であり、予測性能の評価ではない。

## 成果物・操作

以下は `workdirs/2D_RadAz/radaz_transport_velocity_20260929/` 内。

- `cache_amendment_20260930/inputs/`: 入力H5。
- 同ディレクトリの`cache_manifest.json`, `tests.json`, `tests.log`, `transition.json`,
  `gpu_nondeterminism_diagnostic.json`, `bundle.json`: 数値・復元・hash検証。
- `launch_cached_20260930_101814.json`: PID、起動時刻、実行ファイル、コマンド。
- `train_cached_20260930_101814.stdout.log` / `.stderr.log`: 新学習ログ。
- `jobs/corr_uez42/status.json`: 現在の進捗。epoch9以降はruntime hashも記録する。
- `cache_amendment_20260930/resume_report.json`: 再開後の実測速度と検証時の進捗。

進捗表示は従来の `python run_radaz_transport_velocity.py status` を継続利用できる。
以降の再開は `python run_radaz_transport_velocity_cached.py train` を使う。
epoch境界で停止する場合の`PAUSE_AFTER_EPOCH`とexecution lockは旧方式と共通。
学習終了後にTEST評価や別実験を自動起動する機能は追加していない。
