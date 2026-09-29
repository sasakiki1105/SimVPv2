# 実装・技術確認（2026-09-30）

利用者指定の5量 ne,ni,phi,gamma_corr,u_ez を入出力するモデルを実装した。
4量版の凍結コードは変更していない。まず組合せ1案を60epoch、seed42/43で学習し、
入力別の寄与は後で調べる。過去10/未来10、source6条件と分割はPROTOCOL.md参照。

## 確認結果

- 新規8/8 unit tests PASS。H5速度の時刻・単位・周期端点除外・padding、
  未来ラベルと入力の分離、損失とgradient、4量版との初期共有予測一致、
  gamma/速度の直接帰還、非有限値拒否、実trainerの停止再開一致を確認。
- 実データGPU smokeはE10_B20 TRAIN [1400,1420)、3 optimizer steps。
  入力shape [1,10,7,260,256]（5量＋条件2）、出力 [1,10,5,260,256]。
  パラメータ13,293,445、GPU最大割当4180.67MiB。
- gamma/速度readoutのgradient L1は0.8543/0.2115で非ゼロ。
  損失・gradient・診断値は有限。
- 2block=20未来frame=300nsの自律推論で、5量すべての予測値が次入力と完全一致。
  E/Bは固定し、checkpoint保存復元後のrolloutも完全一致。
- source TRAINの速度global RMSは199848.84922945377 m/s、
  有効格子数631,603,200。ゼロoffset・無clip。VAL/TESTを使わず計算。
- RAM63.7GiB中34.7GiB空き、GPU8GiB、ローカルdisk198.2GiB空きを起動前に確認。

3step時点の直接gammaと場由来gammaの不整合は大きい。今回のsmokeは
経路の技術確認であり、予測性能・物理的整合・汎化を達成したという結果ではない。
smoke時間0.276秒/stepは初期化・gradient確認を含み、本学習ETAには使用しない。

## 再現と進捗

OpenSTL Pythonでtests、prepare、smoke、sealを順に実行。
詳細はworkdirs/2D_RadAz/radaz_transport_velocity_20260929/smoke.json。
bundle/contract/smokeのhashはPREPARED.sha256。bundleは親の凍結資産も照合する。
学習進捗は `python run_radaz_transport_velocity.py status`。
jobs/corr_uez42/、jobs/corr_uez43/にstatus.json、epochs.jsonl、last.ptを保存。
学習は逐次実行し、epoch末で保存・PAUSE_AFTER_EPOCHを確認する。
学習完了時のTEST評価・入力ablationの自動投入は行わない。
