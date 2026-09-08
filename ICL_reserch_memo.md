
---

<!-- user-decisions-execution-20260909 -->

# 2026-09-09：利用者決定の実行（queue追加・掃除・アーカイブ・PIC入力準備開始）

利用者の決定：push保留／seed-43をD/P完了後に自動開始／`.tmp.driveupload`等は
gdrive同期失敗の残骸なので削除可／GPU作業は全部やる（順序は任せる）／
**HPCへは絶対に勝手にアクセスしない（恒久・全セッション共通）**／R3用PIC入力を
ローカルに別ファイルで作成（seed設定可能化・injectionのハードコード解消込み）／
closure-identificationのARCHIVED作成／統合済み重複メモはスタブ化。

実施済み：

1. **HPC非アクセス規則**をメモリとroot `AGENTS.md`冒頭のHard rulesに恒久保存。
   サーバー投入物はローカル作成のみ、アップロード・投入は利用者が行う。
2. **seed-43 D複製queue**：`SimVPv2/run_radaz_seed43_replicate.py`を独立プロセス
   （PID 54300）として起動。pilot完了を10分間隔で監視→凍結コマンドの`--seed 43`/
   新ex_nameのみ差し替えでD再学習→addendum凍結定義のnoise floor
   （`evaluate_radaz_seed43_noise_floor.py`、N=source_test 6条件の|gamma skill(D43 vs D42)|
   中央値）まで自動連鎖。再起動コマンド：`python run_radaz_seed43_replicate.py --run`。
3. **ディスク掃除**：`.tmp.driveupload`（56 GB）、`.tmp_pypdf_check`、codex checkpoint
   refs 3本を削除し`git gc --prune=now`。`.git`は12.54 GiB→69.9 MiB。SimVPv2の
   2月の可視化残骸（viz_*、165 MB、追跡済みだったのでgit rmでコミット）も削除。
   **ディスク空き15 GB→69 GB**。
4. **`ARCHIVED_nonlocal_closures.md`作成**：ブランチ名の由来である非局所クロージャ研究の
   終了記録。git履歴774b4a08から局所vs非局所のΔR²表を回収（HFx +0.024が最大改善、
   Pparp2 −0.054が最大悪化、9対象中改善4/悪化5→一貫した改善なし＝実質的な放棄理由）。
   復元コマンドと未解決事項（Box-Cox λ<0、4変種のマニフェスト欠如、時間的非局所性と
   T20/T26枠組みの接続）も記録。
5. **重複メモのスタブ化**：`ICL_research_memo_claude_2026-09-03_04.md`は統合確認済みの
   ため出典スタブへ縮約（全文は`git show 0154ab70:`で取得可能）。
6. **R3用PIC入力の準備**：PEPAPICのJulia内部（moment_tensorパーサー欠落、seed設定経路、
   injectionハードコード箇所、入力スキーマ）の調査を開始。調査完了後に
   (i) moment_tensor出力実装、(ii) JSON seedの導入、(iii) injectionの入力化、
   (iv) fresh native 22.5 kV/m realization用入力JSON作成を行う。**投入は利用者**。

GPU使用計画（概算、決定順序：pilot→seed43→per-input再計算→B/C再設計）：
D残り約6.7 h（20.2 min/epoch×20）→P約20–25 h→固定評価~0.5 h→seed43 D約20 h＋
noise floor ~0.5 h（ここまで自動）。その後factorial 5cell+§7分解のlast+per-input
再計算（推論のみ約1–2 h、wrapperはseed43学習中に準備）。B/C再設計は
2 cell×2 seed×60 epoch≈80 h で、D42/D43でsignatureのseed感度を先に確認し、
判別的null・新校正閾値を凍結してから開始する。
