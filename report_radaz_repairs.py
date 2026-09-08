"""Generate the Japanese correction report from the actual v3 results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "workdirs/2D_RadAz/radaz_corrected_A_v3"


def main():
    results = json.loads((OUT / "results.json").read_text())
    text = """# A 周辺の修正結果（2026-09-08）

コード・評価・実験計画・研究メモを修正し、既存 A の best/last を再評価した。
正規化方法の変更で誤差は大きく改善するが、全条件での時間輸送予測は解決していない。
旧 A2 STOP は歴史的判定として保存し、修正版を用いて旧実験を後から PASS に変更していない。

## 実装した修正

| 指摘 | 修正 |
|---|---|
| BN の推論統計が結果を支配 | 入力窓ごとの独立正規化と source-train 入力のみの校正を実装。既存重みについて全 3 規約を併記。次の全 cell は同じ GroupNorm を使用 |
| 学習・評価モデルの設定不一致 | 共通 factory に統一。B の方位角 downsample を含む全設定を渡し、A/B/C の実 config で state の完全一致を検証 |
| O が加重中央値になっていない | truth Gamma² による逆 CDF 加重中央値へ修正。旧無加重版は明示的 legacy replay のみ |
| 絶対値 O 誤差が符号反転を隠す | 符号付き O 誤差・O 比の 1 からの距離・符号反転重み・mask coverage・振幅誤差を別々に保存 |
| 平均スペクトルだけでは予測能力を検証できない | 各時刻の modal flux、全流束、ne/Ey/phi 複素係数を copy と比較。lead/window 別を保存 |
| 径方向平均した場の積を局所流束と解釈 | 局所積を band 内で平均する loss/evaluator を実装。片側 FFT の係数、全 mode の総流束、n>64 の残差を明示 |
| 全体 validation loss で条件差を見落とす | 毎 epoch の条件×channel ログ、条件均等平均・中央値・最悪条件を追加。新実験の主 checkpoint は全 cell 共通の完了 epoch 60 に固定 |
| C を完全な容量対照と解釈 | 解像度×encoder channel の 2×2（A/B/C/D）に整理し、内部容量・受容野・decoder の差を明記。3 training seed を固定 |
| alias と径方向 branch の断定 | メモ本文と解析スクリプトを訂正。電子移流仮定と波の固有周波数、windowed 重心と固有 mode を区別 |
| 6 条件を独立反復と解釈、小標本 p 値 | 全 720 順列の記述的相関を追加。PIC realization と training seed を区別。未使用データによる確認実験を別途定義 |
| 文書 hash だけでは実装を固定できない | code/config/manifest/checkpoint/使用入力の fingerprint を保存。実行コードを executed_source に保存 |
| 旧 loss 重み調整が source-test を使用 | **追加発見**。旧コードは 1800–1819 を読み込んでいた。新しい局所積 loss の重みを source-train の 24 窓だけで再計算。旧結果は未使用 test と呼ばない |
| ゼロ振幅で cross loss の勾配が NaN になり得る | sqrt の前で product を clamp し、完全ゼロ予測でも有限勾配となることを検証 |
| 60 epoch の snapshot が保存されない | 完了 epoch と内部 index を明示。新規指定の 60 は index 59 で保存。sanity validation は除外 |
| drop_path=0 が完全な無効化にならない | **実機検証で追加発見**。v3 の schedule は 0 から設定値までとし、0 指定を尊重。旧 checkpoint の履歴再現は legacy を維持 |

旧データに基づく探索であり、新しい独立実験による原因の確証ではない。
新計画・コードの対応は [RADAZ_V3_PROTOCOL.md](../../../RADAZ_V3_PROTOCOL.md)、
元の監査は [REPORT.md](../../../../research_results/audits/radaz_A_2026-09-08/REPORT.md) を参照。

## 再評価結果

各値は **6 source 条件の中央値**。同じ 10 窓×10 出力を全規約と copy で比較した。
Gamma は局所積による n=1..64 の modal flux。時間評価では時間平均前の配列を使う。
Skill=1−SSE_model/SSE_copy、正が copy より良い。Skill の中央値は NRMSE の中央値から計算した比ではない。

"""
    labels = {"copy": "copy", "best/running": "best / 保存 BN", "best/input": "best / 入力ごと",
              "best/source_train_calibrated": "best / train 校正 BN", "last/running": "last / 保存 BN",
              "last/input": "last / 入力ごと", "last/source_train_calibrated": "last / train 校正 BN"}
    for split, title in (("source_validation", "Source-validation（1600–1799）"), ("source_test", "Source-test（1800–1999、開発使用済み）")):
        text += f"### {title}\n\n"
        text += "| 規約 | field 相対誤差 | 平均 Gamma NRMSE | 時間 Gamma NRMSE | 時間 Gamma skill | copy 超えの条件数 |\n|---|---:|---:|---:|---:|---:|\n"
        for variant, row in results["evaluations"][split].items():
            s = row["summary"]
            vals = [s[k]["median"] for k in ("field_aggregate", "mean_modal_Gamma_nrmse", "gamma_time_nrmse", "gamma_skill_vs_copy")]
            count = s["gamma_skill_vs_copy"]["conditions_beating_copy"]
            text += "| " + labels[variant] + " | " + " | ".join(f"{v:.6f}" for v in vals) + f" | {count}/6 |\n"
        text += "\n"
    text += """同じ last の重みで source-test の時間 Gamma NRMSE 中央値は 0.706183 → 0.148532 へ下がる。
しかし copy 超えは 3/6 条件であり、E10_B10・E10_B30・E40_B20 では上回らない。
source-validation と source-test で改善条件数も異なる。保存 BN の均等校正だけでは入力ごとの正規化ほど改善しなかった。
BN は重大な交絡である一方、単独の修正で失敗が全て消えるという証拠はない。

時間平均 Gamma では copy 自体が非常に強い一方、ne/Ey/phi の複素係数 forecast skill は入力ごとの正規化で高い。
これらは異なる予測対象であり、場・複素係数の改善から輸送時系列も成功したとは推論しない。
条件別・lead 別・window 別の全数値は [results.json](results.json)、各時刻の spectra は同じフォルダの NPZ に保存した。

## 次の実験の準備

`radaz_arch_v3_plan/bundle.json` に 4 config×3 training seed の 12 コマンドを保存。
全 cell の GroupNorm、局所積 loss、学習条件、終端 checkpoint 規約を統一した。
新 loss 重みは power=3.2805313248982754e-6、cross=1.96292373867782e-5。
これは source-train の Gaussian 摂動に対する **出力勾配**の中央値を各 5% にする規約であり、実学習の parameter 勾配比ではない。

本格的な 12 本の学習および新 PIC の生成は実行していない。
独立 realization・未使用条件での確認は実験として残る。ソフトウェアの修正によって既存データを未使用データに戻すことはできない。

## 検証

- 11 件の回帰テスト：符号、加重中央値、Parseval、時間順序、NumPy と学習 loss の一致、ゼロ予測勾配、BN の batch/order 独立性・非破壊性、A/B/C の checkpoint、source-only frame guard、条件別ログ、snapshot、provenance。
- 新 A/B/C/D の全てで実 source-train フレームを使った forward/backward：parameter 勾配が有限、出力 shape 正常、train/eval 出力一致。optimizer 更新と checkpoint 保存は行っていない。
- 既存 A の全 84 条件・規約・split の組合せを再評価。best/last の重みは変更していない。

実行例（SimVPv2 を作業ディレクトリとし、OpenSTL 環境を使う）：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:PYTHONPATH='.'
python -m unittest discover -s tests -p test_radaz_corrections.py -v
python tests/smoke_radaz_v3_training.py
python evaluate_radaz_corrected_A.py --output workdirs/2D_RadAz/radaz_corrected_A_v3_repeat
```

評価済みフォルダは上書きしない。現在のコードで再実行する場合は別フォルダを指定する。
歴史的指標の再現には明示的な `--replay-legacy` が必要で、旧結果と別名に出力する。
"""
    (OUT / "REPAIR_REPORT.md").write_text(text, encoding="utf-8")
    memo = ROOT.parent / "ICL_reserch_memo.md"
    marker = "# 2026-09-08：A 周辺の実装修正と再評価（監査対応完了）"
    current = memo.read_text(encoding="utf-8")
    if marker not in current:
        current += "\n\n---\n\n" + marker + "\n\n" + """評価・学習設定・統計処理・物理的解釈を修正した。詳細は
[修正結果](SimVPv2/workdirs/2D_RadAz/radaz_corrected_A_v3/REPAIR_REPORT.md) と
[次の実験計画](SimVPv2/RADAZ_V3_PROTOCOL.md) に記録した。

旧 A2 STOP は旧定義に対する記録として維持する。checkpoint 軌跡から「学習された物理構造の再分配」を
単独原因と断定する記述も弱める。重みと BN buffer が同時に変わっており、今回の固定重み介入で
推論統計だけでも失敗形状が大きく変化することが確認されたためである。

主な修正は、モデル構築の共通化、局所積による輸送 loss、加重中央値と符号評価、時間別の copy 比較、
条件別 validation ログ、GroupNorm を揃えた A/B/C/D×3 seed、完了 epoch 60 の固定終端評価である。
旧 gradient calibration が source-test を使用していた追加の問題を修正し、source-train 24 窓で再計算した。
ゼロ振幅での NaN 勾配、snapshot の epoch index、drop_path=0 が完全に無効でなかった挙動も修正した。

修正版の局所・時間別 Gamma NRMSE 中央値（source-test）は、同じ last 重みで保存 BN の 0.706183 から
入力ごとの正規化では 0.148532 に低下した。copy は 0.157431 だが、条件別に copy を上回るのは **3/6**。
全面的な物理予測の成功とはしない。best/last・3 推論規約・copy を両 split で全て報告した。
旧 A1/A2 の閾値をこの新定義へ当て直して PASS に変える操作はしていない。

11 件の回帰テストと新 4 cell の実データ forward/backward が完了した。12 本の本学習と独立 PIC の生成は未実行。
既存データは探索用であり、未使用条件・独立 realization での確認は次の実験として残る。
"""
        memo.write_text(current, encoding="utf-8")
    print("Written", OUT / "REPAIR_REPORT.md")


if __name__ == "__main__":
    main()
