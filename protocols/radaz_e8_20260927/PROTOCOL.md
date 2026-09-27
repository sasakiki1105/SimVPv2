# E8: matched continuation with a time-resolved cross-phase objective, 2026-09-27

Status: prospective matched comparison on already inspected development data (source6, one PIC
realization per condition). Gated by the frozen 2026-09-16 post-B/C plan ("field prediction succeeds
but time-resolved O loses to copy/AR"), which the completed B/C aggregate, the Gram decomposition
(organisation term 51–94 % of transport SSE) and few-mode Step A (error concentrated on the n≤16
transport-carrying modes, identical across A/B/C) satisfied. No frozen B/C, D/P or v2 file is modified.
User decisions (2026-09-27): backbone = B; three arms; adapters later under the same B backbone.

## Question and discriminating null

Does replacing the time-averaged organisation constraint by a **time-resolved** one, with everything
else held fixed (starting weights, data order, update count, optimiser, LR), reduce the time-resolved
signed-O error of the transport-carrying modes? The null is **CTRL**: continuation of the same weights
with field MSE only. **XA** (time-averaged cross term, the P-type constraint alone) separates "having a
cross term" from "having it per frame".

| arm | loss | role |
|---|---|---|
| CTRL | field MSE | null: drift of continuation itself |
| XT | field MSE + lambda_xt · cross_time (per-frame normalised complex cross, n=1..32) | primary test |
| XA | field MSE + lambda_xa · cross_avg (T-averaged, identical to P's cross term) | separates per-frame from cross-term |

## Fixed conditions

- Start: `radaz_bc_Donly_B_GN_seed{42,43}_60ep_20260912/checkpoints/last.ckpt` (epoch index 59),
  **weights only** (model.* tensors). No optimizer/scheduler state is restored; both arms start the
  continuation identically.
- Config `configs/custom/pepapic/SimVP_gSTA_radaz_e8_B_10ep.py` = B config except: epoch 10,
  lr 1e-4, sched step with decay_epoch 100 (constant LR for the run, warmup 0), a snapshot at completed
  epoch 5 (the completed-10 state is `last.ckpt`), cross max_mode 32. Batch 1, GN, float32, D-only data, same source manifest.
- Data order: the B/C isolated per-epoch permutation generator with epoch index offset **60**
  (permutation seeds 60..69), identical across arms and never overlapping B/C's 0..59.
- Update count: 10 × 9,486 = 94,860 steps per run, all arms.
- Loss module: `openstl/methods/pepapic_spectral_loss_e8.py` (subclass; cross term only; per-frame
  normalisation `sqrt(Pn_t Pe_t)` and per-frame truth-power mask kappa = 0.001 for XT; parent's
  T-average for XA). Injected at run time without registering it as a submodule, so E8 checkpoints
  keep exactly the B/C state_dict structure.
- lambda: calibrated once on TRAIN (`calibrate_radaz_e8.py`): the first 200 windows of permutation
  epoch 60, seed 42, evaluated with the B42 and B43 terminal models; lambda_arm = 0.10 ×
  mean(field MSE) / mean(cross_arm) pooled over both models. Recorded in `calibration.json`; the same
  lambda is used for both seeds. Not tuned on any evaluation split.
- Primary checkpoint: `last.ckpt` after epoch index 9. Snapshot 5 is diagnostic only. No selection.
- Pause takes effect at the next completed epoch; failure stops the queue; explicit resume restores a
  partial E8 run from its own last checkpoint (Lightning), never re-loads the source weights.

## Evaluation (frozen)

`evaluate_radaz_e8.py` reuses the B/C evaluator definitions unchanged (181 all-phase starts per case,
source_validation and source_test, six source cases, truth-only mask |O|>0.05 and Γ²>0.001·Σ, Γ²
weights, copy/input-mean/AR(10) nulls, spectra NPZ in the B/C format). E8 rows are checked against the
same input/target hashes as B/C.

Per seed and split, for each condition: `CTRL−XT = E_O(CTRL) − E_O(XT)`, `CTRL−XA`, `XA−XT` on the
time-resolved signed-O RMSE. **Primary scalar** = mean over seeds of the median over the six
conditions of `CTRL−XT` on source_test; positive favours XT. Direction hypothesis: positive.
Reported alongside, never collapsed: per-condition signs and counts, per-seed values, O skill vs copy
and AR, **field skill vs copy** (safeguard; no threshold, value and change reported), Γ skill vs AR,
source_validation, the Gram O-term share and the inside-T10 share (Step A reading applied to E8
outputs), the three losing conditions (E10_B20, E10_B30, E40_B20), snapshot-5 vs last.

No effect-size threshold, p value, interval, or reuse of any earlier calibration scale as a gate.
Six conditions are one realization each; windows/leads are not independent samples.

## Pre-stated reading

| observation | supports | next |
|---|---|---|
| XT lowers O; field skill change small | objective was the limiting factor for this backbone | fix the loss, then condition transfer |
| XT lowers O; field skill clearly worse | competing objectives; not a rescue | re-weighting is a separate protocol |
| XT ≈ CTRL | this backbone cannot extract per-frame cross-phase from 10-frame history | state insufficiency: Step B / tensor line |
| XA ≈ CTRL and XT > XA | per-frame resolution matters, not the cross term itself | — |

## Leakage and multiplicity

lambda from TRAIN only; mode range n=1..32 fixed a priori; source_test untouched until evaluation;
contrasts 3 × 2 seeds × 2 splits, all reported. Continuation is not bit-deterministic
(`deterministic=False`); the two seeds bound that. Interruption history is recorded per job.

## Resources and artifacts

B cell ≈ 25.5 min/epoch → 6 runs × 10 epochs ≈ 25.5 h plus evaluations (~1 h). Checkpoints
≈ 157 MB × 4 per run ≈ 3.8 GB, evaluation NPZ ≈ 1.8 GB. **Free disk at freeze: ~9.8 GB**; the runner
stops below 3 GiB free. Display stays on the iGPU.

Order: CTRL42 → XT42 → XA42 → CTRL43 → XT43 → XA43, each followed by its evaluation; then aggregate.
Scripts: `radaz_e8_experiment.py`, `train_radaz_e8.py`, `calibrate_radaz_e8.py`, `evaluate_radaz_e8.py`,
`run_radaz_e8.py`, `freeze_radaz_e8.py`, `tests/test_radaz_e8.py`. Output
`workdirs/2D_RadAz/radaz_e8_20260927/`. Hashes in `bundle.json`; local commit before launch is
versioning, not an external timestamp.
