# 2026-09-09: exploratory matched-last factorial reanalysis

This protocol is fixed before this reanalysis, after inspecting the old results.
It is NOT an independent confirmatory experiment. No old gate is reopened.
The accompanying JSON freezes code, configurations, checkpoints and data metadata.
Commit the protocol, JSON and new evaluator before running inference.

Question: do the old conditioning/physics-loss comparisons survive a matched last
checkpoint and per-input normalization? Discriminating nulls are persistence and
the matched U-D cell. A physics-loss benefit must survive comparison with D;
an amplitude-only explanation is probed by a fixed validation-fitted rescaling.

Models: U-D, C-D, U-P, C-P and U-Pv2; ONLY last.ckpt, epoch index 59.
Original training/configurations are preserved. U-P/C-P used the old q loss and
U-Pv2 used the old integer loss; these are NOT the new local-loss P pilot.
Primary inference: batch_input on the translator (per example/channel spatial
statistics). Sensitivity: original stored BatchNorm at the SAME checkpoint.
All run on CPU, float32 inference, two intra-op threads, one inter-op thread;
GPU is reserved for D/P. No BN calibration or test-time optimizer updates.

Data: the existing axis-factorial manifest, source E10/20/30/40_B20 and E10_B10/B30,
old holdouts E22.5_B20, E25_B20, E10_B15 and E10_B25. All have already been seen.
Use the old source TRAIN affine normalization and condition normalization.
Source validation [1600,1800), source test and old holdout test [1800,2000).
Each interval uses ten disjoint 10-to-10 windows, stride 20, phase 0, dt=15ns.
Phase 0 is explicitly a MATCHED LEGACY-WINDOW diagnostic, not an all-origin
population claim. No window/condition is selected after seeing its score.
Changing the origin grid is a separate analysis; no CI is computed here.

Observables: radaz-local-signed-time-v3.1, valid x=257, y=256, four radial bands
0.9..11.9mm, local Fourier products BEFORE radial pooling, central difference Ey,
integer modes 1..64, Gamma_n=-2 Re(ne_n Ey_n*)/B, full flux includes all modes.
Report excluded flux. Gamma is an ExB transport proxy, not measured particle current.

Primary descriptive endpoints: median per-condition gamma_time_nrmse and
gamma_skill_vs_copy, separately for six source-test and four old holdout-test
conditions. Also report each condition, mean, min/max, full-flux skills, signed O,
log-amplitude error, normalized field MSE and complex-coefficient skills.
Denominator-zero scores are null, never a pass. Include SI RMSE and zero-relative
normalization to reveal cases with nearly exact persistence.

Decomposition: for each sample/time/band/mode, A=sqrt(Pn Pe), O=Re(C)/A (zero
when A=0). Resolve delta Gamma into amplitude, organization and interaction
terms exactly; retain all nine Gram contributions, including cancellations.
These are algebraic attribution diagnostics, not separately observed mechanisms.

Secondary calibration: for EACH model/inference policy fit two GLOBAL positive
amplitude factors on SIX SOURCE VALIDATION conditions only, using modes 1..64.
For each field q in {ne,Ey}, minimize the equal-condition sum
  sum_c [ sum_bins (a_q sqrt(Pq_pred)-sqrt(Pq_true))^2 / sum_bins Pq_true ].
Closed-form nonnegative slope through the origin, no clipping/hyperparameter grid.
Freeze the two coefficients before opening that variant's test intervals.
Apply Pn*=a_ne^2, Pe*=a_Ey^2, C/Gamma/flux*=a_ne*a_Ey and corresponding
complex coefficients. O and phase are unchanged; this is a spectral diagnostic,
not a retrained model nor a corrected full-field forecast. No test-specific fits.
Validation power-fit sufficient statistics are saved as fitting diagnostics;
calibrated forecast scores are reported only on the two test groups.

Factorial contrasts are computed PER CONDITION before taking the median/mean:
conditioning=[(CD-UD)+(CP-UP)]/2, physics=[(UP-UD)+(CP-CD)]/2,
interaction=CP-CD-UP+UD; U-Pv2-U-D and U-Pv2-U-P are separate contrasts.
For skill, positive favors the first effect; for NRMSE, negative favors it.
Report BOTH policies and both raw/calibrated variants, without choosing a winner.

Multiplicity: 5 models x 2 inference policies x (6 validation + 6 test + 4
holdout) = 160 raw condition rows, 100 calibrated test counterparts; copy is shared
across models/policies. Five named contrasts x two endpoints x two inference
policies x two calibration states x two test groups = 80 descriptive aggregate
contrasts. Other diagnostics are explicitly secondary; no p-values or discovery
counts, no new model/seed/PIC realization, and no superiority claim from seed one.

Execution is resumable from per-case NPZ/JSON artifacts ONLY if their protocol
SHA256 matches. Input/target hashes verify identical windows across models and
policies. Partial files/status never mean complete; missing model/case is an error.
The final REPORT and memo append identify exploratory status and limitations.
