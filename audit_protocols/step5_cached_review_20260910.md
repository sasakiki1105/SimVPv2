# Step 5: CPU review of already saved results (2026-09-10)

The user authorized GPU-free work during seed43 training. This is an exploratory
follow-up to already inspected results, not a new preregistered hypothesis test.
Do not modify original results, checkpoints, normalization, splits, or thresholds.
No model inference, H5 frame loading, training, or remote access is needed here.

Inputs: all 160 records and NPZ files of factorial_reaudit_20260909, with its
protocol hash and saved spectra hashes checked. Five cells (U-D, C-D, U-P, C-P,
U-Pv2), two policies (input/running), six source validation conditions, six source
test conditions, and four previously inspected holdout test conditions. Last
epoch index 59; ten disjoint 10-to-10 windows in each interval. Frame intervals
and local spectra definitions remain those of the original reanalysis protocol.
The complete D42/P42 pilot result and its spectra supply a separate descriptive
signature reference. They are different architectures/loss interventions from
the old five models and must not be pooled as extra seeds.

CPU budget: one numerical thread, BelowNormal process priority on Windows, one
NPZ at a time. Hide all CUDA devices before importing numerical libraries; never
import torch. Measure wall time, CPU time, and process peak working set. Read
roughly 334 MiB of existing factorial artifacts; write compact JSON and Markdown.

1. Reproduce the recorded time-resolved modal Gamma NRMSE/skills and all nine
   amplitude/organization/interaction Gram terms from saved spectra. Compare
   input vs running at identical weights, cases and windows. Compare the five
   named factorial contrasts by condition before aggregation. Preserve source
   validation, source test and known holdout as separate groups.
2. For every raw record define A=sqrt(Pn*Pe), r=abs(C)/A, delta=angle(C), using
   r=0 and delta=0 when C=0. Compute all eight combinations choosing each factor
   from prediction or truth. Use Gamma=-2*A*r*cos(delta)/B. Report NRMSE relative
   to truth for each combination, never call the single-factor errors additive
   shares. Evaluate both time-resolved spectra and spectra averaged over sample
   and lead BEFORE factorization, clearly separated. All-true must give zero;
   all-predicted must reconstruct the saved prediction. Report low/zero cross
   bins and explain that a zero cross spectrum has no identifiable phase.
3. Report sum(A_pred)/sum(A_true), signed weighted O metrics, and mean coherence
   for every case/policy. Report both time-resolved and ensemble definitions.
   Reuse the two existing source-validation-only amplitude calibration factors;
   do not refit. Recompute calibrated test NRMSE/skills and factorization, and
   compare calibrated vs raw per condition. No calibration scores on validation.
4. For the six source conditions compute descriptive Spearman rho between n0
   and the EXISTING v3 signed O_ratio_weighted_median; report O_signed_rmse as a
   separate diagnostic. No p values, significance, selected subsets, or gate
   decisions. These are local signed signatures, not the old sign-insensitive
   Gate A signature. D43 is pending, so no seed-stability conclusion yet.

The old section 7 used best checkpoints, normalized q-grid complex interpolation,
radial field pooling BEFORE products, and time-ensemble factors. Saved local
power/cross spectra cannot reconstruct the missing cross-node or cross-mode
products needed to reproduce it exactly. This task therefore does not reproduce
the historical 33-59% number under its original definition. The current matched
input/running contrast isolates inference policy; it cannot isolate the effects
of simultaneously changing checkpoint and observable definition relative to the
old table. Exact legacy replay would require extra inference/field spectra and
is a separate, more expensive archival task.

The remaining D42/D43 comparison requires a completed epoch-59 D43 checkpoint
and local signed O extraction; the existing noise evaluator only saves flux
skills/n0. Prepare the extraction specification now; execute after seed43/noise
completion and check consistency with its checkpoint hashes. B/C training stays
conditional on this review and later design decisions; never launch it here.
