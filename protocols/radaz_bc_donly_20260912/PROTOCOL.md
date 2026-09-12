# D-only B/C architecture intervention, 2026-09-12

Status: separate, prospective B/C computation on already inspected development data.
This protocol does not revive the old architecture Gate A or change the D/P primary endpoint.
The user authorized proceeding after the design review. No server access or remote push.

## Question, hypotheses and limits

Does changing azimuthal latent sampling improve time-resolved signed organisation
more than increasing encoder/decoder latent channel width? Are improvements useful
relative to simple input-only/AR prediction, and consistent across the two training runs?

The discriminating architecture null is C: extra channel width can perform as well
as, or better than, the B intervention. The forecasting nulls are O persistence,
input-history mean O, and condition-specific scalar AR(10). A model that merely
reproduces average spectra need not beat these time-resolved nulls. They are included
in the primary results, not introduced after seeing B/C.

B/C both succeeding does not uniquely establish capacity deficiency; both failing
does not uniquely establish temporal/mode-family deficiency. B changes sampling in
the encoder AND decoder; a literal Nyquist cutoff or translator-only mechanism is
not identified. C changes hid_S but not every internal width. Highest n0 is a single
operating condition, so an E10_B30 improvement is not universal high-n0 causality.

## Data, windows and leakage controls

Reuse exactly the six source cases and normalization of radaz_paired_pilot_v2/source_manifest.json:
E10_B20, E20_B20, E30_B20, E40_B20, E10_B10, E10_B30. Source paths, sizes and mtimes
are checked by the original 125-asset frozen pilot verification before each stage.
No condition holdout, R2 data, new PIC realization, server or new data transfer.

Array indices: train [0,1600), source_validation [1600,1800), source_test [1800,2000).
History 10 frames -> 10 future frames, sampling 15 ns, leads 15..150 ns. Windows
cannot cross split boundaries. Main evaluation pools ALL valid starts start+0..180:
181 windows per case/split, including all 20 phases of the old stride-20 lattice.
Physical times recurring in overlapping windows are repeated forecast-origin/lead
observations, not independent samples. Source_validation is secondary to source_test.
The old starts 0,20,...,180 are a labelled secondary reproduction only.

The source-test and these conditions have been used in development; no independent
confirmatory or causal population inference. A's old phase-0 results and all-phase
nulls informed this design. Null AR coefficients use late TRAIN [1200,1600) only,
which is shorter than the neural model's full train interval. O AR selects a single
ridge across all six validation conditions, minimizing their equal-weight average
truth-weighted O MSE on all 181 starts. Candidates 1e-6,1e-4,1e-2,1,100; chosen 1e-4.
Its predicted O is clipped to [-1,1] as fixed before the CPU audit. No test selection.
O input mean means averaging per-frame O, not taking O of mean input fields.
Gamma/full-flux AR diagnostics retain the earlier frozen train/validation fitted
scalar AR models and their earlier ridge choices; no new test fitting.

## Models and training budget

All models use float32, three normalized output fields ne/ni/phi, field MSE ONLY,
translator GN (8 groups), hid_T=256, N_T=4, N_S=4, kernel 3, batch 1, lr .001,
Adam/OneCycle as in the parent pilot, drop_path=0 with zero_to_max convention,
60 epochs, exactly 9486 train windows/epoch, 569160 optimizer steps. Optimizer,
scheduler and field normalization otherwise match the original D command.

| Cell | azimuth latent | hid_S | radial latent | parameters |
|---|---:|---:|---:|---:|
| A | 64 | 64 | 65 | 13,182,211 |
| B | 128 | 64 | 65 | 13,108,355 |
| C | 64 | 128 | 65 | 39,640,067 |

Reuse completed D42/D43 as historical A references, both with interruption histories.
Run NEW B42, C42, B43, C43. Same seed number does not make architectures' weight
tensors identical. B/C use an isolated torch.Generator with seed+1000003*epoch
to generate the full train-index permutation. Record its hash every epoch and the
sample-index mapping hash. This makes B/C order independent of model initialization
and epoch-boundary restart; it does not retroactively match A's unrecorded orders
or guarantee bit-identical GPU floating point / full RNG trajectories.

Last checkpoint at epoch index 59 is primary. Best and completed-epoch snapshots
5,10,15,20,30,40,50,60 remain diagnostics only. No checkpoint selection, early
stopping based on physical outcomes, loss-weight adjustment or A rerun is scheduled.
Failure stops the queue; explicit resume restores model/optimizer/scheduler from
the last completed checkpoint and retains an interruption record. Pause requests
take effect at the next completed training epoch or between stages.

## Observable definitions and primary reporting

Use periodic central-difference Ey, forward-normalized integer FFT modes 1..64,
four radial bands in [0.0009,0.0119] m, LOCAL ne*conj(Ey) products before radial
averaging. At each individual window/lead/band/mode, let

    Pn = mean_radial |ne_hat|^2; Pe = mean_radial |Ey_hat|^2
    C = mean_radial ne_hat*conj(Ey_hat)
    A = sqrt(Pn*Pe); O = Re(C)/A = r*cos(delta)
    Gamma = -2 Re(C)/B.

No temporal averaging before the main O error. Shared truth-only mask:
|O_true|>0.05 and Gamma_true^2 > .001 * sum_band,mode(Gamma_true^2) for EACH frame.
The numerical constants are inherited diagnostic bin definitions, not success
thresholds calibrated to the current outcome. Weight is raw Gamma_true^2 in retained
bins, then pool across all windows/leads/bands/modes. Do not renormalize each frame
to equal weight. Report coverage, invalid bins/frames and sign flips. Zero predicted
amplitude gives O_pred=0 and is separately counted, not silently excluded. Zero
total truth weight makes the endpoint undefined, never a pass; do not omit invalid
conditions to manufacture the six-condition aggregate.

Per case and training seed:

    E_O = sqrt(sum(w*(O_pred-O_true)^2)/sum(w))
    I_B = E_O(A)-E_O(B); I_C = E_O(A)-E_O(C)
    Delta_BC = I_B-I_C = E_O(C)-E_O(B).

Primary architecture scalar = mean over seeds of median over the six conditions of
Delta_BC on source_test. Positive favors B. Primary competence reporting accompanies
it: casewise E_O for A/B/C and each null, O skill=1-MSE_model/MSE_null for copy and AR,
and A−B/A−C casewise improvements. Ratios with zero null error are undefined. Report
both seeds separately and the number/direction of conditions; no cancellation of
inconsistent seeds into an unqualified success. No binary effect-size threshold,
p value, confidence interval, or use of D/P's N=0.0780 as an O calibration threshold.

This is three fixed architecture contrasts (B−A, C−A, B−C) x 2 seeds x 2 splits,
with six condition rows retained in each. Primary focus is B−C source_test with
null competence context. Secondary diagnostics are ten leads, 20-phase pool versus
fixed phase 0, n0 rank correlation of the DIRECT B/C improvement difference, mean
signed O ratios, Gamma/full-flux error and skill against copy/AR, amplitude, r,
wrapped phase discrepancy (1-cos phase error with phase-defined weight disclosed),
field and local complex-coefficient skill, and amplitude/O/interaction Gram terms.
All six conditions and both splits are kept; these diagnostics are not alternative
primary discoveries or implicit outcome-dependent gates.

The future-truth time-reversal fixture checks that a temporally wrong signal can
have correct averaged spectra; it is an oracle diagnostic, not a deployable null.

## Execution, resources and artifacts

CPU audit completed in 11.203 s. Its phase-0 E10_B30 O RMSE is D42 .16233,
D43 .18659 versus AR .10461, supporting a measurable target for intervention;
E40_B20 also trails AR despite low n0, so do not premise the experiment on a
monotone n0 deficit. GPU smoke uses only E10_B20 train frames [1400,1420), ephemeral
models and 7 optimizer steps (2 warmup + 5 timing), no saved research checkpoint.
Measured median step time A .125 s, B .156 s, C .266 s; measured peak allocated
memory A 3204.5 MiB, B 4104.6 MiB, C 6057.9 MiB. Actual full-run time depends on
loading, validation and I/O. The older 80 h estimate is superseded by approximately
120–150 h for four runs, updated from the first completed B epoch when available.

Before starting long training, test signed/zero/temporal-order fixtures, isolated
epoch permutation/restart behavior, config differences, source-train finite loss/
gradients and one registered all-phase A case through the new evaluator. The latter
is a partial evaluation, not a completed six-condition result, and is frozen here
before opening its new predictions. It does not select/tune B/C hyperparameters.

Run order: B42 train/evaluate -> C42 train/evaluate -> B43 train/evaluate -> C43
train/evaluate -> complete A42/A43 all-phase evaluation -> final aggregation.
The one-case A verification is reused. Evaluation can resume only matching case
files; input/target SHA matches old pilot phase-0 inputs and all-phase null truth.
Each case stores spectra/hash, lead metrics and field diagnostics; aggregate
verifies matched input hashes across architectures before comparing them.

Scripts: audit_radaz_bc_temporal.py, radaz_bc_metrics.py, train_radaz_bc.py,
evaluate_radaz_bc.py, run_radaz_bc.py. Configs are NEW bc_Donly_B/C files, not old
v3_B/C (which enable P). Output workdirs/2D_RadAz/radaz_bc_donly_20260912.
CPU audit/nulls and smoke: ../research_results/audits/bc_temporal_preflight_20260912.
Freeze code/config/null/parent/checkpoint hashes in bundle.json and commit locally
before the registered new evaluation and training. Local git provides versioning,
not an independently authenticated external timestamp; existing no-push instruction
is retained. The runner appends completion to ../ICL_reserch_memo.md automatically.

If the interventions do not improve, report that negative result. Further time-aware
losses, changed translator depth/width, independent PIC data or A fresh replicates
are separate decisions, not automatic jobs in this queue.
