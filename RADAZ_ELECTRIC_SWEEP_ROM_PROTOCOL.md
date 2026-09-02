# RadAz electric-field sweep ROM protocol

## Scientific objective

Construct a reduced-order model that predicts the relaxation at fixed `Bx = 20 mT` after changing `Ez` to `22.5 kV/m`, including dependence on whether the initial state came from `20 kV/m` or `25 kV/m`. Compare a data-only ROM and a physics-informed ROM under an identical prospective split.

## Stage 1 result: smooth parameter interpolation rejected

The state is shared across conditions:

- `L`: 20 frozen-SimVP Fourier latent coordinates.
- `R`: 8 radial phi-envelope observables (4 radial regions x 2 mode bands).
- `T`: 2 modal transport observables.
- Total dimension: 30.

The common representation was fitted using only stationary E20/E30 data. E25 was then treated as an unseen interpolation condition over 24--30 us. No E22.5 data were loaded.

| model | E25 state skill vs persistence | E25 transport skill vs persistence | decision |
|---|---:|---:|---|
| Mixed local Hankel experts | -3.443 | -2.036 | reject |
| Parameter-conditioned neural delay ROM | -8.082 | -2.912 | reject |
| Oracle local E25 Hankel ROM | +0.719 | +0.703 | local closure supported |

The oracle uses E25 transitions and is only an upper-bound diagnostic. Its success in the same E20/E30-fitted representation shows that the common coordinates can express E25 dynamics. The failure is therefore associated with inferring the E25 vector field from E20/E30, not simply with a missing E25 representation.

This result does not prove a bifurcation. It does reject, for the tested state and forecast protocol, the hypothesis that the E25 propagator is obtained by a smooth mixture learned from E20/E30. Phase cancellation between condition-dependent oscillatory propagators contributes to the linear-expert failure, while the nonlinear failure shows that changing the mixer alone is insufficient.

## Stage 2: regime-aware sweep ROM

The next model is trained to represent the non-autonomous transition rather than infer the complete E25 dynamics from stationary endpoints.

### Prospective split

1. Training conditions:
   - stationary E20, E25, and E30;
   - `E20 -> E22.5`, 30--35 us.
2. Development validation:
   - `E20 -> E22.5`, 35--40 us.
3. Locked primary test:
   - the complete `E25 -> E22.5` trajectory now being generated.
4. The primary test must not be used for normalization, representation fitting, early stopping, architecture selection, physics-loss weights, or forecast-horizon selection.

### Dynamics

Use a recurrent state model with the known control history as input:

```text
z[k+1] = z[k] + F_theta(z[k-H:k], Ez[k-H:k])
```

The GRU/delay state carries the branch memory after the field step; no future PIC value is supplied. A regime gate may be added, but it must remain soft and must be selected using only the up-sweep validation.

The rejected smooth-interpolation models remain frozen negative baselines. They are not retuned on E22.5.

### Stage 2 development result (2026-08-10)

The first implementation was initialized from the stationary E25 local neural
ROM and updated with `E20 -> E22.5` over 30.165--34.995 us.  The continuation
over 35.010--39.855 us was a free development rollout.  Stationary E20/E30
features were not mixed into this implementation because their existing saved
features use a different normalization; doing so would violate the common-state
assumption.  The E20-side history is present in the observed beginning of the
up-transition trajectory.

| training rollout | state skill | radial skill | total transport skill | MTSI transport skill | ECDI transport skill |
|---|---:|---:|---:|---:|---:|
| 0.60 us | +0.179 | -0.146 | +0.099 | +0.105 | -4.338 |
| 2.40 us | +0.254 | +0.246 | +0.173 | +0.177 | -2.481 |

The 2.40-us model passes the original aggregate Stage-2 minimum gate, and is a
large improvement over the fixed E25 Hankel and E25-pretrained neural
baselines.  It is nevertheless **not promoted for the branch-sensitive blind
test**, because ECDI-band transport remains worse than persistence.  This
stricter mode-resolved gate is appropriate for the ECDI/MTSI switching
objective.  The locked primary `E25 -> E22.5` data remain uninspected.

### Stage 2b direct and mode-factorized result (2026-08-10)

A 212-dimensional direct physical state was then tested:

- 20 frozen-SimVP latent coordinates;
- 168 global physical Fourier coefficients (`phi`, `ne`, `ni`, `Ey`,
  `n=1--21`, real/imaginary);
- 8 radial phi envelopes;
- 16 radial density--Ey cross-spectrum components.

The single direct-state propagator reversed the previous failure pattern:
ECDI transport became positive (+0.111) and `Ey` skill was +0.221, while MTSI
transport fell to -0.679.  This complementarity motivated a fixed
mode-factorized ROM:

| branch | source model | retained role |
|---|---|---|
| low/MTSI | 30D regime-aware ROM | radial envelopes and MTSI transport |
| high/ECDI | 212D direct physical-state ROM | n=7--21 fields, ECDI state and transport; n=2 diagnostic |

The locked factorized development result is:

| composite state | radial | total transport | MTSI transport | ECDI transport | selected phi | selected Ey |
|---:|---:|---:|---:|---:|---:|---:|
| +0.305 | +0.246 | +0.177 | +0.177 | +0.111 | +0.113 | +0.223 |

All entries are skills versus persistence.  This model passes the
mode-resolved data-only gate and is locked as `READY_FOR_PHYSICS_ABLATION`.
It is not yet promoted to the primary down-history test because the subsequent
physical-field gate remains under development.

## Stage 3: physics-informed ROM

Start from the accepted Stage 2 architecture and data split. Add losses one at a time:

```text
L = L_data
  + lambda_roll L_rollout
  + lambda_E L_(E + grad(phi))
  + lambda_P L_Poisson_floor_hinge
  + lambda_C L_continuity_floor_hinge
```

Order of introduction:

1. `E = -grad(phi)` consistency.
2. Poisson residual using a truth-calibrated floor/hinge, rather than forcing noisy PIC output to zero residual.
3. Continuity/charge conservation only after auditing ionization sources, boundary fluxes, deposition noise, and the discrete derivative stencil on PIC truth.

Every physics weight is selected on the up-sweep development validation. The down-sweep remains untouched until all weights and stopping rules are locked.

### Stage 3a field-gradient ablation result (2026-08-10)

The first physics ablation attached a physical Fourier decoder to the same
30-dimensional Stage-2 state and imposed the spectral relation
`Ey_n + i*k_n*phi_n = 0` with a truth-floor hinge.  On PIC truth, the audited
residual was only 0.409% of `Ey` RMS overall (0.345% for MTSI modes and 1.472%
for ECDI modes), confirming the sign and wave-number convention.

All candidates used identical initialization, minibatch order, data split, and
stopping rule.  Relative to the paired `lambda_E=0` control, `lambda_E=1`
reduced the field-gradient excess hinge by 58.2%, while leaving the 30D state
metrics nearly unchanged.  However, the jointly fine-tuned models had negative
total-transport skill (about -0.092), and the free-rollout physical decoder had
`phi`/`Ey` NRMSE of about 1.007/1.004.  The residual reduction therefore does
not yet constitute a useful field forecast and the Stage-3a model is
**rejected**.

The next physics-informed implementation should place selected physical
Fourier coefficients directly in the recurrent state (or use a recurrent
physical decoder), rather than asking a memoryless decoder to reconstruct all
672 `phi`/`Ey` coefficients from the 30D state.  Poisson and continuity losses
remain deferred until this field representation passes the data-only physical
forecast gate.

### Stage 3b mode-factorized field-gradient result (2026-08-10)

The low/MTSI branch was frozen and the truth-floor field-gradient loss was
applied only to the high physical branch on modes n=2 and n=7--21.  Every
candidate started from the same locked data-only checkpoint and used identical
minibatch ordering.

| lambda_E | selected epoch | residual reduction | composite state skill | MTSI skill | ECDI skill | selected phi skill | selected Ey skill |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0 | 0.0% | +0.305 | +0.177 | +0.111 | +0.113 | +0.223 |
| 0.1 | 4 | 51.6% | +0.310 | +0.177 | +0.081 | +0.143 | +0.212 |
| 1.0 | 15 | 81.5% | +0.316 | +0.177 | +0.082 | +0.166 | +0.202 |

Factorization successfully prevents the physics update from damaging the MTSI
branch, and `lambda_E=1` gives a large reproducible residual reduction.
Nevertheless, its selected `phi`/`Ey` NRMSE values are 1.246/1.218, above the
predeclared climatology gate of 1.0.  Stage 3b is therefore retained as a
**rejected but informative physics ablation**, not a primary-test model.

The next change should improve physical-field amplitude and phase prediction,
especially the n=2 and n=7 amplitudes, before adding Poisson loss.  The current
results favor a recurrent mode-specific physical decoder or complex-valued
amplitude/phase loss rather than increasing the field-gradient weight.

### Stage 2c carrier-envelope and mode-specific expert result (2026-08-10)

The selected physical modes were next represented as complex envelopes after
removing a carrier fitted only from the training tail.  The n=7 carrier was
coherent across the training/development boundary (training-tail frequency
-4.417 MHz and development truth -4.359 MHz), while n=2 showed a larger shift
(-1.096 to -1.335 MHz).  Two high-mode propagators were tested: an autonomous
carrier-envelope ROM and a ROM supplied only with known current/source `Ez`
and elapsed time since the step.  Neither single propagator passed every modal
gate.  The autonomous model predicted the n=2 amplitude, whereas the
time-controlled model predicted the n=7 amplitude.

A development-selected expert manifest was therefore locked with the following
explicit assignments:

| observable | locked expert | skill vs common raw persistence |
|---|---|---:|
| composite low state | factorized low/direct state | +0.305 |
| radial envelope | regime-aware low branch | +0.246 |
| MTSI transport | regime-aware low branch | +0.177 |
| ECDI transport | direct physical-state branch | +0.111 |
| n=2 field coefficients | autonomous carrier branch | amplitude +0.209 |
| n=7 field coefficients | time-controlled carrier branch | amplitude +0.355 |
| selected phi field | mixed carrier experts | +0.349 |
| selected Ey field | mixed carrier experts | +0.347 |

This is disclosed as post-hoc expert selection on the up-sweep development
split, not as evidence from the primary down-history trajectory.  All listed
persistence-skill gates are positive, but selected `phi`/`Ey` NRMSE remains
1.101/1.101, above the climatology gate.  The manifest is therefore locked as
`PROVISIONAL_DATA_ONLY_DIAGNOSTIC`; it is not promoted to another physics-loss
ablation or to the primary test.  The primary `E25 -> E22.5` data remain
unloaded.  The next architecture must model the slow modal amplitude exchange
(including the approximately 1.49-us modulation) jointly, rather than combine
experts selected for individual modes.

### Stage 2d nonstationary modulation result (2026-08-10)

A slow-period fit using only the up-transition training interval
30.165--34.995 us gave a joint n=2/n=7 period of 1.484 us.  Adding its
fundamental and second harmonic as known controls did not give a stable single
propagator: n=2 amplitude skill fell to -1.957.  A mode-resolved audit explains
why a fixed clock is not prospective:

| interval | n=2 fitted period | n=7 fitted period |
|---|---:|---:|
| training, 30.165--34.995 us | 1.001 us | 1.699 us |
| development, 35.010--39.855 us | 1.435 us | 1.461 us |

The development periods above are diagnostic only and were not supplied as
controls.  The slow frequency changes during relaxation and cannot be treated
as one fixed forcing frequency inferred from the early transition.

A single shared GRU with separate n=2, n=7, and remaining-state residual heads
was then trained without a fixed modulation clock.  This produced the first
single field branch to pass the climatology field gate:

| training window | selected phi skill | selected Ey skill | phi NRMSE | Ey NRMSE | n=2 amplitude skill | n=7 amplitude skill |
|---:|---:|---:|---:|---:|---:|---:|
| 1.20 us | +0.639 | +0.607 | 0.820 | 0.855 | -1.055 | -0.687 |
| 2.40 us | +0.219 | +0.294 | 1.206 | 1.145 | -0.983 | -0.286 |

The longer training window therefore does not repair the amplitude forecast
and damages the field forecast.  The 1.20-us separated-head model is retained
as an informative negative result: it establishes that phase/field prediction
can pass the physical-field gate, but the same single propagator does not yet
predict the nonstationary n=2/n=7 amplitude exchange.  Since the predeclared
modal gate fails, no new physics-loss fit is started from either model.  The
next data-model change requires additional allowed transition trajectories or
intermediate restart checkpoints to identify how the slow frequency evolves
with state and electric-field history; increasing physics-loss weights or
rollout length alone is not supported by these results.

### Stage 2e causal state/E-history and amplitude-phase factorization (2026-08-10)

Development then continued without additional HPC/PIC trajectories. A causal
history branch was given nine known electric controls (current/source/delta
`Ez`, transition flag, linear/log elapsed time, and 0.30/1.50/5.00-us decaying
step memories) plus fourteen n=2/n=7 descriptors computed from available or
predicted states. The descriptors include amplitude, complex phase, phase
increment, amplitude increment, amplitude balance, and total amplitude. They
are recomputed from the ROM prediction during free rollout; no future PIC
state is used and no fixed modulation clock is supplied.

Exact azimuthal translations were also implemented in Fourier space. Their
round-trip error was 1.78e-15 in normalized coordinates and their relative
amplitude error was 6.05e-16. Nevertheless, four-shift augmentation made the
coordinate-wise standardized recurrent fit unstable and was rejected. The
unaugmented causal-history model gave the following diagnostic split:

| model | phi skill | Ey skill | phi NRMSE | Ey NRMSE | n=2 amplitude skill | n=7 amplitude skill |
|---|---:|---:|---:|---:|---:|---:|
| four azimuth shifts | +0.124 | +0.554 | 1.277 | 0.910 | -3.404 | -2.401 |
| causal history, no shift | -0.040 | +0.213 | 1.392 | 1.209 | +0.112 | +0.301 |

The no-shift result reverses both modal amplitude skills from negative to
positive, but loses the field gate. This isolates an output-task conflict:
the explicit history identifies the slow amplitude exchange, whereas the
earlier separated-head recurrent model predicts complex field phase much
better.

A locked factorized ROM therefore uses the earlier separated-head model for
phase, all unselected amplitudes, and inter-field phase, and uses the causal
history branch only for prospective n=2/n=7 phi amplitudes. The two causal
predictions are combined by applying each predicted phi-amplitude ratio to all
four field coefficients of the corresponding mode. The ratio is not clipped
and no PIC truth enters the fusion.

| metric | factorized data-only ROM |
|---|---:|
| carrier/composite state skill | +0.332 / +0.305 |
| radial/MTSI/ECDI skill | +0.246 / +0.177 / +0.111 |
| phi/Ey skill | +0.550 / +0.488 |
| phi/Ey NRMSE | 0.916 / 0.976 |
| n=2/n=7 amplitude skill | +0.112 / +0.301 |
| n=2/n=7 frequency error | 0.222 / 0.0327 MHz |
| field-gradient residual / Ey RMS | 0.1246 |

This is the first sweep ROM in the current development sequence to pass every
predeclared data-only and field-climatology gate. The factorization choice is
explicitly disclosed as post-hoc development selection. It is locked before
opening the primary `E25 -> E22.5` trajectory, which remains unread.

### Stage 3c factorized physics-informed comparison (2026-08-10)

Two physics-informed variants were compared against the locked factorized
data-only ROM using the training-truth floor for the spectral residual
`Ey + i*k*phi`.

First, a truth-floor proximal spectral decoder held phi fixed and shrank only
residual excess above the allowed training-PIC floor. This is an output-level
physics-loss-equivalent layer; it does not retrain network weights. Among the
predeclared `lambda_E = 0.01, 0.1, 1.0` candidates, `lambda_E = 1.0` reduced
the excess hinge by 74.0%, improved Ey NRMSE from 0.976 to 0.947, and preserved
all modal gates.

Second, the recurrent phase branch was genuinely fine-tuned end to end with

```
L = L_data + L_complex_modal + lambda_E * L_field_gradient_floor_hinge,
```

while the causal-history amplitude branch was frozen. The same three
`lambda_E` candidates started from the same data-only checkpoint and seed.
The selected candidate was `lambda_E = 1.0` at epoch 15:

| metric | factorized data-only | end-to-end physics | output physics decoder |
|---|---:|---:|---:|
| physics excess reduction | 0% | 41.9% | 74.0% |
| phi NRMSE | 0.916 | 0.898 | 0.916 |
| Ey NRMSE | 0.976 | 0.991 | 0.947 |
| n=2 amplitude skill | +0.112 | +0.112 | +0.112 |
| n=7 amplitude skill | +0.301 | +0.301 | +0.301 |
| n=2 frequency error | 0.222 | 0.190 | 0.222 MHz |
| n=7 frequency error | 0.0327 | 0.0305 | 0.0327 MHz |

The end-to-end candidate passes every preservation gate and is the selected
PINN-like training result. The output decoder is retained as a stronger
physics-projection ablation, not mislabeled as end-to-end training. Neither
variant uses the primary trajectory.

The remaining limitation is identifiability across independent temporal
phases: spatial Fourier translations do not create new slow-envelope phase
responses. The present result establishes causal within-trajectory
generalization on the allowed `E20 -> E22.5` development continuation, not
independence from the transition's initial temporal phase.

A fixed-horizon audit was also completed at 0.15, 0.30, 0.60, 1.20, 3.00,
and the full available 4.845 us. Phi/Ey NRMSE remains below one at every listed
horizon, and both physics variants reduce the field-gradient hinge at every
horizon. Modal amplitude skill is not uniformly positive, however. The n=7
skill is negative through the 3.0-us prefixes, and n=2 is negative at several
intermediate prefixes, even though their full-interval aggregate skills are
+0.112/+0.301. Therefore the model passes the predeclared full-development
gate but has not yet demonstrated uniformly accurate amplitude-exchange timing.
The requested 6.0-us horizon is longer than the available 4.845-us development
continuation and was not extrapolated.

## Required comparison

| model | role |
|---|---|
| Persistence | trivial baseline |
| E25 fixed local ROM | local-ROM baseline |
| Rejected smooth pROMs | negative transfer baselines |
| Regime-aware data-only sweep ROM | primary data-only model |
| Mode-specific carrier expert ROM | provisional data-only diagnostic |
| State/phase-factorized sweep ROM | locked post-hoc data-only development model |
| End-to-end factorized physics ROM | selected PINN-like physics-loss model |
| Truth-floor proximal decoder | output-level physics projection ablation |

Evaluate at horizons 0.15, 0.30, 0.60, 1.20, 3.0, and 6.0 us, plus the full available sweep. Report:

- common-state and physical-field error;
- ECDI/MTSI order parameter and mode-band powers;
- n=2 and n=7 amplitudes and frequencies;
- modal transport and cross-phase;
- approximately 1.49-us modulation amplitude and period;
- mean density and `efy` RMS relaxation;
- Poisson and field-gradient residuals;
- difference between the up- and down-history predictions at the same `Ez = 22.5 kV/m`.

## Acceptance rule before the primary test

The Stage 2 data-only model must remain finite over the full up-sweep validation
and achieve positive skill versus persistence for the 30-dimensional state,
radial envelope, and both MTSI- and ECDI-band transport.  The
physics-informed model must preserve those conditions, produce `phi` and `Ey`
forecasts better than the development climatology (`NRMSE < 1`), and reduce at
least one predeclared physics residual without materially degrading the modal
observables.  Models failing these conditions are retained as negative results
and are not promoted by inspecting the down-sweep test.

### Stage 2f reverse-transition development and bidirectional ROM (2026-08-10)

The allowed `E22.5 -> E20` trajectory was added without opening the blind
primary `E25 -> E22.5` trajectory.  Its source archive and stitched field file
passed a fresh integrity audit: 333 contiguous frames (30.015--34.995 us), 64
ranks per frame, 21,312 raw H5 files, all 15 expected fields, no zero-byte or
non-finite files, no schema/rank mismatch, and a maximum rank-overlap relative
error of `1.66e-16`.

The reverse path was transformed with the frozen E25 normalization and the
same SimVP checkpoint, eight radial bands, and Fourier modes n=0--21 used for
the up transition.  All three normalized channels had zero clipping.  This
produced 314 aligned latent windows and matching physical Fourier targets.
Stationary E20 physical Fourier targets were also cached as an endpoint
constraint.  The primary path remained unread throughout.

Three low-dimensional n=2/n=7 models were tested before retraining the full
state ROM:

| model | useful result | failure establishing the next step |
|---|---|---|
| first-order coupled amplitude ODE | held-out up transition n=2/n=7 skill `+0.190/+0.831` | 35--40-us up continuation `-0.656/-6.977`; held-out down failed |
| second-order amplitude/rate ODE | held-out down n=7 skill `+0.890` | integrated acceleration errors destabilized the other gates |
| coupled delay-amplitude ROM | reverse suffix n=2/n=7 skill `+0.576/+0.393` | long up continuation and cross-direction n=7 transfer failed |

These are retained as informative negative/partial results.  Directional
memory is identifiable, especially for n=7, but n=2/n=7 amplitudes alone are
not a sufficient Markov state for the full long-horizon field ROM.

The reverse trajectory was therefore added directly to the 150-dimensional
carrier-state/history GRU.  The fit used 759 non-augmented windows from E25
stationary, `E20 -> E22.5`, and `E22.5 -> E20`; the original 35--40-us up
continuation remained excluded from training.  The bidirectional amplitude
branch reached best epoch 31.  Relative to the previous up-only amplitude
branch, its full-development n=2/n=7 skills improved from `+0.112/+0.301` to
`+0.188/+0.484`.  The standalone branch still had phi NRMSE 1.108 and was not
accepted as a complete field model.

Replacing only the amplitude branch in the locked phase/amplitude
factorization produced a valid bidirectional development model:

| metric | previous up-only factorized | bidirectional factorized |
|---|---:|---:|
| phi NRMSE | 0.916 | **0.871** |
| Ey NRMSE | 0.976 | **0.919** |
| phi/Ey skill vs persistence | +0.550/+0.488 | **+0.593/+0.546** |
| n=2/n=7 amplitude skill | +0.112/+0.301 | **+0.188/+0.484** |
| n=2/n=7 frequency error [MHz] | 0.222/0.0327 | **0.152/0.0327** |
| field-gradient residual / Ey RMS | 0.1246 | 0.1352 |

All data-only persistence and climatology gates pass after factorization.
This result directly supports adding same-magnitude down-step data for the
blind `E25 -> E22.5` target, while not yet proving translation across absolute
Ez or independence from initial slow phase.

### Stage 3d bidirectional physics comparison (2026-08-10)

The predeclared truth-floor physics comparison was repeated with the new
bidirectional amplitude branch.  The selected post-hoc decoder remains
`lambda_E=1.0`; the selected end-to-end candidate is `lambda_E=1.0`, epoch 25.

| metric | bidirectional data-only | end-to-end physics | output decoder |
|---|---:|---:|---:|
| physics excess reduction | 0% | 49.3% | **74.2%** |
| phi NRMSE | **0.871** | 0.877 | **0.871** |
| Ey NRMSE | 0.919 | 0.959 | **0.895** |
| n=2 amplitude skill | +0.188 | +0.188 | +0.188 |
| n=7 amplitude skill | +0.484 | +0.484 | +0.484 |
| field-gradient residual / Ey RMS | 0.1352 | 0.1379 | **0.0704** |

Both physics variants pass all preservation gates.  The output decoder is the
recommended development configuration because it gives the lowest Ey error
and physics residual; it remains labeled as an output-level physics
projection, not end-to-end PINN training.

The bidirectional horizon audit covers 0.15, 0.30, 0.60, 1.20, 3.00, and the
full 4.845 us.  Phi/Ey NRMSE is below one and the decoder reduces the physics
hinge at every horizon.  n=7 amplitude skill is positive at every horizon.
n=2 is still negative through 1.20 us and becomes positive at 3.00 and 4.845
us, so short-horizon n=2 envelope phase remains an explicit limitation.  A
6-us forecast is not reported because only 4.845 us are available.

New locked development outputs:

- `workdirs/radaz_e22p5_to_e20_transition`
- `workdirs/train_radaz_coupled_amplitude_ode`
- `workdirs/train_radaz_second_order_amplitude_ode`
- `workdirs/train_radaz_delay_amplitude_rom`
- `workdirs/train_radaz_state_history_conditioned_rom_bidirectional_noaug`
- `workdirs/build_radaz_state_phase_factorized_rom_bidirectional`
- `workdirs/evaluate_radaz_factorized_physics_decoder_bidirectional`
- `workdirs/train_radaz_factorized_end_to_end_physics_rom_bidirectional`
- `workdirs/compare_radaz_factorized_physics_horizons_bidirectional`

### Stage 4 pre-primary execution lock (2026-08-11)

The confirmatory protocol was frozen before any local `E25 -> E22.5` primary
input became available.  The primary path was absent when both locks were
written.  The frozen evaluation is:

- target: the 30.015--34.995-us `E25 -> E22.5 kV/m` trajectory at Bx=20 mT;
- preprocessing: the existing E25 normalization, no refit, spatial stride 1,
  model grid 260x256, eight radial bands, and modes n=0--21;
- initialization: the first 40 causal target frames (0.60 us) initialize the
  recurrent states, followed by a completely free rollout;
- model: the locked mode-separated phase branch plus the locked bidirectional
  amplitude/history GRU;
- physics: the already selected truth-floor output decoder at `lambda_E=1`;
- horizons: 0.15, 0.30, 0.60, 1.20, 3.00 us and the full available suffix;
- primary gates: finite fraction 1, positive phi/Ey persistence skill,
  phi/Ey NRMSE below 1, positive n=2/n=7 amplitude skill, and reduced
  field-gradient excess after decoding;
- hysteresis output: E25-history minus allowed E20-history observables at the
  same Ez=22.5 kV/m and matched elapsed time.

The lock explicitly prohibits normalization or representation refitting,
architecture/checkpoint/epoch reselection, and loss/decoder-weight changes
after opening the primary data.  A failed gate must be reported as a failed
confirmatory result rather than repaired by retuning.

`evaluate_radaz_primary_e25_to_e22p5.py` implements the frozen evaluation.
Before the primary arrives it was exercised once on the allowed
`E20 -> E22.5` trajectory with the explicit status
`SELF_TEST_ALLOWED_UP_NOT_PRIMARY`.  The run completed through hash checking,
free rollout, field decoding, horizon metrics, and the equal-history
comparison.  Feeding the same allowed path to both comparison sides produced
exactly zero history difference, as required.  Its n=2 primary-style gate did
not pass, but this is only a mechanical self-test and is not entered as a
primary result or used for model selection.

`prepare_and_evaluate_radaz_primary_e25_to_e22p5.py` is the one-shot intake
runner.  It requires 333 contiguous raw frames beginning at `Macro_tn_2001`,
64 ranks per frame, all 15 expected fields, no zero-byte/non-finite files,
successful rank consolidation and raw comparison, fixed-normalization
metadata, 314 aligned latent/physical Fourier targets, and the exact locked
Ez/Bx/time/grid values.  It then executes the primary evaluator once and
refuses to overwrite an existing confirmatory result.

The execution bundle was independently frozen in
`workdirs/radaz_primary_execution_bundle_lock.json`.  It hashes 29 artifacts,
including the protocol lock, intake/evaluation/preprocessing code, raw H5
auditors, latent config and checkpoint, all allowed representation-fit inputs,
and all ROM evaluation dependencies.
The intake runner verifies every hash before reading primary data.  Current
state: `LOCKED_BEFORE_PRIMARY_INPUTS`; the local primary trajectory is still
unavailable, so no confirmatory score has been computed.

New pre-primary outputs:

- `workdirs/radaz_primary_e25_to_e22p5_evaluation_lock.json`
- `workdirs/radaz_primary_execution_bundle_lock.json`
- `workdirs/evaluate_radaz_primary_e25_to_e22p5_selftest`
