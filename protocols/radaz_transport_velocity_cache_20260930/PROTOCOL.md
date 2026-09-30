# 2026-09-30: five-field training data pipeline amendment

Status: authorized engineering change; scientific protocol unchanged. The user
requested removing repeated CPU gamma computation and resuming current training.
This is not a new scientific selection or performance experiment.

## Unchanged experiment

Parent: `../radaz_transport_velocity_20260929/PROTOCOL.md` and sealed workdir
`workdirs/2D_RadAz/radaz_transport_velocity_20260929/bundle.json`, SHA256
`3036b7b6ff24cbeff5fbedeaf6fbdc617bad8303c5ec1d70bc00c61c66fc7ad2`.
All parent assets remain byte-for-byte unchanged and are verified on resume.

Cases: E10_B20, E20_B20, E30_B20, E40_B20, E10_B10, E10_B30.
TRAIN [0,1600), VAL [1600,1800), TEST [1800,2000); 10 past / 10 future
frames, 15 ns cadence. Cache construction reads TRAIN/VAL only. No recalibration,
TEST opening, case selection, hyperparameter tuning or model selection is added.
The same source-TRAIN normalizations, five physical channels
`ne,ni,phi,gamma_corr,u_ez`, two E/B condition channels, padding and dtype apply.
Gamma remains the saved-field fluctuation-product proxy, not measured particle flux.

Same architecture, loss, Adam, OneCycle, seed42 then43, 60 epochs/seed, batch1,
sampler order, DataLoader generators, num_workers=0 and loss/gradient checks.
Finite-value preprocessing validation moves to cache preparation. Per-step loss
checks remain. Both new physical outputs remain available for autonomous rollout.

## Change and equality gate

Use unchanged `load_frames`/`make_target` to prepare five-channel normalized
float32 H5 datasets (one per case and split). Load them into host RAM once.
Training windows only slice cached frames and append fixed E/B conditions.
No gamma derivatives/products or repeated data finite scans during each epoch.
H5 preparation is frame-local: observing future frames during offline preparation
does not mix their information into earlier frames. No temporal filtering occurs.

Every one of 10,800 TRAIN/VAL frames is compared bitwise against independent
singleton legacy recomputation; seven complete input/target windows per case/split
also compare legacy window arithmetic. Disk SHA256s and source identities are
checked before launch. Cache size is about 13.4 GiB, loaded once for both seeds;
no Windows worker-process data duplication is introduced.

Pause legacy training only after validation and checkpoint save at an epoch
boundary. Archive that checkpoint, hash it, and compare three subsequent real-GPU
Adam updates from identical saved model, optimizer, scheduler and RNG states using
old and new inputs. Require bit-identical inputs, labels, model/Adam state,
scheduler and RNG. A CPU test compares uninterrupted legacy training with a
legacy-to-cache epoch transition, including stochastic forward and validation.
These technical checks do not advance or overwrite the scientific checkpoint.

## Provenance and restart

The parent scientific checkpoint identity/kind stay unchanged for existing
evaluation tools. Runtime changes have a separate immutable `bundle.json` and
`runtime_amendment_sha256` in future checkpoints/logs. The only accepted first
legacy checkpoint is the explicitly audited checkpoint SHA; later cached
checkpoints must match the same runtime amendment. Optimizer, scheduler, global
step and torch/CUDA RNG restore unchanged. New seed43 follows the original path.

The original runner already recreates its private DataLoader generator on resume;
the sampler owns a separate epoch-specific generator and num_workers remains0.
This amendment preserves that behavior. It does not retroactively change RNG
policy or claim universal deterministic behavior across different hardware.

Commands (from SimVPv2, OpenSTL Python):

```
python run_radaz_transport_velocity_cached.py build
python -m unittest discover -s tests -p test_radaz_transport_velocity_cache.py -v
python run_radaz_transport_velocity_cached.py check
python run_radaz_transport_velocity_cached.py seal
python run_radaz_transport_velocity_cached.py train
python run_radaz_transport_velocity.py status
```

`check`, `seal` and `train` use the original execution lock. Remove the deliberate
`PAUSE_AFTER_EPOCH` marker only after all gates pass. Subsequent resumes use the
cached runner. No scientific evaluation is automatically launched after training.
Timing during cache construction shares the machine with training; observed
post-resume step/epoch times, rather than that microbenchmark, determine speedup.
