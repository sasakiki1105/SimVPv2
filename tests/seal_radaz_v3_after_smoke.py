"""Record the pre-training DropPath amendment and seal the tested bundle."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "workdirs/2D_RadAz/radaz_arch_v3_plan"
path = out / "bundle.json"
bundle = json.loads(path.read_text())
smoke = json.loads((out / "training_smoke.json").read_text())
if bundle["commands_executed"] or set(smoke["cells"]) != set("ABCD"):
    raise RuntimeError("Only an untrained, successfully smoke-tested plan can be amended here")
for command in bundle["commands"]:
    ex = command[command.index("--ex_name")+1]
    work = root / "workdirs/2D_RadAz" / ex
    if work.exists():
        raise RuntimeError("A planned workdir already exists: " + str(work))
previous = out / "bundle.pre_smoke.json"
if previous.exists():
    raise RuntimeError("This specific pre-training amendment is already recorded")
previous.write_bytes(path.read_bytes())
bundle["amendment"] = {
    "previous_bundle_sha256": hashlib.sha256(previous.read_bytes()).hexdigest(),
    "reason": "Full-data train/eval smoke exposed legacy nonzero DropPath at configured zero; new configs explicitly use zero_to_max.",
    "research_training_started": False,
    "validation": "All four cells passed finite-parameter-gradient and train/eval equality checks"}
for name in bundle["sha256"]:
    bundle["sha256"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
for p in (Path(__file__), root / "tests/test_radaz_corrections.py", root / "tests/smoke_radaz_v3_training.py",
          out / "training_smoke.json"):
    bundle["sha256"][str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
print("Sealed tested, untrained v3 bundle with pre-smoke history preserved")
