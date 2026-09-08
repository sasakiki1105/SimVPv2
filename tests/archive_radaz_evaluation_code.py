"""Archive the exact executed code before subsequent training-only corrections."""
import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "workdirs/2D_RadAz/radaz_corrected_A_v3"
protocol = json.loads((out / "protocol.json").read_text())
for name, expected in protocol["code_config_manifest_sha256"].items():
    source = root / name
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Code already differs: " + name)
    destination = out / "executed_source" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
print("Archived exact evaluation code/config/manifest")
