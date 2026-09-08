"""Source validation diagnostics; never reads test fields or selects epochs."""
import json
from pathlib import Path

import numpy as np
import torch
from openstl.datasets.dataloader_pepapic_h5 import _condition_values


class SourceValidationDiagnostics:
    def __init__(self, manifest_path, device):
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.cases = [c for c in manifest["cases"] if c["role"] == "source"]
        norm = manifest["normalization"]
        self.low = torch.tensor(norm["low"], dtype=torch.float64, device=device)
        self.span = torch.tensor(norm["high"], dtype=torch.float64, device=device) - self.low
        self.conditions = torch.tensor(np.stack([_condition_values(
            c, manifest["condition_channels"], normalization=manifest["condition_normalization"])
            for c in self.cases]), device=device)
        # Per condition/channel: prediction SSE, truth physical energy,
        # persistence SSE, normalized element count.
        self.sums = torch.zeros(len(self.cases), len(self.low), 4, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, inputs, pred, target):
        channels = len(self.low)
        if inputs.shape[2] != channels + self.conditions.shape[1]:
            raise ValueError("Validation diagnostics require the manifest condition channels")
        for x, p, t in zip(inputs, pred, target):
            vector = x[0, channels:, 0, 0]
            distance = (self.conditions - vector).abs().amax(dim=1)
            index = int(distance.argmin())
            if distance[index] > 1e-5:
                raise ValueError("Validation contains an unknown/non-source condition")
            # RadAz model pads 257 valid nodes to 260; never score the padding.
            p, t = p[:, :, :257].double(), t[:, :, :257].double()
            c = x[-1:, :channels, :257].double().expand_as(t)
            physical_t = t * self.span[None, :, None, None] + self.low[None, :, None, None]
            self.sums[index, :, 0] += ((p - t)**2).sum(dim=(0, 2, 3))
            self.sums[index, :, 1] += physical_t.square().sum(dim=(0, 2, 3))
            self.sums[index, :, 2] += ((c - t)**2).sum(dim=(0, 2, 3))
            self.sums[index, :, 3] += t.shape[0] * t.shape[2] * t.shape[3]

    def finalize(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.sums)
        sums = self.sums.cpu().numpy()
        span = self.span.cpu().numpy()
        rows = {}
        for case, s in zip(self.cases, sums):
            if np.any(s[:, 3] == 0):
                raise ValueError("Every source condition must be represented in validation")
            rows[case["case_key"]] = {
                "normalized_mse_by_channel": (s[:, 0] / s[:, 3]).tolist(),
                "physical_relative_rmse_by_channel": [float(np.sqrt(a*b*b/d)) if d > 0 else None
                    for a, b, d in zip(s[:, 0], span, s[:, 1])],
                "skill_vs_copy_by_channel": [float(1-a/b) if b > 0 else None for a, b in zip(s[:, 0], s[:, 2])],
                "elements_per_channel": s[:, 3].astype(int).tolist()}
        mse = [np.mean(r["normalized_mse_by_channel"]) for r in rows.values()]
        return {"per_condition": rows, "balanced_mse_mean": float(np.mean(mse)),
                "balanced_mse_median": float(np.median(mse)), "balanced_mse_worst": float(np.max(mse)),
                "worst_mse_condition": list(rows)[int(np.argmax(mse))],
                "sampling": "all source validation loader windows; overlapping windows are descriptive, not independent replicates"}
