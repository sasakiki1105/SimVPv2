"""Shared training/evaluation construction and explicit gSTA normalization.

Defaults reproduce historical checkpoints. ``batch_input`` uses independent
input-window statistics at inference; it never updates buffers or enables
DropPath. Group normalization is an opt-in architecture for NEW training.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import DropPath

from .simvp_model import SimVP_Model


class InputBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x):
        if self.training:
            if x.shape[0] != 1:
                raise ValueError("batch_input training requires batch size one; use group for larger batches")
            return super().forward(x)
        # One group per channel: each item is normalized independently over H,W.
        # Unlike batch statistics over N,H,W this is invariant to inference
        # batching, order, and the presence of other operating conditions.
        return F.group_norm(x, self.num_features, self.weight, self.bias, self.eps)


def configure_translator_norm(model, kind="batch", groups=8):
    if kind not in ("batch", "batch_input", "group"):
        raise ValueError(f"Unknown translator_norm: {kind}")
    if kind == "batch":
        return model
    if not hasattr(model, "hid"):
        raise ValueError("Expected a SimVP translator")
    replacements = []
    for name, module in model.hid.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            replacements.append((name, module))
    if not replacements:
        raise ValueError("Requested BatchNorm replacement but translator has no BatchNorm")
    for name, old in replacements:
        if kind == "batch_input":
            new = InputBatchNorm2d(old.num_features, old.eps, old.momentum,
                                   old.affine, old.track_running_stats)
            new.load_state_dict(old.state_dict())
        else:
            if groups <= 0 or old.num_features % groups:
                raise ValueError("translator_norm_groups must divide every gSTA channel width")
            new = nn.GroupNorm(groups, old.num_features, eps=old.eps, affine=old.affine)
            if old.affine:
                with torch.no_grad():
                    new.weight.copy_(old.weight)
                    new.bias.copy_(old.bias)
        new.train(old.training)
        if old.affine:
            new.to(device=old.weight.device, dtype=old.weight.dtype)
        parent_name, _, leaf = name.rpartition(".")
        parent = model.hid.get_submodule(parent_name) if parent_name else model.hid
        setattr(parent, leaf, new)
    return model


def build_simvp_model(config):
    config = {k: v for k, v in config.items() if not k.startswith("__")}
    if config.get("in_shape") is None:
        raise ValueError("in_shape must be resolved before model construction")
    model = SimVP_Model(**config)
    schedule = config.get("drop_path_schedule", "legacy")
    if schedule not in ("legacy", "zero_to_max"):
        raise ValueError("drop_path_schedule must be legacy or zero_to_max")
    if schedule == "zero_to_max":
        # Historical MidMetaNet starts linspace at 0.01 even when the configured
        # maximum is zero. New training must honor zero as fully disabled.
        maximum = float(config.get("drop_path", 0.0))
        if not 0 <= maximum < 1:
            raise ValueError("drop_path must lie in [0,1)")
        blocks = model.hid.enc
        for block, rate in zip(blocks, torch.linspace(0, maximum, len(blocks))):
            for module in block.modules():
                if isinstance(module, DropPath):
                    module.drop_prob = float(rate)
    return configure_translator_norm(model, config.get("translator_norm", "batch"),
                                    int(config.get("translator_norm_groups", 8)))


@torch.inference_mode()
def calibrate_batch_norm(model, inputs):
    """Equal-weight source-TRAIN input calibration, returning buffers only.

    All upstream BN layers use per-input statistics during calibration, exactly
    as batch-size-one training does. No dropout, targets, optimizer, or original
    model mutation. Caller owns/fixes source membership and frame provenance.
    """
    probe = configure_translator_norm(copy.deepcopy(model).eval(), "batch_input")
    accum = {}
    hooks = []
    for name, module in probe.named_modules():
        if isinstance(module, InputBatchNorm2d):
            accum[name] = {"mean": None, "var": None, "count": 0}

            def collect(mod, args, name=name):
                x = args[0].detach().double()
                if x.shape[0] != 1:
                    raise ValueError("BN calibration requires one source window per batch")
                mean = x.mean(dim=(0, 2, 3)).cpu()
                # Stored BN variance uses the unbiased estimator.
                var = x.var(dim=(0, 2, 3), unbiased=True).cpu()
                row = accum[name]
                row["mean"] = mean if row["mean"] is None else row["mean"] + mean
                row["var"] = var if row["var"] is None else row["var"] + var
                row["count"] += 1

            hooks.append(module.register_forward_pre_hook(collect))
    try:
        for x in inputs:
            probe(x)
    finally:
        for hook in hooks:
            hook.remove()
    buffers = {}
    for name, row in accum.items():
        if not row["count"]:
            raise ValueError("Empty BN calibration input")
        buffers[name + ".running_mean"] = (row["mean"] / row["count"]).float()
        buffers[name + ".running_var"] = (row["var"] / row["count"]).float()
        buffers[name + ".num_batches_tracked"] = torch.tensor(row["count"], dtype=torch.long)
    if not buffers:
        raise ValueError("No BatchNorm layers to calibrate")
    return buffers
