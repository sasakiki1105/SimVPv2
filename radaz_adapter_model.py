"""Tensor-state FiLM adapter on a FROZEN SimVP backbone (tensor_adapter_design_20260928.md §2).

No frozen OpenSTL file is modified: the wrapper re-implements SimVP_Model.forward using the
backbone's own encoder, translator blocks and decoder, and applies a per-stage FiLM
(scale, shift) after every translator block.  The FiLM heads are zero-initialised, so at
initialisation every arm reproduces the backbone output exactly (F0 == adapted model).

All arms share ONE adapter architecture with a fixed padded input dimension, so parameter
counts are identical across arms (F_const / F_noise / FA / F_mom / F_hat / F_raw / F_res).
"""
import hashlib
import torch
import torch.nn as nn

ADAPTER_INPUT_DIM = 40      # padded state dimension: max over arms (A 8 + NG4 32 = 40 mean/RMS × 4 bands)


class TemporalStateEncoder(nn.Module):
    """[B, T, d] -> [B, hidden]: per-frame linear -> GELU -> 1D conv over time -> mean pool -> linear."""
    def __init__(self, input_dim=ADAPTER_INPUT_DIM, hidden=64, history=10):
        super().__init__()
        self.input_dim, self.hidden, self.history = int(input_dim), int(hidden), int(history)
        self.frame = nn.Linear(self.input_dim, self.hidden)
        self.temporal = nn.Conv1d(self.hidden, self.hidden, kernel_size=3, padding=1)
        self.out = nn.Linear(self.hidden, self.hidden)
        self.act = nn.GELU()

    def forward(self, aux):
        if aux.ndim != 3 or aux.shape[1] != self.history or aux.shape[2] != self.input_dim:
            raise ValueError(f'adapter input must be [B,{self.history},{self.input_dim}], got {tuple(aux.shape)}')
        h = self.act(self.frame(aux))                     # [B, T, hidden]
        h = self.act(self.temporal(h.transpose(1, 2)))    # [B, hidden, T]
        return self.out(h.mean(dim=-1))                   # [B, hidden]


class FiLMHead(nn.Module):
    """hidden -> (1+gamma, beta) per channel; zero-initialised = identity."""
    def __init__(self, hidden, channels):
        super().__init__()
        self.channels = int(channels)
        self.net = nn.Linear(int(hidden), 2 * self.channels)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, features, code):
        gamma, beta = self.net(code).chunk(2, dim=-1)
        gamma = gamma.view(features.shape[0], self.channels, 1, 1)
        beta = beta.view(features.shape[0], self.channels, 1, 1)
        return (1.0 + gamma) * features + beta


def state_sha256(module):
    h = hashlib.sha256()
    for k, v in module.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class AdaptedSimVP(nn.Module):
    """Frozen SimVP_Model + trainable temporal-state FiLM adapter on the translator stages."""
    def __init__(self, backbone, hidden=64, history=10, input_dim=ADAPTER_INPUT_DIM):
        super().__init__()
        if getattr(backbone.hid, 'film', None) is not None:
            raise ValueError('backbone already carries a FiLM path; the adapter expects condition_film=False')
        if not hasattr(backbone.hid, 'enc') or not hasattr(backbone, '_match_skip'):
            raise ValueError('unexpected backbone structure')
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.backbone_sha256 = state_sha256(backbone)
        blocks = list(self.backbone.hid.enc)
        channels = [b.out_channels for b in blocks]
        self.encoder = TemporalStateEncoder(input_dim, hidden, history)
        self.film = nn.ModuleList([FiLMHead(hidden, c) for c in channels])

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()          # frozen backbone stays in eval (GroupNorm: no running stats; drop_path 0)
        return self

    def adapter_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith('backbone.')]

    def adapter_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('backbone.')}

    def forward(self, x_raw, aux):
        bb = self.backbone
        B, T, C, H, W = x_raw.shape
        if C != bb.in_channels + bb.condition_dim:
            raise ValueError(f'Expected {bb.in_channels + bb.condition_dim} input channels, got {C}')
        fields = x_raw[:, :, :bb.in_channels] if bb.condition_dim else x_raw
        x = fields.reshape(B * T, bb.in_channels, H, W)
        embed, skip = bb.enc(x)
        _, C_, H_, W_ = embed.shape
        z = embed.view(B, T, C_, H_, W_).reshape(B, T * C_, H_, W_)
        code = self.encoder(aux)
        for block, head in zip(bb.hid.enc, self.film):
            z = head(block(z), code)
        out_T = bb.out_seq_length
        hid = z.reshape(B * out_T, C_, H_, W_)
        skip = bb._match_skip(skip, B, T, out_T)
        Y = bb.dec(hid, skip)
        return Y.reshape(B, out_T, bb.out_channels, H, W)
