import torch
from torch import nn

from openstl.modules import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                             HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                             SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock)


class SimVP_Model(nn.Module):
    r"""SimVP Model

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, aft_seq_length=None,
                 simvp_direct_aft_seq=False, out_channels=None,
                 condition_dim=0, condition_film=False,
                 condition_hidden_dim=64, **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        self.pre_seq_length = T
        self.aft_seq_length = int(aft_seq_length) if aft_seq_length is not None else T
        self.out_seq_length = self.aft_seq_length if simvp_direct_aft_seq else T
        self.condition_dim = int(condition_dim)
        if self.condition_dim < 0 or self.condition_dim >= C:
            raise ValueError(
                f"condition_dim must be in [0, C), got condition_dim={self.condition_dim}, C={C}"
            )
        self.in_channels = C - self.condition_dim
        self.out_channels = (
            int(out_channels) if out_channels is not None else self.in_channels
        )
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False
        self.enc = Encoder(
            self.in_channels, hid_S, N_S, spatio_kernel_enc,
            act_inplace=act_inplace,
        )
        self.dec = Decoder(hid_S, self.out_channels, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        model_type = 'gsta' if model_type is None else model_type.lower()
        if model_type == 'incepu':
            self.hid = MidIncepNet(
                T*hid_S, hid_T, N_T,
                channel_out=self.out_seq_length*hid_S,
                output_seq_length=self.out_seq_length)
        else:
            self.hid = MidMetaNet(T*hid_S, hid_T, N_T,
                channel_out=self.out_seq_length*hid_S,
                output_seq_length=self.out_seq_length,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path,
                condition_dim=self.condition_dim if condition_film else 0,
                condition_hidden_dim=int(condition_hidden_dim))

    @staticmethod
    def _match_skip(skip, batch_size, in_seq_length, out_seq_length):
        if out_seq_length == in_seq_length:
            return skip

        b_t, c, h, w = skip.shape
        skip_seq = skip.reshape(batch_size, in_seq_length, c, h, w)
        if out_seq_length < in_seq_length:
            skip_seq = skip_seq[:, :out_seq_length]
        else:
            repeat = out_seq_length - in_seq_length
            tail = skip_seq[:, -1:].expand(batch_size, repeat, c, h, w)
            skip_seq = torch.cat([skip_seq, tail], dim=1)
        return skip_seq.reshape(batch_size*out_seq_length, c, h, w)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        if C != self.in_channels + self.condition_dim:
            raise ValueError(
                f"Expected {self.in_channels + self.condition_dim} input channels, got {C}"
            )
        condition = None
        if self.condition_dim:
            fields = x_raw[:, :, :self.in_channels]
            condition = x_raw[:, :, self.in_channels:].mean(dim=(1, 3, 4))
        else:
            fields = x_raw
        x = fields.reshape(B*T, self.in_channels, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z, condition=condition)
        out_T = self.out_seq_length
        hid = hid.reshape(B*out_T, C_, H_, W_)
        skip = self._match_skip(skip, B, T, out_T)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, out_T, self.out_channels, H, W)
        return Y


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3,5,7,11],
                 groups=8, channel_out=None, output_seq_length=None, **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        self.channel_in = channel_in
        self.channel_out = channel_out if channel_out is not None else channel_in
        self.output_seq_length = output_seq_length
        enc_layers = [gInception_ST(
            channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1,N2-1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers = [
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups)]
        for i in range(1,N2-1):
            dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_in,
                              incep_ker=incep_ker, groups=groups)
                if self.channel_out == channel_in else
                gInception_ST(2*channel_hid, channel_hid//2, self.channel_out,
                              incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x, condition=None):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2-1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1,self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1) )

        out_T = self.output_seq_length if self.output_seq_length is not None else T
        y = z.reshape(B, out_T, C, H, W)
        return y


class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'convmixer':
            self.block = ConvMixerSubBlock(in_channels, kernel_size=11, activation=nn.GELU)
        elif model_type == 'convnext':
            self.block = ConvNeXtSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'hornet':
            self.block = HorNetSubBlock(in_channels, mlp_ratio=mlp_ratio, drop_path=drop_path)
        elif model_type in ['mlp', 'mlpmixer']:
            self.block = MLPMixerSubBlock(
                in_channels, input_resolution, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type in ['moga', 'moganet']:
            self.block = MogaSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path_rate=drop_path)
        elif model_type == 'poolformer':
            self.block = PoolFormerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'swin':
            self.block = SwinSubBlock(
                in_channels, input_resolution, layer_i=layer_i, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path)
        elif model_type == 'uniformer':
            block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
            self.block = UniformerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path, block_type=block_type)
        elif model_type == 'van':
            self.block = VANSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'vit':
            self.block = ViTSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'tau':
            self.block = TAUSubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        else:
            assert False and "Invalid model_type in SimVP"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 channel_out=None, output_seq_length=None,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1,
                 condition_dim=0, condition_hidden_dim=64):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        self.channel_in = channel_in
        self.channel_out = channel_out if channel_out is not None else channel_in
        self.output_seq_length = output_seq_length
        self.condition_dim = int(condition_dim)
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, self.channel_out, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        self.enc = nn.Sequential(*enc_layers)
        if self.condition_dim:
            film_channels = [channel_hid] * (N2 - 1) + [self.channel_out]
            self.film = nn.ModuleList([
                FiLMConditioner(
                    self.condition_dim, channels, int(condition_hidden_dim)
                )
                for channels in film_channels
            ])
        else:
            self.film = None

    def forward(self, x, condition=None):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        if self.film is not None:
            if condition is None or condition.shape != (B, self.condition_dim):
                raise ValueError(
                    "FiLM-conditioned MidMetaNet requires condition shape "
                    f"{(B, self.condition_dim)}, got "
                    f"{None if condition is None else tuple(condition.shape)}"
                )

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if self.film is not None:
                z = self.film[i](z, condition)

        out_T = self.output_seq_length if self.output_seq_length is not None else T
        y = z.reshape(B, out_T, C, H, W)
        return y


class FiLMConditioner(nn.Module):
    """Feature-wise affine modulation initialized as the identity map."""

    def __init__(self, condition_dim, channels, hidden_dim):
        super().__init__()
        self.channels = int(channels)
        self.net = nn.Sequential(
            nn.Linear(int(condition_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2 * self.channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, condition):
        gamma, beta = self.net(condition).chunk(2, dim=-1)
        gamma = gamma.view(features.shape[0], self.channels, 1, 1)
        beta = beta.view(features.shape[0], self.channels, 1, 1)
        return (1.0 + gamma) * features + beta
