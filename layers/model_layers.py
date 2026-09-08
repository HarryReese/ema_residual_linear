from argparse import Namespace

import torch
from mamba_ssm import Mamba
from torch import nn
from torch.nn import functional as F


class RevIN(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.eps = 1e-5
        self.affine_weight = nn.Parameter(torch.ones(channels))
        self.affine_bias = nn.Parameter(torch.zeros(channels))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        return (x - self.mean) / self.stdev * self.affine_weight + self.affine_bias

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        return x * self.stdev + self.mean


class EMADecomposition(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.register_buffer(
            "_cached_weights",
            torch.empty(0, dtype=torch.double),
            persistent=False,
        )
        self.register_buffer(
            "_cached_divisor",
            torch.empty(0, dtype=torch.double),
            persistent=False,
        )

    def _coefficients(
        self,
        length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_is_valid = (
            self._cached_weights.shape == (1, length, 1)
            and self._cached_weights.device == device
            and self._cached_divisor.shape == (1, length, 1)
            and self._cached_divisor.device == device
        )
        if not cache_is_valid:
            powers = torch.flip(
                torch.arange(length, dtype=torch.double, device=device),
                dims=(0,),
            )
            weights = torch.pow(1.0 - self.alpha, powers)
            divisor = weights.clone()
            weights[1:] *= self.alpha
            self._cached_weights = weights.reshape(1, length, 1)
            self._cached_divisor = divisor.reshape(1, length, 1)
        return self._cached_weights, self._cached_divisor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        weights, divisor = self._coefficients(length, x.device)
        trend = torch.cumsum(x.to(torch.double) * weights, dim=1)
        trend = (trend / divisor).to(torch.float32)
        return x - trend


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_len: int,
        stride: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding = nn.ReplicationPad1d((0, stride))
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.padding(x.transpose(1, 2))
        patches = x.unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.stride,
        )
        batch, channels, patch_count, _ = patches.shape
        z = self.value_embedding(
            patches.reshape(batch * channels * patch_count, self.patch_len)
        )
        return self.dropout(self.norm(z)).reshape(
            batch,
            channels,
            patch_count,
            -1,
        )


class MambaMixer(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.mamba = Mamba(
            d_model=args.d_model,
            d_state=args.mamba_d_state,
            d_conv=args.mamba_d_conv,
            expand=args.mamba_expand,
        )
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.mamba(x))


class ChannelMLP(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        hidden_channels = args.enc_in * args.pm_channel_factor
        self.norm = nn.LayerNorm(args.d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(args.enc_in, hidden_channels),
            nn.GELU(),
            nn.Dropout(args.pm_dropout),
            nn.Linear(hidden_channels, args.enc_in),
        )
        self.dropout = nn.Dropout(args.pm_dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.norm(z).permute(0, 2, 3, 1)
        h = self.channel_mlp(h)
        return self.dropout(h.permute(0, 3, 1, 2).contiguous())


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.dropout(self.w1(x) * F.silu(self.w2(x))))


class SeasonalLayer(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.beta_couple = nn.Parameter(
            torch.tensor(args.channel_beta_init, dtype=torch.float32)
        )
        self.v_mixer = ChannelMLP(args)
        self.t_mixer = MambaMixer(args)

        self.t_norm = nn.LayerNorm(args.d_model)
        self.ffn_norm = nn.LayerNorm(args.d_model)
        self.ffn = SwiGLU(args.d_model, args.d_ff, args.dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z + self.beta_couple * self.v_mixer(z)
        batch, channels, patches, width = z.shape
        z = z.reshape(batch * channels, patches, width)
        z = z + self.t_mixer(self.t_norm(z))
        z = z + self.ffn(self.ffn_norm(z))
        return z.reshape(batch, channels, patches, width)


class ForecastBranch(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        patch_len = args.seasonal_patch_len
        stride = args.seasonal_stride
        patch_count = (args.seq_len + stride - patch_len) // stride + 1
        self.patch_embedding = PatchEmbedding(
            patch_len,
            stride,
            args.d_model,
            args.dropout,
        )
        self.encoder = nn.ModuleList(
            [SeasonalLayer(args) for _ in range(args.e_layers)]
        )
        self.proj = nn.Linear(patch_count * args.d_model, args.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.patch_embedding(x)
        for layer in self.encoder:
            z = layer(z)
        batch, channels, patches, width = z.shape
        return self.proj(
            z.reshape(batch, channels, patches * width)
        ).transpose(1, 2).contiguous()
