from argparse import Namespace

import torch
from torch import nn

from layers.model_layers import EMADecomposition, ForecastBranch, RevIN


class BestShipMotionModel(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.rev_norm = RevIN(args.enc_in)
        self.decomposition = EMADecomposition(args.ema_alpha)
        self.residual_branch = ForecastBranch(args)
        self.linear_branch = nn.Linear(args.seq_len, args.pred_len)
        nn.init.zeros_(self.linear_branch.weight)
        nn.init.zeros_(self.linear_branch.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rev_norm.normalize(x)
        residual = self.decomposition(x)
        prediction = self.residual_branch(residual)
        prediction = prediction + self.linear_branch(x.transpose(1, 2)).transpose(1, 2)
        return self.rev_norm.denormalize(prediction)
