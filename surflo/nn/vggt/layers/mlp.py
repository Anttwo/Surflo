# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional

from torch import Tensor, nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AdaLNMLPBlock(nn.Module):
    """
    MLP block with AdaLN normalization.
    Adapted from: 
    https://github.com/nicolas-dufour/plonk/blob/76d46410910c9dfec9e19ed371450ebc7051cdf3/plonk/models/networks/mlp.py#L54

    Args:
        dim (int): The dimension of the input and output.
        mlp_ratio (float): The ratio of the hidden dimension to the input dimension.
    """
    def __init__(self, dim: int, mlp_ratio: float=4.0):
        super().__init__()
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.ada_map = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(dim, dim * 3),
        )
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)

        nn.init.zeros_(self.mlp.fc2.weight)
        nn.init.zeros_(self.mlp.fc2.bias)

    def forward(self, x, y):
        gamma, mu, sigma = self.ada_map(y).chunk(3, dim=-1)
        x_res = (1 + gamma) * self.ln(x) + mu
        x = x + self.mlp(x_res) * sigma
        return x
