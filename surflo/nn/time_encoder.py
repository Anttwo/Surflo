import torch
from torch import nn, Tensor
from typing import Optional, Callable
import math


class TimeFrequencyEncoder(nn.Module):
    """
    """

    def __init__(
        self,
        num_freq: Optional[int] = 512,
        max_period: float = 10000.0,
        input_factor: float = 1000.0,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.num_freq = num_freq
        self.max_period = max_period
        self.input_factor = input_factor
        self.log_max_period = math.log(max_period)
        self.output_dim = num_freq * 2
        
        freqs = self.input_factor * torch.exp(
            -self.log_max_period * torch.arange(
                start=0, end=self.num_freq, dtype=torch.float32,
            ) / self.num_freq
        )  # (num_freq,)
        
        self.register_buffer("freqs", freqs)

    def forward(self, t: Tensor) -> Tensor:
        """Forward pass of the TimeFrequencyEncoder.

        Args:
            t (Tensor): Spatial points to encode, shape (..., 1)

        Returns:
            Tensor: Frequency domain representation of the spatial points, shape (..., in_dim * (1 + 2 * num_freq))
        """

        assert t.shape[-1] == 1
        
        args = t * self.freqs  # (..., num_freq)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (..., 2 * num_freq)
        
        return embedding


class TimeEncoder(nn.Module):
    def __init__(
        self,
        frequency_encoder: TimeFrequencyEncoder,
        mlp_ratio: float = 1.0,
        out_dim: Optional[int] = 512,
        act_layer: Callable[..., nn.Module] = nn.SiLU,
        drop: float = 0.0,
        bias: bool = True,
        n_hidden_layers: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.frequency_encoder = frequency_encoder
        
        frequency_dim = self.frequency_encoder.output_dim
        mlp_hidden_dim = int(out_dim * mlp_ratio)
        
        # MLP layers to encode the time embedding
        hidden_layers = []

        # Projection from frequency dimension to hidden dimension
        hidden_layers.append(nn.Linear(frequency_dim, mlp_hidden_dim, bias=bias))
        hidden_layers.append(act_layer())
        hidden_layers.append(nn.Dropout(drop))

        # Hidden layers
        for _ in range(n_hidden_layers):
            hidden_layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim, bias=bias))
            hidden_layers.append(act_layer())
            hidden_layers.append(nn.Dropout(drop))

        # Projection from hidden dimension to output dimension
        hidden_layers.append(nn.Linear(mlp_hidden_dim, out_dim, bias=bias))
        hidden_layers.append(nn.Dropout(drop))
        
        # MLP
        self.mlp = nn.Sequential(*hidden_layers)
        
    def forward(self, t: Tensor) -> Tensor:
        out = self.frequency_encoder(t)
        out = self.mlp(out)
        return out