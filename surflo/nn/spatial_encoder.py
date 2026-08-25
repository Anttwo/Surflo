import logging
from typing import Callable, Optional, Union, List
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class IdentityEncoder(nn.Module):
    """Just returns input as is."""

    def __init__(
        self,
        in_dim: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()
        self.output_dim = in_dim

    def forward(self, x: Tensor) -> Tensor:
        return x


class FrequencyEncoder(nn.Module):
    """Frequency encoder that encodes a spatial point into a frequency domain representation.
    The frequency domain representation is a concatenation of the original spatial point and the frequency domain representation of the spatial point.
    The point needs to be normalized before being encoded into the frequency domain.

    When spatial_mean and spatial_std are provided, performs channel-wise centering and
    normalization: ``(x - mean) / std * target_std``.  When they are ``None`` (the default),
    no normalization is applied (equivalent to the old ``spatial_scale=1.0`` behaviour).

    Args:
        spatial_mean (Optional[List[float]]): Per-channel mean of the data distribution.
        spatial_std (Optional[List[float]]): Per-channel std of the data distribution.
        target_std (float): Desired std in the normalized (flow) space.
        in_dim (int): The dimension of the spatial point.
        num_freq (Optional[int]): The number of frequency bands.
    """

    def __init__(
        self,
        in_dim: int = 3,
        num_freq: Optional[int] = 128,
        min_freq: float = 0.0,
        max_freq: float = 10.0,
        spatial_mean: Optional[List[float]] = None,
        spatial_std: Optional[List[float]] = None,
        target_std: float = 0.5,
        spatial_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        if spatial_scale is not None:
            logger.warning(
                "FrequencyEncoder: 'spatial_scale' is deprecated. "
                "Use 'spatial_mean' / 'spatial_std' / 'target_std' instead."
            )
        self.in_dim = in_dim
        self.num_freq = num_freq
        self.output_dim = in_dim * (1 + 2 * num_freq)

        self.min_freq = min_freq
        self.max_freq = max_freq

        self._normalize = spatial_mean is not None and spatial_std is not None
        if self._normalize:
            self.register_buffer("spatial_mean", torch.tensor(spatial_mean, dtype=torch.float32))
            self.register_buffer("spatial_std", torch.tensor(spatial_std, dtype=torch.float32))
            self.target_std = target_std

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the FrequencyEncoder.

        Args:
            x (Tensor): Spatial points to encode, shape (..., in_dim)

        Returns:
            Tensor: Frequency domain representation of the spatial points, shape (..., in_dim * (1 + 2 * num_freq))
        """
        if self._normalize:
            x = (x - self.spatial_mean) / self.spatial_std * self.target_std

        # Compute frequency bands; points are normalized to the range [0, 1]
        freq_bands = 2.0 ** torch.linspace(self.min_freq, self.max_freq, self.num_freq, device=x.device, dtype=x.dtype)  # (num_freq,)
        x_freq = 2 * torch.pi * ((1.0 + x) / 2.0).unsqueeze(-1) * freq_bands  # (..., in_dim, num_freq)
        x_freq = x_freq.view(*x_freq.shape[:-2], -1)  # (..., in_dim * num_freq)

        # Compute sine and cosine of the frequency bands
        x_freq = torch.sin(torch.cat([x_freq, x_freq + torch.pi / 2.0], dim=-1))  # (..., in_dim * num_freq * 2)

        # Concatenate the original normalized point with the frequency domain representation
        x_freq = torch.cat([x, x_freq], dim=-1)  # (..., in_dim * (1 + 2 * num_freq))

        return x_freq
    
    
class GaussianFrequencyEncoder(nn.Module):
    """Gaussian frequency encoder using random Fourier features.

    When ``spatial_mean`` and ``spatial_std`` are provided, performs channel-wise
    centering and normalization: ``(x - mean) / std * target_std``.
    When they are ``None`` (the default), no normalization is applied
    (equivalent to the old ``spatial_scale=1.0`` behaviour — suitable for
    inputs that are already in flow space).

    Args:
        spatial_mean (Optional[List[float]]): Per-channel mean of the data distribution.
        spatial_std (Optional[List[float]]): Per-channel std of the data distribution.
        target_std (float): Desired std in the normalized (flow) space.
        in_dim (int): Dimension of the input spatial point.
        num_freq (Optional[int]): Number of random Fourier features.
        sigma (float | list[float]): Std of the Gaussian used to sample frequency directions.
    """

    def __init__(
        self,
        in_dim: int = 3,
        num_freq: Optional[int] = 512,
        sigma: Union[float, List[float]] = 10.,
        spatial_mean: Optional[List[float]] = None,
        spatial_std: Optional[List[float]] = None,
        target_std: float = 0.5,
        spatial_scale: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if spatial_scale is not None:
            logger.warning(
                "GaussianFrequencyEncoder: 'spatial_scale' is deprecated. "
                "Use 'spatial_mean' / 'spatial_std' / 'target_std' instead."
            )
        self.in_dim = in_dim
        self.num_freq = num_freq
        self.output_dim = in_dim + num_freq * 2

        self._normalize = spatial_mean is not None and spatial_std is not None
        if self._normalize:
            self.register_buffer("spatial_mean", torch.tensor(spatial_mean, dtype=torch.float32))
            self.register_buffer("spatial_std", torch.tensor(spatial_std, dtype=torch.float32))
            self.target_std = target_std
        
        self.sigma = sigma
        
        W = torch.randn(self.num_freq, self.in_dim)
        
        if isinstance(sigma, float):
            W = W * sigma

        else:
            self.n_sigma = len(sigma)
            self.freq_per_sigma = self.num_freq // self.n_sigma

            for i_sigma in range(self.n_sigma):
                start_idx = i_sigma * self.freq_per_sigma
                if i_sigma == self.n_sigma - 1:
                    end_idx = self.num_freq
                else:
                    end_idx = start_idx + self.freq_per_sigma
                W[start_idx:end_idx, :] = sigma[i_sigma] * W[start_idx:end_idx, :]

        self.register_buffer("W", W)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the GaussianFrequencyEncoder.

        Args:
            x (Tensor): Spatial points to encode, shape (..., in_dim)

        Returns:
            Tensor: Frequency domain representation, shape (..., in_dim + 2 * num_freq)
        """
        if self._normalize:
            x = (x - self.spatial_mean) / self.spatial_std * self.target_std
        # else: assume input is already in normalized space
        
        # Project the spatial point onto the frequency directions
        x_proj = x @ self.W.T  # (..., num_freq)
        
        # Concatenate the original spatial point with the frequency domain representations (sine and cosine)
        return torch.cat(
            [
                x,  # (..., in_dim)
                torch.sin(2 * torch.pi * x_proj),  # (..., num_freq)
                torch.cos(2 * torch.pi * x_proj),  # (..., num_freq)
            ],
            dim=-1,
        )  # (..., in_dim + 2 * num_freq) = (..., output_dim)


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        frequency_encoder: nn.Module,
        estimate_in_freq_enc: bool,
        mlp_ratio: float = 1.0,
        out_dim: Optional[int] = 2048,
        act_layer: Callable[..., nn.Module] = nn.ReLU,  # TODO: Should be GELU?
        drop: float = 0.0,
        bias: bool = True,
        n_hidden_layers: int = 4,
        add_time_as_input: bool = True,
    ) -> None:
        super().__init__()

        mlp_hidden_dim = int(out_dim * mlp_ratio)

        if not estimate_in_freq_enc:
            self.frequency_encoder = frequency_encoder
        else:
            self.frequency_encoder = IdentityEncoder(in_dim=771)  # x is already freq encoded (3 + 2*128*3)

        # Add time dimension if needed
        self.add_time_as_input = add_time_as_input
        frequency_dim = self.frequency_encoder.output_dim
        if add_time_as_input:
            frequency_dim = frequency_dim + 257  # add time dimension (freq encoded)

        # MLP to encode the spatial points
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
        self.mlp = nn.Sequential(*hidden_layers)

    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        x = self.frequency_encoder(x)
        if self.add_time_as_input:
            assert t is not None
            t = FrequencyEncoder(in_dim=1)(t)
            x = torch.cat((x, t), dim=-1)  # Concatenate time dimension
        x = self.mlp(x)
        return x


class GaussianSpatialEncoder(nn.Module):
    """
    Gaussian spatial encoder that encodes a spatial point into token representations.
    The spatial point will first be encoded into the frequency domain using random Fourier features,
    then passed through an MLP to produce the token representations.

    Args:
        frequency_encoder (GaussianFrequencyEncoder): The frequency encoder to use.
        mlp_ratio (float, optional): The ratio of the hidden dimension to the output dimension. Defaults to 1.0.
        out_dim (Optional[int], optional): The output token dimension. Defaults to 1024.
        act_layer (Callable[..., nn.Module], optional): The activation function to use. Defaults to nn.SiLU.
        drop (float, optional): The dropout rate. Defaults to 0.0.
        bias (bool, optional): Whether to use bias in the linear layers. Defaults to True.
        n_hidden_layers (int, optional): The number of additional hidden layers in the MLP. Defaults to 0.
    """
    def __init__(
        self,
        frequency_encoder: GaussianFrequencyEncoder,
        mlp_ratio: float = 1.0,
        out_dim: Optional[int] = 512,
        act_layer: Callable[..., nn.Module] = nn.SiLU,
        drop: float = 0.0,
        bias: bool = True,
        n_hidden_layers: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()

        # MLP hidden dimension
        mlp_hidden_dim = int(out_dim * mlp_ratio)
        
        # Frequency encoder
        self.frequency_encoder = frequency_encoder
        frequency_dim = self.frequency_encoder.output_dim

        # MLP layers to encode the spatial points
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

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Forward pass of the Gaussian spatial encoder.

        Args:
            x (Tensor): The spatial points to encode, shape (..., in_dim)

        Returns:
            Tensor: The token representations of the spatial points, shape (..., out_dim)
        """

        # Encoding the spatial points into the frequency domain
        x = self.frequency_encoder(x)  # (..., in_dim) -> (..., in_dim + 2 * num_freq)

        # Applying the MLP to the embedded points
        x = self.mlp(x)  # (..., in_dim + 2 * num_freq) -> (..., out_dim)

        return x
