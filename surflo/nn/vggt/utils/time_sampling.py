"""
Time sampling functions for Flow Matching models.

This module provides different sampling strategies for the time variable t in flow matching.
Each sampler class should have a `sample` method that takes a batch size and device,
and returns a tensor of sampled time values in the range [0, 1].
Each sampler should also provide a `get_time_grid` method for inference time grids.
"""

import torch


class UniformTimeSampler:
    """Sample time uniformly from [0, 1]."""
    
    def sample(self, batch_size: int, device: torch.device, generator: torch.Generator = None) -> torch.Tensor:
        """
        Sample time uniformly from [0, 1].

        Args:
            batch_size (int): Number of samples to generate.
            device (torch.device): Device to place the tensor on.
            generator (torch.Generator, optional): Random number generator for reproducibility.

        Returns:
            torch.Tensor: Sampled time values of shape (batch_size,).
        """
        return torch.rand(batch_size, device=device, generator=generator)

    def get_time_grid(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """
        Get a monotonic time grid for inference.

        Args:
            num_steps (int): Number of inference steps.
            device (torch.device): Device to place the tensor on.

        Returns:
            torch.Tensor: Time grid of shape (num_steps + 1,).
        """
        return torch.linspace(0.0, 1.0, num_steps + 1, device=device)


class LogitNormalTimeSampler:
    """Sample time from a logit-normal distribution."""
    
    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        """
        Initialize the logit-normal time sampler.
        
        Args:
            mu (float): Mean of the underlying normal distribution.
            sigma (float): Standard deviation of the underlying normal distribution.
        """
        self.mu = mu
        self.sigma = sigma
    
    def sample(self, batch_size: int, device: torch.device, generator: torch.Generator = None) -> torch.Tensor:
        """
        Sample time from a logit-normal distribution.

        The logit-normal distribution is obtained by applying the sigmoid function
        to samples from a normal distribution N(mu, sigma^2).

        Args:
            batch_size (int): Number of samples to generate.
            device (torch.device): Device to place the tensor on.
            generator (torch.Generator, optional): Random number generator for reproducibility.

        Returns:
            torch.Tensor: Sampled time values of shape (batch_size,).
        """
        # Sample from normal distribution
        x = torch.randn(batch_size, device=device, generator=generator) * self.sigma + self.mu
        # Apply sigmoid to get values in [0, 1]
        return torch.sigmoid(x)
    def get_time_grid(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """
        Get a monotonic time grid for inference.

        Args:
            num_steps (int): Number of inference steps.
            device (torch.device): Device to place the tensor on.

        Returns:
            torch.Tensor: Time grid of shape (num_steps + 1,).
        """
        return torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    
    # None linear sampling. Too check later
    # def get_time_grid(self, num_steps: int, device: torch.device) -> torch.Tensor:
    #     """
    #     Get a monotonic time grid for inference.

    #     Args:
    #         num_steps (int): Number of inference steps.
    #         device (torch.device): Device to place the tensor on.

    #     Returns:
    #         torch.Tensor: Time grid of shape (num_steps + 1,).
    #     """
    #     if num_steps <= 1:
    #         return torch.tensor([0.0, 1.0], device=device)
    #     eps = 1e-6
    #     u = torch.linspace(eps, 1.0 - eps, num_steps + 1, device=device)
    #     normal = torch.distributions.Normal(
    #         loc=torch.tensor(self.mu, device=device),
    #         scale=torch.tensor(self.sigma, device=device),
    #     )
    #     x = normal.icdf(u)
    #     return torch.sigmoid(x)


class MeanFlowTimeSampler:
    """Sample (t, r) pairs for Improved Mean Flow training.

    Both t and r are drawn from a logit-normal distribution. With probability
    `data_proportion`, r is set equal to t (standard flow matching sample).
    The pair is sorted so that t >= r.
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0, data_proportion: float = 0.5):
        self.mu = mu
        self.sigma = sigma
        self.data_proportion = data_proportion

    def sample(
        self, batch_size: int, device: torch.device, generator: torch.Generator = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample (t, r) pairs.

        Args:
            batch_size (int): Number of samples to generate.
            device (torch.device): Device to place the tensor on.
            generator (torch.Generator, optional): Random number generator for reproducibility.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (t, r) each of shape (batch_size,), with t >= r.
        """
        t = torch.sigmoid(torch.randn(batch_size, device=device, generator=generator) * self.sigma + self.mu)
        r = torch.sigmoid(torch.randn(batch_size, device=device, generator=generator) * self.sigma + self.mu)

        # For data_proportion fraction, set r = t (standard FM sample)
        mask = torch.rand(batch_size, device=device, generator=generator) < self.data_proportion
        r[mask] = t[mask]

        # Ensure t >= r
        t_out = torch.max(t, r)
        r_out = torch.min(t, r)
        return t_out, r_out

    def get_time_grid(self, num_steps: int, device: torch.device) -> torch.Tensor:
        return torch.linspace(0.0, 1.0, num_steps + 1, device=device)
