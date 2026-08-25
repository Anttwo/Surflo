"""Flow-matching velocity wrapper for the Surflo model.

Wraps :class:`~surflo.model.surface_net.SurfaceNet` so it can be integrated by
an ODE solver: it converts the network's target/velocity prediction into a
velocity field (and optionally applies classifier-free guidance).
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
from flow_matching.path import AffineProbPath
from flow_matching.utils import ModelWrapper

_log = logging.getLogger(__name__)


class VelocityModel(ModelWrapper):
    """A wrapper around a denoiser model (SurfaceNet in this file) to convert its target prediction into a velocity prediction."""

    def __init__(self, denoiser: nn.Module, path: AffineProbPath, prediction_mode: str, use_cfg: bool = False, guidance_scale: float = 0.0):
        super().__init__(model=denoiser)
        self.path = path
        assert prediction_mode in ["target", "velocity"], f"Invalid prediction mode: {prediction_mode}"
        self.prediction_mode = prediction_mode
        self.use_cfg = use_cfg
        self.guidance_scale = guidance_scale
        _log.info(f"Instantiating VelocityModel with prediction mode: {self.prediction_mode}")

    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        compressed_tokens: Optional[torch.Tensor] = None,
        compressed_camera_tokens: Optional[torch.Tensor] = None,
        compressed_tokens_uncond: Optional[torch.Tensor] = None,
        compressed_camera_tokens_uncond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Since the wrapper doesn't support batching but the wrapped model does, we need to reshape the inputs accordingly.
        """
        # Get the conditional prediction
        prediction = super().forward(
            x=x, t=t, 
            compressed_tokens=compressed_tokens, 
            compressed_camera_tokens=compressed_camera_tokens,
            **kwargs
        ).squeeze(0)
                
        # If using CFG, get the unconditional prediction
        if self.use_cfg:
            prediction_uncond = super().forward(
                x=x, t=t, 
                compressed_tokens=compressed_tokens_uncond, 
                compressed_camera_tokens=compressed_camera_tokens_uncond,
                **kwargs
            ).squeeze(0)
        
        # Target prediction
        if self.prediction_mode == "target":
            velocity_prediction = self.path.target_to_velocity(x_1=prediction, x_t=x, t=t)            
            if self.use_cfg:
                velocity_prediction_uncond = self.path.target_to_velocity(x_1=prediction_uncond, x_t=x, t=t)
        
        # Velocity prediction
        elif self.prediction_mode == "velocity":
            velocity_prediction = prediction
            if self.use_cfg:
                velocity_prediction_uncond = prediction_uncond
        
        else:
            raise ValueError(f"Invalid prediction mode: {self.prediction_mode}")
        
        # Apply CFG formula: pred = pred_cond + guidance_scale * (pred_cond - pred_uncond)
        if self.use_cfg:
            velocity_prediction = velocity_prediction + self.guidance_scale * (velocity_prediction - velocity_prediction_uncond)
        
        return velocity_prediction

