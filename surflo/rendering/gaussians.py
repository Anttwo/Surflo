from typing import Optional

import torch
from surflo.structures.cameras import Camera
from surflo.structures.struct_utils import fov2focal


class Gaussians():
    def __init__(
        self,
        means: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        colors_sh: Optional[torch.Tensor] = None,
        active_sh_degree: int = 0,
    ):
        """
        Gaussians class for storing gaussians parameters.

        Args:
            means (torch.Tensor): Means of the gaussians. Shape (N, 3).
            rotations (torch.Tensor): Rotations of the gaussians. Shape (N, 4).
            scales (torch.Tensor): Scales of the gaussians. Shape (N, 3).
            opacities (torch.Tensor): Opacities of the gaussians. Shape (N,).
            colors (torch.Tensor): Colors of the gaussians. Shape (N, 3).
            colors_sh (torch.Tensor, optional): Non-DC SH bands. Shape
                (N, (deg + 1)^2 - 1, 3). The DC term is derived from ``colors``
                by the renderer.
            active_sh_degree (int): SH bands the renderer may use. Deliberately
                **not** inferred from ``colors_sh``: that buffer is allocated at
                full size up front while the degree can ramp independently (SH
                warmup), so inferring it silently overrides the schedule. Pass it
                explicitly whenever ``colors_sh`` is set, and carry it over when
                rebuilding a Gaussians from another one.
        """
        super().__init__()
        self.means = means
        self.rotations = rotations
        self.scales = scales
        self.opacities = opacities
        self.colors = colors
        self.colors_sh = colors_sh
        self.active_sh_degree = int(active_sh_degree)

    @property
    def device(self):
        return self.means.device


def get_intrinsics_matrix(camera: Camera) -> torch.Tensor:
    """
    Get the intrinsics matrix for the camera.

    Args:
        camera (Camera): The camera to get the intrinsics matrix for.
    Returns:
        torch.Tensor: The intrinsics matrix.
    """
    
    fx = fov2focal(camera.FoVx, camera.image_width)
    fy = fov2focal(camera.FoVy, camera.image_height)
    
    return torch.tensor(
        [
            [fx, 0, camera.image_width / 2],
            [0, fy, camera.image_height / 2],
            [0, 0, 1],
        ]
    )
    
