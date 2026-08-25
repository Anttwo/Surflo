from typing import List, Tuple, Optional
import torch
from surflo.structures.cameras import (
    Camera, 
    get_cameras_spatial_extent,
    transform_points_world_to_view,
    transform_points_to_pixel_space,
)


def get_interpolated_value_from_pixel_coordinates(
    value_img:torch.Tensor,
    pix_coords:torch.Tensor,
    interpolation_mode:str='bilinear',
    padding_mode:str='border',
    align_corners:bool=True,
):
    """
    Get value for pixel coordinates, by interpolating the value_img.

    Args:
        value_img (torch.Tensor): Has shape (H, W, C).
        pix_coords (torch.Tensor): Has shape (N, 2).
        interpolation_mode (str, optional): Defaults to 'bilinear'.
        padding_mode (str, optional): Defaults to 'border'.
        align_corners (bool, optional): Defaults to True.
        
    Returns:
        torch.Tensor: Has shape (N, C).
    """
    height, width = value_img.shape[:2]
    n_points = pix_coords.shape[0]
    
    # Scale and shift pixel coordinates to the range [-1, 1]
    factors = 0.5 * torch.tensor([[width-1, height-1]], dtype=torch.float32).to(pix_coords.device)  # (1, 2)
    scaled_pix_coords = pix_coords / factors - 1.  # (N, 2)
    scaled_pix_coords = scaled_pix_coords.view(1, -1, 1, 2)  # (1, N, 1, 2)

    # Interpolate the value
    interpolated_value = torch.nn.functional.grid_sample(
        input=value_img.permute(2, 0, 1)[None],  # (1, C, H, W)
        grid=scaled_pix_coords,  # (1, N, 1, 2)
        mode=interpolation_mode,
        padding_mode=padding_mode,  # 'reflection', 'zeros'
        align_corners=align_corners,
    )  # (1, C, N, 1)
    
    # Reshape to (N, C)
    interpolated_value = interpolated_value.reshape(-1, n_points).permute(1, 0)
    return interpolated_value


class AdaptiveTSDF:
    def __init__(
        self,
        points:torch.Tensor,
        trunc_margin:float,
        znear:float=None,
        zfar:float=None,
        initial_sdf_value:float=-1.0,
        use_binary_opacity:bool=False,
    ):
        """
        A class for computing a TSDF field from a set of points and a collection of posed depth maps.
        
        Args:
            points (torch.Tensor): Points at which to compute the TSDF. Has shape (N, 3).
            trunc_margin (float): Truncation margin for the TSDF.
            znear (float): Near clipping plane. If not provided, will use camera.znear when integrating.
            zfar (float): Far clipping plane. If not provided, will use camera.zfar when integrating.
            use_binary_opacity (bool): Whether to use a binary opacity field or a TSDF field.
                Please note that the TSDF field can be approximated into a binary opacity field 
                by using a TSDF with softmax weighting and high temperature.
        """
        
        assert trunc_margin >= 0, "Truncation margin must be positive"
        assert points.shape[1] == 3, "Points must have shape (N, 3)"
        assert (znear is None) or (znear > 0), "znear must be positive"
        assert (zfar is None) or (zfar > znear), "zfar must be greater than znear"
        
        self._n_points = points.shape[0]
        self._points = points
        self._trunc_margin = trunc_margin if not use_binary_opacity else 1.0
        self._znear = znear
        self._zfar = zfar

        # Initialize the field values
        self._use_binary_opacity = use_binary_opacity
        if self._use_binary_opacity:
            self._tsdf = torch.ones(self._n_points, 1, device=points.device)
        else:
            self._tsdf = initial_sdf_value * torch.ones(self._n_points, 1, device=points.device)
        self._weights = torch.zeros(self._n_points, 1, device=points.device)
        self._colors = torch.zeros(self._n_points, 3, device=points.device)
        
    @property
    def device(self):
        return self._points.device
    
    def integrate(
        self, 
        img:torch.Tensor, 
        depth:torch.Tensor,
        camera:Camera, 
        obs_weight=1.0,
        override_points:torch.Tensor=None,
        interpolate_depth:bool=True,
        interpolation_mode:str='bilinear',
        padding_mode:str='border',
        align_corners:bool=True,
        weight_by_softmax:bool=False,
        softmax_temperature:float=1.0,
    ):
        """
        Integrate a new observation into the TSDF.
        
        Args:
            img (torch.Tensor): Image. Has shape (H, W, 3) or (3, H, W).
            depth (torch.Tensor): Depth. Has shape (H, W), (H, W, 1) or (1, H, W).
            camera (GSCamera): Camera.
            obs_weight (float): Weight for the observation.
            override_points (torch.Tensor): Points for integration. Has shape (N, 3). If None, will use the points provided in the constructor.
            interpolate_depth (bool): Whether to interpolate the depth.
            interpolation_mode (str): Interpolation mode.
            padding_mode (str): Padding mode for interpolation.
            align_corners (bool): Whether to align corners for interpolation.
            weight_by_softmax (bool): Whether to weight the interpolation by the softmax.
            softmax_temperature (float): Temperature for the softmax.
        """
        
        # Reshape image and depth to (H, W, 3) and (H, W) respectively
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        depth = depth.squeeze()
        H, W = depth.shape
        
        points = self._points if override_points is None else override_points
        assert points.shape[0] == self._n_points, f"Points must have shape ({self._n_points}, 3)"
        
        # Transform points to view space
        view_points = transform_points_world_to_view(
            points=points.view(1, self._n_points, 3),
            cameras=[camera],
        )[0]  # (N, 3)
        
        # Project points to pixel space
        pix_points = transform_points_to_pixel_space(
            points=view_points.view(1, self._n_points, 3),
            cameras=[camera],
            points_are_already_in_view_space=True,
            keep_float=True,
        )[0]  # (N, 2)
        int_pix_points = pix_points.round().long()  # (N, 2)
        pix_x, pix_y, pix_z = pix_points[..., 0], pix_points[..., 1], view_points[..., 2]
        int_pix_x, int_pix_y = int_pix_points[..., 0], int_pix_points[..., 1]
        
        # Remove points outside view frustum and outside depth range
        valid_mask = (
            (pix_x >= 0) & (pix_x <= W-1) 
            & (pix_y >= 0) & (pix_y <= H-1) 
            & (pix_z > (camera.znear if self._znear is None else self._znear)) 
            & (pix_z < (camera.zfar if self._zfar is None else self._zfar))
        )  # (N,)
        
        if valid_mask.sum() > 0:
            # Get depth and image values at pixel locations
            packed_values = torch.cat(
                [
                    -torch.ones(len(valid_mask), 1, device=self.device),  # Depth values
                    torch.zeros(len(valid_mask), 3, device=self.device)  # Image values
                ], 
                dim=-1
            )  # (N, 4)
            if interpolate_depth:
                packed_values[valid_mask] = get_interpolated_value_from_pixel_coordinates(
                    value_img=torch.cat([depth.unsqueeze(-1), img], dim=-1),  # (H, W, 4)
                    pix_coords=pix_points[valid_mask],
                    interpolation_mode=interpolation_mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                )  # (N_valid, 4)
            else:
                packed_values[valid_mask] = torch.cat([depth.unsqueeze(-1), img], dim=-1)[int_pix_y[valid_mask], int_pix_x[valid_mask]]  # (N_valid, 4)
            depth_values = packed_values[..., :1]  # (N, 1)
            img_values = packed_values[..., 1:]  # (N, 3)
            valid_mask = valid_mask & (depth_values[..., 0] > 0.)  # (N,)
            
            # Compute distance
            sdf = ((depth_values - pix_z.unsqueeze(-1)) / self._trunc_margin).clamp_max(1.)  # (N, 1)
            # if not self._use_binary_opacity:
            #     valid_mask = valid_mask & (sdf[..., 0] >= -1.)
            valid_mask = valid_mask & (sdf[..., 0] >= -1.)
            
            # Compute observation weight
            _obs_weight = obs_weight
            if weight_by_softmax:
                _obs_weight = _obs_weight * torch.exp(sdf / softmax_temperature)  # (N_valid, 1)
                
            # Update Field Values
            new_weights = self._weights + _obs_weight  # (N, 1)
            if self._use_binary_opacity:
                new_tsdf = torch.minimum(self._tsdf, (sdf < 0.).float())
            else:
                new_tsdf = (self._tsdf * self._weights + sdf * _obs_weight) / new_weights  # (N, 1)        
            new_colors = (self._colors * self._weights + img_values * _obs_weight) / new_weights  # (N, 3)
            new_colors = new_colors.clamp(min=0., max=1.)  # (N, 3)
            
            # Update field values
            new_weights = torch.where(valid_mask.unsqueeze(-1), new_weights, self._weights)  # (N, 1)
            new_tsdf = torch.where(valid_mask.unsqueeze(-1), new_tsdf, self._tsdf)  # (N, 1)
            new_colors = torch.where(valid_mask.unsqueeze(-1), new_colors, self._colors)  # (N, 3)
            self._weights = new_weights.detach()
            self._tsdf = new_tsdf.detach()
            self._colors = new_colors.detach()
        
        else:
            new_weights = self._weights
            new_tsdf = self._tsdf
            new_colors = self._colors

        return {
            "weights": new_weights,
            "tsdf": new_tsdf,
            "colors": new_colors,
        }

    def return_field_values(self):
        output_pkg = {
            "weights": self._weights,
            "tsdf": self._tsdf,
            "colors": self._colors,
        }
        return output_pkg


def fuse_depth_maps(
    pivots:torch.Tensor,
    cameras:List[Camera],
    images:torch.Tensor,
    depths:torch.Tensor,
    trunc_margin_factor:float,
    scene_radius:Optional[float]=None,
    initial_sdf_value:float=-1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get initial SDF values for a given set of pivots, cameras, and predictions.
    Uses the depth maps to compute the SDF values with TSDF fusion.

    Args:
        pivots (torch.Tensor): Pivots to compute SDF values for
        cameras (List[Camera]): List of cameras
        images (torch.Tensor): Images. Has shape (N, 3, H, W).
        depths (torch.Tensor): Depths. Has shape (N, H, W).
        trunc_margin_factor (float): Truncation margin factor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: TSDF values and colors for the pivots.
            The tensors have shape (P,) and (P, 3) respectively.
    """
    if scene_radius is None:
        scene_radius = get_cameras_spatial_extent(cameras)['radius']
    
    # Create TSDF volume
    tsdf_volume = AdaptiveTSDF(
        points=pivots,
        trunc_margin=trunc_margin_factor * scene_radius,
        znear=None,
        zfar=None,
        initial_sdf_value=initial_sdf_value,
        use_binary_opacity=False,
    )
    
    # Fuse depth maps
    for i in range(len(cameras)):
        camera_i = cameras[i]
        rgb_i = images[i]
        depth_i = depths[i]
        
        tsdf_volume.integrate(
            img=rgb_i, 
            depth=depth_i,
            camera=camera_i, 
            obs_weight=1.0,
            interpolate_depth=True,
            interpolation_mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        
    field = tsdf_volume.return_field_values()
    pivots_sdf = field['tsdf']
    pivots_colors = field['colors']
    
    return pivots_sdf, pivots_colors
