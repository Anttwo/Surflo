"""FFM: the Surflo model.

Encodes a variable number of unposed RGB images with a frozen VGGT-1B backbone
and a Perceiver compressor into a single fixed-size global latent, then decodes
an oriented surface point cloud with a per-point flow-matching ODE.

The training ``forward`` (standard flow-matching, velocity/target modes) is
included so this build can be trained by ``training/``. The mesh/render helpers
(``meshify*`` / ``render_*`` / ``fuse_depth``) and the raw-image training-time
``preprocess_inputs`` alignment remain omitted: training consumes preprocessed
(cached) batches only. The submodule structure (``vggt``, ``surface_net``,
``path``) is preserved so that checkpoints trained with the original code load
unchanged.
"""
import logging
from typing import Optional, Union

import torch
import torch.nn as nn
from einops import rearrange
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from hydra.utils import instantiate
from huggingface_hub import PyTorchModelHubMixin

from surflo.data.utils import transform_points
from surflo.utils.geometry import depths_to_points_parallel_batched
from surflo.structures.multi_cameras import (
    get_multi_cameras_from_intrinsics_and_extrinsics,
    get_cameras_spatial_extent as get_multi_cameras_spatial_extent,
)
from surflo.nn.vggt.models.vggt import VGGT
from surflo.nn.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from surflo.model.flow import VelocityModel

_log = logging.getLogger(__name__)


class FFM(nn.Module, PyTorchModelHubMixin):
    """Feed-Forward Meshing (FFM) model that combines VGGT for feature extraction and a model for predicting surface points."""

    def __init__(
        self,
        SurfaceNet: nn.Module,
        spatial_mean: list[float],
        spatial_std: list[float],
        target_std: float,
        estimate_in_freq_enc,
        scheduler: dict,
        t_sampler: Optional[dict] = None,
        overfit_mode=False,
        SDF=None,
        sample_from_vggt_world_points_with_sd: float | None = 0.1,
        use_voxel_matching: bool = False,
        voxel_match_n_voxels_per_axis: int = 10,
        prediction_mode: str = "velocity",  # "target", "velocity"
        compute_loss_in_velocity: bool = False,  # If true, compute the loss between predicted and target velocities when prediction_mode is "target". Only used for training, has no effect during inference.
        cache_vggt_predictions_when_overfitting: bool = False,
        use_learned_tokens: bool = False,
        estimate_normals: bool = True,
        normals_shift: float = 0.01,
        unconditional_ratio: float = 0.1,  # Classifier-free guidance: ratio of data without VGGT features
        use_mean_flow: bool = False,  # Enable Improved Mean Flow (iMF) training
        compile: bool = False,  # Whether to torch.compile blocks
        spatial_scale: Optional[float] = None,  # Deprecated, kept for backward compat
        per_scene_normalize: bool = True,
        scene_normalize_mode: str = "median_dist_to_medianpoint",  # "std", "median_dist_to_barycenter", or "median_dist_to_medianpoint"
        renormalize_after_cull: bool = False,
    ):  # argument SDF exists just for compatibility
        super().__init__()
        self.use_mean_flow = use_mean_flow

        if spatial_scale is not None:
            _log.warning(
                "FFM: 'spatial_scale' is deprecated. "
                "Use 'spatial_mean' / 'spatial_std' / 'target_std' instead."
            )

        # Load VGGT model for feature extraction
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        self.estimate_in_freq_enc = estimate_in_freq_enc
        self.sample_from_vggt_world_points_with_sd = sample_from_vggt_world_points_with_sd
        self.use_voxel_matching = use_voxel_matching
        self.voxel_match_n_voxels_per_axis = voxel_match_n_voxels_per_axis
        
        # Classifier-free guidance
        assert 0.0 <= unconditional_ratio <= 1.0, f"unconditional_ratio must be in [0.0, 1.0], got {unconditional_ratio}"
        self.unconditional_ratio = unconditional_ratio
        
        assert prediction_mode in ["target", "velocity"], f"Invalid prediction mode: {prediction_mode}"
        self.prediction_mode = prediction_mode
        self.compute_loss_in_velocity = compute_loss_in_velocity
        
        self.register_buffer("spatial_mean", torch.tensor(spatial_mean, dtype=torch.float32))
        self.register_buffer("spatial_std", torch.tensor(spatial_std, dtype=torch.float32))
        self.target_std = target_std
        self.per_scene_normalize = per_scene_normalize
        _valid_modes = ("std", "median_dist_to_barycenter", "median_dist_to_medianpoint")
        assert scene_normalize_mode in _valid_modes, f"Invalid scene_normalize_mode: {scene_normalize_mode}. Must be one of {_valid_modes}"
        self.scene_normalize_mode = scene_normalize_mode
        self.renormalize_after_cull = renormalize_after_cull
        for param in self.vggt.parameters():
            param.requires_grad = False
        self.vggt.eval()

        # Define FFM model
        # Takes (vggt_feature, query_points) as input and outputs surface points
        self.surface_net = instantiate(SurfaceNet, _recursive_=True)  # Recursively instantiates modules under the network

        if self.estimate_in_freq_enc:
            from surflo.nn.spatial_encoder import FrequencyEncoder

            self.frequency_encoder = FrequencyEncoder(
                spatial_mean=spatial_mean,
                spatial_std=spatial_std,
                target_std=target_std,
                in_dim=3,
            ).to(self.device)
            
        self.estimate_normals = estimate_normals
        avg_world_scale = sum(spatial_std) / len(spatial_std)
        self.normals_shift_world_space = normals_shift * avg_world_scale
        if self.estimate_normals:
            assert not self.estimate_in_freq_enc, "Normals estimation is not supported when using frequency space prediction"

        self.overfit_mode = overfit_mode
        self.cache_vggt_predictions_when_overfitting = cache_vggt_predictions_when_overfitting
        if self.cache_vggt_predictions_when_overfitting:
            _log.warning(f"Caching VGGT predictions is activated. This should be activated ONLY WHEN OVERFITTING WITH FIXED CAMERA POSES.")
            # self.vggt_cache = {}

        if use_learned_tokens:
            assert cache_vggt_predictions_when_overfitting, (
                "use_learned_tokens requires cache_vggt_predictions_when_overfitting=True "
                "(fixed cameras, single scene)"
            )
            self.surface_net.setup_learned_tokens()

        # Define flow path
        self.path = AffineProbPath(scheduler=instantiate(scheduler))
        
        # Define time sampler
        if t_sampler is None:
            # Default to uniform sampling
            from surflo.nn.vggt.utils.time_sampling import UniformTimeSampler
            self.t_sampler = UniformTimeSampler()
        else:
            self.t_sampler = instantiate(t_sampler)

        if compile:
            self._compile_blocks()

    def _compile_blocks(self):
        """In-place torch.compile on transformer blocks and SurfaceNet sub-modules.

        VGGT backbone blocks are compiled individually (fullgraph) since the
        backbone has its own control flow around them.

        For SurfaceNet, we compile the compressor and field_decoder as whole
        modules (allowing cross-block fusion) but leave the top-level
        SurfaceNet.forward uncompiled — its _build_aggregated_tokens does a
        large cat/copy that triggers triton bugs with whole-module compilation.
        """
        # The DDP dynamo backend partitions compiled graphs at allreduce
        # boundaries.  Our compiled sub-modules (frozen backbone blocks,
        # compressor, decoder) contain no allreduce ops, so the partitioner
        # adds no value.  Worse, it can place symbolic-int size nodes at
        # partition boundaries, which crashes AOT autograd when the graph is
        # recompiled for eval mode.  Disable it.
        torch._dynamo.config.optimize_ddp = False

        block_kwargs = dict(fullgraph=True)
        count = 0

        # --- Frozen VGGT backbone: compile individual blocks ---
        aggregator = self.vggt.aggregator
        for block in aggregator.frame_blocks:
            block.compile(**block_kwargs)
            count += 1
        for block in aggregator.global_blocks:
            block.compile(**block_kwargs)
            count += 1

        patch_embed = aggregator.patch_embed
        if hasattr(patch_embed, "blocks"):
            for block in patch_embed.blocks:
                block.compile(**block_kwargs)
                count += 1


        # --- Trainable SurfaceNet: compile compressor & field_decoder as whole modules ---
        snet = self.surface_net
        snet_kwargs = dict(fullgraph=True)
        compiled_parts = []

        if hasattr(snet, "compressor"):
            snet.compressor.compile(**snet_kwargs)
            compiled_parts.append("compressor")

        if self.use_mean_flow:
            _log.info("Skipping torch.compile on field_decoder (compiled modules are incompatible with torch.func.jvp)")
        else:
            snet.field_decoder.compile(**snet_kwargs)
            compiled_parts.append("field_decoder")

        if hasattr(snet, "camera_tokens_compressor"):
            snet.camera_tokens_compressor.compile(**snet_kwargs)
            compiled_parts.append("camera_tokens_compressor")


    def _apply_unconditional_masking(self, aggregated_tokens_list: list, batch_size: int) -> list:
        """
        Apply classifier-free guidance by randomly zeroing out VGGT features for some samples.
        This is used during training to enable unconditional trajectory learning.

        Args:
            aggregated_tokens_list (list): List of aggregated tokens from VGGT. Each element has shape (B, N, ?, D) or None.
            batch_size (int): Batch size B.

        Returns:
            list: Modified aggregated_tokens_list with some batches zeroed out based on unconditional_ratio.
        """
        # Get device from the first non-None tensor
        device = next(t for t in aggregated_tokens_list if t is not None).device

        # Determine which samples in the batch should be unconditional
        unconditional_mask = torch.rand(batch_size, device=device) < self.unconditional_ratio  # (B,)

        if unconditional_mask.any():

            # Create a copy of the list to avoid modifying the original (skip None entries)
            aggregated_tokens_list = [tokens.clone() if tokens is not None else None for tokens in aggregated_tokens_list]

            # Zero out tokens for unconditional samples (skip None entries)
            for layer_idx, tokens in enumerate(aggregated_tokens_list):
                if tokens is not None:
                    tokens[unconditional_mask] = 0.0

        return aggregated_tokens_list

    def _compute_scene_stats(
        self, vggt_world_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-scene centering and scale from VGGT world points.

        The centering depends on ``self.scene_normalize_mode``:

        * ``"std"`` / ``"median_dist_to_barycenter"`` — mean (barycenter).
        * ``"median_dist_to_medianpoint"`` — coordinate-wise median.

        The scale depends on ``self.scene_normalize_mode``:

        * ``"std"`` — per-axis standard deviation (the default).
        * ``"median_dist_to_barycenter"`` — median L2 distance to the mean.
          Gives a single **isotropic** scale per scene that is robust to
          outliers and preserves the scene's aspect ratio.
        * ``"median_dist_to_medianpoint"`` — median L2 distance to the
          coordinate-wise median point.  Same isotropic behaviour but the
          center itself is also robust to outliers.

        Args:
            vggt_world_points: (B, N, H, W, 3) or (B, N_flat, 3)

        Returns:
            scene_center: (B, 1, 3)
            scene_scale:  (B, 1, 3)
        """
        pts = vggt_world_points.reshape(vggt_world_points.shape[0], -1, 3)

        if self.scene_normalize_mode == "median_dist_to_medianpoint":
            scene_center = pts.median(dim=1, keepdim=True).values  # (B, 1, 3)
        else:
            scene_center = pts.mean(dim=1, keepdim=True)  # (B, 1, 3)

        if self.scene_normalize_mode in ("median_dist_to_barycenter", "median_dist_to_medianpoint"):
            dists = torch.norm(pts - scene_center, dim=-1)  # (B, N_flat)
            median_dist = dists.median(dim=1, keepdim=True).values  # (B, 1)
            scene_scale = median_dist.unsqueeze(-1).expand_as(scene_center).clamp(min=1e-6)  # (B, 1, 3)

        elif self.scene_normalize_mode == "std":
            scene_scale = pts.std(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1, 3)

        else:
            raise ValueError(f"Invalid scene_normalize_mode: {self.scene_normalize_mode}")

        return scene_center, scene_scale

    def _compute_post_cull_scene_stats(
        self,
        vggt_world_points: torch.Tensor,
        pre_cull_mean: torch.Tensor,
        pre_cull_std: torch.Tensor,
        cull_radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute per-scene stats from VGGT points that survive culling.

        Uses the *pre-cull* mean/std to determine which points lie inside the
        culling radius, then computes fresh mean and scale from the survivors
        only.  This ensures that after culling, the normalization maps the
        remaining points to exactly match ``target_std``.

        Falls back to pre-cull stats for any batch element where fewer than 2
        points survive (shouldn't happen in practice).

        Returns:
            scene_mean:  (B, 1, 3)
            scene_scale: (B, 1, 3)
        """
        pts = vggt_world_points.reshape(vggt_world_points.shape[0], -1, 3)  # (B, N, 3)
        B = pts.shape[0]

        if isinstance(cull_radius, torch.Tensor):
            cr = cull_radius.view(B, 1)
        else:
            cr = cull_radius

        std_dist = torch.norm(
            (pts - pre_cull_mean) / pre_cull_std, dim=-1
        )  # (B, N)
        mask = std_dist < cr  # (B, N)

        new_means = []
        new_stds = []
        for i in range(B):
            culled = pts[i][mask[i]]  # (N_culled, 3)
            if culled.shape[0] < 2:
                new_means.append(pre_cull_mean[i])   # (1, 3)
                new_stds.append(pre_cull_std[i])      # (1, 3)
                continue
            if self.scene_normalize_mode == "median_dist_to_medianpoint":
                m = culled.median(dim=0, keepdim=True).values  # (1, 3)
            else:
                m = culled.mean(dim=0, keepdim=True)  # (1, 3)
            if self.scene_normalize_mode in ("median_dist_to_barycenter", "median_dist_to_medianpoint"):
                d = torch.norm(culled - m, dim=-1).median().clamp(min=1e-6)
                s = d.unsqueeze(0).unsqueeze(0).expand(1, 3)  # (1, 3)
            else:  # "std"
                s = culled.std(dim=0, keepdim=True).clamp(min=1e-6)  # (1, 3)
            new_means.append(m)
            new_stds.append(s)

        return (
            torch.stack(new_means, dim=0),  # (B, 1, 3)
            torch.stack(new_stds, dim=0),   # (B, 1, 3)
        )

    def _normalize_3d(
        self,
        xyz: Union[torch.Tensor, float],
        scene_mean: Optional[Union[torch.Tensor, float]] = None,
        scene_std: Optional[Union[torch.Tensor, float]] = None,
    ) -> Union[torch.Tensor, float]:
        """Channel-wise centering + normalization for 3-channel (XYZ) data.
        When scene_mean/scene_std are provided they override the global buffers."""
        mean = scene_mean if scene_mean is not None else self.spatial_mean
        std = scene_std if scene_std is not None else self.spatial_std
        return (xyz - mean) / std * self.target_std

    def _denormalize_3d(
        self,
        xyz: Union[torch.Tensor, float],
        scene_mean: Optional[Union[torch.Tensor, float]] = None,
        scene_std: Optional[Union[torch.Tensor, float]] = None,
    ) -> Union[torch.Tensor, float]:
        """Inverse of ``_normalize_3d``."""
        mean = scene_mean if scene_mean is not None else self.spatial_mean
        std = scene_std if scene_std is not None else self.spatial_std
        return xyz / self.target_std * std + mean

    def lift_points_to_flow_space(
        self,
        points: torch.Tensor,
        scene_mean: Optional[torch.Tensor] = None,
        scene_std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Lift N-D points to normalized flow space.
        N should be 3 or 6.  For 6-D (R6) points the same mean/std is applied
        independently to both the first and second set of 3 channels.

        Args:
            points (torch.Tensor): N-D points to lift to flow space. Shape (B, P, N).
            scene_mean / scene_std: optional per-scene overrides of shape (B, 1, 3).

        Returns:
            torch.Tensor: Lifted points. Shape (B, P, D).
        """
        if self.estimate_in_freq_enc:
            return self.frequency_encoder(points)  # (B, P, D)
        else:
            if points.shape[-1] == 3:
                return self._normalize_3d(points, scene_mean, scene_std)
            elif points.shape[-1] == 6:
                return torch.cat([
                    self._normalize_3d(points[..., :3], scene_mean, scene_std),
                    self._normalize_3d(points[..., 3:], scene_mean, scene_std),
                ], dim=-1)
            else:
                raise ValueError(f"Expected 3 or 6 channels, got {points.shape[-1]}")
        
    def unlift_points_from_flow_space(
        self,
        points: torch.Tensor,
        scene_mean: Optional[torch.Tensor] = None,
        scene_std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Unlift points from normalized flow space to N-D space.
        N should be 3 or 6.

        Args:
            points (torch.Tensor): Points in flow space. Shape (B, P, D).
            scene_mean / scene_std: optional per-scene overrides of shape (B, 1, 3).

        Returns:
            torch.Tensor: Unlifted points. Shape (B, P, N).
        """
        if self.estimate_in_freq_enc:
            N = 6 if self.estimate_normals else 3
            raw = points[..., :N]
        else:
            raw = points
        if raw.shape[-1] == 3:
            return self._denormalize_3d(raw, scene_mean, scene_std)
        elif raw.shape[-1] == 6:
            return torch.cat([
                self._denormalize_3d(raw[..., :3], scene_mean, scene_std),
                self._denormalize_3d(raw[..., 3:], scene_mean, scene_std),
            ], dim=-1)
        else:
            raise ValueError(f"Expected 3 or 6 channels, got {raw.shape[-1]}")
        
    def get_points_normals_from_r6_points(self, r6_points: torch.Tensor) -> torch.Tensor:
        """
        Get points and normals from 6D points.

        Args:
            r6_points (torch.Tensor): 6D points. Shape (..., 6).
            shift (float, optional): Shift amount. Defaults to 0.01.

        Returns:
            torch.Tensor: Points and normals. Shape (..., 3) and (..., 3) respectively.
        """
        points = r6_points[..., :3]  # (..., 3)
        normals = (r6_points[..., 3:] - points)  # (..., 3)
        normals = torch.nn.functional.normalize(normals, dim=-1)  # (..., 3)
        return points, normals
    
    def get_r6_points_from_points_normals(
        self, 
        points: torch.Tensor, 
        normals:torch.Tensor, 
    ) -> torch.Tensor:
        """
        Get 6D points from points and normals.

        Args:
            points (torch.Tensor): Points in 3D space. Shape (..., 3).
            normals (torch.Tensor): Normals in 3D space. Shape (..., 3).

        Returns:
            torch.Tensor: 6D points. Shape (..., 6).
        """
        
        r6_points = torch.cat(
            [
                points,  # (..., 3)
                points + self.normals_shift_world_space * normals,  # (..., 3)
            ],
            dim=-1,
        )  # (..., 6)
        return r6_points
        
    def sample_from_source_distribution(
        self,
        n_points: int,
        batch_size: Optional[int] = None,
        preprocessed_batch: Optional[dict] = None,
        vggt_world_points: Optional[torch.Tensor] = None,
        cull_radius: Optional[float] = None,
        pure_noise_std: float = 0.5,
        generator: Optional[torch.Generator] = None,
        scene_mean: Optional[torch.Tensor] = None,
        scene_std: Optional[torch.Tensor] = None,
        cull_mean: Optional[torch.Tensor] = None,
        cull_std: Optional[torch.Tensor] = None,
        # Accepted but unused: culling reads cull_mean / cull_std.
        scene_center: Optional[torch.Tensor] = None,
        scene_radius: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample points from source distribution, conditioned on VGGT world points.
        VGGT world points can be provided directly, or from a preprocessed batch.
        If using a pure Gaussian noise source distribution, it is not necessary to provide VGGT world points;
        only the batch_size is required.
        
        As a result, only one of batch_size, preprocessed_batch or vggt_world_points must be provided.

        Args:
            n_points (int): Number of points to sample.
            batch_size (int): Batch size.
            preprocessed_batch (dict): Preprocessed batch containing VGGT tokens and world points.
            vggt_world_points (torch.Tensor): VGGT world points. Shape (B, N, H, W, 3).
            cull_radius (float): Number of standard deviations from spatial_mean
                beyond which VGGT points are discarded.
            pure_noise_std (float): Standard deviation of the pure Gaussian noise.

        Returns:
            torch.Tensor: Sampled points. Shape (B, P, 3) or (B, P, D).
        """
        assert (
            (batch_size is not None)
            or (preprocessed_batch is not None)
            or (vggt_world_points is not None)
        ), "Either batch_size, preprocessed_batch or vggt_world_points must be provided"
        
        # If VGGT world points are provided directly, use them
        if vggt_world_points is not None:
            B = vggt_world_points.shape[0]
        
        # If preprocessed batch is provided, use the VGGT world points stored in it
        elif preprocessed_batch is not None:
            if "images" in preprocessed_batch:
                B = preprocessed_batch["images"].shape[0]
            else:
                B = preprocessed_batch["extrinsics"].shape[0]
            vggt_world_points = preprocessed_batch.get("vggt_world_points", None)
            
        # Else, just use the provided batch size for sampling Gaussian noise
        else:
            assert self.sample_from_vggt_world_points_with_sd is None
            B = batch_size

        P = n_points
        D = 3 if (not self.estimate_in_freq_enc) else self.frequency_encoder.output_dim
        device = self.device
        
        # Case 1: Source distribution is pure Gaussian noise
        if self.sample_from_vggt_world_points_with_sd is None:  # Start from completely random noise
            x0 = pure_noise_std * torch.randn(B, P, D, device=device, generator=generator)  # (B, P, 3) or (B, P, D)
        
        # Case 2: Source distribution is VGGT points + Gaussian noise
        else:
            # Sample a subset of VGGT points with size P
            x_vggt = vggt_world_points.contiguous().view(B, -1, 3)  # (B, N*H*W, 3)

            # FIXME: If n_points > N*H*W, we need to interpolate in pointmaps
            if P > x_vggt.shape[1]:
                raise ValueError(f"n_points ({P}) > N*H*W ({x_vggt.shape[1]})")
            
            # Cull VGGT points using dataset-level spatial statistics.
            # When renormalize_after_cull is active, cull_mean/cull_std hold the
            # *pre-cull* stats (used for the culling criterion) while scene_mean/
            # scene_std hold the *post-cull* stats (used for normalization).
            if (cull_radius is None) and (preprocessed_batch is not None):
                cull_radius = preprocessed_batch.get("cull_radius", None)
            if cull_radius is not None:
                if isinstance(cull_radius, torch.Tensor):
                    cr = cull_radius.view(B, 1)
                else:
                    cr = cull_radius
                _cull_mean = cull_mean if cull_mean is not None else (scene_mean if scene_mean is not None else self.spatial_mean)
                _cull_std = cull_std if cull_std is not None else (scene_std if scene_std is not None else self.spatial_std)
                std_dist = torch.norm(
                    (x_vggt - _cull_mean) / _cull_std, dim=-1
                )  # (B, N*H*W)
                radius_mask = std_dist < cr  # (B, N*H*W)

                vggt_points = []
                for i in range(B):
                    vggt_points_i_to_add = x_vggt[i][radius_mask[i]]  # (N_i_sampled, 3)
                    n_valid = vggt_points_i_to_add.shape[0]
                    if n_valid == 0:
                        _log.warning(
                            f"No VGGT points inside culling radius for batch index {i}. "
                            f"Falling back to sampling from all VGGT points."
                        )
                        sample_idx = torch.randint(
                            0,
                            x_vggt[i].shape[0],
                            (P,),
                            device=x_vggt.device,
                            generator=generator,
                        )
                        sampled_points_i = x_vggt[i][sample_idx]
                    elif n_valid >= P:
                        # If more than P points are available, sample without replacement
                        sample_idx = torch.randperm(n_valid, device=x_vggt.device, generator=generator)[:P]
                        sampled_points_i = vggt_points_i_to_add[sample_idx]
                    else:
                        # If less than P points are available, sample with replacement
                        sample_idx = torch.randint(0, n_valid, (P,), device=x_vggt.device, generator=generator)
                        sampled_points_i = vggt_points_i_to_add[sample_idx]

                    vggt_points.append(sampled_points_i)

                x_vggt = torch.stack(vggt_points, dim=0)  # (B, P, 3)
            
            # Case 3: Voxel-matched sampling between target points and VGGT points
            # Use the same voxel grid to keep source points near target points.
            if self.use_voxel_matching and preprocessed_batch is not None:
                assert (
                    self.use_voxel_matching
                    and (preprocessed_batch is not None)
                    and ("target_3d_points" in preprocessed_batch)
                ), f"Voxel matching requires preprocessed batch with target_3d_points."
                n_voxels_per_axis = self.voxel_match_n_voxels_per_axis
                voxel_count = n_voxels_per_axis ** 3
                matched_points = []

                n_std = 3.0
                min_corner = (self.spatial_mean - n_std * self.spatial_std).view(1, 3)
                max_corner = (self.spatial_mean + n_std * self.spatial_std).view(1, 3)
                voxel_size = (max_corner - min_corner) / float(n_voxels_per_axis)

                for i in range(B):
                    target_points_i = preprocessed_batch["target_3d_points"][i]  # (P, 3)
                    if target_points_i.shape[0] != P:
                        raise ValueError(
                            f"target_3d_points count ({target_points_i.shape[0]}) != n_points ({P})"
                        )
                    vggt_points_i = x_vggt[i]  # (N, 3)
                    vggt_rel = (vggt_points_i - min_corner) / voxel_size
                    vggt_idx = torch.floor(vggt_rel).to(torch.int64).clamp(0, n_voxels_per_axis - 1)
                    vggt_flat = (
                        vggt_idx[:, 0]
                        + n_voxels_per_axis * (vggt_idx[:, 1] + n_voxels_per_axis * vggt_idx[:, 2])
                    )

                    target_rel = (target_points_i - min_corner) / voxel_size
                    target_idx = torch.floor(target_rel).to(torch.int64).clamp(0, n_voxels_per_axis - 1)
                    target_flat = (
                        target_idx[:, 0]
                        + n_voxels_per_axis * (target_idx[:, 1] + n_voxels_per_axis * target_idx[:, 2])
                    )

                    counts_all = torch.bincount(vggt_flat, minlength=voxel_count)
                    start_all = torch.cumsum(counts_all, dim=0) - counts_all
                    vggt_sorted_idx = torch.argsort(vggt_flat)

                    target_counts = counts_all[target_flat]
                    has_match = target_counts > 0
                    rand_offsets = torch.floor(
                        torch.rand(P, device=device, generator=generator) * target_counts.clamp_min(1)
                    ).to(torch.long)
                    pick_positions = start_all[target_flat] + rand_offsets

                    selected_indices = torch.empty(P, dtype=torch.long, device=device)
                    if has_match.any():
                        selected_indices[has_match] = vggt_sorted_idx[pick_positions[has_match]]
                    if (~has_match).any():
                        selected_indices[~has_match] = torch.randint(
                            0,
                            vggt_points_i.shape[0],
                            (int((~has_match).sum().item()),),
                            device=device,
                            generator=generator,
                        )

                    matched_points.append(vggt_points_i[selected_indices])

                x_vggt = torch.stack(matched_points, dim=0)  # (B, P, 3)

            # Else, sample VGGT points randomly (we use same indices across the batch for now)
            else:
                sampled_pt_idx = torch.randperm(x_vggt.shape[1], device=x_vggt.device, generator=generator)[:P]
                x_vggt = x_vggt[:, sampled_pt_idx]  # (B, P, 3)
            
            # Lift the sampled VGGT points to flow space
            x_vggt = self.lift_points_to_flow_space(x_vggt, scene_mean, scene_std)  # (B, P, D)
            
            # Add noise to the sampled VGGT points
            noise = torch.empty_like(x_vggt)
            noise.normal_(generator=generator)
            x0 = x_vggt + noise * self.sample_from_vggt_world_points_with_sd  # (B, P, D)
        
        if self.estimate_normals:
            # Sample normals from a standard Gaussian distribution
            x0_normals = torch.empty_like(x0)
            x0_normals.normal_(generator=generator)  # (B, P, 3)
            
            # Handle normals with very small norms by clamping coordinates to +/- epsilon
            epsilon = 1e-6
            x0_normals = torch.where(
                x0_normals > 0.,
                x0_normals.clamp_min(epsilon),
                x0_normals.clamp_max(-epsilon),
            )  # (B, P, 3)
            
            # Normalize normals
            # (Sampling from a Gaussian and normalizing is equivalent 
            # to uniformly sampling on the unit sphere)
            x0_normals = torch.nn.functional.normalize(x0_normals, dim=-1)  # (B, P, 3)
            
            # Add normals to points in flow space.
            # The effective shift must match the target R6 gap that goes through
            # lift_points_to_flow_space: gap = shift_world * target_std * n / std.
            # If not using scene_std, we use the default global std.
            # To normalize the shift, we use a mean of 0 as shifts are independent of the distribution center.
            if scene_std is not None:
                # If using scene_std, we normalize the shift with the scene_std and mean of 0.
                effective_shift = self._normalize_3d(xyz=self.normals_shift_world_space, scene_mean=0., scene_std=scene_std)  # (B, 1, 3)
            else:
                # If not using scene_std, we use the default global std and mean of 0.
                effective_shift = self._normalize_3d(xyz=self.normals_shift_world_space, scene_mean=0.)  # (3,)

            x0_normals = x0 + effective_shift * x0_normals  # (B, P, 3)
            
            # Concatenate to obtain 6D points
            x0 = torch.cat([x0, x0_normals], dim=-1)  # (B, P, 6)
        
        return x0

    def forward(self, batch: dict, generator: Optional[torch.Generator] = None) -> dict:
        """Training forward pass (standard flow matching).

        Takes a *preprocessed* batch (from :class:`FfmDl3dvPreprocessedDataset`),
        samples a flow time ``t``, builds the interpolated point ``x_t`` between a
        source sample ``x_0`` and the GT target ``x_1``, runs the denoiser and
        returns the tensors consumed by ``training.losses.FlowLoss``.

        Only the standard (non mean-flow) branch is implemented.

        Returns a dict with ``source_points``, ``target_points``, ``t``,
        ``surface_estimates``, ``velocity_estimates``, ``conditional_velocity``,
        ``vggt_extrinsics`` and ``vggt_intrinsics``.
        """
        if self.use_mean_flow:
            raise NotImplementedError(
                "Mean-flow (iMF) training is not part of the released method; "
                "set use_mean_flow=false."
            )

        with torch.no_grad():
            # Preprocess inputs: reconstruct VGGT tokens / align GT points on GPU.
            # Only the cached path is supported (training data is preprocessed).
            if "cached_aggregated_tokens" in batch:
                batch = self.preprocess_from_cached(batch)
            elif not batch.get("has_been_preprocessed", False):
                raise NotImplementedError(
                    "FFM.forward requires a preprocessed (cached) batch. Raw-image "
                    "training preprocessing is not supported; use "
                    "scripts/preprocess.py to cache VGGT tokens."
                )
            aggregated_tokens_list = batch["aggregated_tokens_list"]
            patch_start_idx = batch["patch_start_idx"]

            # Per-scene normalization stats (None when disabled -> global buffers used)
            scene_mean, scene_std = None, None
            cull_mean, cull_std = None, None
            cull_radius = batch.get("cull_radius", None)
            if self.per_scene_normalize:
                scene_mean, scene_std = self._compute_scene_stats(
                    batch["vggt_world_points"]
                )  # (B, 1, 3), (B, 1, 3)

                # Recompute stats from culled VGGT points so the normalization
                # maps the post-cull distribution to exactly target_std.
                if self.renormalize_after_cull and cull_radius is not None:
                    cull_mean, cull_std = scene_mean, scene_std
                    scene_mean, scene_std = self._compute_post_cull_scene_stats(
                        batch["vggt_world_points"], cull_mean, cull_std, cull_radius,
                    )

            # Get points from target distribution and lift them to flow space.
            x1 = batch["target_points"]  # (B, P, 3) or (B, P, 6)
            if self.estimate_normals:
                assert x1.shape[-1] == 6, f"Expected 6D target points, got {x1.shape[-1]}D"
            x1 = self.lift_points_to_flow_space(x1, scene_mean, scene_std)  # (B, P, D)

            # Sample points from source distribution.
            x0 = self.sample_from_source_distribution(
                n_points=x1.shape[1],
                preprocessed_batch=batch,
                generator=generator,
                scene_mean=scene_mean,
                scene_std=scene_std,
                cull_mean=cull_mean,
                cull_std=cull_std,
            )  # (B, P, D)

            # Reshape to (B*P, ...) to get per-point samples.
            B = x0.shape[0]
            x0 = rearrange(x0, "B P C -> (B P) C")
            x1 = rearrange(x1, "B P C -> (B P) C")

            # Sample time.
            t = self.t_sampler.sample(x0.shape[0], device=self.device, generator=generator)

            # Interpolate along the (conditional OT) path.
            sample = self.path.sample(t=t, x_0=x0, x_1=x1)

            # Reshape back to (B, P, ...).
            x0 = rearrange(sample.x_0, "(B P) C -> B P C", B=B)
            xt = rearrange(sample.x_t, "(B P) C -> B P C", B=B)
            x1 = rearrange(sample.x_1, "(B P) C -> B P C", B=B)
            t = rearrange(sample.t, "(B P) -> B P", B=B)

        # Classifier-free guidance: disable VGGT features for some samples.
        if self.training and self.unconditional_ratio > 0.0:
            aggregated_tokens_list = self._apply_unconditional_masking(aggregated_tokens_list, B)

        # Denoiser forward pass.
        predictions = self.surface_net(
            xt, t, aggregated_tokens_list, patch_start_idx,
            vggt_world_points=batch["vggt_world_points"],
            scene_mean=scene_mean, scene_std=scene_std,
            cull_radius=cull_radius, cull_mean=cull_mean, cull_std=cull_std,
        )

        if self.prediction_mode == "target":
            surface_points = predictions
            if self.compute_loss_in_velocity:
                # Convert predicted target to velocity for the loss.
                surface_points_flat = rearrange(surface_points, "B P C -> (B P) C")
                xt_flat = rearrange(xt, "B P C -> (B P) C")
                t_flat = rearrange(t, "B P -> (B P) 1")
                velocities_flat = self.path.target_to_velocity(x_1=surface_points_flat, x_t=xt_flat, t=t_flat)
                velocities = rearrange(velocities_flat, "(B P) C -> B P C", B=B)
                dxt = rearrange(sample.dx_t, "(B P) C -> B P C", B=B)
            else:
                velocities = None
                dxt = None
        elif self.prediction_mode == "velocity":
            surface_points = None
            velocities = predictions
            dxt = rearrange(sample.dx_t, "(B P) C -> B P C", B=B)
        else:
            raise ValueError(f"Invalid prediction mode: {self.prediction_mode}")

        return {
            "source_points": x0,
            "target_points": x1,
            "t": t,
            "surface_estimates": surface_points,
            "velocity_estimates": velocities,
            "conditional_velocity": dxt,
            "vggt_extrinsics": batch["vggt_extrinsics"],
            "vggt_intrinsics": batch["vggt_intrinsics"],
        }

    @torch.no_grad()
    def inference(
        self,
        images: Optional[torch.Tensor] = None,
        query_points=None,
        aggregated_tokens_list=None,
        patch_start_idx=None,
        world_points=None,
        return_intermediates=False,
        num_steps=100,
        num_query_points=2**12,
        cull_radius=None,
        compressed_tokens=None,
        compressed_camera_tokens=None,
        guidance_scale: float = 0.0,
        compressed_tokens_uncond=None,
        compressed_camera_tokens_uncond=None,
        generator: Optional[torch.Generator] = None,
        n_latent_tokens: Optional[int] = None,
        # Deprecated — kept for backward compat.
        scene_center=None,
        scene_radius=None,
    ):
        """
        Estimate the surface points by solving the ODE of the flow conditioned on input images, starting from the query points.
        Doesn't support batching yet.

        Args:
            images (torch.Tensor, optional): Input images of shape (N, 3, H, W). Can be None when
                aggregated_tokens_list, patch_start_idx, and world_points are provided directly.
            query_points (torch.Tensor): Query points of shape (P, 3) or (P, 6).
            guidance_scale (float): Classifier-free guidance scale. 0.0 = no guidance (conditional only, guidance disabled).
                                   Higher values (e.g., 2.5) = stronger conditioning.
                                   Formula: pred = pred_cond + guidance_scale * (pred_cond - pred_uncond)

        Returns:
            torch.Tensor: Estimated points of shape (P, 3) or (P, 6). If return_intermediates is True, returns a list of intermediate surface points at each step, so (num_steps, P, 3) or (num_steps, P, 6).
        """

        if (aggregated_tokens_list is None) or (patch_start_idx is None) or (world_points is None):
            assert images is not None, "images must be provided when aggregated_tokens_list/patch_start_idx/world_points are not given"
            with torch.amp.autocast("cuda", dtype=self.dtype):
                predictions = self.vggt(images.unsqueeze(0), return_aggregated_tokens=True)
                aggregated_tokens_list = predictions["aggregated_tokens_list"]
                patch_start_idx = predictions["patch_start_idx"]

                vggt_extrinsics, vggt_intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
                vggt_depth = predictions["depth"]
                assert type(vggt_intrinsics) is torch.Tensor
                world_points = depths_to_points_parallel_batched(
                    vggt_intrinsics,
                    vggt_extrinsics,
                    vggt_depth,
                    to_world=True,
                )

        # Per-scene normalization stats
        scene_mean, scene_std = None, None
        cull_mean, cull_std = None, None
        if self.per_scene_normalize and world_points is not None:
            scene_mean, scene_std = self._compute_scene_stats(world_points)
            if self.renormalize_after_cull and cull_radius is not None:
                cull_mean, cull_std = scene_mean, scene_std
                scene_mean, scene_std = self._compute_post_cull_scene_stats(
                    world_points, cull_mean, cull_std, cull_radius,
                )

        if query_points is None:
            query_points = self.sample_from_source_distribution(
                n_points=num_query_points,
                batch_size=1,
                vggt_world_points=world_points,
                cull_radius=cull_radius,
                generator=generator,
                scene_mean=scene_mean,
                scene_std=scene_std,
                cull_mean=cull_mean,
                cull_std=cull_std,
            ).squeeze(0)  # (num_query_points, D)

        if images is not None:
            assert len(images.shape) == 4 and len(query_points.shape) == 2, f"Invalid shapes: {images.shape}, {query_points.shape}"
        else:
            assert len(query_points.shape) == 2, f"Invalid query_points shape: {query_points.shape}"
        
        # Classifier-free guidance: check if we need to run both conditional and unconditional inference
        use_cfg = guidance_scale != 0.0

        if use_cfg:
            _log.info(f"Using classifier-free guidance with scale={guidance_scale}")
            
        surface_points = self._run_inference_core(
            query_points=query_points,
            aggregated_tokens_list=aggregated_tokens_list,
            patch_start_idx=patch_start_idx,
            world_points=world_points,
            num_steps=num_steps,
            return_intermediates=return_intermediates,
            compressed_tokens=compressed_tokens,
            compressed_camera_tokens=compressed_camera_tokens,
            use_cfg=use_cfg,
            guidance_scale=guidance_scale,
            compressed_tokens_uncond=compressed_tokens_uncond,
            compressed_camera_tokens_uncond=compressed_camera_tokens_uncond,
            scene_mean=scene_mean,
            scene_std=scene_std,
            n_latent_tokens=n_latent_tokens,
            cull_radius=cull_radius,
            cull_mean=cull_mean,
            cull_std=cull_std,
        )
        
        assert type(surface_points) is torch.Tensor
        _log.info(f"Inference completed with {num_steps} steps. {surface_points.shape=}")
        if not return_intermediates:
            surface_points = surface_points.unsqueeze(0)  # (1, P, 3) or (1, P, 6)
        surface_points = self.unlift_points_from_flow_space(surface_points, scene_mean, scene_std)

        if self.estimate_normals:
            assert surface_points.shape[-1] == 6, f"Expected 6D points, got {surface_points.shape[-1]}D"
            surface_points, surface_normals = self.get_points_normals_from_r6_points(r6_points=surface_points)  # (P, 3), (P, 3)
            if not return_intermediates:
                surface_points = surface_points.squeeze(0)  # (P, 3)
                surface_normals = surface_normals.squeeze(0)  # (P, 3)
            return surface_points, surface_normals
        else:
            if not return_intermediates:
                surface_points = surface_points.squeeze(0)  # (P, 3)
            return surface_points

    def _run_inference_core(
        self,
        query_points: torch.Tensor,
        aggregated_tokens_list: list,
        patch_start_idx: int,
        world_points: torch.Tensor,
        num_steps: int,
        return_intermediates: bool,
        compressed_tokens: Optional[torch.Tensor],
        compressed_camera_tokens: Optional[torch.Tensor],
        use_cfg: bool = False,
        guidance_scale: float = 0.0,
        compressed_tokens_uncond: Optional[torch.Tensor] = None,
        compressed_camera_tokens_uncond: Optional[torch.Tensor] = None,
        scene_mean: Optional[torch.Tensor] = None,
        scene_std: Optional[torch.Tensor] = None,
        n_latent_tokens: Optional[int] = None,
        cull_radius: Optional[torch.Tensor] = None,
        cull_mean: Optional[torch.Tensor] = None,
        cull_std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Core inference logic that can be run conditionally or unconditionally.

        Args:
            unconditional (bool): If True, zero out VGGT features for unconditional generation.

        Returns:
            torch.Tensor: Surface points in flow space.
        """
        # Apply unconditional masking if requested
        if self.surface_net.use_learned_tokens:
            # Learned tokens are expanded to batch size inside SurfaceNet.forward;
            # pre-compute them here so the ODE solver doesn't recompute each step.
            B = aggregated_tokens_list[self.surface_net.intermediate_layer_idx[0]].shape[0]
            compressed_tokens = self.surface_net.learned_compressed_tokens.unsqueeze(0).expand(B, -1, -1)
            compressed_camera_tokens = None
            if self.surface_net.encode_camera_tokens_separately:
                compressed_camera_tokens = self.surface_net.learned_compressed_camera_tokens.unsqueeze(0).expand(B, -1, -1)
            compressed_tokens_uncond = None
            compressed_camera_tokens_uncond = None
        else:
            if use_cfg:
                # Zero out all VGGT features
                aggregated_tokens_list_uncond = [torch.zeros_like(tokens) if tokens is not None else None for tokens in aggregated_tokens_list]

                # Get the compressed unconditional tokens
                # FIXME: Using a single zero token should be enough, no?
                if compressed_tokens_uncond is None:
                    compressed_tokens_uncond, _ = self.surface_net.get_compressed_tokens(
                        aggregated_tokens_list_uncond, patch_start_idx,
                        vggt_world_points=world_points,
                        scene_mean=scene_mean, scene_std=scene_std,
                        n_latent_tokens=n_latent_tokens,
                        cull_radius=cull_radius, cull_mean=cull_mean, cull_std=cull_std,
                    )
                if (compressed_camera_tokens_uncond is None) and self.surface_net.use_camera_tokens and self.surface_net.encode_camera_tokens_separately:
                    compressed_camera_tokens_uncond = self.surface_net.get_compressed_camera_tokens(
                        aggregated_tokens_list_uncond, patch_start_idx,
                    )

            # Get the compressed tokens if not provided
            if compressed_tokens is None:
                compressed_tokens, _ = self.surface_net.get_compressed_tokens(
                    aggregated_tokens_list, patch_start_idx,
                    vggt_world_points=world_points,
                    scene_mean=scene_mean, scene_std=scene_std,
                    n_latent_tokens=n_latent_tokens,
                    cull_radius=cull_radius, cull_mean=cull_mean, cull_std=cull_std,
                )
            if (compressed_camera_tokens is None) and self.surface_net.use_camera_tokens and self.surface_net.encode_camera_tokens_separately:
                compressed_camera_tokens = self.surface_net.get_compressed_camera_tokens(aggregated_tokens_list, patch_start_idx)

        timesteps = self.t_sampler.get_time_grid(num_steps, device=query_points.device)

        # --- Mean flow: custom Euler loop passing h at each step ---
        # Training uses the paper convention z_t = (1-t)*data + t*noise,
        # so inference goes from t=1 (noise) to t=0 (data).
        if self.use_mean_flow:
            x = query_points  # (P, D) — noise
            intermediates = [x] if return_intermediates else None

            # Reverse time grid: go from t=1 (noise) → t=0 (data)
            mf_timesteps = timesteps.flip(0)

            for i in range(num_steps):
                t_curr = mf_timesteps[i]
                t_next = mf_timesteps[i + 1]
                h_step = t_curr - t_next  # positive (stepping backward in t)

                # Forward through surface_net (u-head only)
                result = self.surface_net(
                    x, t_curr, aggregated_tokens_list, patch_start_idx,
                    vggt_world_points=world_points,
                    compressed_tokens=compressed_tokens,
                    compressed_camera_tokens=compressed_camera_tokens,
                    h=h_step,
                    u_head_only=True,
                    scene_mean=scene_mean,
                    scene_std=scene_std,
                    cull_radius=cull_radius,
                    cull_mean=cull_mean,
                    cull_std=cull_std,
                )
                u = result[0] if isinstance(result, tuple) else result

                # CFG if enabled
                if use_cfg:
                    u_uncond = self.surface_net(
                        x, t_curr, aggregated_tokens_list, patch_start_idx,
                        vggt_world_points=world_points,
                        compressed_tokens=compressed_tokens_uncond,
                        compressed_camera_tokens=compressed_camera_tokens_uncond,
                        h=h_step,
                        u_head_only=True,
                        scene_mean=scene_mean,
                        scene_std=scene_std,
                        cull_radius=cull_radius,
                        cull_mean=cull_mean,
                        cull_std=cull_std,
                    )
                    u_uncond = u_uncond[0] if isinstance(u_uncond, tuple) else u_uncond
                    u = u + guidance_scale * (u - u_uncond)

                # Euler step (subtract because we go from t=1 toward t=0)
                x = x - h_step * u

                if return_intermediates:
                    intermediates.append(x)

            if return_intermediates:
                return torch.stack(intermediates, dim=0)  # (num_steps+1, P, D)
            return x

        # --- Standard flow matching: use ODESolver ---
        velocity_model = VelocityModel(
            denoiser=self.surface_net, path=self.path, prediction_mode=self.prediction_mode,
            use_cfg=use_cfg, guidance_scale=guidance_scale,
        )

        solver = ODESolver(velocity_model=velocity_model)
        surface_points = solver.sample(
            x_init=query_points,
            method="euler",
            step_size=None,
            time_grid=timesteps,
            aggregated_tokens_list=aggregated_tokens_list,
            patch_start_idx=patch_start_idx,
            return_intermediates=return_intermediates,
            vggt_world_points=world_points,
            compressed_tokens=compressed_tokens,
            compressed_camera_tokens=compressed_camera_tokens,
            compressed_tokens_uncond=compressed_tokens_uncond,
            compressed_camera_tokens_uncond=compressed_camera_tokens_uncond,
            scene_mean=scene_mean,
            scene_std=scene_std,
            cull_radius=cull_radius,
            cull_mean=cull_mean,
            cull_std=cull_std,
        )

        return surface_points
    
    @torch.no_grad()
    def batched_inference(
        self,
        images: Optional[torch.Tensor] = None,
        query_points=None,
        aggregated_tokens_list=None,
        patch_start_idx=None,
        world_points=None,
        return_intermediates=False,
        num_steps=50,
        num_query_points=99_999,
        num_points_per_batch=33_333,
        cull_radius=None,
        guidance_scale: float = 0.0,
        generator: Optional[torch.Generator] = None,
        n_latent_tokens: Optional[int] = None,
        # Deprecated — kept for backward compat.
        scene_center=None,
        scene_radius=None,
    ):
        """
        Batched inference with optional classifier-free guidance.
        
        Args:
            images (torch.Tensor, optional): Input images of shape (N, 3, H, W). Can be None when
                aggregated_tokens_list, patch_start_idx, and world_points are provided directly.
            guidance_scale (float): Classifier-free guidance scale. 0.0 = no guidance.
        """  
        if query_points is not None:
            num_query_points = query_points.shape[0]
        
        n_batches = (num_query_points + num_points_per_batch - 1) // num_points_per_batch

        surface_points_list = []
        if self.estimate_normals:
            surface_normals_list = []

        for batch_idx in range(n_batches):
            batch_start = batch_idx * num_points_per_batch
            batch_end = min(batch_start + num_points_per_batch, num_query_points)
            
            if query_points is None:
                batch_query_points = None
            else:
                batch_query_points = query_points[batch_start:batch_end]
            
            batch_surface_results = self.inference(
                images=images,
                query_points=batch_query_points,
                aggregated_tokens_list=aggregated_tokens_list,
                patch_start_idx=patch_start_idx,
                world_points=world_points,
                return_intermediates=return_intermediates,
                num_steps=num_steps,
                num_query_points=batch_end - batch_start,
                cull_radius=cull_radius,
                guidance_scale=guidance_scale,
                generator=generator,
                n_latent_tokens=n_latent_tokens,
            )
            if self.estimate_normals:
                batch_surface_points, batch_surface_normals = batch_surface_results
            else:
                batch_surface_points = batch_surface_results
            
            surface_points_list.append(batch_surface_points)
            if self.estimate_normals:
                surface_normals_list.append(batch_surface_normals)

        surface_points = torch.cat(surface_points_list, dim=-2)
        if self.estimate_normals:
            surface_normals = torch.cat(surface_normals_list, dim=-2)
            return surface_points, surface_normals
        else:
            return surface_points

    @torch.no_grad()
    def preprocess_from_cached(
        self,
        batch: dict,
        transform_colmap_points: bool = False,
    ) -> dict:
        """Reconstruct VGGT token list from cached tensor and align GT points on GPU.

        This replaces :meth:`preprocess_inputs` when the batch comes from
        :class:`FfmDl3dvPreprocessedDataset`.  Everything that ``preprocess_inputs``
        does (run VGGT, compute alignment, transform points) is already pre-computed
        and stored in the ``.pt`` cache files — except for the actual GT-point
        alignment which we do here so that fresh random GT samples can be drawn
        each epoch.
        """
        if batch.get("has_been_preprocessed", False):
            return batch

        # -- Reconstruct sparse aggregated_tokens_list (24 entries, mostly None) --
        cached_tokens = batch["cached_aggregated_tokens"]  # (B, n_used, N, S, D)
        layer_indices = self.surface_net.intermediate_layer_idx  # e.g. [4, 11, 17, 23]
        n_total_layers = 24
        aggregated_tokens_list: list = [None] * n_total_layers
        for i, layer_idx in enumerate(layer_indices):
            aggregated_tokens_list[layer_idx] = cached_tokens[:, i]  # (B, N, S, D)

        batch["aggregated_tokens_list"] = aggregated_tokens_list
        batch["patch_start_idx"] = self.vggt.aggregator.patch_start_idx

        # -- Alignment matrices --
        L = batch["alignment_L"]  # (B, 3, 3)
        T = batch["alignment_T"]  # (B, 3)

        # -- Align target 3D points: Y = X @ L + T --
        batch["target_3d_points"] = transform_points(
            X=batch["target_3d_points"], L=L, T=T,
        )

        # -- Align chamfer evaluation points (if present) --
        if "chamfer_target_3d_points" in batch:
            batch["chamfer_target_3d_points"] = transform_points(
                X=batch["chamfer_target_3d_points"], L=L, T=T,
            )

        # -- Align target normals (rotation only, then re-normalize) --
        if "target_normals" in batch:
            target_normals = transform_points(
                X=batch["target_normals"], L=L, T=torch.zeros_like(T),
            )
            batch["target_normals"] = torch.nn.functional.normalize(target_normals, dim=-1)

        # -- Build N-D target points (6D if normals, else 3D) --
        if self.estimate_normals:
            batch["target_points"] = self.get_r6_points_from_points_normals(
                points=batch["target_3d_points"], normals=batch["target_normals"],
            )
        else:
            batch["target_points"] = batch["target_3d_points"]

        # -- Transform scene extent --
        alignment_scaling = torch.linalg.svdvals(L).mean(dim=-1)  # (B,)
        batch["scene_radius"] = batch["scene_radius"] * alignment_scaling.reshape(-1)
        batch["scene_center"] = transform_points(
            X=batch["scene_center"], L=L, T=T,
        )

        # -- Optionally transform COLMAP extrinsics / world points --
        if transform_colmap_points:
            batch_size = L.shape[0]
            R = L / alignment_scaling.reshape(-1, 1, 1)
            batch["extrinsics"][:, :, :3, :3] = torch.einsum(
                "bnij,bjk->bnik", batch["extrinsics"][:, :, :3, :3], R,
            )
            batch["extrinsics"][:, :, :3, 3] *= alignment_scaling.reshape(-1, 1, 1)
            batch["extrinsics"][:, :, :3, 3] -= torch.einsum(
                "bnij,bj->bni", batch["extrinsics"][:, :, :3, :3], T,
            )

            if "colmap_world_points" in batch:
                old_shape = batch["colmap_world_points"].shape
                batch["colmap_world_points"] = transform_points(
                    X=batch["colmap_world_points"].reshape(batch_size, -1, 3),
                    L=L, T=T,
                ).reshape(old_shape)

        # -- Propagate VGGT-predicted cameras (already in VGGT space) --
        if "vggt_extrinsics" not in batch:
            batch["vggt_extrinsics"] = batch.get("vggt_extrinsics", None)
        if "vggt_intrinsics" not in batch:
            batch["vggt_intrinsics"] = batch.get("vggt_intrinsics", None)

        # VGGT world points (optional, needed for source distribution sampling)
        if "vggt_world_points" not in batch:
            batch["vggt_world_points"] = None

        batch["has_been_preprocessed"] = True
        return batch

    @torch.no_grad()
    def preprocess_images(
        self,
        images: torch.Tensor,
        cull_radius: Optional[Union[float, torch.Tensor]] = None,
    ) -> dict:
        """Build a preprocessed, GT-free batch directly from raw images.

        Intended for inference: given a few images of a scene, run VGGT once and
        assemble a batch dict with the same keys as the training pipeline
        produces after :meth:`preprocess_from_cached`, minus everything that
        depends on GT data (``target_*``, ``chamfer_*``, COLMAP extrinsics,
        ``alignment_L/T``…).

        Since there is no COLMAP frame to align against, everything is already
        expressed in the VGGT frame — no alignment transform is applied.

        Args:
            images: Either ``(N, 3, H, W)`` for a single scene or
                ``(B, N, 3, H, W)`` for a batch of scenes. Values should be in
                ``[0, 1]`` (same convention as :meth:`meshify_by_tsdf`).
            cull_radius: Optional culling radius to store under the
                ``"cull_radius"`` key (matches dataloader behaviour). Can be a
                scalar, or a tensor of shape ``(B,)``.

        Returns:
            dict: batch ready to be fed to :meth:`forward` (it will short-circuit
            both ``preprocess_from_cached`` and ``preprocess_inputs`` thanks to
            the ``"has_been_preprocessed"`` flag).
        """
        # -- Normalize input shape to (B, N, 3, H, W) --
        if images.ndim == 4:
            images = images.unsqueeze(0)
        assert images.ndim == 5 and images.shape[2] == 3, (
            f"Expected images of shape (N, 3, H, W) or (B, N, 3, H, W), got {tuple(images.shape)}"
        )
        images = images.to(self.device)
        B, N, _, H, W = images.shape

        # -- Run VGGT (mirrors meshify_by_tsdf / preprocess_inputs) --
        with torch.amp.autocast("cuda", dtype=self.dtype):
            predictions = self.vggt(images, return_aggregated_tokens=True)

        aggregated_tokens_list = predictions["aggregated_tokens_list"]
        patch_start_idx = predictions["patch_start_idx"]

        # -- Recover VGGT cameras --
        vggt_extrinsics, vggt_intrinsics = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:],
        )  # (B, N, 3, 4), (B, N, 3, 3)
        assert isinstance(vggt_intrinsics, torch.Tensor)
        vggt_depth = predictions["depth"]  # (B, N, H, W, 1)

        # -- Backproject VGGT depth into world points --
        vggt_world_points = depths_to_points_parallel_batched(
            vggt_intrinsics,
            vggt_extrinsics,
            vggt_depth,
            to_world=True,
        )  # (B, N, H, W, 3)

        # -- Scene extent from VGGT cameras (no COLMAP / GT available) --
        cameras = get_multi_cameras_from_intrinsics_and_extrinsics(
            intrinsics=vggt_intrinsics,
            extrinsics=vggt_extrinsics,
            images=images,
            data_device=str(self.device),
        )
        # NOTE: must use the ``MultiCameras``-aware overload here. The
        # ``List[Camera]`` one imported at module top does
        # ``torch.cat([c.camera_center.view(1, 3) for c in cameras])``, which
        # silently collapses/reshapes ``camera_center`` (shape ``(B, N, 3)``)
        # and yields a garbage scene extent.
        scene_extent = get_multi_cameras_spatial_extent(cameras)
        scene_center = scene_extent["avg_cam_center"].to(torch.float32)  # (B, 1, 3)
        scene_radius = scene_extent["radius"].to(torch.float32).reshape(B)  # (B,)

        # -- Assemble the preprocessed batch (GT-free) --
        batch: dict = {
            "images": images,
            "rgb_images": images,
            "frame_num": torch.tensor(images.shape[1], dtype=torch.int32),
            "aggregated_tokens_list": aggregated_tokens_list,
            "patch_start_idx": patch_start_idx,
            "vggt_extrinsics": vggt_extrinsics,
            "vggt_intrinsics": vggt_intrinsics,
            "vggt_depth": vggt_depth,
            "vggt_world_points": vggt_world_points,
            "scene_center": scene_center,
            "scene_radius": scene_radius,
            "has_been_preprocessed": True,
            "vggt_cameras": cameras,
        }

        # -- Confidence maps --
        if "depth_conf" in predictions:
            batch["vggt_depth_conf"] = predictions["depth_conf"]  # (B, N, H, W)

        if cull_radius is not None:
            if not isinstance(cull_radius, torch.Tensor):
                cull_radius = torch.as_tensor(
                    cull_radius, dtype=torch.float32, device=self.device,
                )
            if cull_radius.ndim == 0:
                cull_radius = cull_radius.expand(B)
            batch["cull_radius"] = cull_radius.to(self.device)

        return batch


