"""SurfaceNet: the Surflo denoiser.

Aggregates VGGT patch/camera tokens, compresses them into a fixed-size global
latent with a Perceiver compressor, and decodes a per-query-point flow field
(velocity or target) conditioned on that latent and the flow time ``t``.
"""
from typing import List, Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat


class SurfaceNet(nn.Module):
    """A network that predicts surface points from VGGT features, query points and flow time t. To be used within the FFM model."""

    def __init__(
        self,
        intermediate_layer_idx: List[int],
        compressor: nn.Module,
        decoder: nn.Module,
        project_aggregated_tokens: bool,
        project_aggregated_tokens_dim: int,
        use_camera_tokens: bool,
        encode_camera_tokens_separately: bool=True,
        camera_tokens_compressor: nn.Module=None,
        use_3d_positional_encoding_for_patch_tokens: bool=True,
        tokens_3d_positional_encoder: nn.Module=None,
        mask_tokens_outside_cull_radius: bool=False,
        use_mask_to_cull_tokens: bool=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.intermediate_layer_idx = list(intermediate_layer_idx)
        self.project_aggregated_tokens = project_aggregated_tokens
        self.aggregated_tokens_dim = project_aggregated_tokens_dim if project_aggregated_tokens else 2048

        # Compressor, decoder
        self.compressor = compressor
        self.field_decoder = decoder

        self.use_camera_tokens = use_camera_tokens
        self.encode_camera_tokens_separately = (
            self.use_camera_tokens
            and encode_camera_tokens_separately
            and (camera_tokens_compressor is not None)
        )
        if use_camera_tokens:
            if self.encode_camera_tokens_separately:
                self.camera_tokens_compressor = camera_tokens_compressor

        if project_aggregated_tokens:
            self.token_projector = nn.Linear(2048, project_aggregated_tokens_dim, bias=True)
            if self.encode_camera_tokens_separately:
                self.camera_token_projector = nn.Linear(2048, project_aggregated_tokens_dim, bias=True)
        
        self.use_3d_positional_encoding_for_patch_tokens = use_3d_positional_encoding_for_patch_tokens
        if use_3d_positional_encoding_for_patch_tokens:
            assert tokens_3d_positional_encoder is not None
            self.tokens_3d_positional_encoder = tokens_3d_positional_encoder

        self.mask_tokens_outside_cull_radius = mask_tokens_outside_cull_radius
        # When True, build a (-inf / 0) attn_mask that forbids the compressor's
        # cross-attention from attending to masked keys. This is the "strict"
        # path but disqualifies FlashAttention in PyTorch SDPA (falls back to
        # memory-efficient attention, ~1.5-2x slower on H100/bf16).
        # When False, only zero out masked tokens (K=V=0) and skip the attn
        # mask entirely; the output is still zero-contribution from masked
        # tokens, but softmax probability mass is no longer renormalized.
        # FlashAttention remains enabled.
        self.use_mask_to_cull_tokens = use_mask_to_cull_tokens

        self.use_learned_tokens = False

    def setup_learned_tokens(self):
        """Replace compressor outputs with learned nn.Parameter tokens for fast overfitting.

        When enabled, the compressor and all modules feeding into it are deleted,
        and the decoder receives learned nn.Parameter tokens directly.
        """
        self.use_learned_tokens = True

        # Read shapes before deleting modules
        num_latent_tokens, embed_dim = self.compressor.latent_tokens.shape
        self.learned_compressed_tokens = nn.Parameter(
            torch.randn(num_latent_tokens, embed_dim) * 1e-2
        )

        if self.encode_camera_tokens_separately:
            num_cam_tokens, cam_dim = self.camera_tokens_compressor.latent_tokens.shape
            self.learned_compressed_camera_tokens = nn.Parameter(
                torch.randn(num_cam_tokens, cam_dim) * 1e-2
            )
            del self.camera_tokens_compressor
            if hasattr(self, "camera_token_projector"):
                del self.camera_token_projector

        # Delete all modules in the compressor pipeline (unused → DDP errors)
        del self.compressor
        if hasattr(self, "token_projector"):
            del self.token_projector
        if hasattr(self, "tokens_3d_positional_encoder"):
            del self.tokens_3d_positional_encoder

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float,
        aggregated_tokens_list: list,
        patch_start_idx: int,
        compressed_tokens: Optional[torch.Tensor] = None,
        compressed_camera_tokens: Optional[torch.Tensor] = None,
        vggt_world_points: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
        u_head_only: bool = False,
        scene_mean: Optional[torch.Tensor] = None,
        scene_std: Optional[torch.Tensor] = None,
        cull_radius: Optional[torch.Tensor] = None,
        cull_mean: Optional[torch.Tensor] = None,
        cull_std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the surface model. Input can be batched or unbatched.

        Args:
            x (torch.Tensor, (B, P, D) or (P, D)): Query points x_t to predict surface points. D can be 3 or 6.
            t (torch.Tensor, (B, P) or (B,) or (,)): Time values for the flow path.
            aggregated_tokens_list (list): List of VGGT aggregated tokens for each image in the batch.
            patch_start_idx (int): Start indices of patches in the aggregated tokens.
            compressed_tokens (torch.Tensor, (B, N, D)): Compressed latent tokens. If None, they will be computed.
            compressed_camera_tokens (torch.Tensor, (B, 1, D)): Compressed camera tokens. If None, they will be computed.
            h (torch.Tensor, optional): Step size h = t - r for mean flow conditioning. Shape (B, P) or (B, P, 1).
            u_head_only (bool): If True, skip v-head computation (used during JVP).
            scene_mean / scene_std: optional per-scene overrides of shape (B, 1, 3).
            cull_radius / cull_mean / cull_std: culling params forwarded to token masking.

        Returns:
            torch.Tensor or tuple: Surface points (B, P, D), or (u, v) tuple when mean flow is active.
        """

        if type(t) is float:
            t = torch.tensor(t, device=x.device)
        assert type(t) is torch.Tensor

        if len(x.shape) == 2:
            BATCHFY_MODE = True
            B, P, D = 1, x.shape[0], x.shape[1]
            x = x.unsqueeze(0)  # (1, P, 3) or (1, P, 6)
        else:
            BATCHFY_MODE = False
            B, P, D = x.shape

        if len(t.shape) == 0:
            t = repeat(t, "-> B P 1", B=B, P=P)
        elif len(t.shape) == 1:
            t = repeat(t, "B -> B P 1", P=P)
        elif len(t.shape) == 2:
            t = t.unsqueeze(-1)  # (B, P, 1)

        # Reshape h to match t if provided
        if h is not None:
            if len(h.shape) == 0:
                h = repeat(h, "-> B P 1", B=B, P=P)
            elif len(h.shape) == 1:
                h = repeat(h, "B -> B P 1", P=P)
            elif len(h.shape) == 2:
                h = h.unsqueeze(-1)  # (B, P, 1)

        # Get compressed tokens
        write_ca_mask = None
        if self.use_learned_tokens:
            compressed_tokens = self.learned_compressed_tokens.unsqueeze(0).expand(B, -1, -1)
            if self.encode_camera_tokens_separately:
                compressed_camera_tokens = self.learned_compressed_camera_tokens.unsqueeze(0).expand(B, -1, -1)
        else:
            if compressed_tokens is None:
                compressed_tokens, write_ca_mask = self.get_compressed_tokens(
                    aggregated_tokens_list, patch_start_idx,
                    vggt_world_points=vggt_world_points,
                    scene_mean=scene_mean, scene_std=scene_std,
                    cull_radius=cull_radius, cull_mean=cull_mean, cull_std=cull_std,
                )
            if (compressed_camera_tokens is None) and self.use_camera_tokens and self.encode_camera_tokens_separately:
                compressed_camera_tokens = self.get_compressed_camera_tokens(aggregated_tokens_list, patch_start_idx)

        # Decode field
        result = self.field_decoder(x, t, compressed_tokens, compressed_camera_tokens, h=h, u_head_only=u_head_only, write_ca_mask=write_ca_mask)

        if isinstance(result, tuple):
            u_field, v_field = result
            if BATCHFY_MODE:
                u_field = u_field.squeeze(0)
                v_field = v_field.squeeze(0)
            return u_field, v_field
        else:
            if BATCHFY_MODE:
                result = result.squeeze(0)
            return result

    def get_compressed_tokens(
        self,
        aggregated_tokens_list: list,
        patch_start_idx: torch.Tensor,
        vggt_world_points: Optional[torch.Tensor] = None,
        scene_mean: Optional[torch.Tensor] = None,
        scene_std: Optional[torch.Tensor] = None,
        n_latent_tokens: Optional[int] = None,
        cull_radius: Optional[torch.Tensor] = None,
        cull_mean: Optional[torch.Tensor] = None,
        cull_std: Optional[torch.Tensor] = None,
    ):
        # Aggregate scene tokens
        aggregated_tokens, read_ca_mask = self._build_aggregated_tokens(
            aggregated_tokens_list, patch_start_idx,
            vggt_world_points=vggt_world_points,
            scene_mean=scene_mean, scene_std=scene_std,
            cull_radius=cull_radius, cull_mean=cull_mean, cull_std=cull_std,
        )  # (B, L*N*M, D)

        # Compress aggregated tokens
        compressed_tokens, write_ca_mask = self.compressor(
            aggregated_tokens, n_latent_tokens=n_latent_tokens,
            read_ca_mask=read_ca_mask,
        )  # (B, K, D), optional mask

        return compressed_tokens, write_ca_mask
    
    def get_compressed_camera_tokens(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        patch_start_idx: int,
    ) -> torch.Tensor:
        # Get camera tokens
        camera_tokens = self._build_camera_tokens(aggregated_tokens_list, patch_start_idx)  # (B, N, D)

        # Compress camera tokens (ignore write_ca_mask — camera compressor is not elastic)
        compressed_camera_tokens, _ = self.camera_tokens_compressor(camera_tokens)  # (B, 1, D)

        return compressed_camera_tokens

    def _build_aggregated_tokens(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        patch_start_idx: int,
        override_intermediate_layer_idx: Optional[List[int]] = None,
        vggt_world_points: Optional[torch.Tensor] = None,
        scene_mean: Optional[torch.Tensor] = None,
        scene_std: Optional[torch.Tensor] = None,
        cull_radius: Optional[torch.Tensor] = None,
        cull_mean: Optional[torch.Tensor] = None,
        cull_std: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build the concatenated aggregated tokens, optionally with 3D
        positional encoding and spatial culling mask.

        Returns:
            agg_tokens: (B, T, D) — the aggregated (and optionally masked) tokens.
            read_ca_mask: (B, 1, 1, T) float tensor with 0 for active and
                -inf for masked positions, or None when no masking is applied.
                Suitable as ``attn_mask`` for the compressor's cross-attention.
        """
        aggregated_tokens = []

        if override_intermediate_layer_idx is None:
            override_intermediate_layer_idx = self.intermediate_layer_idx

        # Get patch tokens from the intermediate layers
        n_patch_tokens = 0
        for layer_idx in override_intermediate_layer_idx:
            layer_tokens = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]  # (B, N, M, D)
            layer_tokens = rearrange(layer_tokens, "B N M D -> B (N M) D")  # (B, N*M, D)
            aggregated_tokens.append(layer_tokens)
            n_patch_tokens = n_patch_tokens + layer_tokens.shape[1]  # N*M
        
        # For camera tokens, we only use the last layer.
        # Camera tokens are the first tokens of the sequence.
        if self.use_camera_tokens:
            camera_tokens = aggregated_tokens_list[-1][:, :, 0, :]  # (B, N, D)
            aggregated_tokens.append(camera_tokens)
        
        # Concatenate all tokens
        agg_tokens = torch.cat(aggregated_tokens, dim=1)  # (B, L*N*M, D) or (B, L*N*M+N, D) with L = len(intermediate_layer_idx)

        # Project aggregated tokens to desired dimension
        if self.project_aggregated_tokens:
            agg_tokens = self.token_projector(agg_tokens)

        read_ca_mask = None

        # Build 3D positional encoding for patch tokens
        if self.use_3d_positional_encoding_for_patch_tokens:
            assert vggt_world_points is not None
            
            # Interpolate VGGT world points to the tokens size
            B, N, H, W, _ = vggt_world_points.shape  # (B, N, H, W, 3)
            patch_size = 14
            pos_3d = torch.nn.functional.interpolate(
                input=rearrange(vggt_world_points, "B N H W C -> (B N) C H W"),
                size=(H//patch_size, W//patch_size),
                mode='bilinear',
            )
            pos_3d = rearrange(pos_3d, "(B N) C H W -> B (N H W) C", B=B, N=N)  # (B, N*M, 3)

            # When per-scene normalization is active, pre-normalize the 3D
            # points with per-scene stats and bypass the encoder's internal
            # normalization (which uses global stats).
            if scene_mean is not None and scene_std is not None:
                freq_enc = self.tokens_3d_positional_encoder.frequency_encoder
                target_std = freq_enc.target_std if freq_enc._normalize else 1.0
                pos_enc = (pos_3d - scene_mean) / scene_std * target_std
                saved_normalize = freq_enc._normalize
                freq_enc._normalize = False
                pos_enc = self.tokens_3d_positional_encoder(pos_enc, t=None)
                freq_enc._normalize = saved_normalize
            else:
                pos_enc = self.tokens_3d_positional_encoder(pos_3d, t=None)  # (B, N*M, D)

            # Reshape the positional encoding
            n_patch_tokens_per_image = (H//patch_size) * (W//patch_size)
            pos_enc = pos_enc.reshape(B, N * n_patch_tokens_per_image, pos_enc.shape[-1]).repeat(1, len(override_intermediate_layer_idx), 1)  # (B, L*N*M, D)
            
            # Apply positional encoding to the aggregated tokens
            agg_tokens[:, :n_patch_tokens, :] = agg_tokens[:, :n_patch_tokens, :] + pos_enc

            # --- Spatial culling mask ---
            # When enabled, mask out patch tokens whose 3D position falls
            # outside the culling sphere so the compressor ignores them.
            if self.mask_tokens_outside_cull_radius and cull_radius is not None:
                _mask_mean = cull_mean if cull_mean is not None else (scene_mean if scene_mean is not None else None)
                _mask_std = cull_std if cull_std is not None else (scene_std if scene_std is not None else None)
                if _mask_mean is not None and _mask_std is not None:
                    std_dist = torch.norm(
                        (pos_3d - _mask_mean) / _mask_std, dim=-1
                    )  # (B, N*M)
                    if isinstance(cull_radius, torch.Tensor):
                        cr = cull_radius.view(B, 1)
                    else:
                        cr = cull_radius
                    inside = std_dist < cr  # (B, N*M) bool

                    # Repeat the per-image mask for each layer copy, then
                    # append True for camera tokens (always kept).
                    inside_layers = inside.repeat(1, len(override_intermediate_layer_idx))  # (B, L*N*M)
                    T = agg_tokens.shape[1]
                    n_extra = T - inside_layers.shape[1]  # camera tokens etc.
                    if n_extra > 0:
                        inside_full = torch.cat([
                            inside_layers,
                            inside_layers.new_ones(B, n_extra, dtype=torch.bool),
                        ], dim=1)  # (B, T)
                    else:
                        inside_full = inside_layers

                    # Zero out masked tokens (sets K=V=0 in the compressor's
                    # cross-attention, so masked positions contribute nothing
                    # to the output regardless of the attention mask).
                    agg_tokens = agg_tokens * inside_full.unsqueeze(-1).to(agg_tokens.dtype)

                    # Optionally build an additive attention mask so the
                    # softmax also ignores masked keys (not just their value
                    # contribution). This is stricter but disables the flash
                    # attention backend in PyTorch SDPA.
                    if self.use_mask_to_cull_tokens:
                        read_ca_mask = torch.zeros(
                            B, 1, 1, T, device=agg_tokens.device, dtype=agg_tokens.dtype,
                        )
                        read_ca_mask.masked_fill_(
                            ~inside_full.unsqueeze(1).unsqueeze(1), float('-inf'),
                        )

        return agg_tokens, read_ca_mask
    
    def _build_camera_tokens(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        patch_start_idx: int,
    ) -> torch.Tensor:
        # For camera tokens, we only use the last layer.
        # Camera tokens are the first tokens of the sequence.
        cam_tokens = aggregated_tokens_list[-1][:, :, 0, :]  # (B, N, D)

        # Project camera tokens to desired dimension
        if self.project_aggregated_tokens:
            cam_tokens = self.camera_token_projector(cam_tokens)

        return cam_tokens


