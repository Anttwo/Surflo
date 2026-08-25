import copy
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from surflo.nn.vggt.layers.cross_attention import CrossAttention


class GatedCrossAttention(nn.Module):
    """Cross attention with optional context-dependent per-head value gating (arXiv:2505.06708).

    When enabled, the KV projection is extended to also produce per-head gate logits
    from the context. Values are modulated by sigmoid(gate) before attention, giving
    each context token a learned, input-dependent per-head contribution weight.
    """
    def __init__(self, dim: int, num_heads: int = 16, qk_norm: bool = True, value_gating: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.value_gating = value_gating

        self.q = nn.Linear(dim, dim, bias=True)
        # KV projection: if gating, also produce per-head gate logits from context
        kv_out_dim = dim * 2 + (num_heads if value_gating else 0)
        self.kv = nn.Linear(dim, kv_out_dim, bias=True)

        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        _, M, _ = y.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        kv_out = self.kv(y)
        if self.value_gating:
            kv_flat, gate_score = kv_out.split([C * 2, self.num_heads], dim=-1)
            # gate_score: (B, M, num_heads) -> (B, num_heads, M, 1)
            gate_score = gate_score.permute(0, 2, 1).unsqueeze(-1)
        else:
            kv_flat = kv_out

        kv = kv_flat.reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        # Gate values before attention
        if self.value_gating:
            v = v * torch.sigmoid(gate_score)  # (B, h, M, d)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0
        )  # (B, h, N, d)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class SimpleDecoderBlock(nn.Module):
    """A single decoder block with DiT-style adaptive modulation.

    Each block has its own modulation MLP that maps the conditioning signal to
    all AdaLN parameters (shift, scale, gate) for this block in one shot.

    Cross-attention blocks produce 6 modulation values: (shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp).
    MLP-only blocks produce 3: (shift_mlp, scale_mlp, gate_mlp).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        value_gating: bool = False,
        has_cross_attn: bool = True,
    ):
        super().__init__()
        self.has_cross_attn = has_cross_attn

        # DiT-style: single modulation MLP outputs all AdaLN params for this block
        n_modulation = 6 if has_cross_attn else 3
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, n_modulation * dim),
        )
        # Zero-init so modulation starts as identity (scale=0, shift=0, gate=0)
        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)

        # Cross-attention sub-block (optional)
        if has_cross_attn:
            self.ln_ca = nn.LayerNorm(dim, elementwise_affine=False)
            self.cross_attn = GatedCrossAttention(
                dim=dim,
                num_heads=num_heads,
                qk_norm=qk_norm,
                value_gating=value_gating,
            )

        # MLP sub-block
        self.ln_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        conditioning: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, P, D) query tokens
            context: (B, N, D) latent tokens to cross-attend to
            conditioning: (B, P, D) or (B, 1, D) conditioning signal for AdaLN (time tokens)
            context_mask: (B, 1, 1, N) optional attention mask for context (write CA).
                Contains -inf for masked latent tokens.
        """
        modulation = self.adaln_modulation(conditioning)

        if self.has_cross_attn:
            shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
            # Cross-attention with AdaLN + gated residual
            x = x + gate_ca * self.cross_attn(self.ln_ca(x) * (1.0 + scale_ca) + shift_ca, context, attn_mask=context_mask)
        else:
            shift_mlp, scale_mlp, gate_mlp = modulation.chunk(3, dim=-1)

        # MLP with AdaLN + gated residual
        x = x + gate_mlp * self.mlp(self.ln_mlp(x) * (1.0 + scale_mlp) + shift_mlp)

        return x


class SimpleFieldDecoder(nn.Module):
    """
    Simplified field decoder with DiT-style per-block adaptive modulation.

    Architecture:
      - Spatial encoder encodes query points into tokens
      - Camera conditioning injected as additive bias (separate from time)
      - First `cross_attn_depth` blocks use cross-attention + MLP with AdaLN (shared backbone)
      - Remaining blocks use MLP-only with AdaLN (u-head; v-head is identical copy when use_mean_flow)
      - Time conditioning drives per-block AdaLN (6-way: shift/scale/gate for CA and MLP)
    """

    def __init__(
        self,
        spatial_encoder: nn.Module,
        time_encoder: nn.Module,
        field_dim: int = 6,
        embed_dim: int = 512,
        depth: int = 12,
        cross_attn_depth: Optional[int] = 6,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        use_camera_conditioning: bool = True,
        value_gating: bool = False,
        use_mean_flow: bool = False,
        fp32_output_head: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.spatial_encoder = spatial_encoder
        self.time_encoder = time_encoder
        self.depth = depth
        self.cross_attn_depth = cross_attn_depth if cross_attn_depth is not None else depth
        self.use_camera_conditioning = use_camera_conditioning
        self.use_mean_flow = use_mean_flow
        # When True, the final LayerNorm + Linear head(s) are evaluated with
        # autocast disabled (i.e. in fp32) so the predicted velocity is not
        # quantized to bf16. Purely a precision/fidelity upgrade with no
        # impact on state_dict layout — safe to toggle on/off across
        # checkpoints.
        self.fp32_output_head = fp32_output_head

        # Camera conditioning: separate MLP path (zero-initialized output for stable init)
        if use_camera_conditioning:
            self.camera_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )

        # Shared backbone: cross-attention blocks
        self.shared_blocks = nn.ModuleList([
            SimpleDecoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qk_norm=qk_norm,
                value_gating=value_gating,
                has_cross_attn=True,
            )
            for _ in range(self.cross_attn_depth)
        ])

        # u-head: MLP-only blocks + norm + projection
        n_u_blocks = depth - self.cross_attn_depth
        self.u_blocks = nn.ModuleList([
            SimpleDecoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qk_norm=qk_norm,
                value_gating=value_gating,
                has_cross_attn=False,
            )
            for _ in range(n_u_blocks)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, field_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Mean flow: h encoder and v-head
        if self.use_mean_flow:
            self.h_encoder = copy.deepcopy(time_encoder)

            # v-head: identical to u-head
            self.v_blocks = nn.ModuleList([
                SimpleDecoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    value_gating=value_gating,
                    has_cross_attn=False,
                )
                for _ in range(n_u_blocks)
            ])
            self.v_final_norm = nn.LayerNorm(embed_dim)
            self.v_head = nn.Linear(embed_dim, field_dim)
            nn.init.zeros_(self.v_head.weight)
            nn.init.zeros_(self.v_head.bias)

    def forward(
        self,
        query_points: torch.Tensor,
        t: torch.Tensor,
        latent_tokens: torch.Tensor,
        latent_camera_tokens: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
        u_head_only: bool = False,
        write_ca_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            query_points: (B, P, 3) or (B, P, 6) query points
            t: (B, P, 1) time values
            latent_tokens: (B, N, D) compressed scene tokens
            latent_camera_tokens: (B, 1, D) compressed camera tokens (optional)
            h: (B, P, 1) step size h = t - r for mean flow conditioning (optional)
            u_head_only: If True, skip v-head computation (used during JVP)
            write_ca_mask: (B, 1, 1, N) optional attention mask for latent tokens.
                Contains -inf for masked (inactive) latent tokens from elastic compressor.

        Returns:
            When use_mean_flow=False or u_head_only=True: (B, P, field_dim) u-head output
            When use_mean_flow=True and not u_head_only: tuple of (u_field, v_field)
        """
        # Encode query points into spatial tokens
        spatial_tokens = self.spatial_encoder(query_points)  # (B, P, D)

        # Encode time — used as conditioning for per-block AdaLN
        time_tokens = self.time_encoder(t)  # (B, P, D)

        # Mean flow: add h encoding to time tokens
        if self.use_mean_flow and h is not None:
            time_tokens = time_tokens + self.h_encoder(h)  # (B, P, D)

        # Initialize x with spatial tokens
        x = spatial_tokens

        # Camera conditioning: additive bias on conditioning (broadcasts from B,1,D)
        conditioning = time_tokens
        if self.use_camera_conditioning and latent_camera_tokens is not None:
            conditioning = conditioning + self.camera_proj(latent_camera_tokens)

        # Shared backbone (cross-attention blocks)
        for block in self.shared_blocks:
            x = block(x, context=latent_tokens, conditioning=conditioning, context_mask=write_ca_mask)

        # u-head (MLP-only blocks + projection)
        u = x
        for block in self.u_blocks:
            u = block(u, context=latent_tokens, conditioning=conditioning, context_mask=write_ca_mask)
        if self.fp32_output_head:
            # Run final LayerNorm + head in fp32 so the predicted velocity is
            # not quantized to bf16. We upcast the input *before* disabling
            # autocast — otherwise the LayerNorm's output would be cast back
            # to bf16 and the head would re-quantize on its own.
            with torch.autocast(device_type=u.device.type, enabled=False):
                u_field = self.head(self.final_norm(u.float()))  # (B, P, field_dim)
        else:
            u_field = self.head(self.final_norm(u))  # (B, P, field_dim)

        # v-head (mean flow only, skip when u_head_only for JVP)
        if self.use_mean_flow and not u_head_only:
            v = x
            for block in self.v_blocks:
                v = block(v, context=latent_tokens, conditioning=conditioning, context_mask=write_ca_mask)
            if self.fp32_output_head:
                with torch.autocast(device_type=v.device.type, enabled=False):
                    v_field = self.v_head(self.v_final_norm(v.float()))  # (B, P, field_dim)
            else:
                v_field = self.v_head(self.v_final_norm(v))  # (B, P, field_dim)
            return u_field, v_field

        return u_field
