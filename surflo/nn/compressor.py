import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Union, List, Dict, Any
from einops import rearrange

from surflo.nn.vggt.layers.block import Block
from surflo.nn.vggt.layers.ca_block import CABlock
from surflo.nn.vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter


class Compressor(nn.Module):
    """
    The Compressor applies cross-attention over a sequence of input tokens with variable length,
    in order to compress information from the input tokens into a fixed-length sequence of latent tokens.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_latent_tokens (int): Number of latent tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        ca_block_fn (nn.Module): The block type used for cross attention (CABlock by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
        elastic (bool): Whether to enable elastic latent token dropout (ELiT-style).
    """

    def __init__(
        self,
        embed_dim=512,
        depth=4,
        num_heads=16,
        mlp_ratio=4.0,
        num_latent_tokens=128,
        block_fn=Block,
        ca_block_fn=CABlock,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        qk_norm=True,
        rope_freq=0,  # disable positional encoding
        init_values=0.01,
        sa_to_ca_ratio:int=1,
        elastic:bool=False,
    ):
        super().__init__()

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.n_ca_blocks = depth
        self.n_sa_blocks = int(sa_to_ca_ratio) * depth
        self.sa_to_ca_ratio = int(sa_to_ca_ratio)

        self.ca_blocks = nn.ModuleList(
            [
                ca_block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(self.n_ca_blocks)
            ]
        )

        self.sa_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(self.n_sa_blocks)
            ]
        )

        # Latent tokens
        self.latent_tokens = nn.Parameter(torch.randn(num_latent_tokens, embed_dim))

        # Initialize parameters with truncated normal (clipped at 2 std)
        nn.init.trunc_normal_(self.latent_tokens, std=0.02, a=-0.04, b=0.04)

        self.depth = depth
        self.elastic = elastic
        self.num_latent_tokens = num_latent_tokens


    def forward(
        self, tokens: torch.Tensor, n_latent_tokens: Optional[int] = None,
        read_ca_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tokens (torch.Tensor): Input tokens with shape [B, L, D].
            n_latent_tokens (Optional[int]): Number of latent tokens to use at inference
                when elastic=True. If None, uses all latent tokens. Ignored during training.
            read_ca_mask (Optional[torch.Tensor]): Attention mask for the input
                (key/value) tokens in cross-attention, shape (B, 1, 1, L).
                Contains -inf for masked positions so latent tokens don't
                attend to them. Used for spatial culling of aggregated tokens.

        Returns:
            latent_tokens (torch.Tensor): Compressed latent tokens with shape [B, N, D]
                (or [B, n_latent_tokens, D] at inference with elastic).
            write_ca_mask (Optional[torch.Tensor]): Attention mask for downstream write
                cross-attention, shape (B, 1, 1, N). Contains -inf for masked (inactive)
                latent token positions so decoders don't attend to them. None when no
                masking is needed (non-elastic, or inference with physical dropping).
        """
        B, L, D = tokens.shape
        N = self.num_latent_tokens

        if self.elastic and not self.training:
            # Inference: physically drop tokens for real compute savings.
            # No mask needed — only active tokens exist.
            n = n_latent_tokens if n_latent_tokens is not None else N
            assert 1 <= n <= N, f"n_latent_tokens must be in [1, {N}], got {n}"
            latent_tokens = self.latent_tokens[:n].unsqueeze(0).expand(B, -1, -1)  # (B, n, D)

            for ca_idx in range(self.n_ca_blocks):
                latent_tokens = self._process_cross_attention(ca_idx, latent_tokens, tokens, attn_mask=read_ca_mask)
                for sa_to_ca_idx in range(self.sa_to_ca_ratio):
                    sa_idx = ca_idx * self.sa_to_ca_ratio + sa_to_ca_idx
                    latent_tokens = self._process_self_attention(sa_idx, latent_tokens)

            return latent_tokens, None

        # Expand latent tokens to match batch size
        latent_tokens = self.latent_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

        if self.elastic and self.training:
            # Training: sample per-element mask and build SA attention mask.
            # Only SA needs a mask (column masking) to prevent active tokens from
            # attending to masked tokens. No read-CA mask needed — masked tokens get
            # updated by CA but no active token ever reads them through SA.
            # We zero out masked tokens only at the end.
            active_mask, sa_attn_mask = self._sample_elastic_masks(
                B, N, device=tokens.device, dtype=tokens.dtype
            )

            for ca_idx in range(self.n_ca_blocks):
                latent_tokens = self._process_cross_attention(
                    ca_idx, latent_tokens, tokens, attn_mask=read_ca_mask,
                )
                for sa_to_ca_idx in range(self.sa_to_ca_ratio):
                    sa_idx = ca_idx * self.sa_to_ca_ratio + sa_to_ca_idx
                    latent_tokens = self._process_self_attention(
                        sa_idx, latent_tokens, attn_mask=sa_attn_mask
                    )

            # Zero out masked tokens at the end
            latent_tokens = latent_tokens * active_mask.unsqueeze(-1)

            # Build write-CA mask for downstream decoders: (B, 1, 1, N)
            # -inf for inactive latent tokens so decoders don't attend to them.
            write_ca_mask = torch.zeros(B, 1, 1, N, device=tokens.device, dtype=tokens.dtype)
            write_ca_mask.masked_fill_(~active_mask.bool().unsqueeze(1).unsqueeze(1), float('-inf'))

            return latent_tokens, write_ca_mask

        # Non-elastic path (default)
        for ca_idx in range(self.n_ca_blocks):
            latent_tokens = self._process_cross_attention(ca_idx, latent_tokens, tokens, attn_mask=read_ca_mask)
            for sa_to_ca_idx in range(self.sa_to_ca_ratio):
                sa_idx = ca_idx * self.sa_to_ca_ratio + sa_to_ca_idx
                latent_tokens = self._process_self_attention(sa_idx, latent_tokens)

        return latent_tokens, None

    def _sample_elastic_masks(
        self, B: int, N: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample elastic masks for training.

        Returns:
            active_mask: (B, N) float tensor, 1.0 for active tokens, 0.0 for masked.
            sa_attn_mask: (B, 1, N, N) float tensor for self-attention (column masking).
        """
        # Sample k_b ~ Uniform(1, N) for each batch element
        k = torch.randint(1, N + 1, (B,), device=device)  # (B,) values in [1, N]

        # Build active mask: active[b, i] = (i < k[b])
        indices = torch.arange(N, device=device).unsqueeze(0)  # (1, N)
        active_mask = (indices < k.unsqueeze(1)).float()  # (B, N)

        # Self-attention mask: block attention TO masked tokens (column masking)
        # sa_mask[b, :, i, j] = -inf if j >= k[b], else 0
        inactive_keys = ~(indices < k.unsqueeze(1))  # (B, N) — True for masked tokens
        sa_attn_mask = torch.zeros(B, 1, N, N, device=device, dtype=dtype)
        sa_attn_mask.masked_fill_(inactive_keys.unsqueeze(1).unsqueeze(2), float('-inf'))  # broadcast over query dim

        return active_mask, sa_attn_mask

    def _process_cross_attention(
        self, depth_idx: int, latent_tokens: torch.Tensor, tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None
    ):
        """Process cross attention blocks."""
        latent_tokens = self.ca_blocks[depth_idx](latent_tokens, tokens, attn_mask=attn_mask, pos=pos)
        return latent_tokens

    def _process_self_attention(
        self, depth_idx: int, latent_tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None
    ):
        """Process self attention blocks."""
        latent_tokens = self.sa_blocks[depth_idx](latent_tokens, attn_mask=attn_mask, pos=pos)
        return latent_tokens
