"""
RoPE4D interleaved DiT
======================
The Approach 2 DiT — 12 blocks, even = cross-attention to the grounded patch
tokens, odd = self-attention over the [state ; action] stream, AdaLN on every
block, GELU feed-forward — with ONE change: every attention applies 4D rotary
position embeddings to its queries and keys from continuous (x, y, z, t)
positions supplied at forward time, and the fixed sinusoidal slot embedding on
the stream is removed. RoPE acts on q and k only, so positions shape attention
but are never written into token content.

Positions are built by the policy (flow_matching_rope4d_dit_goal_gmm_policy.py):
  visual keys   : world anchor of the patch (from the grounded trunk), t = obs step
  state tokens  : grasp centre at that obs step,                       t = obs step
  action tokens : grasp centre (+ scaled cumsum of predicted deltas),  t = obs steps + k
so action<->visual cross-attention is modulated by the metric offset between
where the chunk is heading and where each scene patch is ("action grounding"),
and action<->action self-attention by the displacement along the planned path.

The DiT's RoPE is a SEPARATE module from the trunk's RoPE with its own scales
(dit_xyz_scale / dit_time_scale / dit_rope_base_frequency on the policy). The
trunk keeps a weak time coordinate (2 obs steps); the DiT needs a strong one
(16 action steps must be ordered), hence time_scale 18 there by default.

Mirrors MINO's RoPE4DDiT (rope4d_dit.py) but keeps our interleaved layout and
adds an optional key mask so depth-less patches are hidden from cross-attention.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from torch import Tensor

from diffusion_policy.model.flow_matching.cross_attention_dit import (
    AdaLayerNorm, TimestepEncoder,
)
from diffusion_policy.model.flow_matching.rope4d_grounding import (
    RotaryPositionEmbedding4D,
)


class RoPE4DQKAttention(nn.Module):
    """Multi-head attention with 4D RoPE on q (query positions) and k (key
    positions). Self-attention when ``encoder_hidden_states`` is None (keys use
    the query positions); cross-attention otherwise."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
        base_frequency: float = 100.0,
    ):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        inner = num_heads * head_dim
        self.to_q = nn.Linear(dim, inner, bias=bias)
        self.to_k = nn.Linear(dim, inner, bias=bias)
        self.to_v = nn.Linear(dim, inner, bias=bias)
        self.to_out = nn.Linear(inner, dim, bias=bias)
        self.out_drop = nn.Dropout(dropout)       # diffusers Attention has Dropout on to_out
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        self.rope = RotaryPositionEmbedding4D(head_dim, base_frequency=base_frequency)

    def forward(
        self,
        hidden_states: Tensor,                       # (B, N, D)
        encoder_hidden_states: Optional[Tensor],     # (B, S, D) or None
        q_pos: Tensor,                               # (B, N, 4)
        kv_pos: Optional[Tensor] = None,             # (B, S, 4); None -> q_pos (self-attn)
        key_mask: Optional[Tensor] = None,           # (B, S) bool, True = attendable
    ) -> Tensor:
        kv_in = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        B, N, _ = hidden_states.shape
        S = kv_in.shape[1]
        H, Dh = self.num_heads, self.head_dim

        q = self.to_q(hidden_states).reshape(B, N, H, Dh).transpose(1, 2)
        k = self.to_k(kv_in).reshape(B, S, H, Dh).transpose(1, 2)
        v = self.to_v(kv_in).reshape(B, S, H, Dh).transpose(1, 2)

        q = self.rope(self.q_norm(q), q_pos)
        k = self.rope(self.k_norm(k), kv_pos if kv_pos is not None else q_pos)
        v = v.to(q.dtype)

        attn_mask = None
        if key_mask is not None:
            # Guard rows with no attendable key (would give NaN): let them see all.
            km = key_mask | ~key_mask.any(dim=-1, keepdim=True)
            attn_mask = km[:, None, None, :]                     # (B, 1, 1, S) bool
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, N, H * Dh)
        return self.out_drop(self.to_out(x))


class RoPE4DInterleavedBlock(nn.Module):
    """One block: AdaLN(temb) -> attention (cross or self, with RoPE4D) ->
    dropout -> residual -> LayerNorm -> FF -> residual. Same as the baseline
    BasicTransformerBlock minus the sinusoidal slot embedding."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        is_cross: bool,
        dropout: float = 0.1,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        base_frequency: float = 100.0,
    ):
        super().__init__()
        self.is_cross = is_cross
        self.norm1 = AdaLayerNorm(dim)
        self.attn = RoPE4DQKAttention(
            dim, num_heads, head_dim, dropout=dropout, bias=attention_bias,
            base_frequency=base_frequency,
        )
        self.final_dropout = nn.Dropout(dropout) if final_dropout else nn.Identity()
        self.norm3 = nn.LayerNorm(dim, norm_eps, elementwise_affine=False)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn,
                              final_dropout=final_dropout)

    def forward(
        self,
        hidden_states: Tensor,
        temb: Tensor,
        hidden_pos: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_pos: Optional[Tensor] = None,
        encoder_key_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.norm1(hidden_states, temb)
        if self.is_cross:
            a = self.attn(x, encoder_hidden_states, q_pos=hidden_pos,
                          kv_pos=encoder_pos, key_mask=encoder_key_mask)
        else:
            a = self.attn(x, None, q_pos=hidden_pos)
        hidden_states = hidden_states + self.final_dropout(a)
        hidden_states = hidden_states + self.ff(self.norm3(hidden_states))
        return hidden_states


class RoPE4DInterleavedDiT(nn.Module):
    """Baseline interleaved DiT with RoPE4D attention. Same depth/width/heads,
    same AdaLN + output modulation; no additive positional embeddings."""

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        output_dim: int,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        interleave_self_attention: bool = True,
        base_frequency: float = 100.0,
    ):
        super().__init__()
        assert attention_head_dim % 4 == 0, "RoPE4D needs head_dim divisible by 4"
        self.inner_dim = num_attention_heads * attention_head_dim
        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim)
        self.blocks = nn.ModuleList([
            RoPE4DInterleavedBlock(
                self.inner_dim, num_attention_heads, attention_head_dim,
                # Baseline layout: even index = cross-attention, odd = self-attention.
                is_cross=(not interleave_self_attention) or (idx % 2 == 0),
                dropout=dropout, activation_fn=activation_fn,
                attention_bias=attention_bias, norm_eps=norm_eps,
                final_dropout=final_dropout, base_frequency=base_frequency,
            )
            for idx in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)
        print("Total number of RoPE4DInterleavedDiT parameters: ",
              sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(
        self,
        hidden_states: Tensor,            # (B, T, D)  [state ; action]
        encoder_hidden_states: Tensor,    # (B, S, D)  grounded patch tokens
        timestep: Tensor,                 # (B,)
        hidden_pos: Tensor,               # (B, T, 4)
        encoder_pos: Tensor,              # (B, S, 4)
        encoder_key_mask: Optional[Tensor] = None,   # (B, S) bool
    ) -> Tensor:
        temb = self.timestep_encoder(timestep)
        x = hidden_states.contiguous()
        enc = encoder_hidden_states.contiguous()
        for blk in self.blocks:
            x = blk(x, temb, hidden_pos,
                    encoder_hidden_states=enc if blk.is_cross else None,
                    encoder_pos=encoder_pos if blk.is_cross else None,
                    encoder_key_mask=encoder_key_mask if blk.is_cross else None)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        x = self.norm_out(x) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(x)
