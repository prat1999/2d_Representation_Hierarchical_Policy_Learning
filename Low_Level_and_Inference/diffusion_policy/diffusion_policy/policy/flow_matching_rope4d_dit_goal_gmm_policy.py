"""
FlowMatchingRoPE4DDiTGoalGMMPolicy
==================================
Approach 2 (grounded trunk + optional auxiliary goal head) with 3D grounding
and ACTION grounding inside the DiT as well, in the style of MINO's
RoPE4DDiT — but keeping our grounded visual encoder and our interleaved
block layout. Everything else (trunk, state encoder, action encoder/decoder,
flow-matching schedule, auxiliary heads, c1) is inherited unchanged.

What changes vs ``FlowMatchingDiTGoalGMMPolicy``:
  * ``self.model`` is ``RoPE4DInterleavedDiT``: every attention rotates q and
    k by continuous (x, y, z, t) positions; the slot sinusoid is gone.
  * Positions (scaled by dit_xyz_scale / dit_time_scale, t = step/(To+H)):
      visual keys  : the trunk's per-patch world anchors, t = obs step
      state tokens : grasp centre (keypoint ``action_pos_ref_keypoint``) at
                     that obs step, t = obs step
      action step k: grasp centre at the last obs step (+ displacement, see
                     ``action_pos_mode``), t = To + k
  * Depth-less patches can be hidden from cross-attention (key mask).

``action_pos_mode`` (where action token k is placed in 3D):
  gripper : at the current grasp centre; tokens differ only by time.
  cumsum  : grasp centre + gain · cumsum of the UNNORMALISED predicted position
            deltas of the current (noisy) chunk — the estimated path. Noisy
            early in denoising (random walk of noise deltas).
  blended : grasp centre + t · gain · cumsum(...): starts at the gripper at t=0
            and opens out to the estimated path as the chunk denoises. Default.
Positions are rebuilt at every Euler step at inference from the current chunk.

``action_pos_delta_gain``: MimicGen hybrid_delta actions are OSC position
COMMANDS (clipped at ±0.05 m), not realised motion; the controller realises
only ~0.23 of each command at this control rate. Measured on KITCHEN_D1 (20
demos, 763 windows of 16): realised displacement = 0.236 × summed command
per axis, residual 2.2 cm. Without the gain, cumsum/blended would place action
tokens ~4× too far along the path (summed commands reach 1.2 m). 1.0
reproduces MINO's raw-cumsum behaviour.

Two independent RoPE modules: the trunk keeps xyz_scale/time_scale/
base_frequency (weak time for 2 obs steps); the DiT gets its own
dit_xyz_scale (default = trunk's 5), dit_time_scale (default 18: one radian
per step, so 16 action steps are separable) and dit_rope_base_frequency.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
import wandb
from torch import Tensor

from diffusion_policy.common.obs_util import process_observations
from diffusion_policy.model.flow_matching.rope4d_interleaved_dit import (
    RoPE4DInterleavedDiT,
)
from diffusion_policy.policy.flow_matching_dit_goal_gmm_policy import (
    FlowMatchingDiTGoalGMMPolicy,
)


class FlowMatchingRoPE4DDiTGoalGMMPolicy(FlowMatchingDiTGoalGMMPolicy):

    def __init__(
        self,
        shape_meta: dict,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        # ---- DiT RoPE (separate from the trunk's) ----
        dit_xyz_scale: float = 5.0,
        dit_time_scale: float = 18.0,
        dit_rope_base_frequency: float = 100.0,
        dit_mask_invalid_visual_keys: bool = True,
        # ---- action token positions ----
        action_pos_mode: str = "blended",          # gripper | cumsum | blended
        action_pos_ref_keypoint: int = 3,          # grasp centre == EE-frame origin
        action_pos_delta_gain: float = 1.0,        # realised/commanded; 0.236 on KITCHEN_D1
        **kwargs,
    ):
        super().__init__(
            shape_meta=shape_meta, horizon=horizon,
            n_action_steps=n_action_steps, n_obs_steps=n_obs_steps, **kwargs,
        )
        assert not self.add_pos_embed, "RoPE replaces additive positions; set add_pos_embed: False"
        assert action_pos_mode in ("gripper", "cumsum", "blended"), action_pos_mode
        assert 0 <= action_pos_ref_keypoint < self.n_keypoints
        self.dit_xyz_scale = float(dit_xyz_scale)
        self.dit_time_scale = float(dit_time_scale)
        self.dit_mask_invalid_visual_keys = bool(dit_mask_invalid_visual_keys)
        self.action_pos_mode = action_pos_mode
        self.action_pos_ref_keypoint = int(action_pos_ref_keypoint)
        self.action_pos_delta_gain = float(action_pos_delta_gain)
        self._total_steps = n_obs_steps + horizon

        # Swap the baseline DiT for the RoPE4D one; same depth/width/heads/layout.
        cfg = dict(kwargs["diffusion_model_cfg"])
        del self.model
        self.model = RoPE4DInterleavedDiT(
            num_attention_heads=cfg["num_attention_heads"],
            attention_head_dim=cfg["attention_head_dim"],
            output_dim=cfg["output_dim"],
            num_layers=cfg.get("num_layers", 12),
            dropout=cfg.get("dropout", 0.1),
            attention_bias=cfg.get("attention_bias", True),
            activation_fn=cfg.get("activation_fn", "gelu-approximate"),
            norm_eps=cfg.get("norm_eps", 1e-5),
            final_dropout=cfg.get("final_dropout", True),
            interleave_self_attention=cfg.get("interleave_self_attention", True),
            base_frequency=float(dit_rope_base_frequency),
        )
        print(
            f"[FlowMatchingRoPE4DDiTGoalGMMPolicy] RoPE4D inside the DiT: "
            f"dit_xyz_scale={self.dit_xyz_scale}, dit_time_scale={self.dit_time_scale}, "
            f"base={dit_rope_base_frequency}, mask_invalid_keys={self.dit_mask_invalid_visual_keys}, "
            f"action_pos_mode={action_pos_mode}, ref_keypoint={self.action_pos_ref_keypoint}, "
            f"delta_gain={self.action_pos_delta_gain}"
        )

    # ------------------------------------------------------------------ #
    # Position builders (all in raw world metres, scaled here)
    # ------------------------------------------------------------------ #
    def _dit_time(self, step_idx: Tensor) -> Tensor:
        return step_idx.float() / self._total_steps * self.dit_time_scale

    def _visual_pos(self, vis_xyz: Tensor) -> Tensor:
        """(B, To*n_per_step, 3) anchors -> (B, To*n_per_step, 4). Token order is
        (obs_step, cam, patch), so each obs step is a contiguous block."""
        B, N, _ = vis_xyz.shape
        To = self.n_obs_steps
        t = self._dit_time(torch.arange(To, device=vis_xyz.device))
        t = t.repeat_interleave(N // To)[None, :, None].expand(B, -1, 1)
        return torch.cat([vis_xyz.float() * self.dit_xyz_scale, t.to(vis_xyz.dtype)], dim=-1)

    def _state_pos(self, gripper_pts: Tensor) -> Tensor:
        """raw (B, T, K, 3) keypoints -> (B, To, 4): grasp centre per obs step."""
        To = self.n_obs_steps
        g = gripper_pts[:, :To, self.action_pos_ref_keypoint, :].float()     # (B, To, 3)
        t = self._dit_time(torch.arange(To, device=g.device))[None, :, None].expand(g.shape[0], -1, 1)
        return torch.cat([g * self.dit_xyz_scale, t], dim=-1)

    def _action_pos(
        self,
        origin: Tensor,                      # (B, 3) grasp centre at the last obs step, raw metres
        noisy_actions: Optional[Tensor],     # (B, H, A) NORMALISED current chunk, or None
        t_cont: Optional[Tensor],            # (B,) flow time in [0, 1], or None
    ) -> Tensor:
        B = origin.shape[0]
        H, To = self.action_horizon, self.n_obs_steps
        t = self._dit_time(torch.arange(To, To + H, device=origin.device))[None, :, None].expand(B, -1, 1)
        origin = origin.float()
        if self.action_pos_mode == "gripper" or noisy_actions is None or t_cont is None:
            xyz = origin[:, None, :].expand(B, H, 3)
        else:
            with torch.no_grad():
                deltas = self.normalizer["action"].unnormalize(noisy_actions)[..., :3].float()
                cum = torch.cumsum(deltas, dim=1) * self.action_pos_delta_gain    # (B, H, 3)
            if self.action_pos_mode == "blended":
                xyz = origin[:, None, :] + t_cont.float().reshape(B, 1, 1) * cum
            else:                                                               # cumsum
                xyz = origin[:, None, :] + cum
        return torch.cat([xyz * self.dit_xyz_scale, t], dim=-1)

    def _run_rope_dit(self, action_features, vis_tok, state_tokens, t_disc,
                      action_pos, state_pos, visual_pos, key_mask):
        parts, pos = [], []
        if self.has_state and state_tokens is not None:
            parts.append(state_tokens); pos.append(state_pos)
        parts.append(action_features); pos.append(action_pos.to(action_features.dtype))
        hidden = torch.cat(parts, dim=1)
        hidden_pos = torch.cat([p.to(action_features.dtype) for p in pos], dim=1)
        out = self.model(hidden, vis_tok, t_disc, hidden_pos,
                         visual_pos.to(action_features.dtype), key_mask)
        return out[:, -self.action_horizon:]

    # ------------------------------------------------------------------ #
    def compute_loss(self, batch: dict) -> Tensor:
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        B = nactions.shape[0]
        device, dtype = nactions.device, nactions.dtype
        process_observations(nobs, self.observation_mode)

        vis_tok, vis_xyz, vis_valid, grip_tok, grip_xyz = \
            self.visual_encoder.encode_with_positions(nobs, batch["obs"])
        state_tokens = self._state_tokens(nobs, B)

        gp_raw = batch["obs"][self.gripper_key]                                 # (B, T, K, 3)
        visual_pos = self._visual_pos(vis_xyz)
        key_mask = vis_valid if self.dit_mask_invalid_visual_keys else None
        state_pos = self._state_pos(gp_raw)

        noise = torch.randn_like(nactions)
        t = self._sample_time(B, device=device, dtype=dtype)
        t_bc = t[:, None, None]
        noisy_actions = (1 - t_bc) * noise + t_bc * nactions
        velocity_target = nactions - noise
        t_disc = (t * self.num_timestep_buckets).long()

        action_features = self.action_encoder(noisy_actions, t_disc)
        origin = gp_raw[:, self.n_obs_steps - 1, self.action_pos_ref_keypoint, :]
        action_pos = self._action_pos(origin, noisy_actions, t)

        dit_out = self._run_rope_dit(action_features, vis_tok, state_tokens, t_disc,
                                     action_pos, state_pos, visual_pos, key_mask)
        pred_velocity = self.action_decoder(dit_out)
        fm_loss = F.mse_loss(pred_velocity, velocity_target)

        prefix = "train" if self.training else "val"
        aux_loss, aux_log = self._compute_aux_loss(
            vis_tok, vis_xyz, vis_valid, grip_tok, grip_xyz, batch, prefix,
        )
        if aux_loss is None:
            return fm_loss
        if wandb.run is not None:
            wandb.log({f"{prefix}_fm_loss": fm_loss.item(), **aux_log}, commit=False)
        return fm_loss + self.aux_gmm_loss_weight * aux_loss

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        device, dtype = self.device, self.dtype
        process_observations(nobs, self.observation_mode)

        # Trunk, visual/state positions: once per observation.
        vis_tok, vis_xyz, vis_valid, _, _ = self.visual_encoder.encode_with_positions(nobs, obs_dict)
        state_tokens = self._state_tokens(nobs, B)
        gp_raw = obs_dict[self.gripper_key]
        visual_pos = self._visual_pos(vis_xyz)
        key_mask = vis_valid if self.dit_mask_invalid_visual_keys else None
        state_pos = self._state_pos(gp_raw)
        origin = gp_raw[:, self.n_obs_steps - 1, self.action_pos_ref_keypoint, :]

        actions = torch.randn(B, self.action_horizon, self.action_dim, dtype=dtype, device=device)
        dt = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            t_disc = int(t_cont * self.num_timestep_buckets)
            timesteps = torch.full((B,), fill_value=t_disc, device=device)
            t_vec = torch.full((B,), t_cont, dtype=dtype, device=device)

            # Action positions follow the current chunk estimate.
            action_pos = self._action_pos(origin, actions, t_vec)
            action_features = self.action_encoder(actions, timesteps)
            dit_out = self._run_rope_dit(action_features, vis_tok, state_tokens, timesteps,
                                         action_pos, state_pos, visual_pos, key_mask)
            actions = actions + dt * self.action_decoder(dit_out)

        action_pred = self.normalizer["action"].unnormalize(actions)
        return {
            "action": action_pred[:, : self.n_action_steps],
            "action_pred": action_pred,
        }
