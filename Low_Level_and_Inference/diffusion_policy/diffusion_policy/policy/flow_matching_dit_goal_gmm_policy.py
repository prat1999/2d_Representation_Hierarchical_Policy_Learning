"""
FlowMatchingDiTGoalGMMPolicy
============================
Approach 2 ablation: the low-level policy is NOT given the goal. Instead the goal
supervises the visual representation through an auxiliary GMM head.

Differences from ``FlowMatchingDiTImagePolicy``:
  1. ``goal_gripper_pts`` is a training TARGET, never an input. The DiT's
     hidden_states stay ``[state_tokens, action_features]``.
  2. The plain DINOv2 encoder is replaced by ``DINOv2RoPE4DGroundedEncoder``,
     which gives each patch token a world-frame 3D anchor and fuses patches with
     the gripper keypoints via 4D-RoPE self-attention.
  3. A ``GoalGMMHead`` reads the SAME grounded tokens the DiT cross-attends to
     and is trained with the ArticuBot goal-GMM NLL.

Everything downstream of the fork is untouched: the DiT, action encoder/decoder,
flow-matching schedule, and the DiT's own state encoder are all as before, and
none of them receive gradient from the auxiliary loss. The only channel by which
the GMM loss can influence the policy is the grounded visual tokens.

Set ``aux_gmm_loss_weight: null`` to get the no-auxiliary-loss control arm (same
architecture, flow-matching loss only).

``aux_head_type`` selects the auxiliary head:
  * ``"gmm"``        (default) — ``GoalGMMHead`` + ArticuBot NLL, as above.
  * ``"regression"`` — ``GoalRegressionHead``: masked mean over the grounded patch
    tokens ⊕ raw current keypoints → MLP → K×3 displacement of the goal from the
    current grasp centre, per-element MSE. A global-regression control that
    keeps everything else (data, trunk, DiT, c1 knob) identical.
  * ``"pooled_gmm"`` — ``GoalPooledGMMHead``: same pooled input → MLP → N×13, a
    Gaussian mixture with N components and NO anchors (means are displacements
    from the current grasp centre). Trained with the SAME ``goal_gmm_loss`` as
    the dense head by anchoring every component at the grasp centre. Isolates
    "mixture" from "anchored on scene points".
``aux_gmm_loss_weight`` is c1 for whichever head is active.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
import wandb
from torch import Tensor

from diffusion_policy.common.obs_util import process_observations
from diffusion_policy.model.flow_matching.goal_gmm_head import (
    FIXED_VARIANCE, GoalGMMHead, goal_gmm_loss,
)
from diffusion_policy.model.flow_matching.goal_regression_head import (
    GoalRegressionHead, goal_regression_loss, goal_regression_metrics,
    goal_regression_target,
)
from diffusion_policy.model.flow_matching.goal_pooled_gmm_head import (
    GoalPooledGMMHead, goal_pooled_gmm_metrics,
)
from diffusion_policy.model.flow_matching.grounded_encoder import (
    DINOv2RoPE4DGroundedEncoder,
)
from diffusion_policy.policy.flow_matching_dit_image_policy import (
    FlowMatchingDiTImagePolicy,
)


class FlowMatchingDiTGoalGMMPolicy(FlowMatchingDiTImagePolicy):

    def __init__(
        self,
        shape_meta: dict,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        # ---- auxiliary goal-GMM head ----
        # None disables the head entirely (the architecture-matched control arm).
        aux_gmm_loss_weight: Optional[float] = 0.01,
        aux_gmm_hidden_dim: int = 512,
        aux_goal_key: str = "goal_gripper_pts",
        gripper_key: str = "present_gripper_pts",
        # ---- auxiliary head type ----
        # "gmm"        : per-anchor mixture + NLL (default; every earlier run).
        # "regression" : masked mean over patch tokens ⊕ raw current keypoints
        #                -> MLP -> K*3 displacement from the current grasp
        #                centre, per-element MSE. aux_gmm_loss_weight is still c1.
        # "pooled_gmm" : same pooled input -> MLP -> N*13: an N-component
        #                mixture with no anchors, trained with the dense head's
        #                goal_gmm_loss (every component anchored at the grasp
        #                centre). aux_gmm_hidden_dim sizes its MLP.
        aux_head_type: str = "gmm",
        aux_reg_hidden_dim: int = 512,
        aux_reg_ref_keypoint: int = 3,      # grasp centre == EE-frame origin
        aux_pooled_gmm_n_components: int = 516,   # == 4 gripper + 512 patch anchors
        # ---- grounded encoder ----
        patch_size: int = 14,
        n_trunk_layers: int = 2,
        xyz_scale: float = 5.0,
        time_scale: float = 1.0,
        base_frequency: float = 100.0,
        **kwargs,
    ):
        # The parent builds a visual encoder from every rgb/depth-typed key, which
        # is not what we want (depth is geometry, not a camera). It is replaced
        # below; the transient backbone is discarded.
        super().__init__(
            shape_meta=shape_meta,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            **kwargs,
        )

        self.aux_goal_key = aux_goal_key
        self.gripper_key = gripper_key
        self.aux_gmm_loss_weight = aux_gmm_loss_weight
        self.n_keypoints = int(shape_meta["obs"][aux_goal_key]["shape"][0])

        # -- 1. The goal is a target, never an input ------------------------- #
        # Both goal_gripper_pts and present_gripper_pts are 'goal_gripper'-typed
        # so the dataset serves them raw; neither may reach the DiT.
        self.goal_gripper_obs_keys = []
        self.has_goal_gripper = False
        self.has_gmm = False
        if hasattr(self, "goal_gripper_encoder"):
            del self.goal_gripper_encoder

        # -- 2. Grounded visual encoder -------------------------------------- #
        rgb_cam_keys = [
            k for k, a in shape_meta["obs"].items() if a.get("type") == "rgb"
        ]
        assert rgb_cam_keys, "no rgb-typed obs keys found"
        rgb_cam_keys = sorted(rgb_cam_keys)
        img_shape = list(shape_meta["obs"][rgb_cam_keys[0]]["shape"])

        enc_cfg = dict(kwargs.get("visual_encoder_cfg") or {})
        enc_cfg.update(dict(
            cam_keys=rgb_cam_keys,
            n_obs_steps=n_obs_steps,
            embed_dim=kwargs.get("input_embedding_dim", 512),
            crop_shape=kwargs.get("crop_shape", (224, 224)),
            in_channels=img_shape[0],
            image_size=img_shape[1],
            gripper_key=gripper_key,
            n_total_steps=n_obs_steps + horizon,
            n_keypoints=self.n_keypoints,
            patch_size=patch_size,
            n_trunk_layers=n_trunk_layers,
            xyz_scale=xyz_scale,
            time_scale=time_scale,
            base_frequency=base_frequency,
            num_heads=kwargs["diffusion_model_cfg"]["num_attention_heads"],
            head_dim=kwargs["diffusion_model_cfg"]["attention_head_dim"],
        ))
        self.visual_encoder = DINOv2RoPE4DGroundedEncoder(**enc_cfg)

        # -- 3. Auxiliary head ----------------------------------------------- #
        assert aux_head_type in ("gmm", "regression", "pooled_gmm"), (
            f"aux_head_type must be 'gmm', 'regression' or 'pooled_gmm', got {aux_head_type!r}"
        )
        self.aux_head_type = aux_head_type
        self.aux_reg_ref_keypoint = int(aux_reg_ref_keypoint)
        assert 0 <= self.aux_reg_ref_keypoint < self.n_keypoints
        self.gmm_head = None
        self.reg_head = None
        self.pgmm_head = None
        token_dim = kwargs.get("input_embedding_dim", 512)
        if aux_gmm_loss_weight is not None:
            if aux_head_type == "gmm":
                self.gmm_head = GoalGMMHead(
                    token_dim=token_dim,
                    hidden_dim=aux_gmm_hidden_dim,
                    n_keypoints=self.n_keypoints,
                )
            elif aux_head_type == "regression":
                self.reg_head = GoalRegressionHead(
                    token_dim=token_dim,
                    hidden_dim=aux_reg_hidden_dim,
                    n_keypoints=self.n_keypoints,
                )
            else:
                self.pgmm_head = GoalPooledGMMHead(
                    token_dim=token_dim,
                    hidden_dim=aux_gmm_hidden_dim,
                    n_keypoints=self.n_keypoints,
                    n_components=int(aux_pooled_gmm_n_components),
                )

        print(
            f"[FlowMatchingDiTGoalGMMPolicy] goal removed from DiT input; "
            f"aux_head_type={aux_head_type}"
            f"{f' (N={aux_pooled_gmm_n_components})' if aux_head_type == 'pooled_gmm' else ''}, "
            f"aux_gmm_loss_weight={aux_gmm_loss_weight}, "
            f"n_trunk_layers={n_trunk_layers}, xyz_scale={xyz_scale}, "
            f"time_scale={time_scale}, variances={list(FIXED_VARIANCE)}, "
            f"cams={rgb_cam_keys}"
        )

    # ===================================================================== #
    def _state_tokens(self, nobs: dict, batch_size: int) -> Optional[Tensor]:
        """DiT state tokens — unchanged from the baseline, and never touched by
        the auxiliary loss (the trunk has its own gripper encoder)."""
        if not self.has_state:
            return None
        parts = [
            nobs[k][:, : self.n_obs_steps].reshape(batch_size, self.n_obs_steps, -1)
            for k in self.state_obs_keys
            if k in nobs
        ]
        return self.state_encoder(torch.cat(parts, dim=-1))

    def _compute_goal_gmm_loss(
        self,
        vis_tokens: Tensor, vis_xyz: Tensor, vis_valid: Tensor,
        grip_tokens: Tensor, grip_xyz: Tensor,
        goal: Tensor,
    ) -> Tensor:
        """One mixture per obs step over [gripper keypoints ; patch anchors].

        Grouping per obs step matters because ``goal_gripper_pts`` is stored per
        timestep and jumps at phase boundaries; a single pooled mixture would
        supervise one step's anchors against the other step's goal. Folding the
        obs-step axis into the batch makes the per-mixture softmax fall out.
        """
        B, To, K = vis_tokens.shape[0], self.n_obs_steps, self.n_keypoints
        D = vis_tokens.shape[-1]
        n_per_step = vis_tokens.shape[1] // To      # n_cams * N_tok

        # Token order is (obs_step, cam, patch), so obs_step is outermost and this
        # reshape groups correctly.
        vt = vis_tokens.reshape(B * To, n_per_step, D)
        vx = vis_xyz.reshape(B * To, n_per_step, 3)
        vv = vis_valid.reshape(B * To, n_per_step)

        gt_tok = grip_tokens.reshape(B * To, K, D)
        gx = grip_xyz.reshape(B * To, K, 3)
        gv = torch.ones(B * To, K, dtype=torch.bool, device=vt.device)

        # Gripper anchors first, mirroring ArticuBot's prepare_scene_pcd.
        tokens = torch.cat([gt_tok, vt], dim=1)
        anchors = torch.cat([gx, vx], dim=1)
        valid = torch.cat([gv, vv], dim=1)

        goal_t = goal[:, :To].reshape(B * To, K, 3).to(anchors.dtype)

        pred_disp, logits = self.gmm_head(tokens, anchors)
        gt_disp = goal_t[:, None, :, :] - anchors[:, :, None, :]
        return goal_gmm_loss(pred_disp, gt_disp, logits, valid)

    def _compute_goal_reg_loss(
        self,
        vis_tokens: Tensor, vis_valid: Tensor,
        gripper_pts: Tensor, goal: Tensor,
    ):
        """Global-regression control. One prediction per obs step, grouped the
        same way as ``_compute_goal_gmm_loss`` (obs-step axis folded into the
        batch) so each step is scored against its own goal.

        Only the PATCH tokens are pooled; the current keypoints enter as raw
        coordinates. Target = goal - current grasp centre, per-element MSE.
        """
        B, To, K = vis_tokens.shape[0], self.n_obs_steps, self.n_keypoints
        D = vis_tokens.shape[-1]
        n_per_step = vis_tokens.shape[1] // To      # n_cams * N_tok

        vt = vis_tokens.reshape(B * To, n_per_step, D)
        vv = vis_valid.reshape(B * To, n_per_step)
        gp = gripper_pts[:, :To].reshape(B * To, K, 3).to(vt.dtype)
        goal_t = goal[:, :To].reshape(B * To, K, 3).to(vt.dtype)

        pred_disp = self.reg_head(vt, vv, gp)
        loss, target = goal_regression_loss(
            pred_disp, goal_t, gp, ref_idx=self.aux_reg_ref_keypoint,
        )
        metrics = goal_regression_metrics(pred_disp, target, self.aux_reg_ref_keypoint)
        return loss, metrics

    def _compute_goal_pooled_gmm_loss(
        self,
        vis_tokens: Tensor, vis_valid: Tensor,
        gripper_pts: Tensor, goal: Tensor,
    ):
        """Non-dense mixture control. Same pooling/grouping as the regression
        head, same NLL as the dense head: every component is treated as anchored
        at the current grasp centre, so ``goal_gmm_loss`` receives
        pred = mu_n - g_ref and gt = goal - g_ref (broadcast over N), and sees
        mu_n - goal exactly as it does for the dense head.
        """
        B, To, K = vis_tokens.shape[0], self.n_obs_steps, self.n_keypoints
        D = vis_tokens.shape[-1]
        n_per_step = vis_tokens.shape[1] // To

        vt = vis_tokens.reshape(B * To, n_per_step, D)
        vv = vis_valid.reshape(B * To, n_per_step)
        gp = gripper_pts[:, :To].reshape(B * To, K, 3).to(vt.dtype)
        goal_t = goal[:, :To].reshape(B * To, K, 3).to(vt.dtype)

        pred_disp, logits = self.pgmm_head(vt, vv, gp)               # (R,N,K,3), (R,N)
        target = goal_regression_target(goal_t, gp, self.aux_reg_ref_keypoint)   # (R,K,3)
        gt_disp = target[:, None].expand_as(pred_disp)
        valid = torch.ones(logits.shape, dtype=torch.bool, device=logits.device)
        loss = goal_gmm_loss(pred_disp, gt_disp, logits, valid)
        metrics = goal_pooled_gmm_metrics(pred_disp, logits, target)
        return loss, metrics

    # ===================================================================== #
    def compute_loss(self, batch: dict) -> Tensor:
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        device, dtype = nactions.device, nactions.dtype

        process_observations(nobs, self.observation_mode)

        # raw_obs carries metric depth, camera matrices and world-frame gripper
        # keypoints; the normalizer would put depth and state in the wrong frame.
        vis_tok, vis_xyz, vis_valid, grip_tok, grip_xyz = \
            self.visual_encoder.encode_with_positions(nobs, batch["obs"])
        state_tokens = self._state_tokens(nobs, batch_size)

        noise = torch.randn_like(nactions)
        t = self._sample_time(batch_size, device=device, dtype=dtype)
        t_bc = t[:, None, None]
        noisy_actions = (1 - t_bc) * noise + t_bc * nactions
        velocity_target = nactions - noise
        t_disc = (t * self.num_timestep_buckets).long()

        action_features = self.action_encoder(noisy_actions, t_disc)
        if self.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device,
            )
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        dit_out = self._run_dit(action_features, vis_tok, state_tokens, t_disc)
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

    def _compute_aux_loss(
        self,
        vis_tok: Tensor, vis_xyz: Tensor, vis_valid: Tensor,
        grip_tok: Tensor, grip_xyz: Tensor,
        batch: dict, prefix: str,
    ):
        """Dispatch to whichever auxiliary head is built.

        Returns ``(aux_loss, log_dict)``, or ``(None, {})`` when no head exists
        (the control arm). Shared with ``GMMActionDiTPolicy``, which swaps the
        flow-matching main loss for an action-mixture NLL but keeps the same
        auxiliary heads.
        """
        if self.gmm_head is None and self.reg_head is None and self.pgmm_head is None:
            return None, {}
        log = {}
        if self.gmm_head is not None:
            aux_loss = self._compute_goal_gmm_loss(
                vis_tok, vis_xyz, vis_valid, grip_tok, grip_xyz,
                batch["obs"][self.aux_goal_key],
            )
            log[f"{prefix}_goal_gmm_loss"] = aux_loss.item()
        elif self.reg_head is not None:
            aux_loss, reg_metrics = self._compute_goal_reg_loss(
                vis_tok, vis_valid,
                batch["obs"][self.gripper_key], batch["obs"][self.aux_goal_key],
            )
            log[f"{prefix}_goal_reg_loss"] = aux_loss.item()
            log.update({f"{prefix}_{k}": v for k, v in reg_metrics.items()})
        else:
            aux_loss, pgmm_metrics = self._compute_goal_pooled_gmm_loss(
                vis_tok, vis_valid,
                batch["obs"][self.gripper_key], batch["obs"][self.aux_goal_key],
            )
            log[f"{prefix}_goal_pooled_gmm_loss"] = aux_loss.item()
            log.update({f"{prefix}_{k}": v for k, v in pgmm_metrics.items()})
        return aux_loss, log

    # ===================================================================== #
    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        batch_size = next(iter(nobs.values())).shape[0]
        device, dtype = self.device, self.dtype

        process_observations(nobs, self.observation_mode)

        # The trunk has no flow-timestep conditioning, so the grounded tokens are
        # computed once and reused across every Euler step. No auxiliary head
        # is called at inference — they only ever shaped the trunk.
        vis_tok, _, _, _, _ = self.visual_encoder.encode_with_positions(nobs, obs_dict)
        state_tokens = self._state_tokens(nobs, batch_size)

        actions = torch.randn(
            batch_size, self.action_horizon, self.action_dim,
            dtype=dtype, device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            t_disc = int(t_cont * self.num_timestep_buckets)
            timesteps = torch.full((batch_size,), fill_value=t_disc, device=device)

            action_features = self.action_encoder(actions, timesteps)
            if self.add_pos_embed:
                pos_ids = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=device,
                )
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

            dit_out = self._run_dit(action_features, vis_tok, state_tokens, timesteps)
            actions = actions + dt * self.action_decoder(dit_out)

        action_pred = self.normalizer["action"].unnormalize(actions)
        return {
            "action": action_pred[:, : self.n_action_steps],
            "action_pred": action_pred,
        }
