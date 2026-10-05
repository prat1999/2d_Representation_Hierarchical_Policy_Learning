"""
GMMActionDiTPolicy
==================
Ablation: the SAME network as ``FlowMatchingDiTGoalGMMPolicy`` (grounded DINOv2
+ RoPE4D trunk, state encoder, the 12-layer DiT, optional auxiliary goal head)
but with the flow-matching objective replaced by a chunk-level Gaussian mixture
over actions, trained with the ArticuBot NLL. One forward pass at inference.

What changes vs the flow-matching policy — and nothing else:
  * The DiT's 16 action tokens are replaced by N learned COMPONENT QUERIES.
    hidden_states = [state tokens ; N queries], encoder_hidden_states = the
    grounded patch tokens, exactly as before (even blocks cross-attend, odd
    blocks self-attend, AdaLN everywhere).
  * The flow timestep is a constant 0, so every AdaLN becomes a fixed learned
    modulation. The DiT is otherwise untouched (same depth, width, heads).
  * Each query's output goes through a linear layer to a 160-number chunk mean
    (horizon × action_dim) and a second linear layer to one logit. Softmax over
    the N logits gives the mixing weights.
  * Loss: ``goal_gmm_loss`` verbatim (variance ladder + 0.1 uniform-weight
    term) on the NORMALISED action chunk, i.e. component n is
    N(a ; mu_n, sigma² I) over all horizon×action_dim numbers.
  * Inference: mean of the highest-weight component (or a sampled component).

Variance variants (``action_variance_mode``):
  shared     : one sigma² per rung over all 160 numbers (sum of squared errors).
  per_group  : position / rotation / gripper each contribute their MEAN squared
               error, i.e. the error in each group is pre-scaled by
               1/sqrt(group_dims × horizon) before the shared ladder. Equivalent
               to sigma²·48, sigma²·96, sigma²·16 per group for the 10-D
               hybrid_delta layout [pos(3), rot6d(6), gripper(1)].

The auxiliary goal heads (gmm / regression / pooled_gmm) are inherited and
dispatched through ``_compute_aux_loss``; ``aux_gmm_loss_weight=null`` removes
them (grounded encoder, no goal supervision). NOTE: c1 must be re-chosen for
this main loss — the action NLL at init is orders of magnitude above the
flow-matching MSE, so the flow-matching c1 values do not transfer.
"""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import wandb
from torch import Tensor

from diffusion_policy.common.obs_util import process_observations
from diffusion_policy.model.flow_matching.goal_gmm_head import (
    FIXED_VARIANCE, UNIFORM_WEIGHTS_COEFF, goal_gmm_loss,
)
from diffusion_policy.policy.flow_matching_dit_goal_gmm_policy import (
    FlowMatchingDiTGoalGMMPolicy,
)


class GMMActionDiTPolicy(FlowMatchingDiTGoalGMMPolicy):

    QUERY_INIT_STD: float = 0.02
    MEAN_INIT_STD: float = 1e-3      # every component starts at the (normalised) mean action

    def __init__(
        self,
        shape_meta: dict,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        # ---- action mixture ----
        action_gmm_n_components: int = 16,
        action_variance_mode: str = "shared",          # "shared" | "per_group"
        action_group_dims: Sequence[int] = (3, 6, 1),  # pos, rot6d, gripper (per step)
        action_sample_component: bool = False,         # False: argmax weight; True: sample
        action_variances: Sequence[float] = FIXED_VARIANCE,
        action_uniform_weights_coeff: float = UNIFORM_WEIGHTS_COEFF,
        **kwargs,
    ):
        super().__init__(
            shape_meta=shape_meta,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            **kwargs,
        )
        # Flow-matching machinery is unused here; drop it so the parameter count
        # and checkpoints describe only what trains.
        del self.action_encoder
        del self.action_decoder
        if hasattr(self, "position_embedding"):
            del self.position_embedding

        D = kwargs.get("input_embedding_dim", 512)
        hidden = kwargs.get("hidden_size", 512)
        N = int(action_gmm_n_components)
        assert N >= 1
        assert action_variance_mode in ("shared", "per_group"), action_variance_mode
        self.n_components = N
        self.action_variance_mode = action_variance_mode
        self.action_sample_component = bool(action_sample_component)
        self.action_variances = tuple(float(v) for v in action_variances)
        self.action_uniform_weights_coeff = float(action_uniform_weights_coeff)

        # N learned component queries stand in for the 16 action tokens.
        self.component_queries = nn.Embedding(N, D)
        nn.init.normal_(self.component_queries.weight, std=self.QUERY_INIT_STD)

        # Per-query heads: chunk mean (horizon*action_dim) and one logit.
        self.mean_head = nn.Linear(hidden, self.action_horizon * self.action_dim)
        nn.init.normal_(self.mean_head.weight, std=self.MEAN_INIT_STD)
        nn.init.zeros_(self.mean_head.bias)
        self.logit_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.logit_head.weight)          # uniform weights at init
        nn.init.zeros_(self.logit_head.bias)

        # Per-dimension scale for the per_group variant: 1/sqrt(group_dims*H) so
        # that sum_d (scale_d * e_d)^2 == mean squared error within each group.
        self.action_group_dims = tuple(int(d) for d in action_group_dims)
        scale = torch.ones(self.action_horizon, self.action_dim)
        if action_variance_mode == "per_group":
            assert sum(self.action_group_dims) == self.action_dim, (
                f"action_group_dims {self.action_group_dims} must sum to action_dim {self.action_dim}"
            )
            start = 0
            for d in self.action_group_dims:
                scale[:, start:start + d] = 1.0 / (d * self.action_horizon) ** 0.5
                start += d
        self.register_buffer("action_err_scale", scale, persistent=False)

        print(
            f"[GMMActionDiTPolicy] flow matching replaced by an action GMM: "
            f"N={N} components, variance_mode={action_variance_mode}, "
            f"ladder={list(self.action_variances)}, uniform_coeff={self.action_uniform_weights_coeff}, "
            f"chunk={self.action_horizon}x{self.action_dim}, inference="
            f"{'sample' if self.action_sample_component else 'argmax'}"
        )

    # ------------------------------------------------------------------ #
    def _mixture(self, vis_tok: Tensor, state_tokens: Optional[Tensor], batch_size: int):
        """Run the (unchanged) DiT on [state ; component queries] at a constant
        timestep. Returns means (B, N, H, A) in normalised action space and
        logits (B, N)."""
        B, N = batch_size, self.n_components
        queries = self.component_queries.weight[None].expand(B, -1, -1)
        parts = []
        if self.has_state and state_tokens is not None:
            parts.append(state_tokens)
        parts.append(queries)
        hidden = torch.cat(parts, dim=1)
        timesteps = torch.zeros(B, dtype=torch.long, device=hidden.device)
        out = self.model(
            hidden_states=hidden,
            encoder_hidden_states=vis_tok,
            timestep=timesteps,
        )
        q = out[:, -N:]                                                  # (B, N, hidden)
        means = self.mean_head(q).reshape(B, N, self.action_horizon, self.action_dim)
        logits = self.logit_head(q).squeeze(-1)
        return means, logits

    @torch.no_grad()
    def _action_metrics(self, means: Tensor, logits: Tensor, nactions: Tensor) -> Dict[str, float]:
        """Diagnostics in normalised units: per-group MSE of the highest-weight
        component (shows whether one group swamps the exponent) and the
        effective number of components (1 = collapsed)."""
        pi = torch.softmax(logits.float(), dim=-1)
        best = pi.argmax(dim=-1)
        mu = means[torch.arange(means.shape[0], device=means.device), best]   # (B, H, A)
        e2 = (mu - nactions) ** 2
        out, start = {}, 0
        for name, d in zip(("pos", "rot", "grip"), self.action_group_dims):
            out[f"action_mse_{name}_best"] = e2[..., start:start + d].mean().item()
            start += d
        ent = -(pi * torch.log(pi.clamp(min=1e-12))).sum(-1)
        out["action_eff_n"] = ent.exp().mean().item()
        return out

    # ------------------------------------------------------------------ #
    def compute_loss(self, batch: dict) -> Tensor:
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])    # (B, H, A)
        B = nactions.shape[0]
        process_observations(nobs, self.observation_mode)

        vis_tok, vis_xyz, vis_valid, grip_tok, grip_xyz = \
            self.visual_encoder.encode_with_positions(nobs, batch["obs"])
        state_tokens = self._state_tokens(nobs, B)

        means, logits = self._mixture(vis_tok, state_tokens, B)
        target = nactions[:, None].expand_as(means)
        scale = self.action_err_scale.to(means.dtype)
        valid = torch.ones(B, self.n_components, dtype=torch.bool, device=means.device)
        nll = goal_gmm_loss(
            means * scale, target * scale, logits, valid,
            variances=self.action_variances,
            uniform_weights_coeff=self.action_uniform_weights_coeff,
        )

        prefix = "train" if self.training else "val"
        log = {f"{prefix}_action_nll": nll.item()}
        log.update({f"{prefix}_{k}": v for k, v in self._action_metrics(means, logits, nactions).items()})
        aux_loss, aux_log = self._compute_aux_loss(
            vis_tok, vis_xyz, vis_valid, grip_tok, grip_xyz, batch, prefix,
        )
        log.update(aux_log)
        if wandb.run is not None:
            wandb.log(log, commit=False)
        if aux_loss is None:
            return nll
        return nll + self.aux_gmm_loss_weight * aux_loss

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        process_observations(nobs, self.observation_mode)

        vis_tok, _, _, _, _ = self.visual_encoder.encode_with_positions(nobs, obs_dict)
        state_tokens = self._state_tokens(nobs, B)
        means, logits = self._mixture(vis_tok, state_tokens, B)

        pi = torch.softmax(logits.float(), dim=-1)
        if self.action_sample_component:
            idx = torch.multinomial(pi, num_samples=1).squeeze(1)
        else:
            idx = pi.argmax(dim=-1)
        chunk = means[torch.arange(B, device=means.device), idx]          # (B, H, A)
        action_pred = self.normalizer["action"].unnormalize(chunk)
        return {
            "action": action_pred[:, : self.n_action_steps],
            "action_pred": action_pred,
        }
