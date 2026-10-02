"""
Pooled (non-dense) GMM auxiliary head
=====================================
Mixture-density control for ``GoalGMMHead``: a Gaussian mixture over goal poses
with N components but NO anchors.

    masked mean over the grounded PATCH tokens of one observation step
    ⊕ the 4 raw current gripper keypoints (12 numbers)
    → MLP → N × 13   (per component: a K×3 mean as a displacement from the
                      CURRENT GRASP CENTRE, plus one mixing logit)

Same input as ``GoalRegressionHead``; the regression head is the N = 1,
no-logit special case of this one trained with MSE instead of NLL.

Trained with the SAME objective as the dense head (``goal_gmm_loss``: variance
ladder + 0.1 × uniform-weight term) by treating every component as anchored at
the grasp centre g_ref:

    pred_disp_n = mu_n − g_ref          (what the MLP emits)
    gt_disp     = goal − g_ref          broadcast to all N components
    valid       = all True

so the NLL sees mu_n − goal exactly as it does for the dense head, and the only
difference between the two runs is the head.

What this isolates. With N = 516 (4 gripper + 512 patch anchors) the component
count matches the dense head, but here the components are slices of ONE output
vector computed from ONE averaged input, instead of one component per scene
point fed its own features and position. Nothing ties component n to a place
in the scene; specialisation has to come from the output weights alone, and
the uniform-weight term pushes the other way (all means toward the target).
Expect the usual mixture-density pathologies — dead or duplicated components —
which ``goal_pooled_gmm_metrics`` exposes (effective number of components).
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diffusion_policy.model.flow_matching.goal_regression_head import GoalRegressionHead
from diffusion_policy.model.flow_matching.helpers import SimpleMLP


class GoalPooledGMMHead(nn.Module):
    """(patch tokens, valid mask, current keypoints) -> (N, C, K, 3) displacements, (N, C) logits.

    Output layer gets the same SMALL (not zero) init as ``GoalGMMHead`` so every
    component starts on the grasp centre with a finite NLL; the independent
    random rows of the output layer are what break the symmetry between
    components (they all see the identical input).
    """

    OUT_INIT_STD: float = 1e-3

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 512,
        n_keypoints: int = 4,
        n_components: int = 516,
    ):
        super().__init__()
        self.n_keypoints = int(n_keypoints)
        self.n_components = int(n_components)
        self.per_comp = self.n_keypoints * 3 + 1          # 12 mean + 1 logit
        self.mlp = SimpleMLP(
            input_dim=token_dim + self.n_keypoints * 3,
            hidden_dim=hidden_dim,
            output_dim=self.n_components * self.per_comp,
        )
        nn.init.normal_(self.mlp.layer2.weight, std=self.OUT_INIT_STD)
        nn.init.zeros_(self.mlp.layer2.bias)

    def forward(self, tokens: Tensor, valid: Tensor, gripper_pts: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            tokens:      (R, L, D) grounded patch tokens of one obs step
            valid:       (R, L)    bool, depth-valid patches
            gripper_pts: (R, K, 3) raw current keypoints, world metres
        Returns:
            disp:   (R, C, K, 3) component means as displacements from the reference keypoint
            logits: (R, C)       mixing logits (softmax over C gives the weights)
        """
        R = tokens.shape[0]
        pooled = GoalRegressionHead.masked_mean(tokens, valid)           # (R, D)
        x = torch.cat([pooled, gripper_pts.reshape(R, -1)], dim=-1)
        out = self.mlp(x).reshape(R, self.n_components, self.per_comp)
        disp = out[..., :-1].reshape(R, self.n_components, self.n_keypoints, 3)
        return disp, out[..., -1]


@torch.no_grad()
def goal_pooled_gmm_metrics(pred_disp: Tensor, logits: Tensor, target_disp: Tensor) -> Dict[str, float]:
    """Diagnostics in metres plus a collapse indicator.

    Args:
        pred_disp:   (R, C, K, 3) component means relative to the reference
        logits:      (R, C)
        target_disp: (R, K, 3)    goal relative to the same reference (NOT broadcast)
    Returns:
        goal_pgmm_best_err_m : mean keypoint error of the highest-weight component
        goal_pgmm_mean_err_m : mean keypoint error of the weight-averaged mean
        goal_pgmm_eff_n      : exp(entropy(weights)) — 1 = collapsed on one
                               component, C = weights spread evenly
    """
    pi = torch.softmax(logits.float(), dim=-1)                          # (R, C)
    R = pi.shape[0]
    best = pi.argmax(dim=-1)                                             # (R,)
    mu_best = pred_disp[torch.arange(R, device=pi.device), best]         # (R, K, 3)
    mu_bar = (pi[..., None, None] * pred_disp).sum(dim=1)                # (R, K, 3)
    ent = -(pi * torch.log(pi.clamp(min=1e-12))).sum(dim=-1)             # (R,)
    return {
        "goal_pgmm_best_err_m": (mu_best - target_disp).norm(dim=-1).mean().item(),
        "goal_pgmm_mean_err_m": (mu_bar - target_disp).norm(dim=-1).mean().item(),
        "goal_pgmm_eff_n": ent.exp().mean().item(),
    }
