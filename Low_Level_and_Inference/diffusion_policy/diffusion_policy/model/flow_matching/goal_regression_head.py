"""
Goal regression auxiliary head
==============================
Global-regression control for ``GoalGMMHead``. Same inputs, same role (shapes
the grounded visual trunk, discarded at inference), but no per-anchor mixture:

    masked mean over the grounded PATCH tokens of one observation step
    ⊕ the 4 raw current gripper keypoints (12 numbers)
    → MLP → K*3, read as the displacement of every goal keypoint from the
      CURRENT GRASP CENTRE (keypoint ``ref_idx``, the EE-frame origin).

Loss is plain per-element MSE (``F.mse_loss`` default reduction), matching the
flow-matching loss so the weight c1 compares per-element squared errors.

Why the raw keypoints are concatenated: patch tokens carry no position in
their content (RoPE rotates only q and k), and neither does their mean. The
GMM head gets position by concatenating each anchor's xyz; a pooled head has no
anchor, so the current gripper pose is passed explicitly instead. The 4 gripper
tokens are NOT pooled in — they would be 4 of 516 — and consequently receive no
direct gradient from this head (only via attention inside the trunk).

Contrast with the GMM head, which this isolates:
  * every valid token receives the identical gradient (1/N_valid of the pooled
    gradient), so the loss can move the mean of the representation but cannot
    tell specific patches to encode goal-relative geometry;
  * a single regressed vector averages across modes at sub-task boundaries.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diffusion_policy.model.flow_matching.helpers import SimpleMLP


class GoalRegressionHead(nn.Module):
    """(patch tokens, valid mask, current keypoints) -> (N, K, 3) displacement.

    Default ``nn.Linear`` init is fine here: the output starts near zero, i.e.
    "goal == current gripper", a sensible prior. (The GMM head needed a tiny
    output init only because its NLL blows up on metre-scale garbage.)
    """

    def __init__(self, token_dim: int, hidden_dim: int = 512, n_keypoints: int = 4):
        super().__init__()
        self.n_keypoints = n_keypoints
        self.mlp = SimpleMLP(
            input_dim=token_dim + n_keypoints * 3,
            hidden_dim=hidden_dim,
            output_dim=n_keypoints * 3,
        )

    @staticmethod
    def masked_mean(tokens: Tensor, valid: Tensor) -> Tensor:
        """tokens (N, L, D), valid (N, L) bool -> (N, D).

        Rows with no valid token fall back to the unmasked mean so nothing is
        divided by zero and the row still contributes a finite value.
        """
        w = valid.to(tokens.dtype)
        n_valid = w.sum(dim=1, keepdim=True)                      # (N, 1)
        masked = (tokens * w[..., None]).sum(dim=1) / n_valid.clamp(min=1.0)
        plain = tokens.mean(dim=1)
        return torch.where(n_valid > 0, masked, plain)

    def forward(self, tokens: Tensor, valid: Tensor, gripper_pts: Tensor) -> Tensor:
        """
        Args:
            tokens:      (N, L, D) grounded patch tokens of one obs step
            valid:       (N, L)    bool, depth-valid patches
            gripper_pts: (N, K, 3) raw current keypoints, world metres
        Returns:
            (N, K, 3) predicted displacement of each goal keypoint from the
            reference keypoint (see ``goal_regression_loss``).
        """
        N = tokens.shape[0]
        pooled = self.masked_mean(tokens, valid)                 # (N, D)
        x = torch.cat([pooled, gripper_pts.reshape(N, -1)], dim=-1)
        return self.mlp(x).reshape(N, self.n_keypoints, 3)


def goal_regression_target(goal: Tensor, gripper_pts: Tensor, ref_idx: int = 3) -> Tensor:
    """(N, K, 3) goal - current keypoint ``ref_idx`` broadcast over K.

    Single-reference convention: all four goal keypoints relative to ONE
    current point, the grasp centre. This is exactly the GMM head's
    parameterisation at its grasp-centre anchor, and it keeps the goal's rigid
    shape in the output (pairwise distances are preserved), so slot ``ref_idx``
    is pure translation and the other slots read as orientation + aperture.
    """
    return goal - gripper_pts[:, ref_idx:ref_idx + 1, :]


def goal_regression_loss(
    pred_disp: Tensor, goal: Tensor, gripper_pts: Tensor, ref_idx: int = 3,
) -> Tuple[Tensor, Tensor]:
    """Per-element MSE between predicted and true displacement.

    Returns (loss, target) so the caller can log metric-space errors.
    """
    target = goal_regression_target(goal, gripper_pts, ref_idx)
    return F.mse_loss(pred_disp, target), target


@torch.no_grad()
def goal_regression_metrics(pred_disp: Tensor, target: Tensor, ref_idx: int = 3) -> Dict[str, float]:
    """Euclidean errors in metres: mean over keypoints, and the reference
    (grasp-centre) slot alone, i.e. translation error."""
    err = (pred_disp - target).norm(dim=-1)                      # (N, K)
    return {
        "goal_reg_err_m": err.mean().item(),
        "goal_reg_center_err_m": err[:, ref_idx].mean().item(),
    }
