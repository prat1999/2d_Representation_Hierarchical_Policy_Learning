"""Transition label-swap variant of LazyArticuBotDataset (Approach 2 ablation).

PURE SUBCLASS, same contract as LazyArticuBotGtMixDataset: it inherits
LazyArticuBotDataset and only adds behaviour; no base dataset, config, or
existing training script changes. Reachable only when a task config sets
`_target_` to this class and `transition_label_swap: true`.

Behaviour — an exact port of Approach 1's goal label swap
(lfd3d/datasets/npy/npy_dataset.py in the high-level codebase): near a goal
transition, the aux GMM target `goal_gripper_pts` is stochastically replaced by
the goal on the OTHER side of the nearest transition, per frame, train only.

  linear  (default)  p_swap(d) = p_max * (1 - d / radius)     for d <= radius
                     (the v2 formula: the triangle reaches exactly 0 AT
                     d = radius. NOTE this differs from gt_mix's weighting,
                     which still uses the old radius+1 denominator.)
  sigmoid            p_swap(d) = 2 * p_max * sigmoid(-d / tau)  (no window)

where d = |t - t_trans| to the nearest transition. Transitions are frames where
the goal value changes within an episode (np.allclose, atol=1e-6), computed on
the same goal stream the dataset serves (goal_source-aware).

The swap targets ONLY the aux-head key (`swap_goal_key`, default
'goal_gripper_pts'). `present_gripper_pts` — also 'goal_gripper'-typed — is
never touched. Each served obs frame flips independently (fresh Bernoulli per
frame per __getitem__), matching Approach 1's per-frame semantics; the aux GMM
loss builds one mixture per obs step, so per-frame flips create no
mid-sequence target conflict.
"""

import numpy as np
import torch
from pathlib import Path

from diffusion_policy.dataset.lazy_articubot_dataset import LazyArticuBotDataset


class LazyArticuBotSwapDataset(LazyArticuBotDataset):

    def __init__(self,
                 *args,
                 transition_label_swap=False,
                 transition_swap_profile='linear',
                 transition_swap_p_max=0.5,
                 transition_radius=5,
                 transition_swap_tau=1.0,
                 swap_goal_key='goal_gripper_pts',
                 **kwargs):
        # Build the full base dataset exactly as usual (swap kwargs are captured
        # here and never forwarded, so the parent sees nothing new).
        super().__init__(*args, **kwargs)

        self.transition_swap_profile = str(transition_swap_profile)
        self.transition_swap_p_max = float(transition_swap_p_max)
        self.transition_radius = int(transition_radius)
        self.transition_swap_tau = float(transition_swap_tau)
        self.swap_goal_key = swap_goal_key

        self._swap_enabled = False
        if transition_label_swap:
            if self.transition_swap_profile not in ('linear', 'sigmoid'):
                raise ValueError(
                    f"Invalid transition_swap_profile '{self.transition_swap_profile}'. "
                    f"Expected 'linear' or 'sigmoid'.")
            if swap_goal_key not in self.shape_meta['obs']:
                raise KeyError(
                    f"swap_goal_key='{swap_goal_key}' not in shape_meta.obs — "
                    f"nothing to swap.")
            # Same episode-file reconstruction as the gt_mix subclass: the base
            # buffer used sorted(glob("*.h5"))[:max_train_episodes].
            h5_paths = sorted(Path(self.data_dir).glob("*.h5"))[: self.replay_buffer.n_episodes]
            self._precompute_swap_meta(h5_paths)
            self._swap_enabled = True
            desc = (f"profile=linear, p_max={self.transition_swap_p_max}, "
                    f"radius={self.transition_radius}"
                    if self.transition_swap_profile == 'linear' else
                    f"profile=sigmoid, p_max={self.transition_swap_p_max}, "
                    f"tau={self.transition_swap_tau}")
            eligible_thresh = 0.0 if self.transition_swap_profile == 'linear' else 0.01
            n_eligible = int((self._swap_p > eligible_thresh).sum())
            print(f"[LazyArticuBotSwapDataset] label swap ENABLED ({desc}, "
                  f"key={swap_goal_key!r}): {n_eligible}/{len(self._swap_p)} frames "
                  f"eligible (p_swap > {eligible_thresh})")

    # ----------------------------------------------------------------------- #
    # Precompute, per GLOBAL replay-buffer frame, the cross-transition neighbor
    # goal and the Bernoulli swap probability. Reads the goal stream straight
    # from the episode h5 files, honouring goal_source (same key remap as the
    # base buffer), so the swap follows whatever goal schedule is being served.
    # ----------------------------------------------------------------------- #
    def _precompute_swap_meta(self, h5_paths):
        import h5py

        episode_ends = np.asarray(self.replay_buffer.episode_ends)
        total = int(episode_ends[-1]) if len(episode_ends) > 0 else 0

        key_shape = self.shape_meta['obs'][self.swap_goal_key]['shape']  # [4, 3]
        K, C = int(key_shape[0]), int(key_shape[1])

        h5_key = (f'obs/{self.swap_goal_key}' if self.goal_source == 'default'
                  else f'obs/{self.swap_goal_key}_{self.goal_source}')

        self._swap_neighbor_goals = np.zeros((total, K, C), dtype=np.float32)
        self._swap_p = np.zeros((total,), dtype=np.float32)

        profile = self.transition_swap_profile
        p_max = self.transition_swap_p_max
        radius = self.transition_radius
        tau = self.transition_swap_tau

        start = 0
        for ep_i, h5path in enumerate(h5_paths):
            end = int(episode_ends[ep_i])
            T = end - start
            with h5py.File(h5path, 'r') as f:
                if h5_key not in f:
                    raise KeyError(
                        f"transition_label_swap requires {h5_key} but it is "
                        f"missing in {h5path}")
                goals = f[h5_key][:T].astype(np.float32)  # (T, K, C)

            transitions = [t for t in range(1, T)
                           if not np.allclose(goals[t], goals[t - 1], atol=1e-6)]
            if transitions:
                for t in range(T):
                    # Nearest transition; the linear profile only looks within
                    # the radius window, sigmoid considers every transition.
                    best_d, best_t = None, None
                    for tt in transitions:
                        d = abs(t - tt)
                        if profile == 'linear' and d > radius:
                            continue
                        if best_d is None or d < best_d:
                            best_d, best_t = d, tt
                    if best_t is None:
                        continue
                    # Neighbor = goal on the other side of the nearest
                    # transition. best_t is the FIRST frame of the new goal:
                    # frames < best_t carry the old goal (neighbor = upcoming),
                    # frames >= best_t carry the new one (neighbor = previous).
                    neighbor = goals[best_t] if t < best_t else goals[best_t - 1]
                    if profile == 'linear':
                        # v2 triangle: exactly 0 at d == radius; max(...,1)
                        # guards a radius=0 config.
                        p = p_max * (1.0 - best_d / max(radius, 1))
                    else:
                        p = 2.0 * p_max / (1.0 + np.exp(best_d / tau))
                    gi = start + t
                    self._swap_neighbor_goals[gi] = neighbor
                    self._swap_p[gi] = p
            start = end

    # ----------------------------------------------------------------------- #
    # Per-frame independent Bernoulli(p_swap) on the served obs frames of the
    # aux goal key — Approach 1's __getitem__ swap, applied per obs step.
    # ----------------------------------------------------------------------- #
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if getattr(self, "_swap_enabled", False):
            self._apply_label_swap(idx, sample["obs"])
        return sample

    def _apply_label_swap(self, idx, obs):
        goal = obs[self.swap_goal_key]        # torch (To, 4, 3)
        To = int(goal.shape[0])

        # Map obs position j -> global replay-buffer frame, mirroring how the
        # SequenceSampler lays real frames into the padded window
        # (positions [ssi, sei) <- buffer [bsi, bei); pad_before repeats the first).
        bsi, bei, ssi, sei = self.sampler.indices[idx]
        bsi = int(bsi); ssi = int(ssi); sei = int(sei)

        for j in range(To):
            real_pos = min(max(j, ssi), sei - 1)
            gf = bsi + (real_pos - ssi)
            p = float(self._swap_p[gf])
            if p > 0.0 and np.random.random() < p:
                goal[j] = torch.from_numpy(self._swap_neighbor_goals[gf].copy())

    # Label swap is a TRAIN-only augmentation; validation supervises against
    # the true per-frame goal. Base get_validation_dataset returns a shallow
    # copy preserving this subclass — just flip the flag off on the val view.
    def get_validation_dataset(self):
        val_set = super().get_validation_dataset()
        val_set._swap_enabled = False
        return val_set
