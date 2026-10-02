import copy
import numpy as np
import torch
from typing import Dict
from pathlib import Path

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.lazy_replay_buffer import LazyH5Buffer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_welford_image_normalizer,
    get_welford_robot_state_normalizer,
    array_to_stats,
    get_welford_online_action_normalizer,
    get_robot_state_normalizer_from_stat,
    get_online_range_robot_state_normalizer,
    get_identity_normalizer,
)
from diffusion_policy.common.action_util import (
    relative_to_delta_actions,
    delta_to_relative_actions,
    hybrid_relative_to_hybrid_delta_actions,
    hybrid_delta_to_hybrid_relative_actions,
)
from manipulation.utils import (
    rotation_transfer_6D_to_matrix_batch_mino,
    rotation_transfer_matrix_to_6D_batch,
)
from common.data_utils import process_pointmap, process_plucker, process_depth
from diffusion_policy.common.heatmap_augmentation import HeatmapRotationAugmentation


class LazyArticuBotDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            data_dir: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            action_mode='hybrid_delta',
            pointmap_frame='robot_frame',
            ghost_heatmap_last_channel_only=False,
            add_current_heatmap=False,
            heatmap_augmentation=None,
            percentage_training=None,
            goal_source='default',
            gmm_pred_npz_dir=None,
            gmm_pred_key_suffix='rdp',
            identity_normalize_depth=False,
        ):
        super().__init__()

        # Depth normalization. By default depth-typed keys get the Welford
        # standardizer, which is what the ArticuBot depth tasks expect. Encoders
        # that UNPROJECT depth need raw metres instead, and the standard
        # VisualTokenEncoder.encode(nobs) interface only ever sees normalized
        # obs — so set this True to hand them metres and keep the encoder a
        # drop-in (no policy subclass, no raw_obs plumbing). Default False keeps
        # every existing task byte-identical.
        self.identity_normalize_depth = bool(identity_normalize_depth)

        # Goal source selection. 'default' reads obs/goal_gripper_pts as always.
        # The other sources read obs/goal_gripper_pts_<source> — alternative
        # goals injected into the h5 files by generate_non_gmm_goals_for_low_level.py
        # --inject_extra_goals. Only the h5 path is remapped; the model-facing
        # obs key (and shape_meta / normalizers) are unchanged.
        VALID_GOAL_SOURCES = ('default', 'rdp', 'rdp_gripper', 'random', 'fixed_interval', 'awe')
        if goal_source not in VALID_GOAL_SOURCES:
            raise ValueError(
                f"Invalid goal_source '{goal_source}'. Expected one of {VALID_GOAL_SOURCES}"
            )
        self.goal_source = goal_source

        # External GMM prediction source (RDP pipeline). When set, the
        # gmm_goals/gmm_weights-typed obs keys are served from a parallel
        # per-frame npz tree (<dir>/demo_N/<t>.npz containing
        # gmm_all_goals_<sfx> / gmm_all_weights_<sfx>, produced by
        # run_gmm_pred_to_npz.py) instead of the h5 files. The h5 copies of
        # those keys are then never read — the h5 files stay untouched.
        self.gmm_pred_npz_dir = Path(gmm_pred_npz_dir) if gmm_pred_npz_dir else None
        self.gmm_pred_key_suffix = gmm_pred_key_suffix
        if self.gmm_pred_npz_dir is not None and not self.gmm_pred_npz_dir.is_dir():
            raise FileNotFoundError(
                f"gmm_pred_npz_dir does not exist: {self.gmm_pred_npz_dir}")
        # Per-episode sorted npz stems of the prediction tree, built lazily.
        # The numeric filenames can have gaps (e.g. KITCHEN_D1 demo_70 has no
        # 62.npz), and the h5 was built from the SORTED list of existing files
        # — so h5 timestep t corresponds to the t-th sorted stem, not '<t>.npz'.
        self._pred_stems_cache = {}
        self.ghost_heatmap_last_channel_only = ghost_heatmap_last_channel_only
        self.add_current_heatmap = add_current_heatmap

        # Build augmenters — one per heatmap type with the correct border fill.
        # border_fill=0.0 for Gaussian (no signal = keypoint not here)
        # border_fill=1.0 for ghost/distance-field (max distance = keypoint not here)
        aug_cfg = heatmap_augmentation or {}
        self.gaussian_heatmap_aug = HeatmapRotationAugmentation(
            border_fill=0.0, **aug_cfg
        ) if aug_cfg else None
        self.ghost_heatmap_aug = HeatmapRotationAugmentation(
            border_fill=1.0, **aug_cfg
        ) if aug_cfg else None

        self.shape_meta = shape_meta
        self.data_dir = Path(data_dir)
        self.action_mode = action_mode
        self.pointmap_frame = pointmap_frame

        if self.action_mode not in ['hybrid_delta', 'hybrid_relative', 'delta', 'relative', 'absolute']:
            raise ValueError(f"Unsupported action_mode: {action_mode}")
        if self.action_mode == 'absolute' and 'state' not in shape_meta.get('obs', {}):
            raise ValueError(
                "action_mode='absolute' requires 'state' in shape_meta.obs "
                "(used as absolute pose target via state[t+1])."
            )
        if self.pointmap_frame not in ['robot_frame', 'gripper_frame']:
            raise ValueError(f"Unsupported pointmap_frame: {pointmap_frame}")

        h5_paths = sorted(self.data_dir.glob("*.h5"))
        if max_train_episodes is not None:
            h5_paths = h5_paths[:max_train_episodes]

        # -----------------------------------------------------------------------
        # 1. Categorise observation keys by modality
        # -----------------------------------------------------------------------
        self.rgb_keys = []
        self.depth_keys = []
        self.heatmap_keys = []
        self.ghost_heatmap_keys = []
        self.pointmap_keys = []
        self.plucker_keys = []
        self.lowdim_keys = []
        self.matrix_keys = []
        self.goal_gripper_keys = []
        self.gmm_keys = []  # gmm_goals and gmm_weights — loaded as-is, identity normalizer

        for key, attr in shape_meta['obs'].items():
            type_name = attr.get('type')
            if type_name == 'rgb':
                self.rgb_keys.append(key)
            elif type_name == 'depth':
                self.depth_keys.append(key)
            elif type_name == 'pointmap':
                self.pointmap_keys.append(key)
            elif type_name == 'plucker':
                self.plucker_keys.append(key)
            elif type_name == 'heatmap':
                self.heatmap_keys.append(key)
            elif type_name == 'ghost_heatmap':
                self.ghost_heatmap_keys.append(key)
            elif type_name == 'goal_gripper':
                self.goal_gripper_keys.append(key)
            elif type_name in ('gmm_goals', 'gmm_weights'):
                self.gmm_keys.append(key)
            elif type_name == 'low_dim':
                self.lowdim_keys.append(key)
            elif 'intrinsic' in key or 'extrinsic' in key:
                self.matrix_keys.append(key)
            else:
                self.lowdim_keys.append(key)

        all_obs_keys = (
            self.rgb_keys + self.depth_keys + self.heatmap_keys +
            self.ghost_heatmap_keys + self.pointmap_keys + self.plucker_keys +
            self.lowdim_keys + self.matrix_keys + self.goal_gripper_keys +
            self.gmm_keys
        )
        if self.pointmap_frame == 'gripper_frame':
            all_obs_keys.append('gripper_to_world')
        if add_current_heatmap:
            for key in self.ghost_heatmap_keys:
                cam_prefix = key.split('_heatmap')[0]
                all_obs_keys.append(f'{cam_prefix}_present_heatmap_ghost')
            for key in self.heatmap_keys:
                cam_prefix = key.split('_heatmap')[0]
                all_obs_keys.append(f'{cam_prefix}_present_heatmap')

        # -----------------------------------------------------------------------
        # 2. Build LazyH5Buffer — reads only metadata (shapes/dtypes/lengths),
        #    no array data loaded into memory.
        # -----------------------------------------------------------------------
        # In 'absolute' mode the target is state[t+1] (computed at __getitem__ time),
        # but we still register an action key so the LazyH5Buffer/sampler keep their
        # ReplayBuffer contract. We arbitrarily pick action/hybrid; its values are
        # discarded in the absolute branch of __getitem__.
        if self.action_mode == 'absolute':
            self.action_key = 'action/hybrid'
        else:
            self.action_key = 'action/hybrid' if self.action_mode.startswith('hybrid') else 'action/delta'
        
        key_to_h5path = {'action': self.action_key}
        for key in all_obs_keys:
            if self.gmm_pred_npz_dir is not None and key in self.gmm_keys:
                # Served from the external npz tree — skip the h5 read path so
                # the buffer never loads the (stale) h5 copies of these keys.
                continue
            key_to_h5path[key] = f'obs/{key}'
        if self.goal_source != 'default':
            for key in self.goal_gripper_keys:
                key_to_h5path[key] = f'obs/{key}_{self.goal_source}'
            print(f"[LazyArticuBotDataset] goal_source={self.goal_source}: "
                  f"{ {k: key_to_h5path[k] for k in self.goal_gripper_keys} }")

        print(f"Indexing {len(h5_paths)} trajectories from {data_dir} (lazy)...")
        self.replay_buffer = LazyH5Buffer(
            h5_paths=h5_paths,
            key_to_h5path=key_to_h5path,
        )

        # Episode file stems ("demo_N") — used to resolve external npz trees
        # (the prediction tree mirrors the dataset layout demo_N/<t>.npz).
        self._episode_stems = [Path(p).stem for p in h5_paths]

        if self.gmm_pred_npz_dir is not None:
            self._gmm_goals_key = next(
                (k for k, a in shape_meta['obs'].items()
                 if a.get('type') == 'gmm_goals'), None)
            self._gmm_weights_key = next(
                (k for k, a in shape_meta['obs'].items()
                 if a.get('type') == 'gmm_weights'), None)
            if self._gmm_goals_key is None or self._gmm_weights_key is None:
                raise ValueError(
                    "gmm_pred_npz_dir is set but shape_meta has no "
                    "gmm_goals/gmm_weights obs keys — nothing to serve from npz.")
            missing = [s for s in self._episode_stems
                       if not (self.gmm_pred_npz_dir / s).is_dir()]
            if missing:
                raise FileNotFoundError(
                    f"gmm_pred_npz_dir={self.gmm_pred_npz_dir} is missing "
                    f"{len(missing)} demo dir(s), e.g. {missing[:3]} — run the "
                    f"RDP prediction generation (run_gmm_pred_to_npz.py) first.")
            sfx = self.gmm_pred_key_suffix
            print(f"[LazyArticuBotDataset] GMM source = npz tree {self.gmm_pred_npz_dir} "
                  f"(keys gmm_all_goals_{sfx} / gmm_all_weights_{sfx}); "
                  f"h5 gmm keys are NOT read.")

        # LazyH5Buffer silently drops keys absent from every file — fail loudly
        # here instead of with an opaque KeyError mid-training.
        if self.goal_source != 'default':
            for key in self.goal_gripper_keys:
                if key not in self.replay_buffer:
                    raise KeyError(
                        f"goal_source='{self.goal_source}' but '{key_to_h5path[key]}' "
                        f"is missing from the h5 files in {data_dir}. Run "
                        f"generate_non_gmm_goals_for_low_level.py --inject_extra_goals "
                        f"first (the RDP training scripts do this automatically)."
                    )

        # -----------------------------------------------------------------------
        # 3. Setup sampler — identical to the eager dataset
        # -----------------------------------------------------------------------
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask

        key_first_k = {}
        if n_obs_steps is not None:
            for key in all_obs_keys:
                key_first_k[key] = n_obs_steps
        # import pdb; pdb.set_trace()
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

        print(f"Dataset initialized with {len(self.sampler)} training sequences.")

        # -----------------------------------------------------------------------
        # 4. Optional multimodal oversampling (--percentage_training).
        #    When set, frames whose GMM goal distribution has >=2 real modes
        #    (i.e. genuinely-multimodal / transition frames) collectively get
        #    sampling mass `percentage_training`; all single-mode (cruise) frames
        #    share the remaining (1 - percentage_training). Mirrors the high-level
        #    transition_p weighted sampler. Requires a gmm_weights-typed obs key
        #    (e.g. gmm_mode_weights from the modes task). None => uniform sampling.
        # -----------------------------------------------------------------------
        self.percentage_training = percentage_training
        self.gmm_weights_key = None
        for key, attr in shape_meta['obs'].items():
            if attr.get('type') == 'gmm_weights':
                self.gmm_weights_key = key
                break
        self.sample_weights = None
        if percentage_training is not None:
            if self.gmm_weights_key is None or self.gmm_weights_key not in self.replay_buffer:
                print(f"[LazyArticuBotDataset] percentage_training={percentage_training} requested "
                      f"but no gmm_weights obs key found -> falling back to uniform sampling.")
            else:
                self.sample_weights = self._compute_multimodal_sample_weights(
                    self.gmm_weights_key, float(percentage_training)
                )

    def _compute_multimodal_sample_weights(self, gmm_weights_key, p, mode_thresh=1e-6):
        """Per-train-sample weights putting collective mass `p` on multimodal
        (>=2 nonzero modes) frames, (1-p) on the rest. Returns float64 array of
        length len(self.sampler)."""
        # n_modes per GLOBAL frame (read the whole tiny gmm_mode_weights once)
        all_mw = np.asarray(self.replay_buffer[gmm_weights_key][:])   # (total_frames, K)
        n_modes = (all_mw > mode_thresh).sum(axis=1)                  # (total_frames,)
        is_multi_frame = n_modes >= 2

        idx_arr = np.asarray(self.sampler.indices)   # (N, 4): [buf_start, buf_end, ...]
        N = len(idx_arr)
        n_obs = self.n_obs_steps or 1
        is_multi = np.zeros(N, dtype=bool)
        for i in range(N):
            bs = int(idx_arr[i, 0]); be = int(idx_arr[i, 1])
            # multimodal sample if ANY of its obs frames is multimodal
            is_multi[i] = bool(is_multi_frame[bs:min(bs + n_obs, be)].any())

        n_multi = int(is_multi.sum())
        n_other = N - n_multi
        w = np.zeros(N, dtype=np.float64)
        if n_multi == 0 or n_other == 0:
            w[:] = 1.0 / max(N, 1)   # all one class -> uniform (nothing to up/down-weight)
        else:
            w[is_multi] = p / n_multi
            w[~is_multi] = (1.0 - p) / n_other
        print(f"[LazyArticuBotDataset] percentage_training={p}: "
              f"{n_multi}/{N} train samples are multimodal (>=2 modes); "
              f"they collectively get sampling mass {p}.")
        return w

    def get_sample_weights(self):
        """Per-sample weights for a WeightedRandomSampler, or None for uniform."""
        return getattr(self, "sample_weights", None)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        # Validation must never use the train multimodal-oversampling weights
        # (they're indexed to the train sampler and have the wrong length here).
        val_set.sample_weights = None
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        # Disable heatmap augmentation for validation
        if val_set.gaussian_heatmap_aug is not None:
            val_set.gaussian_heatmap_aug = copy.copy(val_set.gaussian_heatmap_aug)
            val_set.gaussian_heatmap_aug.enabled = False
        if val_set.ghost_heatmap_aug is not None:
            val_set.ghost_heatmap_aug = copy.copy(val_set.ghost_heatmap_aug)
            val_set.ghost_heatmap_aug.enabled = False
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # Action
        # [:] reads from disk one file at a time and concatenates — low memory
        # footprint for low-dim data; fine for action/state arrays.
        if 'relative' in self.action_mode:
            normalizer['action'] = get_welford_online_action_normalizer(
                horizon=self.horizon,
                action_dim=self.shape_meta['action']['shape'][0],
                max_samples=len(self.sampler) * self.n_obs_steps,
            )
        elif 'delta' in self.action_mode:
            # stat = array_to_stats(self.replay_buffer['action'][:])
            # normalizer['action'] = get_robot_state_normalizer_from_stat(stat)
            action_dim = self.shape_meta['action']['shape'][0]
            normalizer['action'] = get_online_range_robot_state_normalizer(
                state_dim=action_dim,
                max_samples=len(self.sampler) * self.n_obs_steps,
            )
        elif self.action_mode == 'absolute':
            # Absolute target lives in the same space as obs/state — reuse the
            # robot-state range normalizer.
            action_dim = self.shape_meta['action']['shape'][0]
            normalizer['action'] = get_online_range_robot_state_normalizer(
                state_dim=action_dim,
                max_samples=len(self.sampler) * self.n_obs_steps,
            )

        # Low dim
        for key in self.lowdim_keys:
            if 'state' in key or 'qpos' in key:
                state_dim = self.shape_meta['obs'][key]['shape'][0]
                normalizer[key] = get_online_range_robot_state_normalizer(
                    state_dim=state_dim,
                    max_samples=len(self.sampler) * self.n_obs_steps,
                )
            else:
                stat = array_to_stats(self.replay_buffer[key][:])
                normalizer[key] = get_range_normalizer_from_stat(stat)

        # RGB — fixed normalizer, no data needed
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        # Depth — online Welford normalizer, no data loaded upfront.
        # Stats are accumulated per-channel (1 channel) during the first
        # max_samples training steps.
        # With identity_normalize_depth=True the depth passes through untouched
        # so nobs carries raw METRES, which is what an unprojecting encoder
        # needs. See the note in __init__.
        for key in self.depth_keys:
            if self.identity_normalize_depth:
                normalizer[key] = get_identity_normalizer()
            else:
                normalizer[key] = get_welford_image_normalizer(
                    num_channels=1,
                    max_samples=len(self.sampler) * self.n_obs_steps,
                )

        for key in self.heatmap_keys:
            normalizer[key] = get_identity_normalizer()

        for key in self.ghost_heatmap_keys:
            normalizer[key] = get_identity_normalizer()

        for key in self.goal_gripper_keys:
            normalizer[key] = get_identity_normalizer()

        # GMM distribution keys — identity (probabilities and 3D coords, no normalization)
        for key in self.gmm_keys:
            normalizer[key] = get_identity_normalizer()

        # Plucker / Pointmap / Matrices — identity (geometric data)
        for key in self.pointmap_keys + self.plucker_keys + self.matrix_keys:
            normalizer[key] = get_identity_normalizer()

        return normalizer

    def __len__(self):
        return len(self.sampler)

    def _load_state_window(self, idx: int) -> np.ndarray:
        """Load obs/state across the full sampler window, clamping at the episode
        end and applying the same pad_before/pad_after rules the SequenceSampler
        uses for the action chunk. Returns (horizon, state_dim) float32.

        The sampler is configured with `key_first_k['state'] = n_obs_steps`, so
        sample_sequence(...) only returns the first n_obs_steps frames of state
        (the rest NaN). We bypass that here because absolute targets need state
        for the full horizon.
        """
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.sampler.indices[idx]
        )
        state_arr = self.replay_buffer['state']

        ep_idx = int(np.searchsorted(
            self.replay_buffer.episode_ends, buffer_start_idx, side='right'
        ))
        ep_end = int(self.replay_buffer.episode_ends[ep_idx])

        stop = min(buffer_end_idx, ep_end)
        states = state_arr[buffer_start_idx:stop].astype(np.float32)

        # Pad trajectory tail (sampler does this for the action chunk by repeating
        # the last frame; mirror it so xyz/rot composition aligns).
        valid_n = buffer_end_idx - buffer_start_idx
        if states.shape[0] < valid_n:
            pad_count = valid_n - states.shape[0]
            states = np.concatenate(
                [states, np.repeat(states[-1:], pad_count, axis=0)],
                axis=0,
            )

        # Mirror sampler pad_before / pad_after (repeat first/last frame).
        seq_len = self.horizon
        if (sample_start_idx > 0) or (sample_end_idx < seq_len):
            out = np.zeros((seq_len,) + states.shape[1:], dtype=states.dtype)
            if sample_start_idx > 0:
                out[:sample_start_idx] = states[0]
            if sample_end_idx < seq_len:
                out[sample_end_idx:] = states[-1]
            out[sample_start_idx:sample_end_idx] = states
            states = out

        return states

    def _build_absolute_action_target(self, idx: int, hybrid_action: np.ndarray) -> np.ndarray:
        """Absolute action target: absolute[t] = state[t] ⊕ action/hybrid[t].

        This is the *commanded target pose* sent to the OSC controller at each
        step (world-frame xyz, body-frame rot composed onto current orientation,
        additive gripper). It is the standard absolute-control target derived
        from delta-recorded data — what the same controller would track at
        deployment to reproduce the demonstration.

        Note: this is NOT state[t+1] (the observed next pose). On Coffee_D2 the
        OSC controller under-tracks each commanded delta, so state[t+1] is only
        ~1/4 of the way toward state[t] + action/hybrid[t].
        """
        states = self._load_state_window(idx)               # (H, 10)
        actions = hybrid_action.astype(np.float32)          # (H, 10)
        H = states.shape[0]
        assert actions.shape == states.shape, (
            f"action/state horizon mismatch: {actions.shape} vs {states.shape}"
        )

        # World-frame xyz: just add.
        abs_xyz = states[:, :3] + actions[:, :3]

        # Body-frame rotation composition: R_abs = R_state @ R_delta.
        state_R = rotation_transfer_6D_to_matrix_batch_mino(
            states[:, 3:9].reshape(-1, 6)
        ).reshape(H, 3, 3)
        delta_R = rotation_transfer_6D_to_matrix_batch_mino(
            actions[:, 3:9].reshape(-1, 6)
        ).reshape(H, 3, 3)
        abs_R = state_R @ delta_R                            # (H, 3, 3)
        # Match action_util.py convention: transpose before encoding to 6D.
        abs_6d = rotation_transfer_matrix_to_6D_batch(
            abs_R.reshape(-1, 3, 3).transpose(0, 2, 1)
        ).reshape(H, 6)

        # Gripper: additive (matches action_util.py:308 verification).
        abs_grip = states[:, 9:10] + actions[:, 9:10]

        return np.concatenate([abs_xyz, abs_6d, abs_grip], axis=-1).astype(np.float32)

    def _pred_frame_stems(self, ep: int, ep_len: int):
        """Sorted npz stems for episode ep of the prediction tree. h5 timestep
        t maps to stems[t]; filenames are NOT assumed contiguous (see
        _pred_stems_cache comment in __init__)."""
        stems = self._pred_stems_cache.get(ep)
        if stems is None:
            demo_dir = self.gmm_pred_npz_dir / self._episode_stems[ep]
            stems = sorted((p.stem for p in demo_dir.glob('*.npz')), key=int)
            if len(stems) != ep_len:
                raise RuntimeError(
                    f"prediction npz tree {demo_dir} has {len(stems)} frames "
                    f"but the h5 episode has {ep_len} — trees out of sync?")
            self._pred_stems_cache[ep] = stems
        return stems

    def _fill_gmm_from_pred_npz(self, idx: int, obs_dict: dict):
        """Serve the gmm_goals/gmm_weights obs keys for this sample's obs frames
        from the external per-frame npz tree (gmm_pred_npz_dir/demo_N/<t>.npz).

        Maps each obs position j -> global replay-buffer frame exactly the way
        SequenceSampler lays real frames into the padded window (positions
        [ssi, sei) <- buffer [bsi, bei); pads repeat the edge frames) — the
        same mapping LazyArticuBotGtMixDataset uses for its GT overwrite.
        """
        To = self.n_obs_steps or 1
        bsi, _bei, ssi, sei = self.sampler.indices[idx]
        bsi, ssi, sei = int(bsi), int(ssi), int(sei)
        episode_ends = np.asarray(self.replay_buffer.episode_ends)
        sfx = self.gmm_pred_key_suffix

        goals, weights = [], []
        for j in range(To):
            real_pos = min(max(j, ssi), sei - 1)
            gf = bsi + (real_pos - ssi)
            ep = int(np.searchsorted(episode_ends, gf, side='right'))
            ep_start = 0 if ep == 0 else int(episode_ends[ep - 1])
            local = gf - ep_start
            stems = self._pred_frame_stems(ep, int(episode_ends[ep]) - ep_start)
            d = np.load(
                self.gmm_pred_npz_dir / self._episode_stems[ep] / f"{stems[local]}.npz")
            goals.append(d[f"gmm_all_goals_{sfx}"][0])
            weights.append(d[f"gmm_all_weights_{sfx}"][0])

        obs_dict[self._gmm_goals_key] = np.stack(goals).astype(np.float32)
        obs_dict[self._gmm_weights_key] = np.stack(weights).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        T_slice = slice(self.n_obs_steps)

        obs_dict = {}

        # RGB: (T, H, W, C) -> (T, C, H, W), float32 [0, 1]
        # import pdb; pdb.set_trace()
        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.
            del data[key]

        # Depth: (T, H, W) -> decompress -> (T, 1, H, W)
        for key in self.depth_keys:
            d_float = process_depth(data[key][T_slice], compress=False)
            if d_float.ndim == 3:
                d_float = d_float[:, None, :, :]
            obs_dict[key] = d_float
            del data[key]

        for key in self.heatmap_keys:
            raw = data[key][T_slice]
            if raw.dtype == np.uint8:
                # New dataset: (T, H, W, C) uint8 -> (T, C, H, W) float32 [0, 1]
                h_float = np.moveaxis(raw.astype(np.float32) / 255., -1, 1)
            else:
                # Old dataset: (T, H, W) float16, already [0, 1]
                h_float = raw.astype(np.float32)
                if h_float.ndim == 3:
                    h_float = h_float[:, None, :, :]         # (T, 1, H, W)
            if self.gaussian_heatmap_aug is not None:
                h_float = self.gaussian_heatmap_aug(h_float)
            obs_dict[key] = h_float
            del data[key]

        # Ghost heatmap: (T, H, W, 4) uint8 on disk -> (T, 4, H, W) float32 [0, 1]
        for key in self.ghost_heatmap_keys:
            g = data[key][T_slice].astype(np.float32) / 255.  # (T, H, W, 4)
            if self.ghost_heatmap_last_channel_only:
                g = g[..., -1:]                                # (T, H, W, 1)
            goal = np.moveaxis(g, -1, 1)                       # (T, 4or1, H, W)
            if self.ghost_heatmap_aug is not None:
                goal = self.ghost_heatmap_aug(goal)
            if self.add_current_heatmap:
                # key e.g. "cam0_heatmap_ghost" -> present key "cam0_present_heatmap_ghost"
                cam_prefix = key.split('_heatmap')[0]          # e.g. "cam0"
                present_key = f"{cam_prefix}_present_heatmap_ghost"
                p = data[present_key][T_slice].astype(np.float32) / 255.
                present = np.moveaxis(p, -1, 1)                # (T, 4, H, W)
                obs_dict[key] = np.concatenate([goal, present], axis=1)  # (T, 8, H, W)
            else:
                obs_dict[key] = goal
            del data[key]

        # Gaussian heatmap: append current gripper heatmap channels if requested
        if self.add_current_heatmap:
            for key in self.heatmap_keys:
                cam_prefix = key.split('_heatmap')[0]          # e.g. "cam0"
                present_key = f"{cam_prefix}_present_heatmap"
                p = data[present_key][T_slice].astype(np.float32) / 255.
                present = np.moveaxis(p, -1, 1)                # (T, 4, H, W)
                obs_dict[key] = np.concatenate([obs_dict[key], present], axis=1)  # (T, 8, H, W)


        # Pointmap: (T, 3, H, W) -> decompress
        for key in self.pointmap_keys:
            obs_dict[key] = process_pointmap(data[key][T_slice], compress=False)
            if self.pointmap_frame == 'gripper_frame':
                gripper_to_world = data['gripper_to_world'][T_slice] # (T, 4, 4)
                world_to_gripper = np.linalg.inv(gripper_to_world)  # (T, 4, 4)
                R = world_to_gripper[:, :3, :3]  # (T, 3, 3)
                t = world_to_gripper[:, :3, 3]   # (T, 3)
                T, _, H, W = obs_dict[key].shape
                pts = obs_dict[key].reshape(T, 3, -1)          # (T, 3, H*W)
                mask = np.all(pts == 0, axis=1, keepdims=True)  # (T, 1, H*W)
                transformed = (R @ pts + t[:, :, None])
                obs_dict[key] = np.where(mask, 0.0, transformed).reshape(T, 3, H, W)
            del data[key]
        if self.pointmap_frame == 'gripper_frame':
            del data['gripper_to_world']

        # Plucker: (T, 6, H, W) -> decompress
        for key in self.plucker_keys:
            obs_dict[key] = process_plucker(data[key][T_slice], compress=False)
            del data[key]

        # Low dim & matrices
        for key in self.lowdim_keys + self.matrix_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        # Goal gripper pts: (T, 4, 3) float32 — loaded as-is, no reshape
        for key in self.goal_gripper_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        # GMM distribution: loaded as-is (no reshape, no normalization)
        # gmm_all_goals:   (T, N, 4, 3) float32
        # gmm_all_weights: (T, N)       float32
        # With gmm_pred_npz_dir set, these come from the external npz tree
        # (the h5 copies were never loaded by the buffer).
        if self.gmm_pred_npz_dir is not None and self.gmm_keys:
            self._fill_gmm_from_pred_npz(idx, obs_dict)
        else:
            for key in self.gmm_keys:
                obs_dict[key] = data[key][T_slice].astype(np.float32)
                del data[key]

        # Actions
        # 6. Actions
        if self.action_mode == 'hybrid_relative':
            relative_action = hybrid_delta_to_hybrid_relative_actions(data['action'].astype(np.float32))
            action = torch.from_numpy(relative_action)
        elif self.action_mode == 'relative':
            relative_action = delta_to_relative_actions(data['action'].astype(np.float32))
            action = torch.from_numpy(relative_action)
        elif 'delta' in self.action_mode:
            action = torch.from_numpy(data['action'].astype(np.float32))
        elif self.action_mode == 'absolute':
            # Target is state[t] ⊕ action/hybrid[t] across the horizon — the
            # commanded target pose for the OSC controller (world xyz + body rot
            # + additive gripper). action_key is 'action/hybrid' in this mode,
            # so data['action'] is the hybrid delta we compose with state[t].
            action = torch.from_numpy(self._build_absolute_action_target(
                idx, data['action']
            ))

        # import pdb; pdb.set_trace()
        return {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': action,
        }
