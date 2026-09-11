"""Dataset adapter for pre-processed .npz files (Coffee Task format)."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

from lfd3d.datasets.base_data import BaseDataModule, BaseDataset
from lfd3d.datasets.rgb_text_feature_gen import get_siglip_text_embedding
from lfd3d.utils.data_utils import collate_pcd_fn
from transformers import AutoModel, AutoProcessor


def _load_or_compute_text_embed(
    caption: str, cache_path: str | None, use_text_embed: bool
) -> np.ndarray:
    """
    Return a text embedding for the given caption.
    - use_text_embed=False: return zeros (single-task training; no SigLIP loaded)
    - use_text_embed=True:  load from cache_path if it exists, otherwise run
      SigLIP once and save the result to cache_path for future runs.
    """
    siglip_dim = 1152  # google/siglip-so400m-patch14-384 text output dim

    if not use_text_embed:
        return np.zeros(siglip_dim, dtype=np.float32)

    if cache_path and Path(cache_path).exists():
        return np.load(cache_path).astype(np.float32)

    siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    embed = get_siglip_text_embedding(
        caption, siglip=siglip, siglip_processor=processor, device="cpu"
    )  # (1152,) float32

    if cache_path:
        np.save(cache_path, embed)

    return embed


class NpyDataset(BaseDataset):
    """
    Dataset for per-frame .npz files produced by the Coffee Task data pipeline.

    Expected directory layout:
        data_dir/
            demo_0/
                0.npz
                1.npz
                ...
            demo_1/
                ...

    Each .npz contains:
        point_cloud      (1, N, 3)   scene point cloud
        gripper_pcd      (1, 4, 3)   gripper keypoints at this frame
        goal_gripper_pcd (1, 4, 3)   gripper keypoints at the goal frame
        rgb_agentview    (1, H, W, 3)
        depth_agentview  (1, H, W, 1)
        agentview_intrinsics (1, 3, 3)
        agentview_extrinsics (1, 4, 4)
        ...
    """

    GRIPPER_IDX = np.array([0, 1, 2])

    def __init__(
        self,
        dataset_cfg,
        split: str = "train",
        split_indices=None,
        augment_train=None,
        augment_cfg=None,
    ):
        super().__init__(augment_train=augment_train, augment_cfg=augment_cfg)
        self.dataset_cfg = dataset_cfg
        self.split = split
        self.data_dir = Path(dataset_cfg.data_dir)
        self.task_caption = dataset_cfg.task_caption

        # Goal source selection. "default" reads goal_gripper_pcd from the main
        # per-frame .npz (original behavior). The other sources read
        # goal_gripper_pcd_<source> from a parallel directory tree
        # (extra_goals_dir/demo_X/t.npz) produced by the RDP keypoint pipeline.
        VALID_GOAL_SOURCES = (
            "default",
            "rdp",
            "rdp_gripper",
            "random",
            "fixed_interval",
            # AWE (greedy, th=0.35, gripper-aware) keypoints: goal_gripper_pcd_awe
            # in EXTRA_KEYPOINTS/AWE_EXTRA_KEYPOINTS/EXTRA_KEYPOINTS_awe-greedy-th0.35-grip/<TASK>.
            "awe",
        )
        self.goal_source = dataset_cfg.get("goal_source", "default")
        if self.goal_source not in VALID_GOAL_SOURCES:
            raise ValueError(
                f"Invalid goal_source '{self.goal_source}'. "
                f"Expected one of {VALID_GOAL_SOURCES}"
            )
        self.extra_goals_dir = None
        # Per-demo source-stem -> extra-stem lookup, filled lazily by
        # _extra_goal_path (see there for why we can't just match by name).
        self._extra_pos_cache: dict = {}
        if self.goal_source != "default":
            extra_goals_dir = dataset_cfg.get("extra_goals_dir", None)
            if not extra_goals_dir:
                raise ValueError(
                    f"goal_source='{self.goal_source}' requires dataset.extra_goals_dir "
                    "to point at the EXTRA_KEYPOINTS copy of this dataset."
                )
            self.extra_goals_dir = Path(extra_goals_dir)
            if not self.extra_goals_dir.is_dir():
                raise FileNotFoundError(
                    f"extra_goals_dir does not exist: {self.extra_goals_dir}"
                )
        print(f"[NpyDataset] goal_source={self.goal_source} ({split})")

        # Where the sampler-weight / swap-metadata caches live. Defaults to
        # data_dir (original behavior). Jobs that stage data to node-local
        # scratch can set transition_cache_dir to a persistent location (e.g.
        # the /ocean source dir) so the full-dataset goal scans run only once.
        cache_dir = dataset_cfg.get("transition_cache_dir", None)
        self._transition_cache_dir = Path(cache_dir) if cache_dir else self.data_dir
        # Distinguish caches per goal source; default keeps the legacy names.
        self._goal_src_tag = "" if self.goal_source == "default" else f"_{self.goal_source}"

        # RL Bench datasets use a different camera schema (front_* keys, stored
        # channel-first) than MimicGen (agentview_*, channel-last). When set,
        # _read_primary_camera reads/normalizes the RL Bench `front` camera.
        self.is_rl_bench = dataset_cfg.get("is_rl_bench", False)

        # Camera-less datasets (has_camera: False, e.g. PushT): frames carry no
        # intrinsics/extrinsics (and depth is all zeros), so we omit the
        # rgbs/depths/intrinsics/extrinsics keys from every sample entirely.
        # GoalRegressionModule detects the missing "intrinsics" key and skips
        # pixel metrics and camera-projection visualizations.
        self.has_camera = dataset_cfg.get("has_camera", True)

        # Collect all frame paths, sorted deterministically
        all_frames = []
        for demo_dir in sorted(self.data_dir.iterdir()):
            if not demo_dir.is_dir():
                continue
            frames = sorted(demo_dir.glob("*.npz"), key=lambda p: int(p.stem))
            all_frames.extend(frames)

        if split_indices is not None:
            self.frames = [all_frames[i] for i in split_indices]
        else:
            self.frames = all_frames

        # Compute (or load from cache) the text embedding once in the main process.
        # Workers receive it via pickle — no SigLIP reload per worker.
        cache_path = dataset_cfg.get("text_embed_cache", None)
        use_text_embed = dataset_cfg.get("use_text_embed", False)
        self.text_embed = _load_or_compute_text_embed(
            self.task_caption, cache_path, use_text_embed
        )

        # Weighted sampler support — only computed for train split when enabled.
        self.sample_weights = None
        if split == "train" and dataset_cfg.get("use_weighted_sampler", False):
            p = float(dataset_cfg.get("transition_p", 0.5))
            radius = int(dataset_cfg.get("transition_radius", 10))
            cache_file = (
                self._transition_cache_dir
                / f".sample_weights{self._goal_src_tag}_p{p}_r{radius}.npy"
            )
            if cache_file.exists():
                print(f"[NpyDataset] Loading cached transition weights from {cache_file}")
                self.sample_weights = np.load(cache_file)
                if len(self.sample_weights) != len(self.frames):
                    print(
                        f"[NpyDataset] Cached weights cover {len(self.sample_weights)} "
                        f"frames but dataset has {len(self.frames)} — stale cache "
                        f"(different demo subset/split?). Recomputing."
                    )
                    self.sample_weights = None
            if self.sample_weights is None:
                print(f"[NpyDataset] Computing transition weights (p={p}, radius={radius})...")
                self.sample_weights = self._compute_sample_weights(p=p, transition_radius=radius)
                np.save(cache_file, self.sample_weights)
                print(f"[NpyDataset] Saved weights to {cache_file}")

        # Label swap augmentation — within +/-transition_radius of a goal change,
        # replace the GT goal with the goal on the other side of the transition
        # with a probability that decays linearly with distance from the
        # transition. Train only.
        self._swap_neighbor_goals = None
        self._swap_p = None
        if split == "train" and dataset_cfg.get("transition_label_swap", False):
            # Two fully independent swap-probability profiles:
            #   linear  (default) — the original triangular scheme. Uses
            #             transition_radius: p decays linearly to 0 AT radius.
            #   sigmoid           — smooth profile with NO window. Uses only
            #             transition_swap_tau (temperature): every frame swaps
            #             toward the goal across its nearest transition with
            #                 p_swap(d) = 2 * p_max * sigmoid(-d / tau)
            #             (= p_max at the transition, tails self-limit).
            # transition_radius keeps its role for the WEIGHTED SAMPLER in both
            # profiles; the sigmoid swap itself never reads it.
            profile = str(dataset_cfg.get("transition_swap_profile", "linear"))
            p_max = float(dataset_cfg.get("transition_swap_p_max", 0.5))
            if profile == "linear":
                radius = int(dataset_cfg.get("transition_radius", 10))
                # _v2 marks the formula change (denominator radius+1 -> radius,
                # 2026-08-07). Without the bump, caches written by the old
                # formula would be silently reused for the same (p_max, radius).
                cache_file = (
                    self._transition_cache_dir
                    / f".swap_meta{self._goal_src_tag}_pmax{p_max}_r{radius}_v2.npz"
                )
                compute_desc = f"profile=linear, p_max={p_max}, radius={radius}"
            elif profile == "sigmoid":
                tau = float(dataset_cfg.get("transition_swap_tau", 1.0))
                cache_file = (
                    self._transition_cache_dir
                    / f".swap_meta_sigmoid{self._goal_src_tag}_pmax{p_max}_tau{tau}.npz"
                )
                compute_desc = f"profile=sigmoid, p_max={p_max}, tau={tau}"
            else:
                raise ValueError(
                    f"Invalid transition_swap_profile '{profile}'. "
                    f"Expected 'linear' or 'sigmoid'."
                )
            if cache_file.exists():
                print(f"[NpyDataset] Loading cached swap metadata from {cache_file}")
                arr = np.load(cache_file)
                self._swap_neighbor_goals = arr["neighbor_goals"]
                self._swap_p = arr["p_swap"]
                if len(self._swap_p) != len(self.frames):
                    print(
                        f"[NpyDataset] Cached swap metadata covers {len(self._swap_p)} "
                        f"frames but dataset has {len(self.frames)} — stale cache "
                        f"(different demo subset/split?). Recomputing."
                    )
                    self._swap_neighbor_goals, self._swap_p = None, None
            if self._swap_p is None:
                print(f"[NpyDataset] Computing swap metadata ({compute_desc})...")
                if profile == "linear":
                    self._swap_neighbor_goals, self._swap_p = self._compute_swap_meta(
                        transition_radius=radius, p_max=p_max
                    )
                else:
                    self._swap_neighbor_goals, self._swap_p = self._compute_swap_meta_sigmoid(
                        tau=tau, p_max=p_max
                    )
                np.savez(cache_file, neighbor_goals=self._swap_neighbor_goals, p_swap=self._swap_p)
                print(f"[NpyDataset] Saved swap metadata to {cache_file}")
            # For the sigmoid profile p_swap is nonzero (but tiny) almost
            # everywhere; count frames with non-negligible probability instead.
            eligible_thresh = 0.0 if profile == "linear" else 0.01
            n_eligible = int((self._swap_p > eligible_thresh).sum())
            print(
                f"[NpyDataset] {n_eligible}/{len(self.frames)} frames eligible "
                f"for label swap (p_swap > {eligible_thresh})"
            )

        # Per-frame language conditioning. When enabled, each frame is
        # conditioned on its own `lang_goal` instruction (e.g. RL Bench
        # "...tall dustpan" vs "...short dustpan") instead of the single
        # constant task_caption. Runs for BOTH train and val splits — the model
        # needs the matching instruction at eval time too.
        self._lang_embeds = None     # (n_distinct, 1152) float32
        self._lang_row = None        # (N,) int64 -> row in _lang_embeds
        self._frame_captions = None  # (N,) list[str], used for the returned caption
        if dataset_cfg.get("add_language_cond", False):
            self._setup_language_conditioning()

    def _setup_language_conditioning(self):
        """
        Build per-frame SigLIP text embeddings from each frame's `lang_goal`.

        `lang_goal` is stored per frame in the .npz but is constant within a
        demo (one variation per episode), so we read it once per demo (one
        np.load per demo, not per frame). The distinct caption strings are
        embedded once with SigLIP (canonical max_length=64 padding) and cached
        to `.lang_goal_embeds.npz` in data_dir so re-runs / the other split
        skip the SigLIP load. Workers inherit the finished arrays via pickle —
        SigLIP never runs in a dataloader worker.
        """
        # 1) Per-demo caption (one read per demo; lang_goal is per-episode).
        frame_groups: dict = {}
        for i, f in enumerate(self.frames):
            frame_groups.setdefault(str(f.parent), []).append(i)

        captions: list = [None] * len(self.frames)
        for idxs in frame_groups.values():
            lg = np.load(self.frames[idxs[0]], allow_pickle=True)["lang_goal"]
            caption = str(np.asarray(lg).reshape(-1)[0])
            for i in idxs:
                captions[i] = caption

        distinct = sorted(set(captions))

        # 2) Embed distinct captions, loading/extending an on-disk cache.
        cache_file = self.data_dir / ".lang_goal_embeds.npz"
        embeds_by_caption: dict = {}
        if cache_file.exists():
            arr = np.load(cache_file, allow_pickle=True)
            for c, e in zip(list(arr["captions"]), arr["embeds"]):
                embeds_by_caption[str(c)] = e.astype(np.float32)

        missing = [c for c in distinct if c not in embeds_by_caption]
        if missing:
            print(f"[NpyDataset] Embedding {len(missing)} lang_goal caption(s) with SigLIP...")
            siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
            processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
            for c in missing:
                embeds_by_caption[c] = get_siglip_text_embedding(
                    c, siglip=siglip, siglip_processor=processor, device="cpu"
                ).astype(np.float32)
            all_caps = list(embeds_by_caption.keys())
            all_emb = np.stack([embeds_by_caption[c] for c in all_caps]).astype(np.float32)
            np.savez(cache_file, captions=np.array(all_caps, dtype=object), embeds=all_emb)
            print(f"[NpyDataset] Saved lang_goal embeddings to {cache_file}")

        # 3) Per-frame lookup tables.
        cap_to_row = {c: j for j, c in enumerate(distinct)}
        self._lang_embeds = np.stack(
            [embeds_by_caption[c] for c in distinct]
        ).astype(np.float32)
        self._lang_row = np.array([cap_to_row[c] for c in captions], dtype=np.int64)
        self._frame_captions = captions
        print(
            f"[NpyDataset] Language conditioning ON ({self.split}): "
            f"{len(distinct)} distinct lang_goal(s) over {len(self.frames)} frames: {distinct}"
        )

    def _load_goal(self, frame_path: Path) -> np.ndarray:
        """
        Load the (4, 3) goal gripper keypoints for one frame from the selected
        goal source. For "default" this is goal_gripper_pcd in the main frame
        file; otherwise goal_gripper_pcd_<source> from the parallel
        extra_goals_dir tree (same demo_X/t.npz layout).
        """
        if self.goal_source == "default":
            d = np.load(frame_path, allow_pickle=True)
            return d["goal_gripper_pcd"][0].astype(np.float32)
        extra_path = self._extra_goal_path(frame_path)
        d = np.load(extra_path, allow_pickle=True)
        return d[f"goal_gripper_pcd_{self.goal_source}"][0].astype(np.float32)

    def _extra_goal_path(self, frame_path: Path) -> Path:
        """
        Resolve the extra-goals npz for one source frame.

        The extra trees are not always numbered like the source. KITCHEN_D1
        source demos have gaps (demo_1 has no 62.npz / 362.npz), while the RDP
        and AWE trees for those demos are numbered contiguously 0..T-1 in
        sorted order — the same convention the low-level h5 files use.
        Matching by file name there shifts every goal after a gap by one and
        raises FileNotFoundError on the last frames (this killed the
        KITCHEN_D1 RDP 100-demo run, job 43167255). So:

          * extra demo dir has exactly as many frames as the source demo dir
            -> map the i-th sorted source frame to the i-th sorted extra frame
               (identical to name matching whenever the stems agree);
          * otherwise, if it carries every source stem -> map by name;
          * anything else -> error, never a silently misaligned goal.
        """
        demo_dir = frame_path.parent
        key = str(demo_dir)
        entry = self._extra_pos_cache.get(key)
        if entry is None:
            extra_dir = self.extra_goals_dir / demo_dir.name
            src_stems = sorted(int(p.stem) for p in demo_dir.glob("*.npz"))
            ext_stems = sorted(int(p.stem) for p in extra_dir.glob("*.npz"))
            if len(ext_stems) == len(src_stems):
                lookup = dict(zip(src_stems, ext_stems))
            elif set(src_stems) <= set(ext_stems):
                lookup = {s: s for s in src_stems}
            else:
                raise FileNotFoundError(
                    f"{extra_dir}: {len(ext_stems)} extra-goal frames vs "
                    f"{len(src_stems)} source frames in {demo_dir}, and the "
                    f"extra tree does not contain every source stem — cannot "
                    f"align goal_source='{self.goal_source}'."
                )
            entry = (extra_dir, lookup)
            self._extra_pos_cache[key] = entry
        extra_dir, lookup = entry
        return extra_dir / f"{lookup[int(frame_path.stem)]}.npz"

    def _compute_sample_weights(self, p: float, transition_radius: int) -> np.ndarray:
        """
        Frames within transition_radius of a goal transition get collective
        probability p; all other frames get collective probability (1-p).
        """
        is_near = np.zeros(len(self.frames), dtype=bool)

        # Group frames by demo directory
        frame_groups: dict = {}
        for i, f in enumerate(self.frames):
            frame_groups.setdefault(str(f.parent), []).append((i, f))

        for indexed_frames in frame_groups.values():
            indices = [i for i, _ in indexed_frames]
            paths   = [f for _, f in indexed_frames]

            goals = np.array([
                self._load_goal(path) for path in paths
            ])  # (T, 4, 3)

            transitions = [
                t for t in range(1, len(goals))
                if not np.allclose(goals[t], goals[t - 1], atol=1e-6)
            ]

            for t_trans in transitions:
                for local_i in range(
                    max(0, t_trans - transition_radius),
                    min(len(goals), t_trans + transition_radius + 1),
                ):
                    is_near[indices[local_i]] = True

        n_near  = is_near.sum()
        n_other = len(self.frames) - n_near

        weights = np.zeros(len(self.frames), dtype=np.float32)
        if n_near > 0:
            weights[is_near]  = p / n_near
        if n_other > 0:
            weights[~is_near] = (1.0 - p) / n_other

        return weights

    def _compute_swap_meta(self, transition_radius: int, p_max: float):
        """
        For each frame within +/-transition_radius of a goal change, store the
        "neighbor goal" (the goal on the other side of the nearest transition)
        and the per-frame Bernoulli swap probability:

            p_swap(d) = p_max * (1 - d / transition_radius)

        where d = |t - t_trans|. The triangle now reaches exactly 0 at
        d = transition_radius (it previously decayed to 0 only at radius+1,
        leaving p_max/(r+1) of swap probability at the window edge). Frames
        not in any window have p_swap = 0 and a zero neighbor goal (never
        read).

        Returns:
            neighbor_goals: (N, 4, 3) float32
            p_swap_arr:     (N,)       float32
        """
        n = len(self.frames)
        neighbor_goals = np.zeros((n, 4, 3), dtype=np.float32)
        p_swap_arr = np.zeros((n,), dtype=np.float32)

        frame_groups: dict = {}
        for i, f in enumerate(self.frames):
            frame_groups.setdefault(str(f.parent), []).append((i, f))

        for indexed_frames in frame_groups.values():
            indices = [i for i, _ in indexed_frames]
            paths   = [f for _, f in indexed_frames]

            goals = np.array([
                self._load_goal(path) for path in paths
            ])  # (T, 4, 3)

            transitions = [
                t for t in range(1, len(goals))
                if not np.allclose(goals[t], goals[t - 1], atol=1e-6)
            ]
            if not transitions:
                continue

            for local_t in range(len(goals)):
                # Find the closest transition within radius
                best_d, best_t = None, None
                for t_trans in transitions:
                    d = abs(local_t - t_trans)
                    if d <= transition_radius and (best_d is None or d < best_d):
                        best_d, best_t = d, t_trans
                if best_t is None:
                    continue

                # Neighbor = goal on the other side of the nearest transition.
                # t_trans is the first frame of the new goal, so frames < t_trans
                # carry the previous goal and frames >= t_trans carry the new one.
                if local_t < best_t:
                    neighbor = goals[best_t]            # upcoming goal
                else:
                    neighbor = goals[best_t - 1]        # previous goal

                global_idx = indices[local_t]
                neighbor_goals[global_idx] = neighbor.astype(np.float32)
                # Triangle hits exactly 0 at d == transition_radius. max(...,1)
                # guards a radius=0 config (only d=0 is in-window there anyway).
                p_swap_arr[global_idx] = p_max * (
                    1.0 - best_d / max(transition_radius, 1)
                )

        return neighbor_goals, p_swap_arr

    def _compute_swap_meta_sigmoid(self, tau: float, p_max: float):
        """
        Sigmoid swap profile — fully independent of the linear/triangular one
        and of transition_radius. EVERY frame in a demo that has at least one
        goal transition stores the "neighbor goal" (the goal on the other side
        of its NEAREST transition) and the per-frame Bernoulli swap probability

            p_swap(d) = 2 * p_max * sigmoid(-d / tau)
                      = 2 * p_max / (1 + exp(d / tau))

        where d = |t - t_trans| to the nearest transition and tau is the
        temperature: small tau -> mixing confined to the boundary, large tau ->
        gradual mixing. p_swap(0) = p_max exactly; the tails self-limit (no
        hard window is applied). Reframed as P(label = next goal), this is a
        smooth sigmoid ramp from ~0 before the transition to ~1 after it,
        crossing 0.5*(2*p_max) at the boundary.

        Frames in demos with no transitions keep p_swap = 0.

        Returns:
            neighbor_goals: (N, 4, 3) float32
            p_swap_arr:     (N,)       float32
        """
        n = len(self.frames)
        neighbor_goals = np.zeros((n, 4, 3), dtype=np.float32)
        p_swap_arr = np.zeros((n,), dtype=np.float32)

        frame_groups: dict = {}
        for i, f in enumerate(self.frames):
            frame_groups.setdefault(str(f.parent), []).append((i, f))

        for indexed_frames in frame_groups.values():
            indices = [i for i, _ in indexed_frames]
            paths   = [f for _, f in indexed_frames]

            goals = np.array([
                self._load_goal(path) for path in paths
            ])  # (T, 4, 3)

            transitions = [
                t for t in range(1, len(goals))
                if not np.allclose(goals[t], goals[t - 1], atol=1e-6)
            ]
            if not transitions:
                continue
            trans_arr = np.asarray(transitions)

            for local_t in range(len(goals)):
                # Nearest transition (no window — the sigmoid tail decides).
                d_all = np.abs(local_t - trans_arr)
                j = int(d_all.argmin())
                best_t, best_d = int(trans_arr[j]), int(d_all[j])

                # Neighbor = goal on the other side of the nearest transition.
                # t_trans is the first frame of the new goal, so frames < t_trans
                # carry the previous goal and frames >= t_trans carry the new one.
                if local_t < best_t:
                    neighbor = goals[best_t]            # upcoming goal
                else:
                    neighbor = goals[best_t - 1]        # previous goal

                global_idx = indices[local_t]
                neighbor_goals[global_idx] = neighbor.astype(np.float32)
                p_swap_arr[global_idx] = 2.0 * p_max / (1.0 + np.exp(best_d / tau))

        return neighbor_goals, p_swap_arr

    def _read_primary_camera(self, d):
        """Read the primary camera and normalize to the schema the model expects.

        Returns:
            rgb:   (H, W, 3) uint8
            depth: (H, W)    float32
            K:     (3, 3)    float32  intrinsics
            T:     (4, 4)    float32  extrinsics (world_from_cam)

        Handles two on-disk layouts:
          - MimicGen: `agentview_*` keys, images channel-last  (1, H, W, C)
          - RL Bench: `front_*`    keys, images channel-first (1, C, H, W).
            RL Bench has 4 cameras; we use `front` as the primary view and
            ignore left_shoulder / right_shoulder / wrist (the scene point_cloud
            is already the front+left+right fusion).
        """
        if self.is_rl_bench:
            rgb = d["front_rgb"][0].transpose(1, 2, 0).astype(np.uint8)  # (H,W,3)
            depth = d["front_depth"][0, 0].astype(np.float32)            # (H,W)
            K = d["front_camera_intrinsics"][0].astype(np.float32)       # (3,3)
            T = d["front_camera_extrinsics"][0].astype(np.float32)       # (4,4)
        else:
            rgb = d["rgb_agentview"][0].astype(np.uint8)                 # (H,W,3)
            depth = d["depth_agentview"][0, :, :, 0].astype(np.float32)  # (H,W)
            K = d["agentview_intrinsics"][0].astype(np.float32)          # (3,3)
            T = d["agentview_extrinsics"][0].astype(np.float32)          # (4,4)
        return rgb, depth, K, T

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        d = np.load(self.frames[idx], allow_pickle=True)

        anchor_pcd = d["point_cloud"][0].astype(np.float32)       # (N, 3)
        action_pcd = d["gripper_pcd"][0].astype(np.float32)       # (4, 3)
        if self.goal_source == "default":
            goal_pcd = d["goal_gripper_pcd"][0].astype(np.float32)  # (4, 3)
        else:
            goal_pcd = self._load_goal(self.frames[idx])            # (4, 3)

        # Label swap augmentation: with linearly-decaying probability around
        # transitions, replace the GT goal with the goal across the nearest
        # transition. p_swap is precomputed per-frame in __init__ (zero outside
        # any transition window), so this is a no-op for most frames.
        if self._swap_p is not None and self._swap_p[idx] > 0.0:
            if np.random.random() < float(self._swap_p[idx]):
                goal_pcd = self._swap_neighbor_goals[idx].copy()

        # Normalization: follow base_data.get_normalize_mean_std
        # With normalize=False (default) this is identity (mean=0, std=1)
        if self.dataset_cfg.get("normalize", False):
            pcd_mean = action_pcd.mean(axis=0)
            pcd_std  = anchor_pcd.std(axis=0)
        else:
            pcd_mean = np.zeros(3, dtype=np.float32)
            pcd_std  = np.ones(3, dtype=np.float32)

        action_pcd_norm = (action_pcd - pcd_mean) / pcd_std
        anchor_pcd_norm = (anchor_pcd - pcd_mean) / pcd_std
        cross_displacement = (goal_pcd - action_pcd) / pcd_std  # (4, 3)

        # Dummy features: model won't read them with use_rgb=False, but
        # collate_pcd_fn always expects anchor_feat_pcd alongside anchor_pcd.
        anchor_feat_pcd = np.zeros(
            (anchor_pcd_norm.shape[0], 3), dtype=np.float32
        )

        # Camera data (primary view). Schema-normalized across MimicGen
        # (agentview_*, channel-last) and RL Bench (front_*, channel-first).
        # Omitted entirely for camera-less datasets (has_camera: False).
        camera_items = {}
        if self.has_camera:
            rgb, depth, K, T = self._read_primary_camera(d)

            # Stack start/end frames; data is already a single frame so duplicate
            rgbs   = np.stack([rgb, rgb], axis=0)    # (2, H, W, 3)
            depths = np.stack([depth, depth], axis=0)  # (2, H, W)
            camera_items = {
                # Primary camera
                "rgbs":       rgbs,
                "depths":     depths,
                "intrinsics": K,
                "extrinsics": T,
                # Auxiliary cameras (none)
                "aux_rgbs":       np.zeros((0, 2, rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8),
                "aux_depths":     np.zeros((0, 2, rgb.shape[0], rgb.shape[1]), dtype=np.float32),
                "aux_intrinsics": np.zeros((0, 3, 3), dtype=np.float32),
                "aux_extrinsics": np.zeros((0, 4, 4), dtype=np.float32),
            }

        # Dummy gripper trajectory (10 future steps)
        gripper_trajectory = np.tile(action_pcd_norm[[0]], (10, 1)).astype(np.float32)

        # Per-frame language conditioning: serve this frame's lang_goal caption
        # and embedding when enabled, else fall back to the constant task_caption
        # / self.text_embed (zeros when use_text_embed is False).
        if self._lang_row is not None:
            caption = self._frame_captions[idx]
            text_embed = self._lang_embeds[self._lang_row[idx]]
        else:
            caption = self.task_caption
            text_embed = self.text_embed

        return {
            # Camera keys (rgbs/depths/intrinsics/extrinsics/aux_*) — empty
            # dict when has_camera is False.
            **camera_items,
            # Point clouds
            "action_pcd":       action_pcd_norm,   # (4, 3) → Pointclouds
            "anchor_pcd":       anchor_pcd_norm,   # (N, 3) → Pointclouds(features=anchor_feat_pcd)
            "anchor_feat_pcd":  anchor_feat_pcd,   # (N, 3) zeros
            "gripper_trajectory": gripper_trajectory,  # (10, 3) → Pointclouds
            # Labels
            "cross_displacement": cross_displacement,  # (4, 3) → Pointclouds
            # Text
            "caption":        caption,
            "text_embed":     text_embed,
            "actual_caption": caption,
            # Transforms / normalization
            "start2end":  np.eye(4, dtype=np.float32),
            "pcd_mean":   pcd_mean,
            "pcd_std":    pcd_std,
            "augment_R":  np.eye(3, dtype=np.float32),
            "augment_t":  np.zeros(3, dtype=np.float32),
            "augment_C":  anchor_pcd.mean(axis=0).astype(np.float32),
            # Metadata
            "gripper_idx": self.GRIPPER_IDX,
            "vid_name":    str(self.frames[idx]),
            "data_source": "libero_franka",
        }


class NpyDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size,
        val_batch_size,
        num_workers,
        dataset_cfg,
        seed,
        augment_train=None,
        augment_cfg=None,
    ):
        super().__init__(
            batch_size,
            val_batch_size,
            num_workers,
            dataset_cfg,
            seed,
            augment_train,
            augment_cfg,
        )

    def _generate_splits(self):
        """Split by demo directory: 90% train, 10% val."""
        data_dir = Path(self.dataset_cfg.data_dir)
        demo_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())

        rng = random.Random(self.seed)
        demo_dirs_shuffled = demo_dirs[:]
        rng.shuffle(demo_dirs_shuffled)

        n_val = max(1, int(len(demo_dirs_shuffled) * self.dataset_cfg.val_episode_ratio))
        val_demos  = set(str(d) for d in demo_dirs_shuffled[:n_val])
        train_demos = set(str(d) for d in demo_dirs_shuffled[n_val:])

        # Build frame-level indices relative to the full sorted frame list
        all_frames = []
        for demo_dir in sorted(data_dir.iterdir()):
            if not demo_dir.is_dir():
                continue
            frames = sorted(demo_dir.glob("*.npz"), key=lambda p: int(p.stem))
            all_frames.extend(frames)

        train_indices = [i for i, f in enumerate(all_frames) if str(f.parent) in train_demos]
        val_indices   = [i for i, f in enumerate(all_frames) if str(f.parent) in val_demos]
        return train_indices, val_indices

    def setup(self, stage: str = "fit"):
        self.stage = stage
        train_indices, val_indices = self._generate_splits()

        self.train_dataset = NpyDataset(
            dataset_cfg=self.dataset_cfg,
            split="train",
            split_indices=train_indices,
            augment_train=self.augment_train,
            augment_cfg=self.augment_cfg,
        )
        self.val_datasets = {
            "coffee_task": NpyDataset(
                dataset_cfg=self.dataset_cfg,
                split="val",
                split_indices=val_indices,
                augment_train=None,
                augment_cfg=self.augment_cfg,
            )
        }
        self.val_tags = list(self.val_datasets.keys())
        self.test_datasets = {}

    def train_dataloader(self):
        if self.train_dataset.sample_weights is not None:
            sampler = data.WeightedRandomSampler(
                weights=torch.from_numpy(self.train_dataset.sample_weights),
                num_samples=len(self.train_dataset),
                replacement=True,
            )
            return data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=collate_pcd_fn,
            )
        return super().train_dataloader()
