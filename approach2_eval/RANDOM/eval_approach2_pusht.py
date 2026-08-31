"""
Approach 2 eval on PushT: FlowMatchingDiTGoalGMMPolicy ALONE (no high-level).

Sim-eval counterpart of eval_approach2_2d_dit_low_level.py (Hammer_Cleanup_D1),
for the PushT checkpoint trained on the 206-demo GROOT-style h5 set
(NO_GMM_DATASET/PUSH_T_TASK). Same idea: the goal only supervised the visual
representation through the auxiliary GMM head during training, so at rollout the
policy is fully self-contained:

    cam0 RGB + synthetic depth/K/E + state + gripper keypoints
        -> DINOv2 -> RoPE4D grounded trunk -> DiT -> absolute 2-D setpoints

Everything the policy sees must match the training h5s BYTE-FOR-BYTE in
convention. Those conventions, all verified directly against the h5s:

  * 512 sim units == 1 m (env window 512 rendered to 256 px).
  * state          = agent_pos / 512, float32 metres.  Range over the training
                     set is ~[0.16, 0.88]; the state normalizer range printed at
                     startup must bracket live values or the run is garbage.
  * cam0_image     = PushTImageEnv obs['image']: (3, 256, 256) float32 [0, 1],
                     CHW, NO action marker (the marker only ever lands on the
                     uint8 render cache, which we use for the mp4, not the obs).
  * cam0_depth     = constant 1.0 m plane, float32 metres, (1, 256, 256).
                     Training stored uint16 mm (=1000) and the dataset loader
                     divides by 1000, so runtime feeds 1.0 directly.
  * cam0_intrinsic / cam0_extrinsic = the fixed synthetic top-down camera from
                     inject_pusht_camera_geometry.py (Mimicgen_Inference repo).
                     Constants are recomputed here from the same fit parameters.
  * present_gripper_pts = GRIPPER_TEMPLATE + (x, y, 0): a constant synthetic
                     4-keypoint "gripper" (identity quat).  Template extracted
                     from the training h5s; constant across all 206 demos to
                     5e-8 (float32 rounding).
  * action ('hybrid_delta' mode serves action/hybrid as-is) = ABSOLUTE agent
                     setpoint in metres -- NOT a delta.  env action = pred * 512.

Observation history (--obs_history):
  * 'rolling' (default): a real 2-deep deque of env obs (t-1, t), first frame
    repeated at episode start -- matching the dataset's pad_before=1 and the
    Hammer eval.
  * 'repeat_current': both obs steps are the CURRENT frame t -- no t-1.
    (Tried 2026-08-25; worse than rolling over a full seed, kept as an option.)

Metrics (per episode), as requested:
  * max_coverage : max over the episode of the T-block's normalized covered
                   area, computed with the SAME shapely intersection the env
                   uses internally (env reward is clipped at coverage/0.95, so
                   raw coverage is recomputed here rather than backed out).
  * success      : max_coverage > env.success_threshold (0.95).

Output layout is byte-compatible with the MimicGen Approach 2 evals
(args.json / results.jsonl / summary.json / media/*.mp4 / empty
media_with_goal_overlay/) so the shell-side resume+merge bookkeeping and any
downstream analysis work unchanged.

sys.path note: this file must live in a SUBFOLDER of the PushT repo, and the
PushT repo root must never reach sys.path -- its regular `diffusion_policy`
package (has __init__.py) would shadow the Low_Level repo's NAMESPACE package
of the same name, and the Approach 2 policy class would appear to be missing.
The env modules are byte-identical in both repos (diffed), so both the policy
AND PushTImageEnv are imported from the Low_Level repo's namespace package.
The Low_Level repo is used strictly read-only.
"""

import argparse
import collections
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Constants that mirror the training data (see module docstring for provenance)
# ----------------------------------------------------------------------------
SIM_UNITS_PER_M = 512.0

# From inject_pusht_camera_geometry.py: least-squares pixel->world fit over the
# training images (residual 0.68 mm rms), depth plane fixed at 1.0 m.
PIX_TO_M_X = 0.00390712
PIX_TO_M_Y = 0.00390798
OFF_X      = 0.001747
OFF_Y      = 0.001518
DEPTH_M    = 1.0
IMG_HW     = 256
CX = CY    = IMG_HW / 2.0
FX = DEPTH_M / PIX_TO_M_X
FY = DEPTH_M / PIX_TO_M_Y
CAM_X = OFF_X + CX * PIX_TO_M_X
CAM_Y = OFF_Y + CY * PIX_TO_M_Y

INTRINSIC = np.array([[FX, 0.0, CX],
                      [0.0, FY, CY],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

EXTRINSIC = np.array([[1.0, 0.0,  0.0, CAM_X],
                      [0.0, 1.0,  0.0, CAM_Y],
                      [0.0, 0.0, -1.0, DEPTH_M],
                      [0.0, 0.0,  0.0, 1.0]], dtype=np.float32)

# obs/present_gripper_pts[t] - (state_x, state_y, 0), averaged over the first
# 20 training demos; identical across all 206 demos to 5e-8. Keypoint 3 is the
# agent itself (z=0), which is why state == present_gripper_pts[3, :2] exactly.
GRIPPER_TEMPLATE = np.array([
    [ 0.0223214593,  0.0592343907,  0.0147660999],
    [ 0.0217073548,  0.0314323912, -0.0418133400],
    [-0.0034381154,  0.0170486682,  0.0538988113],
    [ 0.0,           0.0,           0.0         ],
], dtype=np.float32)


# ----------------------------------------------------------------------------
# sys.path bootstrap
# ----------------------------------------------------------------------------
def _bootstrap_paths(args):
    """Prepend the Low_Level repo (diffusion_policy + repo root) and NOTHING
    else. In particular the PushT repo root must stay off sys.path (see module
    docstring)."""
    ll_dp = Path(args.low_level_repo) / "diffusion_policy"
    for r in (str(ll_dp), str(args.low_level_repo)):
        if os.path.isdir(r):
            if r not in sys.path:
                sys.path.insert(0, r)
        else:
            raise FileNotFoundError(f"required sys.path root missing: {r}")


# ----------------------------------------------------------------------------
# policy loading (same shape as the Hammer eval)
# ----------------------------------------------------------------------------
def load_approach2_policy(exp_dir: str, ckpt_name: str, device: str = "cuda"):
    import hydra
    from omegaconf import OmegaConf

    cfg_path = Path(exp_dir) / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"LL hydra config missing: {cfg_path}")
    cfg = OmegaConf.load(str(cfg_path))

    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    ckpt_path = Path(exp_dir) / "checkpoints" / ckpt_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"LL checkpoint missing: {ckpt_path}")
    workspace.load_checkpoint(path=str(ckpt_path))

    policy = copy.deepcopy(workspace.model)
    if OmegaConf.select(workspace.cfg, "training.use_ema", default=False):
        policy = copy.deepcopy(workspace.ema_model)
    policy.eval()
    policy.reset()
    return policy.to(device), cfg


def sanity_check_policy(policy, cfg):
    from omegaconf import OmegaConf

    tgt = str(OmegaConf.select(cfg, "policy._target_", default="")).split(".")[-1]
    aux_w = OmegaConf.select(cfg, "policy.aux_gmm_loss_weight", default=None)
    print(f"[LL] policy={tgt}  aux_gmm_loss_weight={aux_w}  "
          f"encoder={type(getattr(policy, 'visual_encoder', None)).__name__}")
    if tgt != "FlowMatchingDiTGoalGMMPolicy":
        print(f"[LL][WARN] expected FlowMatchingDiTGoalGMMPolicy, got {tgt!r}. "
              "This script feeds no goal, so a goal-conditioned LL would run blind.")

    # The one silent failure mode left: feeding state in the wrong scale. The
    # normalizer was fit on metres (~[0.16, 0.88]); raw sim units (0..512)
    # would sail far outside it.
    try:
        p = policy.normalizer["state"].params_dict
        lo = p["input_stats"]["min"].detach().cpu().numpy()
        hi = p["input_stats"]["max"].detach().cpu().numpy()
        print(f"[LL] state normalizer range: min={lo} max={hi}")
        if np.any(hi > 2.0):
            print("[LL][WARN] state normalizer max > 2 m -- training state may "
                  "not be in metres; conventions in this script would be wrong.")
    except Exception as e:  # noqa: BLE001 -- diagnostics only, never fatal
        print(f"[LL][WARN] could not read state normalizer stats: {e}")


# ----------------------------------------------------------------------------
# obs plumbing
# ----------------------------------------------------------------------------
def build_ll_obs_dict(obs_hist, n_obs_steps: int, device):
    """Stack a REAL 2-deep observation history into (B=1, T=n_obs_steps, ...).

    obs_hist entries are raw PushTImageEnv obs dicts {'image', 'agent_pos'},
    oldest first. Shorter than n_obs_steps only at episode start, where the
    oldest frame is repeated -- matching the dataset's pad_before=1.
    """
    import torch

    hist = list(obs_hist)
    while len(hist) < n_obs_steps:
        hist.insert(0, hist[0])
    hist = hist[-n_obs_steps:]

    T = len(hist)
    img = np.stack([np.asarray(o["image"], dtype=np.float32) for o in hist], 0)
    state = np.stack(
        [np.asarray(o["agent_pos"], dtype=np.float32) / SIM_UNITS_PER_M for o in hist], 0)
    state3d = np.concatenate([state, np.zeros((T, 1), dtype=np.float32)], axis=1)
    gripper = GRIPPER_TEMPLATE[None] + state3d[:, None, :]        # (T, 4, 3)

    out = {
        "cam0_image":          img[None],                                        # (1,T,3,H,W)
        "cam0_depth":          np.full((1, T, 1, IMG_HW, IMG_HW), DEPTH_M, np.float32),
        "cam0_intrinsic":      np.broadcast_to(INTRINSIC, (1, T, 3, 3)).copy(),
        "cam0_extrinsic":      np.broadcast_to(EXTRINSIC, (1, T, 4, 4)).copy(),
        "state":               state[None],                                      # (1,T,2)
        "present_gripper_pts": gripper[None],                                    # (1,T,4,3)
    }
    return {k: torch.from_numpy(v).float().to(device) for k, v in out.items()}


def compute_coverage(env, pymunk_to_shapely):
    """Raw normalized covered area, identical to the env's internal computation
    (its reward is clipped at coverage/success_threshold; this is unclipped)."""
    goal_body = env._get_goal_pose_body(env.goal_pose)
    goal_geom = pymunk_to_shapely(goal_body, env.block.shapes)
    block_geom = pymunk_to_shapely(env.block, env.block.shapes)
    return goal_geom.intersection(block_geom).area / goal_geom.area


def _video_frame(obs, env_action, render_size):
    """uint8 HWC frame for the mp4, built from the CLEAN policy obs (the env's
    own render cache is not used: stock PushTImageEnv draws its action marker
    at `action / 512 * 96` with the 96 HARDCODED, so at render_size=256 the
    cross lands at ~37.5% of its true position). The marker drawn here is at
    the correct scale: it sits exactly on the commanded setpoint."""
    import cv2
    frame = np.ascontiguousarray(
        (np.moveaxis(np.asarray(obs["image"]), 0, -1) * 255.0)
        .clip(0, 255).astype(np.uint8))
    if env_action is not None:
        coord = (np.asarray(env_action) / SIM_UNITS_PER_M * render_size).astype(np.int32)
        marker_size = max(4, int(8 / 96 * render_size))
        thickness = max(1, int(1 / 96 * render_size))
        cv2.drawMarker(frame, coord, color=(255, 0, 0),
                       markerType=cv2.MARKER_CROSS,
                       markerSize=marker_size, thickness=thickness)
    return frame


def _write_mp4(path, frames, fps):
    import imageio
    with imageio.get_writer(str(path), fps=fps, codec="libx264",
                            quality=None, ffmpeg_params=["-crf", "22", "-pix_fmt", "yuv420p"],
                            macro_block_size=1) as w:
        for f in frames:
            w.append_data(f)


# ----------------------------------------------------------------------------
# rollout
# ----------------------------------------------------------------------------
def run_episode(env, policy, pymunk_to_shapely, n_obs_steps, n_action_steps,
                max_steps, device, obs_history="rolling", action_start=0):
    import torch

    obs = env.reset()
    # 'repeat_current': deque holds only frame t, and build_ll_obs_dict's
    # padding repeats it to fill all n_obs_steps -- no t-1 ever enters.
    hist_len = 1 if obs_history == "repeat_current" else n_obs_steps
    obs_hist = collections.deque([obs], maxlen=hist_len)
    render_size = env.render_size
    frames = [_video_frame(obs, None, render_size)]

    max_coverage = compute_coverage(env, pymunk_to_shapely)
    max_reward, success, step = 0.0, False, 0

    while step < max_steps:
        ll_obs = build_ll_obs_dict(obs_hist, n_obs_steps, device)
        with torch.no_grad():
            action_dict = policy.predict_action(ll_obs)
        action_pred = action_dict.get("action_pred", action_dict["action"])
        # Dataset windows put action[0] at obs frame t-1 (obs = first To frames
        # of the horizon window). action_start = n_obs_steps-1 executes from the
        # action aligned with the CURRENT frame; 0 reproduces the policy's own
        # `action` output (and the MimicGen evals' convention), which re-issues
        # the previous step's action first.
        action_seq = action_pred[
            0, action_start:action_start + n_action_steps].detach().cpu().numpy()  # (Ta, 2) metres

        for t_idx in range(action_seq.shape[0]):
            env_action = np.clip(
                action_seq[t_idx].astype(np.float64) * SIM_UNITS_PER_M,
                0.0, SIM_UNITS_PER_M)
            obs, reward, done, _info = env.step(env_action)
            obs_hist.append(obs)                         # real t-1 / t history
            frames.append(_video_frame(obs, env_action, render_size))
            max_reward = max(max_reward, float(reward))
            max_coverage = max(max_coverage, compute_coverage(env, pymunk_to_shapely))
            step += 1
            if done:                                     # coverage > success_threshold
                success = True
                break
            if step >= max_steps:
                break
        if success or step >= max_steps:
            break

    return max_coverage, max_reward, success, frames


# ----------------------------------------------------------------------------
def main():
    default_ll_repo = ("/ocean/projects/cis240052p/pbhowal/"
                       "2d_Representation_Hierarchical_Policy_Learning/"
                       "MimicGen_Uncertainty_Code/Low_Level_Policy/"
                       "2d_Representation_Hierarchical_Policy_Learning/"
                       "Low_Level_and_Inference")
    parser = argparse.ArgumentParser(
        description="Approach 2 eval on PushT: 2D DiT LL alone (no high-level).")
    parser.add_argument("--low_level_exp_dir", type=str, required=True,
        help="LL run dir containing .hydra/config.yaml and checkpoints/.")
    parser.add_argument("--low_level_checkpoint", type=str, default="epoch_99.ckpt")
    parser.add_argument("--low_level_repo", type=str, default=default_ll_repo,
        help="Low_Level_and_Inference repo root (read-only; supplies the "
             "diffusion_policy namespace package incl. env/pusht).")
    parser.add_argument("--n_episodes",     type=int, default=50)
    parser.add_argument("--max_steps",      type=int, default=300)
    parser.add_argument("--seed",           type=int, default=100000)
    parser.add_argument("--n_obs_steps",    type=int, default=2)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--action_start", type=int, default=0,
        help="Index into the predicted horizon to start executing from. "
             "0 = policy's own convention (first action aligns with obs t-1); "
             "n_obs_steps-1 (=1) = execute from the action aligned with the "
             "current frame, as stock diffusion_policy does.")
    parser.add_argument("--obs_history", type=str, default="rolling",
        choices=["repeat_current", "rolling"],
        help="'rolling': real t-1/t history as in training (pad_before=1). "
             "'repeat_current': both obs steps are frame t (no t-1).")
    parser.add_argument("--render_size",    type=int, default=256)
    parser.add_argument("--legacy_env", action=argparse.BooleanOptionalAction, default=True,
        help="PushTEnv legacy physics; the 206 source demos were collected with "
             "legacy=True and the training config's runner had legacy_test: true.")
    parser.add_argument("--output_dir",     type=str, default=None)
    parser.add_argument("--save_videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video_fps",      type=int, default=10)
    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ll_tag = Path(args.low_level_exp_dir).name
        ckpt_tag = Path(args.low_level_checkpoint).stem
        args.output_dir = f"outputs_eval_approach2_pusht/{ll_tag}_{ckpt_tag}/{ts}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] saving results to {output_dir.resolve()}")
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    _bootstrap_paths(args)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[WARN] CUDA unavailable -- running the DiT on CPU will be very slow.")

    # Headless pygame: no display needed for offscreen Surface rendering, but
    # the dummy driver makes that explicit on compute nodes.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv

    env = PushTImageEnv(legacy=args.legacy_env, render_size=args.render_size)

    print(f"[LL] loading Approach 2 policy from "
          f"{args.low_level_exp_dir}/{args.low_level_checkpoint}")
    policy, ll_cfg = load_approach2_policy(
        args.low_level_exp_dir, args.low_level_checkpoint, device=device)
    sanity_check_policy(policy, ll_cfg)
    print(f"[obs] history mode: {args.obs_history} "
          f"({'frame t repeated, no t-1' if args.obs_history == 'repeat_current' else 'real t-1/t deque'})")

    videos_dir = None
    if args.save_videos:
        videos_dir = output_dir / "media"
        videos_dir.mkdir(parents=True, exist_ok=True)
        # Parity with the MimicGen evals' merge bookkeeping; Approach 2 has no
        # goal, so there is nothing to overlay.
        (output_dir / "media_with_goal_overlay").mkdir(parents=True, exist_ok=True)
        print(f"[video] rollouts -> {videos_dir.resolve()} "
              f"(fps={args.video_fps}, h264 crf=22)")

    coverages, rewards, successes = [], [], []
    checked_state_range = False
    with open(output_dir / "results.jsonl", "w") as results_f:
        for ep in range(args.n_episodes):
            seed = args.seed + ep
            np.random.seed(seed)
            torch.manual_seed(seed)
            env.seed(seed)

            cov, r, succ, frames = run_episode(
                env, policy, pymunk_to_shapely,
                args.n_obs_steps, args.n_action_steps, args.max_steps,
                device=device, obs_history=args.obs_history,
                action_start=args.action_start)

            if not checked_state_range:
                # one-time live check against the training range (metres)
                agent_m = np.array(env.agent.position) / SIM_UNITS_PER_M
                if np.any(agent_m < -0.1) or np.any(agent_m > 1.1):
                    print(f"[WARN] live state {agent_m} outside [0,1] m -- "
                          "unit convention mismatch?")
                checked_state_range = True

            video_path = None
            if args.save_videos:
                outcome_tag = "success" if succ else "failure"
                video_path = videos_dir / f"episode_{ep + 1:03d}_seed_{seed}_{outcome_tag}.mp4"
                _write_mp4(video_path, frames, args.video_fps)

            coverages.append(cov)
            rewards.append(r)
            successes.append(succ)
            vtag = f"  video={video_path.name}" if video_path is not None else ""
            print(f"Episode {ep + 1}/{args.n_episodes}  seed={seed}  "
                  f"max_coverage={cov:.4f}  reward={r:.3f}  success={succ}{vtag}")
            results_f.write(json.dumps({
                "episode": ep + 1, "seed": seed,
                "max_coverage": float(cov),
                "reward": float(r), "success": bool(succ),
                "video": str(video_path) if video_path is not None else None,
                "video_with_goal_overlay": None,
            }) + "\n")
            results_f.flush()

    summary = {
        "n_episodes":    args.n_episodes,
        "mean_coverage": float(np.mean(coverages)),
        "std_coverage":  float(np.std(coverages)),
        "mean_reward":   float(np.mean(rewards)),
        "std_reward":    float(np.std(rewards)),
        "success_rate":  float(np.mean(successes)),
        "successes":     int(sum(successes)),
        "args":          vars(args),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    print(f"Mean coverage: {summary['mean_coverage']:.4f} ± {summary['std_coverage']:.4f}")
    print(f"Mean reward:   {summary['mean_reward']:.3f} ± {summary['std_reward']:.3f}")
    print(f"Success rate:  {summary['success_rate']:.2%} "
          f"({summary['successes']}/{args.n_episodes})")
    print(f"\nSaved: {output_dir.resolve()}")
    print("  - args.json\n  - results.jsonl\n  - summary.json")


if __name__ == "__main__":
    sys.exit(main())
