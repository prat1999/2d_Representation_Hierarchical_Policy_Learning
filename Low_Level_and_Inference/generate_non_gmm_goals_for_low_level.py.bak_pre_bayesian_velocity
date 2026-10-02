"""
Run the high-level GMM model on the All_Heatmap_Dataset and save predictions.

For each timestep in each HDF5 file:
  1. Build a point cloud by unprojecting depth images, keeping only articulated
     object pixels using the segmentation mask (objectId == 2).
  2. Use obs/present_gripper_pts as gripper_pcd.
  3. Run the multitask high-level model to get the GMM goal prediction.
  4. Write obs/gmm_pred_goal (T, 4, 3) back into the HDF5 file.

Segmask encoding (PyBullet): objectId = value & 0xFFFFFF
  objectId == 2 → articulated object
  objectId == 1 → robot arm
  0 / -1        → background

Example usage:
  python real_world/run_gmm_on_dataset.py \
      --dataset_dir PATH/TO/All_Heatmap_Dataset \
      --high_level_ckpt_path PATH/TO/model.pth \
      --task articulated \
      --cat_idx 4
"""

import os
import sys
import argparse
import numpy as np
import h5py
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_multitask_high_level_model(path):
    import json
    from omegaconf import OmegaConf
    from test_PointNet2.model_invariant import PointNet2_super_multitask

    ckpt_path = os.path.dirname(path)
    cfg = OmegaConf.create(json.load(open(os.path.join(ckpt_path, "config.json"))))
    args = cfg

    device = torch.device("cuda")
    general_args = args.general
    input_channel = 5 if general_args.add_one_hot_encoding else 3
    if general_args.get("use_rgb", False):
        input_channel += 3
    if general_args.get("use_dino", False):
        input_channel += 1024

    output_dim = 13

    if "category_embedding_type" not in general_args:
        general_args.category_embedding_type = None
    if general_args.category_embedding_type == "one_hot":
        embedding_dim = args.num_categories
    elif general_args.category_embedding_type == "siglip":
        embedding_dim = 768
    else:
        embedding_dim = None

    model = PointNet2_super_multitask(
        num_classes=output_dim,
        keep_gripper_in_fps=general_args.keep_gripper_in_fps,
        input_channel=input_channel,
        first_sa_point=general_args.get("first_sa_point", 2048),
        fp_to_full=general_args.get("fp_to_full", False),
        replace_bn_w_gn=general_args.get("replace_bn_with_gn", False),
        replace_bn_w_in=general_args.get("replace_bn_with_in", False),
        embedding_dim=embedding_dim,
        film_in_sa_and_fp=general_args.get("film_in_sa_and_fp", False),
        embedding_as_input=general_args.get("embedding_as_input", False),
        replace_bn_w_ln=general_args.get("replace_bn_with_ln", False),
    ).to(device)

    model.load_state_dict(torch.load(path, map_location=device)['model'])
    print("Successfully loaded model from:", path)
    model.eval()
    return model, args


def prepare_input(pointcloud, gripper_pcd, args):
    if not args.one_hot_encoding:
        inputs = torch.cat([pointcloud, gripper_pcd], dim=1)
    else:
        pointcloud_one_hot = torch.zeros(pointcloud.shape[0], pointcloud.shape[1], 2).to(pointcloud.device)
        pointcloud_one_hot[:, :, 0] = 1
        pointcloud_ = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
        gripper_pcd_one_hot = torch.zeros(gripper_pcd.shape[0], gripper_pcd.shape[1], 2).to(pointcloud.device)
        gripper_pcd_one_hot[:, :, 1] = 1
        gripper_pcd_ = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
        inputs = torch.cat([pointcloud_, gripper_pcd_], dim=1)
    return inputs


def infer_multitask_high_level_model(inputs, goal_prediction_model, cat_embedding=None, args=None):
    inputs = inputs.to('cuda')
    inputs_ = inputs.permute(0, 2, 1).float().contiguous()
    with torch.no_grad():
        pred_dict = goal_prediction_model(inputs_, cat_embedding, build_grasp=False, articubot_format=True)
    outputs = pred_dict['pred_offsets']
    pred_points = pred_dict['pred_points']
    weights = pred_dict['pred_scores'].squeeze(-1)
    inputs = pred_points
    B, N, _, _ = outputs.shape
    outputs = outputs.view(B, N, 4, 3)

    # Exclude the last 4 gripper points from anchor selection — the model always
    # appends gripper_pcd last, and the original eval code strips them before
    # computing the weighted prediction (output_obj_pcd_only=True at eval time).
    weights = weights[:, :-4]
    outputs = outputs[:, :-4, :, :]
    inputs  = inputs[:, :-4, :]

    probabilities = torch.nn.functional.softmax(weights, dim=1)
    if not args.argmax_weight:
        sampled_index = torch.multinomial(probabilities, num_samples=1).item()
    else:
        sampled_index = torch.argmax(probabilities.squeeze(0))

    displacement_mean = outputs[:, sampled_index, :, :]
    input_point_pos = inputs[:, sampled_index, :]
    prediction = input_point_pos.unsqueeze(1) + displacement_mean  # B, 4, 3

    # Return filtered inputs as anchor_points so sizes match weights/outputs
    return prediction, probabilities, inputs, outputs


ARTICULATED_OBJECT_ID = 2  # objectId in PyBullet segmask for the articulated object


def depth_to_pointcloud(depth_mm, segmask, K, extrinsic):
    """Unproject depth to 3D points in world frame, keeping only articulated object pixels.

    Args:
        depth_mm:  H x W uint16, depth in millimeters
        segmask:   H x W int32, PyBullet segmentation mask
        K:         3x3 float32 intrinsic matrix
        extrinsic: 4x4 float32 camera-to-world transform

    Returns:
        N x 3 float32 point cloud in world frame
    """
    H, W = depth_mm.shape
    depth_m = depth_mm.astype(np.float32) / 1000.0  # mm → meters

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    Z = depth_m

    # Keep only articulated object pixels with valid depth
    object_mask = (segmask & 0xFFFFFF) == ARTICULATED_OBJECT_ID
    valid = (depth_mm > 0) & object_mask

    pts = np.stack([X[valid], Y[valid], Z[valid]], axis=1).astype(np.float32)  # N x 3

    if pts.shape[0] == 0:
        return pts

    # extrinsic in dataset is world-to-camera; invert to get camera-to-world
    cam_to_world = np.linalg.inv(extrinsic)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])                      # N x 4
    pts_world = (cam_to_world @ pts_h.T).T[:, :3]       # N x 3

    return pts_world


def downsample(pcd, num_points):
    """Random downsample to exactly num_points; pad with repeats if too few."""
    if pcd.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    replace = pcd.shape[0] < num_points
    idx = np.random.choice(pcd.shape[0], num_points, replace=replace)
    return pcd[idx]


def visualize_with_viser_interactive(all_timestep_data, show_all_gmm_goals=False):
    """Launch an interactive viser viewer with a timestep slider.

    Args:
        all_timestep_data: list of dicts (one per timestep), each with keys:
            merged_pcd        (N, 3) numpy
            anchor_points     (1, N, 3) tensor
            weights           (1, N) tensor
            prediction        (1, 4, 3) tensor
            gmm_all_components (1, N, 4, 3) tensor
        show_all_gmm_goals: bool, whether to show all N GMM component goals
    """
    import viser
    import time

    server = viser.ViserServer()
    print(f"[viser] Open http://localhost:8080. Use the Timestep slider to scrub predictions. Ctrl+C to exit.")

    T = len(all_timestep_data)
    timestep_slider = server.gui.add_slider("Timestep", min=0, max=T - 1, step=1, initial_value=0)

    def render_timestep(t):
        data = all_timestep_data[t]
        merged_pcd = data['merged_pcd']
        has_gmm = 'weights' in data

        # Object point cloud (grey)
        server.scene.add_point_cloud(
            name="scene_pcd",
            points=merged_pcd,
            colors=np.tile([180, 180, 180], (merged_pcd.shape[0], 1)).astype(np.uint8),
            point_size=0.004,
        )

        if has_gmm:
            weights_np = data['weights'].cpu().numpy().flatten()
            w_norm = (weights_np - weights_np.min()) / (weights_np.max() - weights_np.min() + 1e-8)

            # GMM anchor points: blue=low weight, red=high weight
            anchor_colors = np.zeros((len(w_norm), 3), dtype=np.uint8)
            anchor_colors[:, 0] = (w_norm * 255).astype(np.uint8)
            anchor_colors[:, 2] = ((1 - w_norm) * 255).astype(np.uint8)
            server.scene.add_point_cloud(
                name="gmm_anchors",
                points=data['anchor_points'].cpu().numpy().reshape(-1, 3),
                colors=anchor_colors,
                point_size=0.008,
            )

            # All N GMM goal predictions in dark green
            if show_all_gmm_goals:
                all_goals = data['gmm_all_components'].squeeze(0).cpu().numpy()  # N x 4 x 3
                N = all_goals.shape[0]
                all_goal_pts = all_goals.reshape(N * 4, 3)
                server.scene.add_point_cloud(
                    name="all_gmm_goals",
                    points=all_goal_pts,
                    colors=np.tile([0, 120, 0], (N * 4, 1)).astype(np.uint8),
                    point_size=0.010,
                )

            # Sampled predicted goal in light green
            server.scene.add_point_cloud(
                name="predicted_goal",
                points=data['prediction'].cpu().numpy().reshape(-1, 3),
                colors=np.tile([144, 238, 144], (4, 1)).astype(np.uint8),
                point_size=0.018,
            )

        # Ground truth goal gripper in black
        server.scene.add_point_cloud(
            name="ground_truth_goal_gripper",
            points=data['gt_goal'].reshape(-1, 3),
            colors=np.tile([0, 0, 0], (4, 1)).astype(np.uint8),
            point_size=0.022,
        )

    @timestep_slider.on_update
    def _(_):
        render_timestep(timestep_slider.value)

    render_timestep(0)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[viser] Exiting.")
        server.stop()


def process_demo_dir(demo_dir, goal_policy, cat_embedding, args):
    """Process a demo directory of per-timestep .npz files (pre-computed point clouds).

    Each .npz file contains:
        point_cloud:        (1, N, 3)       float32 – pre-computed object PCD
        gripper_pcd:        (1, 4, 3)       float32 – present gripper points
        goal_gripper_pcd:   (1, 4, 3)       float32 – goal gripper points
        rgb_agentview:      (1, H, W, 3)    uint8
        depth_agentview:    (1, H, W, 1)    float32 meters
        agentview_intrinsics:(1, 3, 3)      float32
        agentview_extrinsics:(1, 4, 4)      float32
        rgb_wrist:          (1, H, W, 3)    uint8
        depth_wrist:        (1, H, W, 1)    float32 meters
        wrist_intrinsics:   (1, 3, 3)       float32
        wrist_extrinsics:   (1, 4, 4)       float32
        state:              (1, 10)          float64
        action:             (1, 10)          float64

    Output H5 layout mirrors All_Heatmap_Dataset exactly:
        action/delta           (T, 10)        float64
        action/hybrid          (T, 10)        float32
        obs/cam0_depth         (T, 256, 256)  uint16  (agentview, mm)
        obs/cam0_extrinsic     (T, 4, 4)      float32
        obs/cam0_image         (T, 256, 256, 3) uint8
        obs/cam0_intrinsic     (T, 3, 3)      float32
        obs/cam1_depth         (T, 256, 256)  uint16  (wrist, mm)
        obs/cam1_extrinsic     (T, 4, 4)      float32
        obs/cam1_image         (T, 256, 256, 3) uint8
        obs/cam1_intrinsic     (T, 3, 3)      float32
        obs/gmm_pred_goal      (T, 4, 3)      float32  ← GMM output
        obs/gmm_all_goals      (T, 2044, 4, 3) float32 ← GMM output
        obs/gmm_all_weights    (T, 2044)       float32  ← GMM output
        obs/goal_gripper_pts   (T, 4, 3)      float32
        obs/point_cloud        (T, N, 3)      float32  (pre-computed PCD)
        obs/present_gripper_pts (T, 4, 3)     float32
        obs/state              (T, 10)        float32
        _physical/cam0_extrinsic (4, 4)       float32  (from t=0)
        _physical/cam0_intrinsic (3, 3)       float32
        _physical/cam1_extrinsic (4, 4)       float32
        _physical/cam1_intrinsic (3, 3)       float32
    """
    if args.no_gmm:
        demo_name_check = os.path.basename(demo_dir)
        out_h5_check = os.path.join(args.no_gmm_output_dir, demo_name_check + '.h5')
        if os.path.exists(out_h5_check):
            print(f"  {demo_name_check}: skipping (already exists at {out_h5_check})")
            return None, None, None

    npz_files = sorted(
        [f for f in os.listdir(demo_dir) if f.endswith('.npz')],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    T_total = len(npz_files)
    T = T_total if args.max_timesteps is None else min(T_total, args.max_timesteps)
    npz_files = npz_files[:T]

    if not args.no_gmm:
        N_ANCHORS = 2044
        gmm_pred_goals  = np.zeros((T, 4, 3),            dtype=np.float32)
        gmm_all_goals   = np.zeros((T, N_ANCHORS, 4, 3), dtype=np.float32)
        gmm_all_weights = np.zeros((T, N_ANCHORS),        dtype=np.float32)

    # Per-timestep accumulators — keys match the obs/ group in All_Heatmap_Dataset
    obs_bufs = {
        'cam0_depth':          [],   # (H, W)      uint16 mm
        'cam0_extrinsic':      [],   # (4, 4)      float32
        'cam0_image':          [],   # (H, W, 3)   uint8
        'cam0_intrinsic':      [],   # (3, 3)      float32
        'cam1_depth':          [],   # (H, W)      uint16 mm
        'cam1_extrinsic':      [],   # (4, 4)      float32
        'cam1_image':          [],   # (H, W, 3)   uint8
        'cam1_intrinsic':      [],   # (3, 3)      float32
        'goal_gripper_pts':    [],   # (4, 3)      float32
        'point_cloud':         [],   # (N, 3)      float32
        'present_gripper_pts': [],   # (4, 3)      float32
        'state':               [],   # (10,)       float32
    }
    act_delta  = []   # action/delta  float64
    all_timestep_data = []

    def _depth_to_mm(d):
        d = np.squeeze(d).astype(np.float32)
        return np.clip(d * 1000.0, 0, 65535).astype(np.uint16)

    for t, fname in enumerate(npz_files):
        data = np.load(os.path.join(demo_dir, fname))
        merged_pcd  = data['point_cloud'][0]      # (N, 3)
        gripper_pcd = data['gripper_pcd'][0]      # (4, 3)
        gt_goal     = data['goal_gripper_pcd'][0] # (4, 3)

        if merged_pcd.shape[0] != args.num_points:
            merged_pcd = downsample(merged_pcd, args.num_points)

        if not args.no_gmm:
            pointcloud_t  = torch.from_numpy(merged_pcd).float().unsqueeze(0).to('cuda')
            gripper_pcd_t = torch.from_numpy(gripper_pcd).float().unsqueeze(0).to('cuda')
            inputs = prepare_input(pointcloud_t, gripper_pcd_t, args)

            with torch.no_grad():
                prediction, weights, anchor_points, gmm_components = infer_multitask_high_level_model(
                    inputs, goal_policy, cat_embedding=cat_embedding, args=args
                )

            gmm_pred_goals[t]  = prediction.squeeze(0).cpu().numpy()
            gmm_all_goals[t]   = gmm_components.squeeze(0).cpu().numpy()
            gmm_all_weights[t] = weights.squeeze(0).cpu().numpy()

        obs_bufs['cam0_depth'].append(_depth_to_mm(data['depth_agentview'][0]))
        obs_bufs['cam0_extrinsic'].append(data['agentview_extrinsics'][0].astype(np.float32))
        obs_bufs['cam0_image'].append(data['rgb_agentview'][0])
        obs_bufs['cam0_intrinsic'].append(data['agentview_intrinsics'][0].astype(np.float32))
        obs_bufs['cam1_depth'].append(_depth_to_mm(data['depth_wrist'][0]))
        obs_bufs['cam1_extrinsic'].append(data['wrist_extrinsics'][0].astype(np.float32))
        obs_bufs['cam1_image'].append(data['rgb_wrist'][0])
        obs_bufs['cam1_intrinsic'].append(data['wrist_intrinsics'][0].astype(np.float32))
        obs_bufs['goal_gripper_pts'].append(gt_goal.astype(np.float32))
        obs_bufs['point_cloud'].append(merged_pcd.astype(np.float32))
        obs_bufs['present_gripper_pts'].append(gripper_pcd.astype(np.float32))
        obs_bufs['state'].append(data['state'][0].astype(np.float32))
        act_delta.append(data['action'][0].astype(np.float64))

        if args.visualize or args.visualize_all_gmm_goals:
            entry = {
                'merged_pcd': merged_pcd,
                'gt_goal':    gt_goal,
            }
            if not args.no_gmm:
                entry['anchor_points']      = anchor_points
                entry['weights']            = weights
                entry['prediction']         = prediction
                entry['gmm_all_components'] = gmm_components
            all_timestep_data.append(entry)

    if (args.visualize or args.visualize_all_gmm_goals) and len(all_timestep_data) > 0:
        visualize_with_viser_interactive(all_timestep_data, show_all_gmm_goals=args.visualize_all_gmm_goals)

    demo_name = os.path.basename(demo_dir)
    if args.no_gmm:
        out_h5 = os.path.join(args.no_gmm_output_dir, demo_name + '.h5')
    else:
        out_h5 = os.path.join(os.path.dirname(demo_dir), demo_name + '.h5')
    act_delta_arr = np.stack(act_delta, axis=0)   # (T, 10) float64

    with h5py.File(out_h5, 'w') as f:
        # action/ group — matches original top-level structure
        f.create_dataset('action/delta',  data=act_delta_arr)
        f.create_dataset('action/hybrid', data=act_delta_arr.astype(np.float32))

        # obs/ group — keys in alphabetical order matching original
        for key, buf in sorted(obs_bufs.items()):
            f.create_dataset(f'obs/{key}', data=np.stack(buf, axis=0))
        if not args.no_gmm:
            f.create_dataset('obs/gmm_pred_goal',   data=gmm_pred_goals)
            f.create_dataset('obs/gmm_all_goals',   data=gmm_all_goals)
            f.create_dataset('obs/gmm_all_weights', data=gmm_all_weights)

        # _physical/ group — static calibration from t=0, matches original
        f.create_dataset('_physical/cam0_extrinsic', data=obs_bufs['cam0_extrinsic'][0])
        f.create_dataset('_physical/cam0_intrinsic', data=obs_bufs['cam0_intrinsic'][0])
        f.create_dataset('_physical/cam1_extrinsic', data=obs_bufs['cam1_extrinsic'][0])
        f.create_dataset('_physical/cam1_intrinsic', data=obs_bufs['cam1_intrinsic'][0])

    if args.no_gmm:
        return None, None, None
    return gmm_pred_goals, gmm_all_goals, gmm_all_weights


# RL Bench camera -> h5 cam index mapping. 4 cameras (vs MimicGen's agentview +
# wrist). Order defines cam0..cam3 in the output h5 and must match the task
# shape_meta (config/task/RL_Bench_Tasks/sweep_to_dustpan_goal_gripper.yaml).
RL_BENCH_CAM_MAP = [
    ("cam0", "front"),
    ("cam1", "left_shoulder"),
    ("cam2", "right_shoulder"),
    ("cam3", "wrist"),
]


def process_demo_dir_rl_bench(demo_dir, args):
    """RL Bench analog of process_demo_dir's --no_gmm path.

    Same consolidation and output h5 schema as process_demo_dir, but reads RL
    Bench per-frame npz instead of MimicGen npz:
      - cameras: front/left_shoulder/right_shoulder/wrist -> cam0/cam1/cam2/cam3
        (RGB stored channel-first float32 [0,255] -> HWC uint8;
         depth float32 meters -> uint16 mm, matching MimicGen)
      - state/action: native 8-D [pos(3), quat(4), gripper(1)] / absolute 8-D,
        written verbatim (no MimicGen 10-D / 6D-rotation conversion)
      - goal_gripper_pcd -> goal_gripper_pts; gripper_pcd -> present_gripper_pts;
        point_cloud passed through.

    Output h5 (per demo):
      action/delta            (T, 8)            float64  (= native action)
      action/hybrid           (T, 8)            float32  (= native action)
      obs/cam{0..3}_image     (T, 128, 128, 3)  uint8
      obs/cam{0..3}_depth     (T, 128, 128)     uint16  (mm)
      obs/cam{0..3}_intrinsic (T, 3, 3)         float32
      obs/cam{0..3}_extrinsic (T, 4, 4)         float32
      obs/goal_gripper_pts    (T, 4, 3)         float32
      obs/point_cloud         (T, N, 3)         float32
      obs/present_gripper_pts (T, 4, 3)         float32
      obs/state               (T, 8)            float32
      _physical/cam{0..3}_extrinsic (4, 4)      float32  (from t=0)
      _physical/cam{0..3}_intrinsic (3, 3)      float32
    """
    demo_name = os.path.basename(demo_dir)
    out_h5 = os.path.join(args.no_gmm_output_dir, demo_name + '.h5')
    if os.path.exists(out_h5):
        print(f"  {demo_name}: skipping (already exists at {out_h5})")
        return

    npz_files = sorted(
        [f for f in os.listdir(demo_dir) if f.endswith('.npz')],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    T_total = len(npz_files)
    T = T_total if args.max_timesteps is None else min(T_total, args.max_timesteps)
    npz_files = npz_files[:T]

    obs_bufs = {}
    for cam_idx, _ in RL_BENCH_CAM_MAP:
        obs_bufs[f'{cam_idx}_image']     = []
        obs_bufs[f'{cam_idx}_depth']     = []
        obs_bufs[f'{cam_idx}_intrinsic'] = []
        obs_bufs[f'{cam_idx}_extrinsic'] = []
    obs_bufs['goal_gripper_pts']    = []
    obs_bufs['point_cloud']         = []
    obs_bufs['present_gripper_pts'] = []
    obs_bufs['state']               = []
    act_buf = []

    def _depth_to_mm(d):
        d = np.squeeze(d).astype(np.float32)
        return np.clip(d * 1000.0, 0, 65535).astype(np.uint16)

    for fname in npz_files:
        # allow_pickle so np.load opens the file; we only read numeric keys
        # (never the object-array lang_goal), so no unpickling happens.
        data = np.load(os.path.join(demo_dir, fname), allow_pickle=True)

        for cam_idx, cam_name in RL_BENCH_CAM_MAP:
            rgb = data[f'{cam_name}_rgb'][0]                       # (3, H, W) float32 [0,255]
            rgb = np.transpose(rgb, (1, 2, 0)).astype(np.uint8)   # (H, W, 3) uint8
            depth = _depth_to_mm(data[f'{cam_name}_depth'][0])    # (H, W) uint16 mm
            K = data[f'{cam_name}_camera_intrinsics'][0].astype(np.float32)  # (3, 3)
            E = data[f'{cam_name}_camera_extrinsics'][0].astype(np.float32)  # (4, 4)
            obs_bufs[f'{cam_idx}_image'].append(rgb)
            obs_bufs[f'{cam_idx}_depth'].append(depth)
            obs_bufs[f'{cam_idx}_intrinsic'].append(K)
            obs_bufs[f'{cam_idx}_extrinsic'].append(E)

        obs_bufs['goal_gripper_pts'].append(data['goal_gripper_pcd'][0].astype(np.float32))
        obs_bufs['point_cloud'].append(data['point_cloud'][0].astype(np.float32))
        obs_bufs['present_gripper_pts'].append(data['gripper_pcd'][0].astype(np.float32))
        obs_bufs['state'].append(data['state'][0].astype(np.float32))   # (8,)
        act_buf.append(data['action'][0].astype(np.float32))            # (8,)

    act_arr = np.stack(act_buf, axis=0)   # (T, 8) float32

    with h5py.File(out_h5, 'w') as f:
        # Native 8-D action written verbatim. delta/hybrid are the same array
        # (no MimicGen hybrid-delta form); the trainer reads action/hybrid.
        f.create_dataset('action/delta',  data=act_arr.astype(np.float64))
        f.create_dataset('action/hybrid', data=act_arr.astype(np.float32))

        for key, buf in sorted(obs_bufs.items()):
            f.create_dataset(f'obs/{key}', data=np.stack(buf, axis=0))

        for cam_idx, _ in RL_BENCH_CAM_MAP:
            f.create_dataset(f'_physical/{cam_idx}_extrinsic', data=obs_bufs[f'{cam_idx}_extrinsic'][0])
            f.create_dataset(f'_physical/{cam_idx}_intrinsic', data=obs_bufs[f'{cam_idx}_intrinsic'][0])


def process_demo_dir_push_t(demo_dir, args):
    """PushT analog of process_demo_dir_rl_bench (--no_gmm consolidation only).

    Reads the PushT per-frame npz schema:
      - single camera: rgb_agentview (1, 256, 256, 3) uint8, channel-last -> cam0.
        depth_agentview is all zeros and there are no camera
        intrinsics/extrinsics, so no depth / K / E datasets are written.
      - state/action: native 2-D pusher xy, written verbatim. state = current
        pusher position; action = ABSOLUTE target xy for the env's P-controller
        (verified: state[t+1] tracks toward action[t]). Same recipe as RL Bench:
        action/delta and action/hybrid hold the same native array and the
        trainer reads action/hybrid as-is under action_mode 'hybrid_delta'.
      - goal_gripper_pcd -> goal_gripper_pts; gripper_pcd -> present_gripper_pts;
        point_cloud passed through.

    Output h5 (per demo):
      action/delta            (T, 2)            float64  (= native absolute action)
      action/hybrid           (T, 2)            float32  (= native absolute action)
      obs/cam0_image          (T, 256, 256, 3)  uint8
      obs/goal_gripper_pts    (T, 4, 3)         float32
      obs/point_cloud         (T, N, 3)         float32
      obs/present_gripper_pts (T, 4, 3)         float32
      obs/state               (T, 2)            float32
    """
    demo_name = os.path.basename(demo_dir)
    out_h5 = os.path.join(args.no_gmm_output_dir, demo_name + '.h5')
    if os.path.exists(out_h5):
        print(f"  {demo_name}: skipping (already exists at {out_h5})")
        return

    npz_files = sorted(
        [f for f in os.listdir(demo_dir) if f.endswith('.npz')],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    T_total = len(npz_files)
    T = T_total if args.max_timesteps is None else min(T_total, args.max_timesteps)
    npz_files = npz_files[:T]

    obs_bufs = {
        'cam0_image':          [],
        'goal_gripper_pts':    [],
        'point_cloud':         [],
        'present_gripper_pts': [],
        'state':               [],
    }
    act_buf = []

    for fname in npz_files:
        data = np.load(os.path.join(demo_dir, fname), allow_pickle=True)
        obs_bufs['cam0_image'].append(data['rgb_agentview'][0].astype(np.uint8))
        obs_bufs['goal_gripper_pts'].append(data['goal_gripper_pcd'][0].astype(np.float32))
        obs_bufs['point_cloud'].append(data['point_cloud'][0].astype(np.float32))
        obs_bufs['present_gripper_pts'].append(data['gripper_pcd'][0].astype(np.float32))
        obs_bufs['state'].append(data['state'][0].astype(np.float32))   # (2,)
        act_buf.append(data['action'][0].astype(np.float32))            # (2,)

    act_arr = np.stack(act_buf, axis=0)   # (T, 2) float32

    with h5py.File(out_h5, 'w') as f:
        # Native 2-D absolute action written verbatim. delta/hybrid are the same
        # array (no MimicGen hybrid-delta form); the trainer reads action/hybrid.
        f.create_dataset('action/delta',  data=act_arr.astype(np.float64))
        f.create_dataset('action/hybrid', data=act_arr.astype(np.float32))

        for key, buf in sorted(obs_bufs.items()):
            f.create_dataset(f'obs/{key}', data=np.stack(buf, axis=0))


# Alternative goal sources produced by the RDP keypoint pipeline. Each per-frame
# npz in the EXTRA_KEYPOINTS tree holds goal_gripper_pcd_<source> (1, 4, 3).
EXTRA_GOAL_SOURCES = ['rdp', 'rdp_gripper', 'random', 'fixed_interval']


def inject_extra_goals_into_h5(h5_dir, extra_goals_dir, max_files=None):
    """Backfill obs/goal_gripper_pts_<source> datasets into existing demo h5 files.

    For each demo_N.h5 in h5_dir, reads extra_goals_dir/demo_N/t.npz (same
    per-frame layout as the main npz tree, verified frame-aligned) and writes
    one (T, 4, 3) float32 dataset per source in EXTRA_GOAL_SOURCES. Idempotent:
    files that already have all sources are skipped, and only missing sources
    are written (append-only — existing datasets are never modified), so this
    is safe to run unconditionally at the start of every training job.
    """
    h5_files = sorted(
        [f for f in os.listdir(h5_dir) if f.endswith('.h5') and f.startswith('demo_')],
        key=lambda x: int(os.path.splitext(x)[0].split('_')[1]),
    )
    if max_files is not None:
        h5_files = h5_files[:max_files]

    n_injected, n_skipped = 0, 0
    for fname in tqdm(h5_files, desc="Injecting extra goal keys"):
        h5_path = os.path.join(h5_dir, fname)
        demo_name = os.path.splitext(fname)[0]

        with h5py.File(h5_path, 'r') as f:
            missing = [s for s in EXTRA_GOAL_SOURCES
                       if f'obs/goal_gripper_pts_{s}' not in f]
            T = f['obs/goal_gripper_pts'].shape[0]
        if not missing:
            n_skipped += 1
            continue

        demo_npz_dir = os.path.join(extra_goals_dir, demo_name)
        if not os.path.isdir(demo_npz_dir):
            raise FileNotFoundError(
                f"{demo_name}: extra-goals npz dir not found: {demo_npz_dir}"
            )
        npz_files = sorted(
            [f2 for f2 in os.listdir(demo_npz_dir) if f2.endswith('.npz')],
            key=lambda x: int(os.path.splitext(x)[0]),
        )
        # h5 may have been generated with --max_timesteps; npz tree must cover it.
        if len(npz_files) < T:
            raise ValueError(
                f"{demo_name}: h5 has T={T} frames but only {len(npz_files)} npz "
                f"files in {demo_npz_dir} — refusing to inject misaligned goals."
            )
        npz_files = npz_files[:T]

        bufs = {s: np.zeros((T, 4, 3), dtype=np.float32) for s in missing}
        for t, nf in enumerate(npz_files):
            data = np.load(os.path.join(demo_npz_dir, nf))
            for s in missing:
                bufs[s][t] = data[f'goal_gripper_pcd_{s}'][0]

        with h5py.File(h5_path, 'a') as f:
            for s in missing:
                f.create_dataset(f'obs/goal_gripper_pts_{s}', data=bufs[s])
        n_injected += 1
        print(f"  {demo_name}: injected {missing}")

    print(f"[inject] done: {n_injected} file(s) updated, {n_skipped} already complete.")


def process_file(h5_path, goal_policy, cat_embedding, args):
    with h5py.File(h5_path, 'r') as f:
        T = f['obs/cam0_depth'].shape[0]
        depths     = {i: f[f'obs/cam{i}_depth'][:]     for i in args.camera_indices}
        segmasks   = {i: f[f'obs/cam{i}_segmask'][:]   for i in args.camera_indices}
        intrinsics = {i: f[f'obs/cam{i}_intrinsic'][:] for i in args.camera_indices}
        extrinsics = {i: f[f'obs/cam{i}_extrinsic'][:] for i in args.camera_indices}
        present_gripper_pts  = f['obs/present_gripper_pts'][:]   # T x 4 x 3
        goal_gripper_pts     = f['obs/goal_gripper_pts'][:]      # T x 4 x 3

    T = T if args.max_timesteps is None else min(T, args.max_timesteps)

    # Pre-allocate output arrays. Anchor count = first_sa_point - 4 gripper pts = 2044.
    # Filled with zeros for timesteps where point cloud is empty (no valid depth).
    N_ANCHORS = 2044
    gmm_pred_goals   = np.zeros((T, 4, 3),            dtype=np.float32)
    gmm_all_goals    = np.zeros((T, N_ANCHORS, 4, 3), dtype=np.float32)  # goal at each anchor
    gmm_all_weights  = np.zeros((T, N_ANCHORS),        dtype=np.float32)  # probability of each anchor

    all_timestep_data = []  # for interactive visualization

    for t in range(T):
        # Merge articulated object point clouds from all cameras
        pcds = []
        for cam_id in args.camera_indices:
            pts = depth_to_pointcloud(
                depths[cam_id][t],
                segmasks[cam_id][t],
                intrinsics[cam_id][t],
                extrinsics[cam_id][t],
            )
            if pts.shape[0] > 0:
                pcds.append(pts)

        if len(pcds) == 0:
            continue

        merged_pcd = np.vstack(pcds)                          # M x 3
        merged_pcd = downsample(merged_pcd, args.num_points)  # num_points x 3

        gripper_pcd = present_gripper_pts[t]                  # 4 x 3

        pointcloud_t  = torch.from_numpy(merged_pcd).float().unsqueeze(0).to('cuda')   # 1 x N x 3
        gripper_pcd_t = torch.from_numpy(gripper_pcd).float().unsqueeze(0).to('cuda')  # 1 x 4 x 3
        inputs = prepare_input(pointcloud_t, gripper_pcd_t, args)  # 1 x (N+4) x C

        with torch.no_grad():
            prediction, weights, anchor_points, gmm_components = infer_multitask_high_level_model(
                inputs, goal_policy, cat_embedding=cat_embedding, args=args
            )

        gmm_pred_goals[t]  = prediction.squeeze(0).cpu().numpy()          # (4, 3)
        gmm_all_goals[t]   = gmm_components.squeeze(0).cpu().numpy()      # (2044, 4, 3)
        gmm_all_weights[t] = weights.squeeze(0).cpu().numpy()             # (2044,)

        if args.visualize or args.visualize_all_gmm_goals:
            all_timestep_data.append({
                'merged_pcd':         merged_pcd,
                'anchor_points':      anchor_points,
                'weights':            weights,
                'prediction':         prediction,
                'gmm_all_components': gmm_components,
                'gt_goal':            goal_gripper_pts[t],   # 4 x 3 numpy
            })

    if (args.visualize or args.visualize_all_gmm_goals) and len(all_timestep_data) > 0:
        visualize_with_viser_interactive(all_timestep_data, show_all_gmm_goals=args.visualize_all_gmm_goals)

    with h5py.File(h5_path, 'a') as f:
        for key, data in [
            ('obs/gmm_pred_goal',  gmm_pred_goals),
            ('obs/gmm_all_goals',  gmm_all_goals),
            ('obs/gmm_all_weights', gmm_all_weights),
        ]:
            if key in f:
                del f[key]
            f.create_dataset(key, data=data)

    return gmm_pred_goals, gmm_all_goals, gmm_all_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir',          type=str, required=True,  help='Path to All_Heatmap_Dataset directory')
    parser.add_argument('--high_level_ckpt_path', type=str, default=None,   help='Path to high-level model checkpoint (.pth)')
    parser.add_argument('--cat_idx',              type=int, default=None,   help='Task category index for SigLIP embedding')
    parser.add_argument('--task',                 type=str, default='articulated')
    parser.add_argument('--camera_indices',       type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--num_points',           type=int, default=4500)
    # args consumed by prepare_input / infer_multitask_high_level_model
    parser.add_argument('--one_hot_encoding',     type=int, default=0)
    parser.add_argument('--output_obj_pcd_only',  type=int, default=0)
    parser.add_argument('--argmax_weight',        type=int, default=1)
    parser.add_argument('--visualize',             action='store_true', help='Visualize anchor points + sampled goal per timestep (viser at localhost:8080)')
    parser.add_argument('--visualize_all_gmm_goals', action='store_true', help='Also visualize all N GMM component goals colored by weight')
    parser.add_argument('--max_files',             type=int, default=None, help='Only process this many files (useful for visualization runs)')
    parser.add_argument('--max_timesteps',         type=int, default=None, help='Only process this many timesteps per file')
    parser.add_argument('--siglip_path',           type=str, default=None, help='Path to siglip_text_features .pt file')
    parser.add_argument('--no_gmm',                action='store_true',    help='Skip GMM inference; consolidate npz files into h5 only (npz format only)')
    parser.add_argument('--no_gmm_output_dir',     type=str, default=None, help='Output directory for h5 files when --no_gmm is used (required with --no_gmm)')
    parser.add_argument('--rl_bench',              action='store_true',    help='RL Bench npz variant: 4 cameras (front/left_shoulder/right_shoulder/wrist) + native 8-D state/action. Requires --no_gmm.')
    parser.add_argument('--push_t',                action='store_true',    help='PushT npz variant: single agentview camera, native 2-D state/action (absolute target xy). Depth is all zeros and there are no camera intrinsics/extrinsics, so none are written. Requires --no_gmm.')
    parser.add_argument('--inject_extra_goals',    action='store_true',    help='Backfill obs/goal_gripper_pts_{rdp,rdp_gripper,random,fixed_interval} into existing h5 files in --dataset_dir from the --extra_goals_dir npz tree, then exit. Idempotent.')
    parser.add_argument('--extra_goals_dir',       type=str, default=None, help='EXTRA_KEYPOINTS npz tree (demo_N/t.npz with goal_gripper_pcd_<source> keys). Required with --inject_extra_goals.')
    args = parser.parse_args()

    # Injection mode: --dataset_dir is the h5 dir; no model or GMM flags involved.
    if args.inject_extra_goals:
        if not args.extra_goals_dir:
            parser.error("--inject_extra_goals requires --extra_goals_dir")
        inject_extra_goals_into_h5(args.dataset_dir, args.extra_goals_dir, args.max_files)
        sys.exit(0)

    # Validate --no_gmm / --no_gmm_output_dir pairing
    if args.no_gmm_output_dir and not args.no_gmm:
        parser.error("--no_gmm_output_dir can only be used together with --no_gmm")
    if args.no_gmm and not args.no_gmm_output_dir:
        parser.error("--no_gmm requires --no_gmm_output_dir")
    if args.push_t and not args.no_gmm:
        parser.error("--push_t requires --no_gmm")
    if args.push_t and args.rl_bench:
        parser.error("--push_t and --rl_bench are mutually exclusive")
    if args.rl_bench and not args.no_gmm:
        parser.error("--rl_bench is an RL Bench variant of the --no_gmm consolidation; pass --no_gmm too")

    if not args.no_gmm:
        if args.high_level_ckpt_path is None:
            parser.error("--high_level_ckpt_path is required when not using --no_gmm")
        if args.cat_idx is None:
            parser.error("--cat_idx is required when not using --no_gmm")
        goal_policy, _ = load_multitask_high_level_model(args.high_level_ckpt_path)
        goal_policy.eval()

        if args.siglip_path is not None:
            siglip_path = args.siglip_path
        else:
            siglip_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'siglip_text_features_w_pick_and_place_w_grasp_and_lift.pt')
        siglip_text_features = torch.load(siglip_path)
        siglip_text_features = siglip_text_features['values'].cuda()
        cat_embedding = siglip_text_features[args.cat_idx].float().to('cuda').unsqueeze(0)
        print(f"cat_idx={args.cat_idx}, task={args.task}")
    else:
        goal_policy = None
        cat_embedding = None

    # Auto-detect dataset format: demo_* subdirectories → npz mode; *.h5 files → h5 mode
    entries = os.listdir(args.dataset_dir)
    demo_dirs = sorted(
        [e for e in entries if e.startswith('demo_') and os.path.isdir(os.path.join(args.dataset_dir, e))],
        key=lambda x: int(x.split('_')[1]),
    )
    h5_files = sorted([
        os.path.join(args.dataset_dir, f)
        for f in entries if f.endswith('.h5')
    ])

    if demo_dirs:
        if args.no_gmm:
            os.makedirs(args.no_gmm_output_dir, exist_ok=True)
        print(f"Detected npz format: {len(demo_dirs)} demo directories (camera_indices ignored)")
        if args.max_files is not None:
            demo_dirs = demo_dirs[:args.max_files]
        for demo_name in tqdm(demo_dirs):
            demo_path = os.path.join(args.dataset_dir, demo_name)
            if args.rl_bench:
                process_demo_dir_rl_bench(demo_path, args)
                print(f"  {demo_name}: consolidated to h5 (rl_bench, no GMM)")
                continue
            if args.push_t:
                process_demo_dir_push_t(demo_path, args)
                print(f"  {demo_name}: consolidated to h5 (push_t, no GMM)")
                continue
            goals, all_goals, all_weights = process_demo_dir(demo_path, goal_policy, cat_embedding, args)
            if not args.no_gmm:
                print(f"  {demo_name}: gmm_pred_goal {goals.shape}, "
                      f"gmm_all_goals {all_goals.shape}, gmm_all_weights {all_weights.shape}")
            else:
                print(f"  {demo_name}: consolidated to h5 (no GMM)")
    else:
        if args.no_gmm:
            parser.error("--no_gmm only supports npz format (demo_* subdirectories), but only .h5 files were found in --dataset_dir")
        if args.max_files is not None:
            h5_files = h5_files[:args.max_files]
        print(f"Detected h5 format: processing {len(h5_files)} files with cameras {args.camera_indices}")
        for h5_path in tqdm(h5_files):
            goals, all_goals, all_weights = process_file(h5_path, goal_policy, cat_embedding, args)
            print(f"  {os.path.basename(h5_path)}: gmm_pred_goal {goals.shape}, "
                  f"gmm_all_goals {all_goals.shape}, gmm_all_weights {all_weights.shape}")
