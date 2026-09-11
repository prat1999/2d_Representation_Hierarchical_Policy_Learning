"""Compare aux_regression_frame=absolute against relative_to_gripper on real data.

Builds the real dataset + policy from the same hydra config the sbatch scripts
use, then runs one pass of optimiser steps per frame from an IDENTICAL seed,
init and batch stream, logging the head's own MSE pre-update each step. This is
what produced the scale table in the header of the
*_auxregression_absolute_c1_*.sh scripts.

The point of the comparison: the absolute target carries a large task-constant
offset (~0.3 m^2 vs ~0.02 for the relative target), so it is worth knowing (a)
how fast the output bias absorbs that constant and (b) whether the loss then
keeps falling THROUGH the constant-predictor floor -- if it plateaus at that
floor the head has learned the task mean and nothing else.

Usage:
    # a staged h5 dir with obs/goal_gripper_pts_awe already injected, i.e. the
    # DEST_DATA_DIR the sbatch scripts build:
    pixi run python theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/verify_aux_regression_frames.py \
        --data_dir /scratch/.../Hammer_Cleanup_D1_Approach2_... \
        --task MimicGen_Tasks/hammercleanup_D1_goal_gmm_aux \
        --steps 60 --batch_size 16
"""
import argparse
import copy
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "diffusion_policy"))

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

OmegaConf.register_new_resolver("eval", eval, replace=True)

CFG_DIR = os.path.join(REPO, "diffusion_policy", "diffusion_policy", "config")
FRAMES = ("absolute", "relative_to_gripper")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="staged h5 dir with obs/goal_gripper_pts_<source> injected")
    ap.add_argument("--task", default="MimicGen_Tasks/hammercleanup_D1_goal_gmm_aux")
    ap.add_argument("--goal_source", default="awe")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        cfg = compose(
            config_name="train_flow_matching_dit_goal_gmm_workspace",
            overrides=[
                f"task={args.task}",
                f"task.dataset.data_dir={args.data_dir}",
                f"+task.dataset.goal_source={args.goal_source}",
                "policy.aux_head_type=regression",
                "policy.aux_gmm_loss_weight=1.0",
                f"dataloader.batch_size={args.batch_size}",
                f"dataloader.num_workers={args.num_workers}",
                "val_dataloader.num_workers=0",
            ],
        )

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()

    # one fixed batch stream, replayed identically for both frames
    torch.manual_seed(0)
    loader = DataLoader(dataset, **cfg.dataloader)
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= args.steps:
            break
    print(f"[verify] {len(dataset)} sequences, {len(batches)} batches of "
          f"{args.batch_size}", flush=True)

    # the three numbers C1 has to be sized against, on these very batches
    g = torch.cat([b["obs"]["goal_gripper_pts"].reshape(-1, 4, 3) for b in batches])
    p = torch.cat([b["obs"]["present_gripper_pts"].reshape(-1, 4, 3) for b in batches])
    floor = ((g - g.reshape(-1, 3).mean(0)) ** 2).sum(-1).mean()
    print(f"[verify] E|g|^2   (absolute target)      = {(g ** 2).sum(-1).mean():.4f}")
    print(f"[verify] E|g-p|^2 (relative target)      = {((g - p) ** 2).sum(-1).mean():.4f}")
    print(f"[verify] E|g-mean g|^2 (const-pred floor) = {floor:.4f}", flush=True)

    results = {}
    for frame in FRAMES:
        c = copy.deepcopy(cfg)
        c.policy.aux_regression_frame = frame
        torch.manual_seed(42)
        np.random.seed(42)
        policy = hydra.utils.instantiate(c.policy)
        policy.set_normalizer(normalizer)
        policy.to(args.device).train()
        opt = hydra.utils.instantiate(cfg.optimizer, params=policy.parameters())

        hist = []
        for step, batch in enumerate(batches):
            b = {"obs": {k: v.to(args.device) for k, v in batch["obs"].items()},
                 "action": batch["action"].to(args.device)}
            # aux BEFORE this step's update, so step 0 is the untrained head
            with torch.no_grad():
                aux_pre = policy._compute_goal_regression_loss(
                    *policy.visual_encoder.encode_with_positions(
                        policy.normalizer.normalize(b["obs"]), b["obs"]),
                    b["obs"][policy.aux_goal_key]).item()
            torch.manual_seed(1000 + step)   # same flow-matching noise and time
            loss = policy.compute_loss(b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            hist.append((loss.item(), aux_pre))
            if step < 4 or (step + 1) % 25 == 0:
                print(f"  [{frame}] step {step:3d}  total {loss.item():8.4f}  "
                      f"aux(pre-step) {aux_pre:8.4f}", flush=True)
        results[frame] = hist
        del policy, opt
        torch.cuda.empty_cache()

    print("\nstep |     absolute total /  aux |   relative total /  aux")
    for i in range(0, len(batches), max(1, len(batches) // 12)):
        a, r = results["absolute"][i], results["relative_to_gripper"][i]
        print(f"{i:4d} | {a[0]:9.4f} / {a[1]:8.4f} | {r[0]:9.4f} / {r[1]:8.4f}")
    for frame, h in results.items():
        aux_min = min(x[1] for x in h)
        verdict = ("learning past the task mean"
                   if aux_min < float(floor) else "PLATEAUED at the const-pred floor")
        print(f"{frame}: aux step0={h[0][1]:.4f} last={h[-1][1]:.4f} min={aux_min:.4f} "
              f"({verdict}); total {h[0][0]:.4f} -> {h[-1][0]:.4f}")


if __name__ == "__main__":
    main()
