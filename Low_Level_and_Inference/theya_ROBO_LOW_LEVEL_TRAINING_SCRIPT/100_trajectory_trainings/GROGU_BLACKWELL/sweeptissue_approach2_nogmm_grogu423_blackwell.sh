#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=16    # matches dataloader.num_workers=16 below
#SBATCH --mem=80G             # explicit floor above DefMemPerCPU(3.5G)*cpus, so worker/page-cache spikes don't hit the cgroup limit
#SBATCH -p dheld
#SBATCH -w grogu-4-23         # pin to the RTX 6000 Blackwell node
#SBATCH --gres=gpu:1
#SBATCH -C 6000Blackwell      # grogu-4-23 is an 8x RTX 6000 Blackwell node, not A6000 -- match the node's AVAIL_FEATURES
#SBATCH -t 24:00:00           # use RESUME_CKPT to continue in a follow-up job if more time is needed
#SBATCH --job-name sweeptissue-d1-approach2-nogmm-grogu423-blackwell
#SBATCH -o /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/sweeptissue-d1-approach2-nogmm-grogu423-blackwell_job_%j.out
#SBATCH -e /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/sweeptissue-d1-approach2-nogmm-grogu423-blackwell_job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# ===========================================================================
# APPROACH 2 *CONTROL ARM* on the real-robot Franka SWEEP_TISSUE task (D1)
#   -> NO auxiliary GMM loss, GMM head NOT constructed at all.
# grogu cluster, partition "dheld", pinned to grogu-4-23 (8x RTX 6000 Blackwell).
# ===========================================================================
# Same architecture as the Approach-2 treatment scripts
# (GROGU_BLACKWELL/sweeptissueD1_approach2_awe_greedy_th0.2_grip_grogu423_blackwell.sh):
# goal removed from the DiT input, DINOv2RoPE4DGroundedEncoder trunk with
# 4D-RoPE gripper/patch fusion. The ONLY difference is the auxiliary head:
#
#     USE_GMM_AUX=1 : total_loss = flow_loss + c1 * gmm_loss
#     USE_GMM_AUX=0 : total_loss = flow_loss                        <-- default here
#
# This is arm B in train_flow_matching_dit_goal_gmm_workspace.yaml's header.
# It is driven by hydra override `policy.aux_gmm_loss_weight=null`, which in
# FlowMatchingDiTGoalGMMPolicy.__init__ leaves `self.gmm_head = None` --
# GoalGMMHead is never instantiated, so there are no GMM parameters in the
# model at all -- and compute_loss() returns fm_loss before ever touching the
# goal (see diffusion_policy/policy/flow_matching_dit_goal_gmm_policy.py:121-122
# and :221-222).
#
# Because the head does not exist, no goal keypoints are needed. This script
# does NOT reference any EXTRA_KEYPOINTS tree and never runs
# generate_non_gmm_goals_for_low_level.py --inject_extra_goals: it stages the
# h5 demos and trains, nothing else. goal_source stays at 'default' (the
# sweep_tissue_h5 demos already carry obs/goal_gripper_pts, which the dataset
# serves because the task yaml's shape_meta declares it -- n_keypoints is read
# from that entry -- but which the policy never reads once the head is gone).
#
# USE_GMM_AUX=1 turns the head back on, supervised by that native
# obs/goal_gripper_pts field. That is NOT the same as the AWE-sourced Approach 2
# treatment run -- for that, use sweeptissueD1_approach2_awe_greedy_th0.2_grip_grogu423_blackwell.sh
# in this same directory, which injects obs/goal_gripper_pts_awe from the
# EXTRA_KEYPOINTS_sweeptissue_awe-greedy-th0.2-grip tree and passes goal_source=awe.
#
# The sweep_tissue h5 tree is produced upstream already cropped/resized to the
# 256x256 shape MimicGen_Tasks/sweeptissue_D1_goal_gmm_aux.yaml expects, so there is
# no --no_gmm conversion step here: this script only stages and trains.
#
# Usage:
#   sbatch sweeptissue_approach2_nogmm_grogu423_blackwell.sh
#       -> control arm: no GMM head
#   USE_GMM_AUX=1 sbatch sweeptissue_approach2_nogmm_grogu423_blackwell.sh
#       -> GMM head on, aux loss at c1=0.1 (native goal_gripper_pts)
#   RESUME_CKPT=/path/to/epoch_N.ckpt sbatch ...                        # resume THIS variant

set -euo pipefail
set -x

export PIXI_HOME="/project_data/held/teswaram/pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# --- the flag ---------------------------------------------------------------
# 0 (default) => GMM head removed entirely (aux_gmm_loss_weight=null).
# 1           => head rebuilt, supervised by the native obs/goal_gripper_pts.
USE_GMM_AUX="${USE_GMM_AUX:-0}"
C1="${C1:-0.1}"            # only used when USE_GMM_AUX=1

# --- paths ------------------------------------------------------------------
SWEEP_TISSUE_H5_DIR="${SWEEP_TISSUE_H5_DIR:-/home/teswaram/data/D1/sweep_tissue_h5}"
REPO_DIR="/home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

if [ "${USE_GMM_AUX}" = "1" ]; then
    ARM_TAG="APPROACH2_gmm_c1_${C1}"
    echo "[config] arm=GMM ON (aux GMM loss, c1=${C1}, native goal_gripper_pts)"
else
    ARM_TAG="APPROACH2_NOGMM"
    echo "[config] arm=CONTROL (GMM head REMOVED, flow-matching loss only)"
fi
echo "[config] SWEEP_TISSUE_H5_DIR=${SWEEP_TISSUE_H5_DIR}"

if [ ! -d "${SWEEP_TISSUE_H5_DIR}" ]; then
    echo "[error] SWEEP_TISSUE_H5_DIR not found: ${SWEEP_TISSUE_H5_DIR}" >&2
    exit 1
fi

RESUME_CKPT="${RESUME_CKPT:-}"

# --- node-local scratch ------------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/scratch/teswaram/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/SweepTissue_D1_${ARM_TAG}"

# --- stage demos -------------------------------------------------------------
# Only the h5 pool is needed -- there is no second tree to intersect with.
THREADS="${RSYNC_THREADS:-32}"
mkdir -p "${DEST_DATA_DIR}"

demos_h5=$(find "${SWEEP_TISSUE_H5_DIR}" -maxdepth 1 -name 'demo_*.h5' -printf '%f\n' | sed 's/\.h5$//' | sort)
n_common=$(echo -n "${demos_h5}" | grep -c . || true)
echo "[stage] staging ${n_common} h5 demos"
if [ "${n_common}" -eq 0 ]; then
    echo "[stage] ERROR: nothing to stage." >&2
    exit 1
fi

stage_start=$(date +%s)

copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export SWEEP_TISSUE_H5_DIR DEST_DATA_DIR

echo "${demos_h5}" | xargs -P "${THREADS}" -I {} \
    bash -c 'copy_one "${SWEEP_TISSUE_H5_DIR}/{}.h5" "${DEST_DATA_DIR}/"'

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} files, $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."
if [ "${staged_count}" -ne "${n_common}" ]; then
    echo "[stage] ERROR: expected ${n_common} files staged, got ${staged_count}." >&2
    exit 1
fi

# --- GMM head on/off ---------------------------------------------------------
if [ "${USE_GMM_AUX}" = "1" ]; then
    GOAL_ARGS=("policy.aux_gmm_loss_weight=${C1}")
else
    # null => FlowMatchingDiTGoalGMMPolicy never constructs GoalGMMHead, and
    # compute_loss returns fm_loss before the goal is ever touched.
    GOAL_ARGS=("policy.aux_gmm_loss_weight=null")
fi

# --- train -------------------------------------------------------------------
cd "${REPO_DIR}"

RESUME_ARGS=()
if [ -n "${RESUME_CKPT}" ]; then
    echo "[resume] resuming training from ${RESUME_CKPT}"
    if [ ! -f "${RESUME_CKPT}" ]; then
        echo "[resume] ERROR: checkpoint not found: ${RESUME_CKPT}" >&2
        exit 1
    fi
    RESUME_ARGS=(training.resume=true "+training.resume_ckpt_path=${RESUME_CKPT}")
else
    echo "[resume] RESUME_CKPT empty -> training from scratch"
fi

RUN_NAME="SweepTissue_D1_${ARM_TAG}_${staged_count}demo_dinov2_DIT_grogu423_blackwell"

USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/project_data/held/teswaram/logs/wandb_cache \
WANDB_DATA_DIR=/project_data/held/teswaram/logs/wandb_data \
PYTHONNOUSERSITE=1 \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_goal_gmm_workspace.yaml \
    task=MimicGen_Tasks/sweeptissue_D1_goal_gmm_aux \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    visual_encoder=dinov2 \
    logging.project=mimicgen_tasks \
    logging.name=${RUN_NAME} \
    name=${RUN_NAME} \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    "${GOAL_ARGS[@]}" \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
