#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=16    # matches dataloader.num_workers=16 below
#SBATCH --mem=80G             # explicit floor above DefMemPerCPU(3.5G)*cpus, so worker/page-cache spikes don't hit the cgroup limit -- jobs 3935330 (OOM) and 3935358 (SIGBUS from a live env swap mid-run on grogu-1-15) predate this Blackwell-pinned script
#SBATCH -p dheld
#SBATCH -w grogu-4-23         # pin to the RTX 6000 Blackwell node
#SBATCH --gres=gpu:1
#SBATCH -C 6000Blackwell      # grogu-4-23 is an 8x RTX 6000 Blackwell node, not A6000 -- match the node's AVAIL_FEATURES
#SBATCH -t 24:00:00
#SBATCH --job-name coffee-prep-d1-approach2-auxregression-relgripper-c1-10-grogu423-blackwell
#SBATCH -o /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/coffee-prep-d1-approach2-auxregression-relgripper-c1-10-grogu423-blackwell_job_%j.out
#SBATCH -e /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/coffee-prep-d1-approach2-auxregression-relgripper-c1-10-grogu423-blackwell_job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# ===========================================================================
# APPROACH 2, REGRESSION auxiliary head on COFFEE_PREPERATION_D1
#   goal_source=awe (greedy, err_th=0.35, WITH gripper)
#   aux_head_type=regression, aux_regression_frame=relative_to_gripper
#
# Identical data pipeline to the sibling
# coffee_preperationD1_approach2_awe_greedy_th0.35_grip_grogu423_blackwell.sh
# (same NO_GMM h5 pool, same EXTRA_KEYPOINTS_awe-greedy-th0.35-grip tree, same
# staging/injection). The ONLY difference is the auxiliary head:
#
#   GMM sibling : total_loss = flow_loss + c1 * gmm_nll(goal | every anchor)
#   this script : total_loss = flow_loss + c1 * MSE(pred, goal - present_gripper_pts)
#
# The head (GoalRegressionHead) mean-pools the SAME grounded token stack the
# GMM head scores per-anchor and regresses one subgoal per (sample, obs step).
# frame=relative_to_gripper makes the target the displacement the gripper still
# has to travel, so the target scale does not depend on where in the workspace
# the episode starts.
#
# --- why C1 is 10, not the GMM arm's 0.1 -----------------------------------
# c1=0.1 was NOT arbitrary for the GMM arm: it keeps the auxiliary term within
# a factor of ~2 of the flow-matching loss for the WHOLE run. Measured from the
# wandb histories of the three c1=0.1 GMM runs on this node
# (outputs/2026.09.01/*_c1_0.1_100demo_dinov2_DIT_grogu423_blackwell_*),
# aux/fm = 0.1*gmm_loss / fm_loss:
#
#     % of run     0     0.2%    1%     5%    10%    25%    50%    75%   100%
#   hammer      0.63    0.65   1.49   0.81   0.68   0.25   0.19   2.11   3.56
#   kitchen     0.67    1.94   1.36   0.39   0.61   0.43  -0.26  -0.11  -0.11
#   coffee      0.72    2.29   0.98   0.86   0.61   0.59  -0.09  -0.47  -0.42
#
# i.e. a band of roughly 0.2-2, centred near 0.6, start to finish.
#
# Reproducing that band with an MSE head needs a different constant, and no
# single constant reproduces it exactly, because the two losses decay
# differently:
#
#   fm_loss        1.39 -> ~0.02-0.04   (falls ~40x over the run)
#   gmm NLL        ~9   -> ~0 or NEGATIVE. It is an unnormalised NLL, so it is
#                  unbounded below and self-extinguishes as the head sharpens --
#                  which is exactly why c1=0.1 never runs away late.
#   regression MSE starts at the target's own scale (measured mean
#                  sum-of-squares of goal_gripper_pts - present_gripper_pts over
#                  the first 5 demos: 0.016 HAMMER_CLEANUP_D1, 0.017 KITCHEN_D1,
#                  0.022 COFFEE_PREPERATION_D1 m^2, call it 0.02) and FLOORS at
#                  the irreducible subgoal ambiguity. It cannot go negative, so
#                  unlike the GMM NLL it does not fade out on its own.
#
# That asymmetry is the whole sizing problem:
#   - matching at init  (0.6 * 1.39 / 0.02)                  => c1 ~= 42
#   - matching at the end, if the MSE floors near 0.004
#                       (0.6 * 0.03 / 0.004)                 => c1 ~= 5
#   c1 ~= 10 is the geometric middle: aux/fm ~= 0.14 at init, rising into the
#   GMM arm's 0.2-2 band as fm collapses, and plateauing there rather than
#   overpowering.
#
# For reference, at the other candidate weights:
#   c1=1    aux/fm ~= 0.0014 at init -- three orders below the GMM arm during
#           the early phase where the trunk is actually being shaped. Safe, but
#           probably indistinguishable from the no-aux control arm early on.
#   c1=50   init-matched to the GMM arm (aux/fm ~= 0.7), but because the MSE
#           floors while fm keeps falling, it can reach 5-15x fm late -- this is
#           the setting that genuinely risks overpowering.
#
# The uncertain input is the MSE floor; the first run settles it. Watch
# train_goal_regression_loss vs train_fm_loss in wandb and rescale:
#   C1=1 sbatch this_script.sh
#   C1=50 sbatch this_script.sh
#
# Caveat: unlike GoalGMMHead (OUT_INIT_STD=1e-3, so mu ~= anchor and the loss
# starts at the target's own scale), GoalRegressionHead uses default Linear
# init, so its step-0 loss is 0.02 PLUS whatever constant the untrained MLP
# emits. Check train_goal_regression_loss on the first few steps and rescale C1
# if it does not land near 0.02.
#
# Everything else (node pinning, --mem floor, batch size, checkpointing)
# is inherited unchanged from the GMM sibling.
#
#   NUM_DEMOS=50 sbatch this_script.sh
# ===========================================================================

set -euo pipefail
set -x

export PIXI_HOME="/project_data/held/teswaram/pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# --- the one knob these sibling scripts vary ------------------------------
GOAL_SOURCE="awe"
SOURCE_TAG="awe_greedy_th0.35_grip"
# --- auxiliary head: regression instead of the GMM mixture ---------------
AUX_HEAD_TYPE="regression"
AUX_REGRESSION_FRAME="relative_to_gripper"
AUX_TAG="auxhead_regression_frame_rel_gripper"
# see the header for why this is 10 and not the GMM arm's 0.1
C1="${C1:-1.0}"

NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[config] goal_source=${GOAL_SOURCE} (${SOURCE_TAG}), aux_head=${AUX_HEAD_TYPE} (${AUX_REGRESSION_FRAME}), c1=${C1}, NUM_DEMOS cap=${NUM_DEMOS}"

# --- paths -----------------------------------------------------------------
SRC_NPZ_DIR="/project_data/held/teswaram/data/D1/COFFEE_PREPERATION_D1"
EXTRA_GOALS_DIR="/project_data/held/teswaram/data/D1/EXTRA_KEYPOINTS_awe-greedy-th0.35-grip/COFFEE_PREPERATION_D1"
LOCAL_NO_GMM_H5_DIR="/project_data/held/teswaram/data/D1/NO_GMM_preds/COFFEE_PREPERATION_D1"
REPO_DIR="/home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# --- resume from checkpoint --------------------------------------------------
RESUME_CKPT="${RESUME_CKPT:-}"

if [ ! -d "${EXTRA_GOALS_DIR}" ]; then
    echo "[error] EXTRA_GOALS_DIR not found: ${EXTRA_GOALS_DIR}" >&2
    exit 1
fi

# --- generate local NO_GMM h5 pool if not already cached --------------------
existing_h5_count=0
if [ -d "${LOCAL_NO_GMM_H5_DIR}" ]; then
    existing_h5_count=$(find "${LOCAL_NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" 2>/dev/null | wc -l)
fi
echo "[gen] existing local *.h5 in ${LOCAL_NO_GMM_H5_DIR}: ${existing_h5_count}"
if [ "${existing_h5_count}" -eq 0 ]; then
    echo "[gen] converting demo_*/ -> demo_*.h5 in ${LOCAL_NO_GMM_H5_DIR}"
    mkdir -p "${LOCAL_NO_GMM_H5_DIR}"
    (
        cd "${REPO_DIR}"
        USE_TF=0 \
        GIT_LFS_SKIP_SMUDGE=1 \
        PYTHONNOUSERSITE=1 \
        pixi run python generate_non_gmm_goals_for_low_level.py \
            --dataset_dir "${SRC_NPZ_DIR}" \
            --no_gmm \
            --no_gmm_output_dir "${LOCAL_NO_GMM_H5_DIR}" \
            --max_files "${NUM_DEMOS}"
    )
else
    echo "[gen] already have ${existing_h5_count} local h5 files -> skipping conversion"
fi
NO_GMM_H5_DIR="${LOCAL_NO_GMM_H5_DIR}"

# --- node-local scratch ------------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/scratch/teswaram/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/Coffee_Preperation_D1_Approach2_${SOURCE_TAG}_${AUX_TAG}"

# --- stage demos present in BOTH the h5 dir and the EXTRA_KEYPOINTS tree ----
THREADS="${RSYNC_THREADS:-32}"
mkdir -p "${DEST_DATA_DIR}"

demos_h5=$(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name 'demo_*.h5' -printf '%f\n' | sed 's/\.h5$//' | sort)
demos_goals=$(find "${EXTRA_GOALS_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' -printf '%f\n' | sort)
demos_common=$(comm -12 <(echo "${demos_h5}") <(echo "${demos_goals}"))
n_common=$(echo -n "${demos_common}" | grep -c . || true)
echo "[stage] h5 demos: $(echo -n "${demos_h5}" | grep -c . || true), goal demos: $(echo -n "${demos_goals}" | grep -c . || true), common: ${n_common}"
if [ "${n_common}" -eq 0 ]; then
    echo "[stage] ERROR: no demo present in both NO_GMM_H5_DIR and EXTRA_GOALS_DIR." >&2
    exit 1
fi

echo "[stage] source : ${NO_GMM_H5_DIR}"
echo "[stage] dest   : ${DEST_DATA_DIR}"
echo "[stage] threads: ${THREADS}"

stage_start=$(date +%s)

copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export NO_GMM_H5_DIR DEST_DATA_DIR

echo "${demos_common}" | xargs -P "${THREADS}" -I {} \
    bash -c 'copy_one "${NO_GMM_H5_DIR}/{}.h5" "${DEST_DATA_DIR}/"'

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} files, $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."
if [ "${staged_count}" -ne "${n_common}" ]; then
    echo "[stage] ERROR: expected ${n_common} files staged, got ${staged_count}." >&2
    exit 1
fi

# --- inject extra goal keys into the staged, node-local copy ---------------
echo "[inject] ensuring obs/goal_gripper_pts_awe exists in staged demos (from EXTRA_KEYPOINTS_awe-greedy-th0.35-grip)"
(
    cd "${REPO_DIR}"
    USE_TF=0 \
    GIT_LFS_SKIP_SMUDGE=1 \
    PYTHONNOUSERSITE=1 \
    pixi run python generate_non_gmm_goals_for_low_level.py \
        --dataset_dir "${DEST_DATA_DIR}" \
        --inject_extra_goals \
        --extra_goals_dir "${EXTRA_GOALS_DIR}"
)

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

RUN_NAME="Coffee_Preperation_D1_APPROACH2_${SOURCE_TAG}_${AUX_TAG}_c1_${C1}_${staged_count}demo_dinov2_DIT_grogu423_blackwell"

USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/project_data/held/teswaram/logs/wandb_cache \
WANDB_DATA_DIR=/project_data/held/teswaram/logs/wandb_data \
PYTHONNOUSERSITE=1 \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_goal_gmm_workspace.yaml \
    task=MimicGen_Tasks/coffee_preperation_goal_gmm_aux \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    +task.dataset.goal_source=${GOAL_SOURCE} \
    policy.aux_gmm_loss_weight=${C1} \
    policy.aux_head_type=${AUX_HEAD_TYPE} \
    policy.aux_regression_frame=${AUX_REGRESSION_FRAME} \
    logging.project=mimicgen_tasks \
    logging.name=${RUN_NAME} \
    name=${RUN_NAME} \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
