#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=16    # matches dataloader.num_workers=16 below
#SBATCH --mem=80G             # explicit floor above DefMemPerCPU(3.5G)*cpus, so worker/page-cache spikes don't hit the cgroup limit
#SBATCH -p all                # genuine A6000 48GB nodes live on "all" (grogu-1-10/20/25/30, grogu-2-5);
                              # dheld has none -- its A6000-labelled grogu-4-13 is really a 12GB 3080Ti.
#SBATCH --gres=gpu:1
#SBATCH -C A6000
#SBATCH --exclude=grogu-4-13  # feature-labelled A6000, actually a 12GB 3080Ti -- would OOM at batch_size=128
#SBATCH -t 1-00:00:00         # partition "all" caps walltime at 24h (MaxTime=1-00:00:00);
                              # use RESUME_CKPT=<ckpt> to continue in a follow-up job.
#SBATCH --job-name square-d2-approach2-auxregression-relgripper-c1-10-grogu-a6000
#SBATCH -o /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/square-d2-approach2-auxregression-relgripper-c1-10-grogu-a6000_job_%j.out
#SBATCH -e /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/square-d2-approach2-auxregression-relgripper-c1-10-grogu-a6000_job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# ===========================================================================
# APPROACH 2, REGRESSION auxiliary head on SQUARE_D2
#   goal_source=awe (greedy, err_th=0.35, WITH gripper)
#   aux_head_type=regression, aux_regression_frame=relative_to_gripper, c1=10
# grogu cluster / dheld partition, pinned to grogu-4-23 (8x RTX 6000 Blackwell).
#
# NEW_TASKS port of
# GROGU_BLACKWELL/coffee_preperationD1_approach2_auxregression_relgripper_c1_10_grogu_a6000.sh
# -- that script's header carries the full c1 derivation, reproduced in brief
# below. Only the dataset paths and the task yaml differ here.
#
# Identical data pipeline to the GMM sibling in this directory
# (squareD2_approach2_awe_greedy_th0.35_grip_grogu_a6000.sh): same NO_GMM
# h5 pool, same EXTRA_KEYPOINTS_awe-greedy-th0.35-grip tree, same staging and
# injection. The ONLY difference is the auxiliary head:
#
#   GMM sibling : total_loss = flow_loss + c1 * gmm_nll(goal | every anchor)
#   this script : total_loss = flow_loss + c1 * MSE(pred, goal - present_gripper_pts)
#
# GoalRegressionHead mean-pools the SAME grounded token stack the GMM head
# scores per-anchor and regresses one subgoal per (sample, obs step).
# frame=relative_to_gripper makes the target the displacement the gripper still
# has to travel, so the target scale does not depend on where in the workspace
# the episode starts.
#
# --- why C1 is 10, not the GMM arm's 0.1 -----------------------------------
# c1=0.1 holds the GMM arm's aux/fm ratio in a band of roughly 0.2-2 for the
# whole run. An MSE head cannot reproduce that with the same constant: the GMM
# NLL is unnormalised and self-extinguishes as the head sharpens, while the MSE
# starts at the target's own scale (~0.02 m^2 measured on the D1 tasks) and
# floors at the irreducible subgoal ambiguity instead of fading. Matching at
# init gives c1 ~= 42, matching at the end (MSE floor ~0.004) gives c1 ~= 5;
# c1 = 10 is the geometric middle -- aux/fm ~= 0.14 at init, rising into the
# GMM arm's band as fm collapses, and plateauing rather than overpowering.
#
# The uncertain input is the MSE floor, and it is task-dependent -- this is a
# new task, so check it on the first run: watch train_goal_regression_loss vs
# train_fm_loss in wandb and rescale if needed.
#   C1=1 sbatch this_script.sh
#   C1=50 sbatch this_script.sh
#
# Also note GoalRegressionHead uses default Linear init (unlike GoalGMMHead's
# OUT_INIT_STD=1e-3), so its step-0 loss is the target scale PLUS whatever
# constant the untrained MLP emits.
# ===========================================================================

set -euo pipefail
set -x

export PIXI_HOME="/project_data/held/teswaram/pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# --- goal source (shared with the GMM sibling) ----------------------------
GOAL_SOURCE="awe"
SOURCE_TAG="awe_greedy_th0.35_grip"

# --- auxiliary head: regression instead of the GMM mixture ---------------
AUX_HEAD_TYPE="regression"
AUX_REGRESSION_FRAME="${AUX_REGRESSION_FRAME:-relative_to_gripper}"
AUX_TAG="auxhead_regression_frame_rel_gripper"
# see the header for why this is 10 and not the GMM arm's 0.1
C1="${C1:-10.0}"

NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[config] goal_source=${GOAL_SOURCE} (${SOURCE_TAG}), aux_head=${AUX_HEAD_TYPE} (${AUX_REGRESSION_FRAME}), c1=${C1}, NUM_DEMOS cap=${NUM_DEMOS}"

# --- paths -----------------------------------------------------------------
# npz sources live in the NEW_TASKS tree under $HOME; the converted h5 pool does
# NOT -- $HOME has ~34G free and one pool runs ~19G, so pools go to /project_data.
SRC_NPZ_DIR="${SRC_NPZ_DIR:-/home/teswaram/data/NEW_TASKS/square_d2}"
EXTRA_GOALS_DIR="/home/teswaram/data/NEW_TASKS/EXTRA_KEYPOINTS_awe-greedy-th0.35-grip/square_d2"
LOCAL_NO_GMM_H5_DIR="${LOCAL_NO_GMM_H5_DIR:-/project_data/held/teswaram/data/NEW_TASKS/NO_GMM_preds/square_d2}"
REPO_DIR="/home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# --- resume from checkpoint --------------------------------------------------
# Empty by default: a fresh variant must NOT resume from a sibling arm's
# checkpoint. Override only when resuming THIS SAME variant:
#   RESUME_CKPT=/path/to/epoch_N.ckpt sbatch this_script.sh
RESUME_CKPT="${RESUME_CKPT:-}"

if [ ! -d "${EXTRA_GOALS_DIR}" ]; then
    echo "[error] EXTRA_GOALS_DIR not found: ${EXTRA_GOALS_DIR}" >&2
    exit 1
fi


# --- generate local NO_GMM h5 pool if not already cached --------------------
# flock-guarded and SHARED by all three square_d2 Approach-2 arms
# (gmm c1=0.1 / regression c1=10 / no-aux control): the demo_N/ npz -> h5
# conversion reads only SRC_NPZ_DIR and depends on neither the goal source nor
# the auxiliary head, so the three may be submitted together and the first one
# in does the work.
mkdir -p "${LOCAL_NO_GMM_H5_DIR}"
(
    flock -x 200
    existing_h5_count=$(find "${LOCAL_NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" 2>/dev/null | wc -l)
    echo "[gen] existing local *.h5 in ${LOCAL_NO_GMM_H5_DIR}: ${existing_h5_count}"
    if [ "${existing_h5_count}" -ge "${NUM_DEMOS}" ]; then
        echo "[gen] already have ${existing_h5_count} local h5 files (>= NUM_DEMOS=${NUM_DEMOS}) -> skipping conversion"
    else
        echo "[gen] converting demo_*/ -> demo_*.h5 in ${LOCAL_NO_GMM_H5_DIR}"
        cd "${REPO_DIR}"
        USE_TF=0 \
        GIT_LFS_SKIP_SMUDGE=1 \
        PYTHONNOUSERSITE=1 \
        pixi run python generate_non_gmm_goals_for_low_level.py \
            --dataset_dir "${SRC_NPZ_DIR}" \
            --no_gmm \
            --no_gmm_output_dir "${LOCAL_NO_GMM_H5_DIR}" \
            --max_files "${NUM_DEMOS}"
    fi
) 200>"${LOCAL_NO_GMM_H5_DIR}.genlock"
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
DEST_DATA_DIR="${SCRATCH_ROOT}/Square_D2_Approach2_${SOURCE_TAG}_${AUX_TAG}"

# --- stage demos present in BOTH the h5 dir and the EXTRA_KEYPOINTS tree ----
# The keypoints tree carries demo_0..demo_99, so this intersection is what
# actually caps the run at 100 demos.
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

RUN_NAME="Square_D2_APPROACH2_${SOURCE_TAG}_${AUX_TAG}_c1_${C1}_${staged_count}demo_dinov2_DIT_grogu_a6000"

USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/project_data/held/teswaram/logs/wandb_cache \
WANDB_DATA_DIR=/project_data/held/teswaram/logs/wandb_data \
PYTHONNOUSERSITE=1 \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_goal_gmm_workspace.yaml \
    task=MimicGen_Tasks/square_d2_goal_gmm_aux \
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
