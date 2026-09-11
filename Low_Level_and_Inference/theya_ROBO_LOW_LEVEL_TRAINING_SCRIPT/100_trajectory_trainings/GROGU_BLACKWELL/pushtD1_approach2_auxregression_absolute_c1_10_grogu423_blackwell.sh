#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=16    # matches dataloader.num_workers=16 below
#SBATCH --mem=160G            # same floor as the awe_dp PushT siblings: job 3935697 was cgroup-OOM-killed mid-epoch at 80G despite plenty of free node RAM
#SBATCH -p dheld
#SBATCH --gres=gpu:1
#SBATCH -C 6000Blackwell      # grogu-4-23 is an 8x RTX 6000 Blackwell node, not A6000
#SBATCH -w grogu-4-23
#SBATCH -t 24:00:00
#SBATCH --job-name pusht-d1-approach2-auxregression-absolute-c1-10-grogu423-blackwell
#SBATCH -o /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/pusht-d1-approach2-auxregression-absolute-c1-10-grogu423-blackwell_job_%j.out
#SBATCH -e /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/pusht-d1-approach2-auxregression-absolute-c1-10-grogu423-blackwell_job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# ===========================================================================
# APPROACH 2, REGRESSION auxiliary head on PUSHT_D1
#   goal_source=awe (GREEDY keypoint selection, err_th=0.35)
#   aux_head_type=regression, aux_regression_frame=ABSOLUTE
#
# The c1=10 twin of
# pushtD1_approach2_auxregression_absolute_c1_1_grogu423_blackwell.sh --
# identical in every respect except the auxiliary loss weight.
#
#   rel_gripper arms : MSE(pred, goal_gripper_pts - present_gripper_pts)
#   absolute arms    : MSE(pred, goal_gripper_pts)          # world frame
#
# i.e. absolute asks WHERE the subgoal is, relative asks HOW FAR the pusher
# still has to travel. Same GoalRegressionHead, same mean-pooled grounded
# token stack; only goal_regression_target() switches branch.
#
# --- C1 on PushT: measured, not inherited ---------------------------------
# Same measurement as the MimicGen regression scripts' headers (first 5 demos,
# greedy-th0.35 goals from the tree this script injects, present_gripper_pts
# from the NO_GMM PUSH_T_D1 h5 pool; mean-over-(row, keypoint) squared L2 --
# exactly what goal_regression_loss() returns for a zero prediction), over
# n=738 rows:
#
#                            E|g|^2   const-pred floor   E|g-p|^2
#   PushT_D1 greedy th0.35    0.614        0.049           0.078
#   PushT_D1 dp     th0.35    0.581        0.030           0.056
#   (for reference, the MimicGen tasks)
#   HAMMER_CLEANUP_D1         0.271        0.022           0.017
#   KITCHEN_D1                0.329        0.040           0.038
#   COFFEE_PREPERATION_D1     0.312        0.051           0.014
#
# PushT sits ~2x the manipulation tasks on the absolute target (0.61 vs
# ~0.3) and 2-4x on the relative one (0.078 vs 0.017-0.038): the pusher
# sweeps a wide planar workspace, so consecutive subgoals are further apart
# than an arm's are. The const-pred floor (0.049) is ~8% of E|g|^2, so as on
# the other tasks nearly all of the absolute target's excess is a task
# constant the head absorbs with its output bias alone.
#
# c1 is kept at the SAME {1, 10} pair as every other regression arm so the
# comparison across tasks and against the GMM arms stays a one-variable
# change. With PushT's relative target ~2x kitchen's, c1=1 is the
# better-matched setting of the two; c1=10 is run for parity.
#
# In wandb, watch train_goal_regression_loss: a fast drop onto ~0.05 in the
# first ~10 steps, then a slow decay THROUGH that floor. A plateau AT ~0.05
# means the head learned the task mean and nothing else, and the rel_gripper
# arm is the better trunk signal. GoalRegressionHead uses default Linear init
# (unlike GoalGMMHead's OUT_INIT_STD=1e-3), so step 0 is E|g|^2 plus whatever
# constant the untrained MLP emits.
#
#   C1=10.0 is this file's default; NUM_DEMOS/NUM_EPOCHS/RESUME_CKPT overridable
# ===========================================================================
set -euo pipefail
set -x

export PIXI_HOME="/project_data/held/teswaram/pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# --- the one knob these sibling scripts vary ------------------------------
GOAL_SOURCE="awe"
# SOURCE_TAG only distinguishes run/log/scratch naming from the awe_dp
# siblings -- it is NOT passed to hydra (goal_source stays "awe" because that
# is the npz/h5 key name in every EXTRA_KEYPOINTS tree).
SOURCE_TAG="awe_greedy_th0.35"
# --- auxiliary head: regression instead of the GMM mixture ---------------
AUX_HEAD_TYPE="regression"
AUX_REGRESSION_FRAME="absolute"
AUX_TAG="auxhead_regression_frame_absolute"
C1="${C1:-10.0}"

NUM_DEMOS="${NUM_DEMOS:-206}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
echo "[config] goal_source=${GOAL_SOURCE} (${SOURCE_TAG}_${AUX_TAG}), c1=${C1}, NUM_DEMOS cap=${NUM_DEMOS}, epochs=${NUM_EPOCHS}"

# --- paths -----------------------------------------------------------------
# NOTE the greedy tree does NOT live under .../D1/PushT/ like the awe_dp trees
# do -- it landed as D1/EXTRA_KEYPOINTS_awe-greedy-th0.35/PushT_npz, following
# the naming of the other tasks' greedy trees. Override if it is ever moved.
SRC_NPZ_DIR="${SRC_NPZ_DIR:-/project_data/held/teswaram/data/D1/PushT/PushT_npz}"
EXTRA_GOALS_DIR="${EXTRA_GOALS_DIR:-/project_data/held/teswaram/data/D1/EXTRA_KEYPOINTS_awe-greedy-th0.35/PushT_npz}"
LOCAL_NO_GMM_H5_DIR="/project_data/held/teswaram/data/D1/NO_GMM_preds/PUSH_T_D1"
REPO_DIR="/home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# --- resume from checkpoint --------------------------------------------------
# Empty by default: a fresh variant should NOT resume from a different
# variant's checkpoint. Override only when resuming THIS SAME variant:
#   RESUME_CKPT=/path/to/epoch_N.ckpt sbatch this_script.sh
RESUME_CKPT="${RESUME_CKPT:-}"

if [ ! -d "${EXTRA_GOALS_DIR}" ]; then
    echo "[error] EXTRA_GOALS_DIR not found: ${EXTRA_GOALS_DIR}" >&2
    exit 1
fi

# --- generate local NO_GMM h5 pool if not already cached ---------------------
# flock-guarded and SHARED with every other PushT_D1 Approach2 script (the
# three awe_dp thresholds and the greedy siblings): h5 generation reads only
# SRC_NPZ_DIR and does not depend on the goal source or the aux head, so all
# of them read/write this one pool and may be submitted together.
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
            --push_t \
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
DEST_DATA_DIR="${SCRATCH_ROOT}/PushT_D1_Approach2_${SOURCE_TAG}_auxhead_regression_frame_absolute"

# --- stage demos present in BOTH the h5 dir and the EXTRA_KEYPOINTS tree ----
THREADS="${RSYNC_THREADS:-32}"
mkdir -p "${DEST_DATA_DIR}"

demos_h5=$(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name 'demo_*.h5' -printf '%f\n' | sed 's/\.h5$//' | sort)
demos_goals=$(find "${EXTRA_GOALS_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' -printf '%f\n' | sort)
demos_common=$(comm -12 <(echo "${demos_h5}") <(echo "${demos_goals}"))
n_common=$(echo -n "${demos_common}" | grep -c . || true)
echo "[stage] h5 demos: $(echo -n "${demos_h5}" | grep -c . || true), goal demos: $(echo -n "${demos_goals}" | grep -c . || true), common: ${n_common}"
# --- cap the staged set to NUM_DEMOS ---------------------------------------
# This MUST happen here, not only at the conversion step above. The PUSH_T_D1
# h5 pool is a SUPERSET (all 206 demos converted, shared with the awe_dp
# siblings), so --max_files/NUM_DEMOS gates conversion only -- with the pool
# already full, conversion is skipped and the intersection below would stage
# every one of the 206 regardless of NUM_DEMOS. Sorting is NUMERIC on the demo
# index so the subset is demo_0..demo_$((NUM_DEMOS-1)) and identical across all
# the PushT arms; a lexicographic head would pick demo_0, demo_1, demo_10,
# demo_100, ... instead.
if [ "${n_common}" -gt "${NUM_DEMOS}" ]; then
    demos_common=$(echo "${demos_common}" | sort -t_ -k2 -n | head -n "${NUM_DEMOS}")
    n_common=$(echo -n "${demos_common}" | grep -c . || true)
    echo "[stage] capped to NUM_DEMOS=${NUM_DEMOS} -> ${n_common} demos (demo_0..demo_$((NUM_DEMOS-1)) by numeric index)"
fi
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
# The greedy tree carries one goal_gripper_pcd_awe (1,4,3) per frame, exactly
# like the awe_dp trees, and has >= as many npz frames as the h5 has rows
# (demo_0: 162 npz vs T=161), so the injector's npz_files[:T] truncation is a
# no-op mismatch and not a data error.
echo "[inject] ensuring obs/goal_gripper_pts_awe exists in staged demos (from EXTRA_KEYPOINTS_awe-greedy-th0.35)"
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

RUN_NAME="PushT_D1_APPROACH2_${SOURCE_TAG}_${AUX_TAG}_c1_${C1}_${staged_count}demo_dinov2_DIT_grogu423_blackwell"

USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/project_data/held/teswaram/logs/wandb_cache \
WANDB_DATA_DIR=/project_data/held/teswaram/logs/wandb_data \
PYTHONNOUSERSITE=1 \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_goal_gmm_workspace.yaml \
    task=PushT_Tasks/push_t_goal_gmm_aux \
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
    training.num_epochs=${NUM_EPOCHS} \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
