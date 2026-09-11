#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=16    # matches dataloader.num_workers=16 below
#SBATCH --mem=80G             # explicit floor above DefMemPerCPU(3.5G)*cpus, so worker/page-cache spikes don't hit the cgroup limit -- jobs 3935329 (OOM) and 3935352 (SIGBUS from a live env swap mid-run on grogu-1-15) predate this Blackwell-pinned script
#SBATCH -p dheld
#SBATCH -w grogu-4-23         # pin to the RTX 6000 Blackwell node
#SBATCH --gres=gpu:1
#SBATCH -C 6000Blackwell      # grogu-4-23 is an 8x RTX 6000 Blackwell node, not A6000 -- match the node's AVAIL_FEATURES
#SBATCH -t 24:00:00
#SBATCH --job-name kitchen-d1-approach2-auxregression-absolute-c1-10-grogu423-blackwell
#SBATCH -o /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/kitchen-d1-approach2-auxregression-absolute-c1-10-grogu423-blackwell_job_%j.out
#SBATCH -e /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/kitchen-d1-approach2-auxregression-absolute-c1-10-grogu423-blackwell_job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# ===========================================================================
# APPROACH 2, REGRESSION auxiliary head on KITCHEN_D1
#   goal_source=awe (greedy, err_th=0.35, WITH gripper)
#   aux_head_type=regression, aux_regression_frame=ABSOLUTE
#
# The absolute-frame twin of
# kitchenD1_approach2_auxregression_relgripper_c1_1_grogu423_blackwell.sh
# Everything -- data pool, EXTRA_KEYPOINTS tree, staging, injection, node
# pinning, batch size, checkpointing -- is inherited unchanged. The ONLY
# difference is the regression target:
#
#   rel_gripper sibling : MSE(pred, goal_gripper_pts - present_gripper_pts)
#   this script         : MSE(pred, goal_gripper_pts)          # world frame
#
# i.e. the head predicts WHERE the subgoal is, not HOW FAR the gripper still
# has to travel. Same GoalRegressionHead, same mean-pooled grounded token
# stack; only goal_regression_target() switches branch (frame="absolute"
# returns the goal untouched). The readout reports world coordinates in both
# frames, so subgoal_pred is directly comparable across the two arms.
#
# --- why the absolute frame is worth a separate arm ------------------------
# The RoPE4D trunk gives every token its position through rotary attention, so
# the pooled feature already knows where the anchors are in the world. Asking
# for the goal in world coordinates therefore does NOT require the head to
# reconstruct the gripper position first, as the relative target implicitly
# does. The cost is that the absolute target carries a large task-constant
# offset (goals sit around [0.5, 0.0, 0.1] m) which the head can fit with its
# output bias alone and which carries no gradient into the trunk.
#
# --- C1: the SAME {1, 10} pair as the relative arms, and why -------------
# Measured on the first 5 demos of each task (awe goals from the same
# EXTRA_KEYPOINTS tree this script injects, present_gripper_pts from the same
# NO_GMM h5 pool), as mean-over-(row, keypoint) squared L2 -- exactly what
# goal_regression_loss() returns for a zero prediction:
#
#                            E|g|^2   const-pred floor   E|g-p|^2
#   HAMMER_CLEANUP_D1         0.271        0.022           0.017
#   KITCHEN_D1                0.329        0.040           0.038
#   COFFEE_PREPERATION_D1     0.312        0.051           0.014
#
# (KITCHEN_D1: 0.33, 0.04, 0.038 m^2.)
#
# The raw absolute target is 9-22x the relative one, but nearly all of that
# excess is the constant: "const-pred floor" is what a predictor that emits
# only the task mean already reaches, and it is back on the relative target's
# own scale. Confirmed by an actual 53-step run of BOTH frames on 4
# HAMMER_CLEANUP_D1 demos (identical seed, init and batch stream; the head's
# own MSE logged pre-update each step):
#
#     step        0      4      8     12     20     36     44   min
#   absolute   0.261  0.079  0.030  0.024  0.019  0.011  0.008  0.008
#   relative   0.096  0.050  0.040  0.018  0.011  0.008  0.008  0.008
#
# Three things to take from that:
#   * the constant is absorbed in ~10 steps, not "the early phase" -- the 15x
#     transient is a handful of steps long, so it does not need its own c1;
#   * from step ~15 on the two frames sit in the SAME 0.008-0.02 band, so the
#     relative arm's c1 analysis transfers verbatim;
#   * absolute reaches 0.008, BELOW its own const-pred floor (0.04), i.e.
#     the head is fitting per-scene structure and not just the task mean --
#     which is the degenerate outcome this frame risks.
#
# So c1 is kept at {1, 10} deliberately, to keep the comparison against the
# relative arms a one-variable change. With fm_loss ~= 1.4 at init falling to
# ~0.02-0.04:
#   c1=1    aux/fm ~= 0.2 for the first couple of steps, ~0.01-0.02 once the
#           constant is gone, rising back into the GMM arms' 0.2-2 band as fm
#           collapses. The better-matched setting of the two.
#   c1=10   aux/fm ~= 2 for those first steps, and since the MSE floors while
#           fm keeps falling it can reach a few x fm late. Run it for parity
#           with the relative c1=10 arm, not because the scale matches.
#
# In wandb, watch train_goal_regression_loss for the signature above: a fast
# drop from ~0.26 onto ~0.04 inside the first ~10 steps, then a slow decay
# THROUGH that floor. If it plateaus AT ~0.04 and stops, the head has
# learned the task mean and nothing else, and the relative arm is the better
# trunk signal. GoalRegressionHead uses default Linear init (unlike
# GoalGMMHead's OUT_INIT_STD=1e-3), so step 0 is E|g|^2 plus whatever constant
# the untrained MLP emits -- 0.26 vs 0.20 in the run above.
#
#   C1=10 sbatch this_script.sh
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
AUX_REGRESSION_FRAME="absolute"
AUX_TAG="auxhead_regression_frame_absolute"
# see the header for the absolute-frame scale measurements behind this
C1="${C1:-10.0}"

NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[config] goal_source=${GOAL_SOURCE} (${SOURCE_TAG}), aux_head=${AUX_HEAD_TYPE} (${AUX_REGRESSION_FRAME}), c1=${C1}, NUM_DEMOS cap=${NUM_DEMOS}"

# --- paths -----------------------------------------------------------------
SRC_NPZ_DIR="/project_data/held/teswaram/data/D1/KITCHEN_D1"
EXTRA_GOALS_DIR="/project_data/held/teswaram/data/D1/EXTRA_KEYPOINTS_awe-greedy-th0.35-grip/KITCHEN_D1"
LOCAL_NO_GMM_H5_DIR="/project_data/held/teswaram/data/D1/NO_GMM_preds/KITCHEN_D1"
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
DEST_DATA_DIR="${SCRATCH_ROOT}/Kitchen_D1_Approach2_${SOURCE_TAG}_${AUX_TAG}"

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

RUN_NAME="kitchen_D1_APPROACH2_${SOURCE_TAG}_${AUX_TAG}_c1_${C1}_${staged_count}demo_dinov2_DIT_grogu423_blackwell"

USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/project_data/held/teswaram/logs/wandb_cache \
WANDB_DATA_DIR=/project_data/held/teswaram/logs/wandb_data \
PYTHONNOUSERSITE=1 \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_goal_gmm_workspace.yaml \
    task=MimicGen_Tasks/kitchen_goal_gmm_aux \
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
