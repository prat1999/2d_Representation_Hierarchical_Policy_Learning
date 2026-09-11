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
#SBATCH --job-name coffee-prep-d1-approach2-awe-greedy-th0.35-grip-actionmode-absolute-grogu423-blackwell
#SBATCH -o /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/coffee-prep-d1-approach2-awe-greedy-th0.35-grip-actionmode-absolute-grogu423-blackwell_job_%j.out
#SBATCH -e /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/coffee-prep-d1-approach2-awe-greedy-th0.35-grip-actionmode-absolute-grogu423-blackwell_job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# ===========================================================================
# APPROACH 2 (GMM-as-auxiliary-loss) on COFFEE_PREPERATION_D1 -- ACTION_MODE ABLATION
#   goal_source=awe (greedy, err_th=0.35, WITH gripper), aux_head=gmm (default), c1=0.1
#   action_mode=ABSOLUTE   <-- the ONLY difference from
#     coffee_preperationD1_approach2_awe_greedy_th0.35_grip_grogu423_blackwell.sh
#     (that sibling uses the repo default action_mode=hybrid_delta and is
#     already fully trained + evaluated locally:
#       epoch_20=27.3% epoch_40=44.7% epoch_60=52.7% epoch_80=60.0% epoch_99=68.0%
#     see /data/theya/models/approach2_models/
#     15.26.32_Coffee_Preperation_D1_APPROACH2_awe_greedy_th0.35_grip_c1_0.1_..._coffee_preperation_goal_gmm_aux/)
#
# Sibling of kitchenD1_approach2_awe_greedy_th0.35_grip_actionmode_absolute_grogu423_blackwell.sh
# -- see that script's header for the full action_mode=absolute rationale
# (Equivariant Diffusion Policy paper, Table 1: plain Diffusion Policy with
# ABSOLUTE pose actions beats RELATIVE on this exact MimicGen/robosuite task
# family, e.g. Coffee Preparation D1 @100 demos: Abs=65 vs Rel=42; fixed
# robot-base/camera frame in robosuite/MimicGen is exactly the regime where
# this is expected to transfer).
#
# action_mode=absolute is a real, already-implemented option in
# lazy_articubot_dataset.py (composes state[t] (+) hybrid_delta[t] into the
# commanded OSC pose target, requires 'state' in shape_meta.obs -- present
# here). From-scratch training run: no checkpoint reuse possible.
#
# NOTE: the eval-side fix for action_mode='absolute' checkpoints (the env
# action conversion previously hard-coded hybrid_delta semantics) lives in
# a worktree at .claude/worktrees/eval-absolute-action-fix -- pass
# INFERENCE_ROOT=<that worktree> to eval.sh when evaluating this run's
# checkpoints, same as for the kitchen_D1 sibling.
#
# Everything else -- data pool, EXTRA_KEYPOINTS tree, staging, injection,
# node pinning, batch size, checkpointing, c1 -- is inherited unchanged
# from the hybrid_delta sibling, so the comparison is a clean one-variable
# (action_mode) change.
# ===========================================================================

set -euo pipefail
set -x

export PIXI_HOME="/project_data/held/teswaram/pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# --- the one knob these sibling scripts vary ------------------------------
GOAL_SOURCE="awe"
SOURCE_TAG="awe_greedy_th0.35_grip"
C1=0.1
# --- the variable under test in THIS script -------------------------------
ACTION_MODE="${ACTION_MODE:-absolute}"

NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[config] goal_source=${GOAL_SOURCE} (${SOURCE_TAG}), c1=${C1}, action_mode=${ACTION_MODE}, NUM_DEMOS cap=${NUM_DEMOS}"

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
DEST_DATA_DIR="${SCRATCH_ROOT}/Coffee_Preperation_D1_Approach2_${SOURCE_TAG}_actionmode_${ACTION_MODE}"

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

RUN_NAME="Coffee_Preperation_D1_APPROACH2_${SOURCE_TAG}_c1_${C1}_actionmode_${ACTION_MODE}_${staged_count}demo_dinov2_DIT_grogu423_blackwell"

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
    action_mode=${ACTION_MODE} \
    logging.project=mimicgen_tasks \
    logging.name=${RUN_NAME} \
    name=${RUN_NAME} \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
