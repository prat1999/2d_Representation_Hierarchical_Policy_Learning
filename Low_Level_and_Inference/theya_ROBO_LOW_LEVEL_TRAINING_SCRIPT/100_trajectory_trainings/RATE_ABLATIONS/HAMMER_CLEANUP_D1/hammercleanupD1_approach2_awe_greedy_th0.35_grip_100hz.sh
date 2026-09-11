#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=16    # matches dataloader.num_workers=16 below
#SBATCH --mem=160G            # bumped from 80G after job 3937301 (5hz sibling) was oom-killed on this shared node
#SBATCH -p dheld
#SBATCH --gres=gpu:1
#SBATCH -w grogu-4-23         # pin to the RTX 6000 Blackwell node
#SBATCH -C 6000Blackwell      # pin to the RTX 6000 Blackwell node
#SBATCH -t 48:00:00
#SBATCH --job-name hammer-cleanup-d1-approach2-awe-greedy-th0.35-grip-100hz
#SBATCH -o /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/hammer-cleanup-d1-approach2-awe-greedy-th0.35-grip-100hz_job_%j.out
#SBATCH -e /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/hammer-cleanup-d1-approach2-awe-greedy-th0.35-grip-100hz_job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# ===========================================================================
# RATE ABLATION on HAMMER_CLEANUP_D1  --  APPROACH 2 (GMM-as-auxiliary-loss),
# goal_source=awe (greedy, err_th=0.35, WITH gripper), control rate=100Hz.
#
# Sibling of the other RATE_ABLATIONS/HAMMER_CLEANUP_D1/*.sh scripts -- same
# pipeline and goal_source, only the control-rate h5 source pool differs
# (RATE_H5_DIR below). Also a sibling, minus the rate axis, of
# ../../GROGU_LOCAL_A6000/hammercleanupD1_approach2_awe_greedy_th0.35_grip_grogu.sh.
#
# Unlike that script, the demo_N/ npz -> h5 conversion step is skipped
# entirely: /project_data/held/teswaram/data/rate_ablations/h5/hammer_cleanup_d1_100hz/
# is populated directly (demo_N.h5, goal_source-independent) by a separate
# rate-resampling job, so this script only needs to stage that pool and inject
# the awe-greedy-th0.35-grip keypoints on top of it, same as the other
# Approach2 scripts.
#
# NOTE (as of 2026-09-02): hammer_cleanup_d1_100hz/ is still being populated
# (partial coverage) while hammer_cleanup_d1_{5,10,50}hz/ are complete
# (.done_100 marker present, 100/100 demos). The staging step below only
# requires the h5/EXTRA_KEYPOINTS intersection to be non-empty, so this script
# is safe to launch against a partially-populated 100hz pool -- it will just
# train on however many demos are common at launch time. Re-run later (or wait
# for .done_100) for full 100-demo coverage.
# ===========================================================================
#     total_loss = flow_loss + c1 * gmm_loss
#
#   NUM_DEMOS=50 sbatch this_script.sh

set -euo pipefail
set -x

export PIXI_HOME="/project_data/held/teswaram/pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# --- the one knob these sibling scripts vary --------------------------------
GOAL_SOURCE="awe"
SOURCE_TAG="awe_greedy_th0.35_grip"
C1=0.1
RATE_HZ="100"

NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[config] goal_source=${GOAL_SOURCE} (${SOURCE_TAG}), c1=${C1}, rate=${RATE_HZ}hz, NUM_DEMOS cap=${NUM_DEMOS}"

# --- paths -------------------------------------------------------------------
RATE_H5_DIR="/project_data/held/teswaram/data/rate_ablations/h5/hammer_cleanup_d1_100hz"
EXTRA_GOALS_DIR="/project_data/held/teswaram/data/rate_ablations/npz/EXTRA_KEYPOINTS_awe-greedy-th0.35-grip/hammer_cleanup_d1_100hz"
REPO_DIR="/home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# --- resume from checkpoint --------------------------------------------------
RESUME_CKPT="${RESUME_CKPT:-}"

if [ ! -d "${RATE_H5_DIR}" ]; then
    echo "[error] RATE_H5_DIR not found: ${RATE_H5_DIR}" >&2
    exit 1
fi
if [ ! -d "${EXTRA_GOALS_DIR}" ]; then
    echo "[error] EXTRA_GOALS_DIR not found: ${EXTRA_GOALS_DIR}" >&2
    exit 1
fi

rate_h5_count=$(find "${RATE_H5_DIR}" -maxdepth 1 -name '*.h5' 2>/dev/null | wc -l)
echo "[src] ${rate_h5_count} h5 files present in ${RATE_H5_DIR} (100hz pool)"

# --- node-local scratch --------------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/scratch/teswaram/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/Hammer_Cleanup_D1_Approach2_${SOURCE_TAG}_${RATE_HZ}hz"

# --- stage demos present in BOTH the rate h5 dir and the EXTRA_KEYPOINTS tree ----
THREADS="${RSYNC_THREADS:-32}"
mkdir -p "${DEST_DATA_DIR}"

demos_h5=$(find "${RATE_H5_DIR}" -maxdepth 1 -name 'demo_*.h5' -printf '%f\n' | sed 's/\.h5$//' | sort)
demos_goals=$(find "${EXTRA_GOALS_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' -printf '%f\n' | sort)
demos_common=$(comm -12 <(echo "${demos_h5}") <(echo "${demos_goals}"))
n_common=$(echo -n "${demos_common}" | grep -c . || true)
echo "[stage] h5 demos: $(echo -n "${demos_h5}" | grep -c . || true), goal demos: $(echo -n "${demos_goals}" | grep -c . || true), common: ${n_common}"
if [ "${n_common}" -eq 0 ]; then
    echo "[stage] ERROR: no demo present in both RATE_H5_DIR and EXTRA_GOALS_DIR." >&2
    exit 1
fi

# apply NUM_DEMOS cap deterministically (sorted order)
demos_common=$(echo "${demos_common}" | head -n "${NUM_DEMOS}")
n_common=$(echo -n "${demos_common}" | grep -c . || true)

echo "[stage] source : ${RATE_H5_DIR}"
echo "[stage] dest   : ${DEST_DATA_DIR}"
echo "[stage] threads: ${THREADS}"
echo "[stage] staging ${n_common} demos (after NUM_DEMOS=${NUM_DEMOS} cap)"

stage_start=$(date +%s)

copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export RATE_H5_DIR DEST_DATA_DIR

echo "${demos_common}" | xargs -P "${THREADS}" -I {} \
    bash -c 'copy_one "${RATE_H5_DIR}/{}.h5" "${DEST_DATA_DIR}/"'

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} files, $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."
if [ "${staged_count}" -ne "${n_common}" ]; then
    echo "[stage] ERROR: expected ${n_common} files staged, got ${staged_count}." >&2
    exit 1
fi

# --- inject extra goal keys into the staged, node-local copy -----------------
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

# --- train ---------------------------------------------------------------------
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

RUN_NAME="Hammer_Cleanup_D1_APPROACH2_${SOURCE_TAG}_c1_${C1}_${staged_count}demo_${RATE_HZ}hz_dinov2_DIT_grogu"

USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/project_data/held/teswaram/logs/wandb_cache \
WANDB_DATA_DIR=/project_data/held/teswaram/logs/wandb_data \
PYTHONNOUSERSITE=1 \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_goal_gmm_workspace.yaml \
    task=MimicGen_Tasks/hammercleanup_D1_goal_gmm_aux \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    +task.dataset.goal_source=${GOAL_SOURCE} \
    policy.aux_gmm_loss_weight=${C1} \
    logging.project=mimicgen_tasks \
    logging.name=${RUN_NAME} \
    name=${RUN_NAME} \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    training.num_epochs=100 \
    dataloader.pin_memory=false \
    val_dataloader.pin_memory=false \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
