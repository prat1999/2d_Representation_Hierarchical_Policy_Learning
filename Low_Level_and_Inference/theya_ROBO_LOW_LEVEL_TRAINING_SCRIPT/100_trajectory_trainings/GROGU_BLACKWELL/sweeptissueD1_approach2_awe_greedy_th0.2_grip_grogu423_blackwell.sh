#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=16    # matches dataloader.num_workers=16 below
#SBATCH --mem=160G            # raised from 80G -- job 3935697 was cgroup-OOM-killed mid-epoch-0 despite 80G working for sibling tasks; node has ~700G free so plenty of headroom
#SBATCH -p dheld
#SBATCH -w grogu-4-23         # pin to the RTX 6000 Blackwell node
#SBATCH --gres=gpu:1
#SBATCH -C 6000Blackwell      # grogu-4-23 is an 8x RTX 6000 Blackwell node, not A6000 -- match the node's AVAIL_FEATURES
#SBATCH -t 24:00:00
#SBATCH --job-name sweep-tissue-d1-approach2-awe-greedy-th0.2-grip-grogu423-blackwell
#SBATCH -o /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/sweep-tissue-d1-approach2-awe-greedy-th0.2-grip-grogu423-blackwell_job_%j.out
#SBATCH -e /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/sweep-tissue-d1-approach2-awe-greedy-th0.2-grip-grogu423-blackwell_job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# ===========================================================================
# APPROACH 2 (GMM-as-auxiliary-loss) on SWEEP_TISSUE_D1  —  goal_source=awe (greedy, err_th=0.2, WITH gripper)
# Pinned to grogu-4-23 (8x RTX 6000 Blackwell). Modeled on the sibling
# ../GROGU_BLACKWELL/hammercleanupD1_approach2_awe_greedy_th0.35_grip_grogu423_blackwell.sh --
# same pipeline; only task/paths/SOURCE_TAG (th0.2 for this dataset) differ.
#
# Uses the pre-converted h5 pool at /home/teswaram/data/D1/sweep_tissue_h5
# directly (demo_*.h5 already present, no NO_GMM npz->h5 conversion needed).
# The h5/goals intersection used for staging naturally caps at however many
# demos exist in both trees at run time.
# ===========================================================================

set -euo pipefail
set -x

export PIXI_HOME="/project_data/held/teswaram/pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# --- the one knob these sibling scripts vary ------------------------------
GOAL_SOURCE="awe"
SOURCE_TAG="awe_greedy_th0.2_grip"
C1=0.1

echo "[config] goal_source=${GOAL_SOURCE} (${SOURCE_TAG}), c1=${C1}"

# --- paths -----------------------------------------------------------------
EXTRA_GOALS_DIR="/project_data/held/teswaram/data/D1/EXTRA_KEYPOINTS_sweeptissue_awe-greedy-th0.2-grip/sweep_tissue_npz"
NO_GMM_H5_DIR="/home/teswaram/data/D1/sweep_tissue_h5"
REPO_DIR="/home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# --- resume from checkpoint --------------------------------------------------
RESUME_CKPT="${RESUME_CKPT:-}"

if [ ! -d "${EXTRA_GOALS_DIR}" ]; then
    echo "[error] EXTRA_GOALS_DIR not found: ${EXTRA_GOALS_DIR}" >&2
    exit 1
fi

existing_h5_count=$(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" 2>/dev/null | wc -l)
echo "[gen] using pre-converted h5 pool ${NO_GMM_H5_DIR}: ${existing_h5_count} files"
if [ "${existing_h5_count}" -eq 0 ]; then
    echo "[gen] ERROR: no h5 files found in ${NO_GMM_H5_DIR}" >&2
    exit 1
fi

# --- node-local scratch ------------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/scratch/teswaram/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/Sweep_Tissue_D1_Approach2_${SOURCE_TAG}"

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
echo "[inject] ensuring obs/goal_gripper_pts_awe exists in staged demos (from EXTRA_KEYPOINTS_sweeptissue_awe-greedy-th0.2-grip)"
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

RUN_NAME="Sweep_Tissue_D1_APPROACH2_${SOURCE_TAG}_c1_${C1}_${staged_count}demo_dinov2_DIT_grogu423_blackwell"

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
    +task.dataset.goal_source=${GOAL_SOURCE} \
    policy.aux_gmm_loss_weight=${C1} \
    logging.project=mimicgen_tasks \
    logging.name=${RUN_NAME} \
    name=${RUN_NAME} \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
