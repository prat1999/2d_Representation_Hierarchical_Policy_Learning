#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=16    # matches dataloader.num_workers=16 below -- was 12, which under-provisioned the 16 workers and (combined with co-located jobs) OOM-killed the sibling kitchen job (3935329) at start of epoch 1; applied here preventatively for any future submit/resume of this script
#SBATCH --mem=80G             # explicit floor above DefMemPerCPU(3.5G)*cpus, so worker/page-cache spikes don't hit the cgroup limit
#SBATCH -p all
#SBATCH --gres=gpu:1
#SBATCH -C A6000              # real RTX A6000 48GB nodes (grogu-1-10/25/30, grogu-2-5) live on partition "all", not "dheld" -- dheld nodes are heterogeneous (3090/24GB, mislabeled "A6000" node actually 3080ti/12GB, 2080ti) so this pins to genuine 48GB cards, matching the batch_size=128 sizing
#SBATCH -t 24:00:00       # partition "all" caps walltime at 24h; use RESUME_CKPT to continue in a follow-up job if more time is needed
#SBATCH --job-name coffee-prep-d1-approach2-awe-greedy-th0.35-grip-grogu
#SBATCH -o /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/coffee-prep-d1-approach2-awe-greedy-th0.35-grip-grogu_job_%j.out
#SBATCH -e /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/coffee-prep-d1-approach2-awe-greedy-th0.35-grip-grogu_job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# ===========================================================================
# APPROACH 2 (GMM-as-auxiliary-loss) on COFFEE_PREPERATION_D1  —  goal_source=awe (greedy, err_th=0.35, WITH gripper)
# grogu cluster / dheld partition (8x A6000 48GB per node). Sibling of
# coffee_preperationD1_approach2_awe_greedy_th0.35_grogu.sh (the "_nogrip" variant) --
# see that script's header for the full design rationale (GMM-as-aux-loss).
# This script differs only in EXTRA_GOALS_DIR / SOURCE_TAG: it points at the
# newer EXTRA_KEYPOINTS_awe-greedy-th0.35-grip tree (gripper state included
# in the keypoints), which as of 2026-09-01 has full 100-demo coverage for
# this task.
# ===========================================================================
#     total_loss = flow_loss + c1 * gmm_loss
#
#   - The raw demo_N/ npz -> h5 conversion is goal_source-independent, so this
#     script reuses the SAME LOCAL_NO_GMM_H5_DIR as the _nogrip sibling
#     script (skips reconversion if that pool already exists).
#   - Only the first 100 demos are used: NUM_DEMOS caps h5 conversion to the
#     first 100 demo_N/ dirs, and the h5-pool/goals-tree intersection used for
#     staging naturally caps at 100 even though the grip keypoints tree has
#     101 demos for COFFEE_PREPERATION_D1.
#   - GPU is an A6000 (48GB) not an H100 (80GB). Measured scaling for this
#     architecture is ~0.24 GiB/sample over a ~2.75 GiB floor, i.e. ~34 GiB at
#     batch_size=128 -- comfortable headroom under 48GB, so batch size is left
#     at 128. Drop it (e.g. 64) if you see CUDA OOM.
#
#   NUM_DEMOS=50 sbatch this_script.sh

set -euo pipefail
set -x

export PIXI_HOME="/project_data/held/teswaram/pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# --- the one knob these sibling scripts vary ------------------------------
GOAL_SOURCE="awe"
SOURCE_TAG="awe_greedy_th0.35_grip"
C1=0.1

NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[config] goal_source=${GOAL_SOURCE} (${SOURCE_TAG}), c1=${C1}, NUM_DEMOS cap=${NUM_DEMOS}"

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
DEST_DATA_DIR="${SCRATCH_ROOT}/Coffee_Preperation_D1_Approach2_${SOURCE_TAG}"

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

RUN_NAME="Coffee_Preperation_D1_APPROACH2_${SOURCE_TAG}_c1_${C1}_${staged_count}demo_dinov2_DIT_grogu"

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
    logging.project=mimicgen_tasks \
    logging.name=${RUN_NAME} \
    name=${RUN_NAME} \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
