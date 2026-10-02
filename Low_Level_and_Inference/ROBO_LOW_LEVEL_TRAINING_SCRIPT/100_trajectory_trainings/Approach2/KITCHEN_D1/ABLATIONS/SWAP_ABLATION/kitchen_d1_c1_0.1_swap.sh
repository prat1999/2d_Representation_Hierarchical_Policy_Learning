#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 48:00:00 # 48-hour budget
#SBATCH --job-name kitchen-d1-c1-0.1-swap
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# ===========================================================================
# APPROACH 2 on KITCHEN_D1 — c1 = 0.1 + Approach 1's TRANSITION LABEL SWAP
# ===========================================================================
# Identical to Approach2/KITCHEN_D1/BAYESIAN_FOLDER/kitchen_d1_c1_0.1.sh EXCEPT
# the task is kitchen_goal_gmm_aux_swap, whose dataset subclass
# (LazyArticuBotSwapDataset) stochastically swaps the aux GMM target
# goal_gripper_pts to the goal across the nearest transition — the exact
# Approach 1 swap: linear triangular profile, p_max=0.5, radius=5, per-frame
# Bernoulli, train only. The DiT never sees the goal (Approach 2), so the swap
# affects ONLY the auxiliary GMM head's supervision.
#
# Ablation question: does the aux head need the explicit 50/50 flip to go
# bimodal at changepoints, or does dataset-level ambiguity suffice? Baseline =
# the plain c1_0.1 run (same data, same seed, same every-other-knob).
#
# NUM_DEMOS defaults to 100. Override at submission time:
#   NUM_DEMOS=200 sbatch this_script.sh

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- auxiliary loss weight (matches the baseline being ablated against) ---
C1=0.1

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[config] c1=${C1}, NUM_DEMOS=${NUM_DEMOS} (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5), SWAP=linear p_max=0.5 r=5"

# --- paths ---------------------------------------------------------------
NO_GMM_H5_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/LOW_LEVEL_GROOT_TRAINING_DATASET/NO_GMM_DATASET/Kitchen_D1"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

RESUME_CKPT="${RESUME_CKPT:-}"

existing_h5_count=$(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" 2>/dev/null | wc -l)
echo "[data] existing *.h5 in ${NO_GMM_H5_DIR}: ${existing_h5_count}"
if [ "${existing_h5_count}" -lt "${NUM_DEMOS}" ]; then
    echo "[data] ERROR: need ${NUM_DEMOS} h5 files, found ${existing_h5_count}." >&2
    echo "[data] Run generate_non_gmm_goals_for_low_level.py --no_gmm first." >&2
    exit 1
fi

# --- node-local scratch --------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
SRC_DATA_DIR="${NO_GMM_H5_DIR}"
DEST_DATA_DIR="${SCRATCH_ROOT}/Kitchen_D1_Approach2_SWAP_${NUM_DEMOS}demo"

# --- stage dataset (only NUM_DEMOS files) --------------------------------
THREADS="${RSYNC_THREADS:-32}"
echo "[stage] source : ${SRC_DATA_DIR}"
echo "[stage] dest   : ${DEST_DATA_DIR}"
mkdir -p "${DEST_DATA_DIR}"
stage_start=$(date +%s)

copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export SRC_DATA_DIR DEST_DATA_DIR

seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1 ".h5"}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DATA_DIR}/$1" "${DEST_DATA_DIR}/"' _ {}

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
echo "[stage] done in $(( $(date +%s) - stage_start ))s. ${staged_count} files, $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."
if [ "${staged_count}" -ne "${NUM_DEMOS}" ]; then
    echo "[stage] ERROR: expected ${NUM_DEMOS} files staged, got ${staged_count}." >&2
    exit 1
fi

# --- train ---------------------------------------------------------------
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

RUN_NAME="kitchen_D1_APPROACH2_c1_${C1}_SWAP_${NUM_DEMOS}demo"

# Keep every knob identical to the plain c1_0.1 baseline; the swap is the only
# difference, so the comparison isolates it.
USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/wandb_cache \
WANDB_DATA_DIR=/ocean/projects/cis240052p/pbhowal/wandb_data \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_goal_gmm_workspace.yaml \
    task=MimicGen_Tasks/kitchen_goal_gmm_aux_swap \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    policy.aux_gmm_loss_weight=${C1} \
    logging.project=mimicgen_tasks \
    logging.name=${RUN_NAME} \
    name=${RUN_NAME} \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
