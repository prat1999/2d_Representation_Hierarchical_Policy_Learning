#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 48:00:00 # 48-hour budget
#SBATCH --job-name coffee-prep-d1-c1-0.03
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# ===========================================================================
# APPROACH 2 on COFFEE_PREPERATION_D1  —  c1 = 0.03
# ===========================================================================
# The low-level policy is NOT given the goal. Instead goal_gripper_pts
# supervises a GMM head that reads the same 3D-grounded visual tokens the DiT
# cross-attends to, so the goal shapes the visual representation rather than
# being an input.
#
#     total_loss = flow_loss + c1 * gmm_loss
#
# c1 = policy.aux_gmm_loss_weight, set at the bottom of this script. It is the
# ONLY thing that differs between the sibling scripts in this folder.
#
# Choosing c1 — measured at init on this dataset:
#     fm_loss  ~ 1.37     |grad_fm  -> shared trunk| ~ 1.47
#     gmm_loss ~ 10.05    |grad_gmm -> shared trunk| ~ 3.78  (per unit c1)
#   c1 = 0.136 equalises the two loss magnitudes
#   c1 = 0.389 equalises the gradient each delivers to the shared trunk
#   c1 = 0.1   -> auxiliary gradient is ~26% of the flow gradient: a real
#                 signal, but subordinate to the actual task. Start here.
# Both terms are logged separately as train_fm_loss / train_goal_gmm_loss, so
# the real balance is visible during the run. fm_loss falls faster than
# gmm_loss, so the auxiliary term gets relatively stronger over training.
#
# Uses the SAME NO_GMM h5 dataset as the goal_gripper baseline; the extra keys
# this approach needs (cam*_depth, cam*_intrinsic, cam*_extrinsic,
# present_gripper_pts) are already present in those files.
#
# Trains from scratch — the architecture differs from the goal_gripper baseline,
# so those checkpoints will not load. At ~550 steps/epoch (batch 128, 100 demos)
# the full 100 epochs is expected to fit inside the 48-hour budget in a single
# job. checkpoint_every=5, so if it does not, resume with:
#   RESUME_CKPT=/path/to/epoch_N.ckpt sbatch this_script.sh
#
# NUM_DEMOS defaults to 100. Override at submission time:
#   NUM_DEMOS=200 sbatch this_script.sh

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- auxiliary loss weight (the one knob these sibling scripts vary) -----
C1=0.03

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[config] c1=${C1}, NUM_DEMOS=${NUM_DEMOS} (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

# --- paths ---------------------------------------------------------------
NO_GMM_H5_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/LOW_LEVEL_GROOT_TRAINING_DATASET/NO_GMM_DATASET/COFFEE_PREPERATION_D1"
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
DEST_DATA_DIR="${SCRATCH_ROOT}/Coffee_Preperation_D1_Approach2_${NUM_DEMOS}demo"

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

RUN_NAME="coffee_prep_D1_approach2_c1_${C1}_${NUM_DEMOS}demo"

# batch_size=128 matches the goal_gripper baseline. Measured scaling for this
# architecture is ~0.24 GiB/sample over a ~2.75 GiB floor, i.e. ~34 GiB at 128 —
# comfortable on an 80 GB H100.
#
# Keep every knob below identical across the sibling c1 scripts; a comparison
# across differing batch size / epochs / seed says nothing about c1.
USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/wandb_cache \
WANDB_DATA_DIR=/ocean/projects/cis240052p/pbhowal/wandb_data \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_goal_gmm_workspace.yaml \
    task=MimicGen_Tasks/coffee_preperation_goal_gmm_aux \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    policy.aux_gmm_loss_weight=${C1} \
    logging.project=mimicgen_tasks \
    logging.name=${RUN_NAME} \
    name=${RUN_NAME} \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
