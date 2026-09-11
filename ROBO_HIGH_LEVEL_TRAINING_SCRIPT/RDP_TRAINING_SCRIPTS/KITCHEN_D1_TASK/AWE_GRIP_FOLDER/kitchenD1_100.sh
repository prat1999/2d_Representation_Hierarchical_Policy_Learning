#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 16:00:00 # Estimated time, 16hour max. DD-HH:MM.
#SBATCH --job-name kitchen-d1-high-level-awe-grip-100demo
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning/ROBO_HIGH_LEVEL_TRAINING_SCRIPT/RDP_TRAINING_SCRIPTS/KITCHEN_D1_TASK/AWE_GRIP_FOLDER/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning/ROBO_HIGH_LEVEL_TRAINING_SCRIPT/RDP_TRAINING_SCRIPTS/KITCHEN_D1_TASK/AWE_GRIP_FOLDER/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# 100-DEMO variant: train articubot (GMM cross-displacement high-level policy)
# on KITCHEN_D1 using AWE-grip goals (goal_gripper_pcd_awe) from the
# AWE_EXTRA_KEYPOINTS tree (awe-greedy-th0.35-grip), but only on the FIRST
# NUM_DEMOS demos (demo_0 .. demo_(N-1)). Run dirs carry a _${NUM_DEMOS}demo
# suffix so these are never confused with the full-dataset trainings.
#
# The AWE tree currently holds demo_0 .. demo_100 only; NUM_DEMOS > 101 fails
# the staging sanity check until more demos are uploaded.
#
# NUM_DEMOS defaults to 100. Override at submission time:
#   NUM_DEMOS=200 sbatch this_script.sh
#
# Training knobs (batch size, sampler, label swap, val/ckpt cadence) are kept
# identical to RDP_FOLDER/kitchenD1_100.sh so the goal sources are directly
# comparable.

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# "awe" -> NpyDataset reads goal_gripper_pcd_awe from extra_goals_dir/demo_X/t.npz.
# Must be listed in VALID_GOAL_SOURCES in src/lfd3d/datasets/npy/npy_dataset.py.
GOAL_SOURCE="awe"

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[demo_limit] using first NUM_DEMOS=${NUM_DEMOS} demos (demo_0 .. demo_$((NUM_DEMOS-1)))"

# --- paths ---------------------------------------------------------------
SRC_DATA_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/KITCHEN_D1"
EXTRA_SRC_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/EXTRA_KEYPOINTS/AWE_EXTRA_KEYPOINTS/EXTRA_KEYPOINTS_awe-greedy-th0.35-grip/KITCHEN_D1"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning"

# --- resume from checkpoint ----------------------------------------------
# Optional. Resumes full Lightning state (weights + optimizer + scheduler +
# epoch) via the +resume_from override in scripts/train.py. Empty (default)
# trains from scratch. To resume a preempted/cancelled run:
#   RESUME_CKPT=/path/to/periodic-epoch=epoch=NN.ckpt sbatch this_script.sh
RESUME_CKPT="${RESUME_CKPT:-}"

# Persistent cache dir for sampler weights / swap metadata, one per demo count
# so different NUM_DEMOS runs never share (length-incompatible) caches. Cache
# file names carry the goal-source tag (_awe), so this never collides with the
# RDP/random/fixed-interval caches even if the dir were shared.
CACHE_DIR="${EXTRA_SRC_DIR}/.transition_cache_${NUM_DEMOS}demo"
mkdir -p "${CACHE_DIR}"

# Pick a node-local scratch dir. We want the per-job isolated subdir
# (/local/slurm-<jobid>/local/) so concurrent jobs on the same node never
# collide. PSC's SLURM doesn't always pre-create that subtree on this cluster,
# so we force-create it ourselves when SLURM_JOB_ID is set.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/KITCHEN_D1_${NUM_DEMOS}demo"
DEST_EXTRA_DIR="${SCRATCH_ROOT}/KITCHEN_D1_extra_goals_awe_${NUM_DEMOS}demo"

# --- stage dataset (only NUM_DEMOS demo dirs) ----------------------------
# Parallel copy: split the requested demo dirs across N rsync workers via
# xargs -P. Override with RSYNC_THREADS=N env var.
THREADS="${RSYNC_THREADS:-32}"

echo "[stage] source : ${SRC_DATA_DIR}"
echo "[stage] extra  : ${EXTRA_SRC_DIR}"
echo "[stage] dest   : ${DEST_DATA_DIR}"
echo "[stage] extra dest: ${DEST_EXTRA_DIR}"
echo "[stage] threads: ${THREADS}"
echo "[stage] dirs   : demo_0 .. demo_$((NUM_DEMOS-1))  (${NUM_DEMOS} dirs)"
mkdir -p "${DEST_DATA_DIR}" "${DEST_EXTRA_DIR}"

stage_start=$(date +%s)

# Per-entry rsync wrapper. Exit code 24 ("vanished files") is a benign warning —
# usually rsync's own temp files (.<name>.XXXXXX) left behind by a previously
# interrupted sync. Treat it as success so it doesn't trip xargs/set -e.
copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export SRC_DATA_DIR DEST_DATA_DIR EXTRA_SRC_DIR DEST_EXTRA_DIR

# Generate exactly the demo dir names we want and feed to xargs. Each rsync
# handles one top-level demo_* dir and stays resumable per-entry, so re-running
# the script skips already-copied demos cheaply.
seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DATA_DIR}/$1" "${DEST_DATA_DIR}/"' _ {}

# Stage the matching AWE extra-goal demo dirs (small; ~16KB per frame).
seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${EXTRA_SRC_DIR}/$1" "${DEST_EXTRA_DIR}/"' _ {}

# Sanity check: every staged demo dir must actually contain .npz frames — an
# empty demo dir would silently shrink the training set, so fail fast.
staged_count=$(find "${DEST_DATA_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' | wc -l)
nonempty_count=$(
    find "${DEST_DATA_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' \
        -exec sh -c 'ls "$1"/*.npz >/dev/null 2>&1' _ {} \; -print | wc -l
)
extra_nonempty_count=$(
    find "${DEST_EXTRA_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' \
        -exec sh -c 'ls "$1"/*.npz >/dev/null 2>&1' _ {} \; -print | wc -l
)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} dirs (${nonempty_count} with npz, ${extra_nonempty_count} extra-goal dirs with npz), $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."
if [ "${staged_count}" -ne "${NUM_DEMOS}" ]; then
    echo "[stage] ERROR: expected ${NUM_DEMOS} demo dirs staged, got ${staged_count}." >&2
    exit 1
fi
if [ "${nonempty_count}" -ne "${NUM_DEMOS}" ]; then
    echo "[stage] ERROR: ${NUM_DEMOS} demo dirs requested but only ${nonempty_count} contain .npz frames." >&2
    exit 1
fi
if [ "${extra_nonempty_count}" -ne "${NUM_DEMOS}" ]; then
    echo "[stage] ERROR: only ${extra_nonempty_count}/${NUM_DEMOS} AWE extra-goal demo dirs contain .npz frames — the ${GOAL_SOURCE} goal tree is incomplete (still uploading?)." >&2
    exit 1
fi

# --- train ---------------------------------------------------------------
cd "${REPO_DIR}"

# Build the hydra resume override only if a checkpoint was requested.
RESUME_ARGS=()
if [ -n "${RESUME_CKPT}" ]; then
    echo "[resume] resuming training from ${RESUME_CKPT}"
    if [ ! -f "${RESUME_CKPT}" ]; then
        echo "[resume] ERROR: checkpoint not found: ${RESUME_CKPT}" >&2
        exit 1
    fi
    # Hydra-level single quotes are required: the checkpoint filename contains
    # '=' (periodic-epoch=epoch=NN.ckpt), which otherwise breaks the override
    # grammar ("mismatched input '=' expecting <EOF>").
    RESUME_ARGS=("+resume_from='${RESUME_CKPT}'")
else
    echo "[resume] RESUME_CKPT empty -> training from scratch"
fi

USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/wandb_cache \
WANDB_DATA_DIR=/ocean/projects/cis240052p/pbhowal/wandb_data \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python scripts/train.py \
    model=articubot \
    dataset=Kitchen \
    dataset.data_dir="${DEST_DATA_DIR}" \
    model.use_rgb=False \
    model.in_channels=4 \
    training.batch_size=164 \
    wandb.entity=pbhowal-carnegie-mellon-university \
    "hydra.run.dir=logs/train_KITCHEN_D1_GOAL_SWAP_AWE_GRIP_${NUM_DEMOS}demo/$(date +%Y-%m-%d/%H-%M-%S)" \
    +dataset.goal_source="${GOAL_SOURCE}" \
    +dataset.extra_goals_dir="${DEST_EXTRA_DIR}" \
    +dataset.transition_cache_dir="${CACHE_DIR}" \
    +dataset.use_weighted_sampler=True \
    +dataset.transition_p=0.5 \
    +dataset.transition_radius=5 \
    +dataset.transition_label_swap=True \
    +dataset.transition_swap_p_max=0.5 \
    "resources.gpus=[0]" \
    +training.checkpoint_every_n_epochs=5 \
    training.check_val_every_n_epochs=15 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
