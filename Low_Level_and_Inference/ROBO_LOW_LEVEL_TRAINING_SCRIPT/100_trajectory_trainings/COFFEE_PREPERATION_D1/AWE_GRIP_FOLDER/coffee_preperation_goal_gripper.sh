#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 16:00:00 # fresh 100-epoch run: the RDP variant did 86 epochs in ~11.2h
                    # (~7.8 min/epoch), so 100 epochs ~= 13h + inject/stage margin.
#SBATCH --job-name coffee-prep-d1-goal-gripper-awe-grip-100demo-dinov2
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/100_trajectory_trainings/COFFEE_PREPERATION_D1/AWE_GRIP_FOLDER/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/100_trajectory_trainings/COFFEE_PREPERATION_D1/AWE_GRIP_FOLDER/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# 100-demo goal_gripper baseline on COFFEE_PREPERATION_D1, conditioned on AWE
# goals (obs/goal_gripper_pts_awe; AWE greedy, th=0.35, gripper-aware) instead
# of the default goal_gripper_pts.
#
# Same pipeline as RDP_FOLDER/coffee_preperation_goal_gripper.sh, but the
# injection step reads the AWE npz tree
#   EXTRA_KEYPOINTS/AWE_EXTRA_KEYPOINTS/EXTRA_KEYPOINTS_awe-greedy-th0.35-grip/COFFEE_PREPERATION_D1
# and injects ONLY the awe key (--extra_goal_sources awe) into the first
# NUM_DEMOS NO_GMM h5 files. Idempotent/append-only: files that already carry
# obs/goal_gripper_pts_awe are skipped. Training selects the key via
# +task.dataset.goal_source=awe.
#
# The AWE tree currently holds demo_0 .. demo_99 only, so NUM_DEMOS > 100 fails
# the pre-check below until more demos are uploaded.
#
# NUM_DEMOS defaults to 100. Override at submission time:
#   NUM_DEMOS=200 sbatch this_script.sh

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# h5 key suffix (obs/goal_gripper_pts_awe) and npz key (goal_gripper_pcd_awe).
# Must be listed in VALID_GOAL_SOURCES (lazy_articubot_dataset.py) and
# EXTRA_GOAL_SOURCES (generate_non_gmm_goals_for_low_level.py).
GOAL_SOURCE="awe"
# Human-facing tag for run / W&B names (matches the high-level AWE_GRIP runs).
RUN_TAG="awe_grip"

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[demo_limit] using first NUM_DEMOS=${NUM_DEMOS} demos (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

# --- paths ---------------------------------------------------------------
SRC_NPZ_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/COFFEE_PREPERATION_D1"
EXTRA_GOALS_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/EXTRA_KEYPOINTS/AWE_EXTRA_KEYPOINTS/EXTRA_KEYPOINTS_awe-greedy-th0.35-grip/COFFEE_PREPERATION_D1"
NO_GMM_H5_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/LOW_LEVEL_GROOT_TRAINING_DATASET/NO_GMM_DATASET/COFFEE_PREPERATION_D1"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# --- resume from checkpoint ----------------------------------------------
# Empty by default: a fresh goal-source variant should NOT resume from another
# variant's checkpoint. Override at submission time if resuming a previous run
# OF THIS SAME VARIANT (training.num_epochs=100 below is an absolute stop):
#   RESUME_CKPT=/path/to/epoch_X.ckpt sbatch this_script.sh
RESUME_CKPT="${RESUME_CKPT:-}"

# Tag the W&B run so a resume leg is distinguishable from the original.
if [ -n "${RESUME_CKPT}" ]; then
    RESUME_TAG="_resumeE$(basename "${RESUME_CKPT}" .ckpt | grep -oE '[0-9]+' || echo X)"
else
    RESUME_TAG=""
fi

# --- pre-check: AWE tree must cover the requested demos --------------------
# The AWE tree is still being uploaded; fail fast before touching any h5 file.
awe_demo_count=0
for i in $(seq 0 $((NUM_DEMOS - 1))); do
    ls "${EXTRA_GOALS_DIR}/demo_${i}"/*.npz >/dev/null 2>&1 && awe_demo_count=$((awe_demo_count + 1))
done
echo "[precheck] AWE demo dirs with npz: ${awe_demo_count}/${NUM_DEMOS} in ${EXTRA_GOALS_DIR}"
if [ "${awe_demo_count}" -ne "${NUM_DEMOS}" ]; then
    echo "[precheck] ERROR: AWE tree is incomplete for demo_0..demo_$((NUM_DEMOS-1)) (still uploading?)." >&2
    exit 1
fi

# --- generate npz -> h5 if missing ---------------------------------------
# Conversion is per-demo idempotent. We trigger it only if the NO_GMM h5 dir
# doesn't already have at least NUM_DEMOS files (demo_0.h5 .. demo_(N-1).h5
# need to exist for the staging step to succeed).
src_demo_count=$(find "${SRC_NPZ_DIR}" -mindepth 1 -maxdepth 1 -type d -name "demo_*" 2>/dev/null | wc -l)
existing_h5_count=0
if [ -d "${NO_GMM_H5_DIR}" ]; then
    existing_h5_count=$(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" 2>/dev/null | wc -l)
fi
echo "[gen] src demo_*/ in ${SRC_NPZ_DIR}: ${src_demo_count}"
echo "[gen] existing *.h5 in ${NO_GMM_H5_DIR}: ${existing_h5_count}"

if [ "${existing_h5_count}" -ge "${NUM_DEMOS}" ]; then
    echo "[gen] already have ${existing_h5_count} h5 files (≥ NUM_DEMOS=${NUM_DEMOS}) → skipping h5 generation"
else
    echo "[gen] converting demo_*/ → demo_*.h5 (need at least NUM_DEMOS=${NUM_DEMOS}; have ${existing_h5_count})"
    mkdir -p "${NO_GMM_H5_DIR}"
    (
        cd "${REPO_DIR}"
        USE_TF=0 \
        GIT_LFS_SKIP_SMUDGE=1 \
        PYTHONNOUSERSITE=1 \
        PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
        pixi run python generate_non_gmm_goals_for_low_level.py \
            --dataset_dir "${SRC_NPZ_DIR}" \
            --no_gmm \
            --no_gmm_output_dir "${NO_GMM_H5_DIR}"
    )
    echo "[gen] done. *.h5 count now: $(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" | wc -l)"
fi

# --- inject the awe goal key -----------------------------------------------
# Idempotent and append-only: files that already have obs/goal_gripper_pts_awe
# are skipped. Runs on the /ocean h5 files BEFORE staging so the scratch copies
# carry the key. Frames are aligned by sorted order (h5 timestep t <-> t-th
# sorted npz), which is how the h5 files were built.
echo "[inject] ensuring obs/goal_gripper_pts_${GOAL_SOURCE} exists in first ${NUM_DEMOS} h5 files"
(
    cd "${REPO_DIR}"
    USE_TF=0 \
    GIT_LFS_SKIP_SMUDGE=1 \
    PYTHONNOUSERSITE=1 \
    PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
    pixi run python generate_non_gmm_goals_for_low_level.py \
        --dataset_dir "${NO_GMM_H5_DIR}" \
        --inject_extra_goals \
        --extra_goals_dir "${EXTRA_GOALS_DIR}" \
        --extra_goal_sources "${GOAL_SOURCE}" \
        --max_files "${NUM_DEMOS}"
)

# --- node-local scratch --------------------------------------------------
# Pick a node-local scratch dir. Always prefer the per-job isolated subdir
# (/local/slurm-<jobid>/local/) so SLURM auto-cleans on job end and concurrent
# jobs on the same node never collide. PSC's SLURM doesn't always pre-create
# that subtree, so we force-create it ourselves when SLURM_JOB_ID is set. Fall
# back to $LOCAL or /tmp only if no per-job dir exists.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
SRC_DATA_DIR="${NO_GMM_H5_DIR}"
DEST_DATA_DIR="${SCRATCH_ROOT}/Coffee_Preperation_D1_Low_Level_${NUM_DEMOS}demo"

# --- stage dataset (only NUM_DEMOS files) --------------------------------
THREADS="${RSYNC_THREADS:-32}"

echo "[stage] source : ${SRC_DATA_DIR}"
echo "[stage] dest   : ${DEST_DATA_DIR}"
echo "[stage] threads: ${THREADS}"
echo "[stage] files  : demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5  (${NUM_DEMOS} files)"
mkdir -p "${DEST_DATA_DIR}"

stage_start=$(date +%s)

# Per-entry rsync wrapper. Exit code 24 (vanished files) is benign.
copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export SRC_DATA_DIR DEST_DATA_DIR

# Generate exactly the demo filenames we want and feed to xargs.
seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1 ".h5"}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DATA_DIR}/$1" "${DEST_DATA_DIR}/"' _ {}

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} files, $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."
if [ "${staged_count}" -ne "${NUM_DEMOS}" ]; then
    echo "[stage] ERROR: expected ${NUM_DEMOS} files staged, got ${staged_count}." >&2
    exit 1
fi

# --- train ---------------------------------------------------------------
cd "${REPO_DIR}"

# Build hydra resume overrides only if a checkpoint was requested. The '+' on
# resume_ckpt_path is required because that key isn't in the base config.
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

USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/wandb_cache \
WANDB_DATA_DIR=/ocean/projects/cis240052p/pbhowal/wandb_data \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_workspace.yaml \
    task=MimicGen_Tasks/coffee_preperation_goal_gripper \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    +task.dataset.goal_source=${GOAL_SOURCE} \
    visual_encoder=dinov2 \
    logging.project=mimicgen_tasks \
    logging.name=coffee_preperation_D1_goal_gripper_${RUN_TAG}_${NUM_DEMOS}demo_dinov2_DIT${RESUME_TAG} \
    name=coffee_preperation_D1_goal_gripper_${RUN_TAG}_${NUM_DEMOS}demo_dinov2_DIT${RESUME_TAG} \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    training.num_epochs=100 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
