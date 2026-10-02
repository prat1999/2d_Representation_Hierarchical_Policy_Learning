#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 30:00:00 # fresh 100-epoch budget: ~17.3 min/epoch on H100
                    # (alpha runs: 100 epochs in ~29h) + margin.
#SBATCH --job-name kitchen-d1-wca-beta3-ablation
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# BETA ABLATION on KITCHEN_D1 (100 demos, DINOv2, serial WCA) — BETA=3 variant.
#
# Same treatment as the alpha ablation scripts one directory up, but the swept
# knob is the NEW prior-temperature beta on the WCA log-weight term:
#
#   logits = ALPHA * QK^T/sqrt(d) + BETA * log(w_j)
#
# beta*log(w) = log(w^beta), so beta is a temperature on the high-level GMM
# prior: beta=1 is the original formula (all alpha-sweep runs), beta>1 SHARPENS
# the prior toward hard top-1 selection by GMM weight, beta<1 flattens it,
# beta=0 removes it entirely (pure content cross-attention). This family
# (beta = 2, 3, 4) probes "trust a sharpened high-level prior" while content
# attention stays at FULL strength: ALPHA defaults to 1.0 here.
#
# Launch matrix (env overrides, same pattern as the alpha scripts):
#   sbatch this_script.sh                       # top-2500, alpha=1.0, beta=3
#   BETA=2 sbatch this_script.sh                # top-2500, beta=2
#   BETA=1 sbatch this_script.sh                # reproduces the original formula
#   TOP_K=6 sbatch this_script.sh               # sharpened prior on the old top-6
#
# MEMORY NOTE: goal tokens + per-block WCA K/V activations scale linearly in
# TOP_K * batch_size. At TOP_K=2500 the extra fp32 activation memory is roughly
# ~25 GB at batch 64 and ~50 GB at batch 128 — batch 128 is borderline on an
# 80 GB H100, so BATCH_SIZE defaults to 64 here. Will NOT fit a V100-32.

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- ablation knobs -------------------------------------------------------
BETA="${BETA:-3}"              # wca_beta: prior temperature on log(w) (1.0 = original)
ALPHA="${ALPHA:-1.0}"          # wca_alpha value (used only when USE_ALPHA=true)
USE_ALPHA="${USE_ALPHA:-true}" # false => content term at full strength (same as alpha=1)
TOP_K="${TOP_K:-2500}"         # gmm_top_k candidates passed to the policy
BATCH_SIZE="${BATCH_SIZE:-64}" # see MEMORY NOTE above
NUM_DEMOS="${NUM_DEMOS:-100}"

if [ "${USE_ALPHA}" = "true" ]; then
    ALPHA_TAG="alpha${ALPHA}"
else
    ALPHA_TAG="noalpha"
fi
RUN_NAME="groot_GMM_WCA_${NUM_DEMOS}demo_dinov2_Kitchen_D1_top${TOP_K}_${ALPHA_TAG}_beta${BETA}_bs${BATCH_SIZE}"

echo "[ablation] TOP_K=${TOP_K}  USE_ALPHA=${USE_ALPHA}  ALPHA=${ALPHA}  BETA=${BETA}  BATCH_SIZE=${BATCH_SIZE}"
echo "[demo_limit] using first NUM_DEMOS=${NUM_DEMOS} demos (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

# --- resume from checkpoint ----------------------------------------------
# Resume the full training state (model + EMA + optimizer + epoch counter).
# num_epochs is an ABSOLUTE target (resumed runs stop at 100, not +100).
# Defaults to the last checkpoint of the 2026.08.15 beta3 run (epoch_80).
# Override at submission time (RESUME_CKPT= for a fresh run):
#   RESUME_CKPT=/path/to/epoch_N.ckpt sbatch this_script.sh
RESUME_CKPT="${RESUME_CKPT:-/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/outputs/2026.08.15/06.43.09_groot_GMM_WCA_100demo_dinov2_Kitchen_D1_top2500_alpha1.0_beta3_bs64_kitchen_D1_gmm_goal/checkpoints/epoch_80.ckpt}"

# Tag the W&B run so a resume leg is distinguishable from the original.
if [ -n "${RESUME_CKPT}" ]; then
    RESUME_TAG="_resumeE$(basename "${RESUME_CKPT}" .ckpt | grep -oE '[0-9]+' || echo X)"
else
    RESUME_TAG=""
fi
RUN_NAME="${RUN_NAME}${RESUME_TAG}"
echo "[ablation] final run name: ${RUN_NAME}"

# --- paths ---------------------------------------------------------------
SRC_DATA_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/LOW_LEVEL_WITH_GMM_DATASET_GROOT_STYLE_DATASET/D2/KITCHEN_D1"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# Pick a node-local scratch dir. Always prefer the per-job isolated subdir
# (/local/slurm-<jobid>/local/) so SLURM auto-cleans on job end and concurrent
# jobs on the same node never collide.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/KITCHEN_D1_Low_Level_${NUM_DEMOS}demo"

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
    task=MimicGen_Tasks/kitchen_D1_gmm_goal \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    visual_encoder=dinov2 \
    policy.use_goal_cross_attention=true \
    policy.use_weighted_cross_attention=true \
    policy.use_alpha="${USE_ALPHA}" \
    policy.wca_alpha="${ALPHA}" \
    policy.wca_beta="${BETA}" \
    policy.gmm_top_k="${TOP_K}" \
    logging.project=MimicGen_GMM_Low_Level_Policy \
    logging.name="${RUN_NAME}" \
    name="${RUN_NAME}" \
    training.checkpoint_every=10 \
    training.num_epochs=100 \
    dataloader.batch_size="${BATCH_SIZE}" \
    dataloader.num_workers=16 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
