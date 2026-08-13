#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12    # 12 CPU cores for npz load + npz write workers
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100 (needed by the high-level GMM forward pass)
#SBATCH -t 12:00:00
#SBATCH --job-name push-t-sigmoid-tau2.0-pred-gen
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning/ROBO_GMM_DATASET_GEN_SCRIPT/ABLATION/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning/ROBO_GMM_DATASET_GEN_SCRIPT/ABLATION/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# Generate per-frame GMM prediction npz files for PushT_Task from the
# SIGMOID-SWAP ablation high-level checkpoint (goal swap augmentation with
# the smooth sigmoid profile, tau=2.0, instead of the linear/triangular
# window). Same pipeline as ../../RDP_DATAGEN/kitchenD1.sh but with
# --key_suffix sigmoid (this checkpoint was trained on the STANDARD goals,
# not RDP — keys become gmm_pred_goal_sigmoid / gmm_all_goals_sigmoid /
# gmm_all_weights_sigmoid so they can never collide with the _rdp tree).
#
# NUM_DEMOS defaults to 206 = ALL PushT demos: the low-level WCA PushT
# experiments train on all 206 demos (unlike kitchen's first-100).
#
# Pipeline:
#   1. Stage the source npz tree onto the node's /local SSD.
#   2. Run scripts/run_gmm_pred_to_npz.py writing to a /local npz dir.
#   3. rsync the prediction tree back to the durable /ocean location
#      D2/ABLATIONS/SIGMOID_EXPERIMENTS_GMM_PREDICTIONS/PushT_tau2.0/
#      (with a free-space guard first).
#
# CKPT_PATH below is set EXPLICITLY (same convention as the other gen
# scripts) — edit the path to switch checkpoints. The script fails fast if
# the file does not exist.

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

GOAL_SOURCE="sigmoid"

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-206}"
echo "[demo_limit] first NUM_DEMOS=${NUM_DEMOS} demos (demo_0 .. demo_$((NUM_DEMOS-1)))"

# --- paths ---------------------------------------------------------------
SRC_NPZ_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/PushT_Task"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning"
FINAL_OCEAN_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/ABLATIONS/SIGMOID_EXPERIMENTS_GMM_PREDICTIONS/PushT_tau2.0"

# --- checkpoint (EDIT HERE to change) --------------------------------------
CKPT_PATH="${REPO_DIR}/logs/ABLATION/SIGMOID_EXPERIMENTS/PUSH_T_TASK/train_PushT_Task_GOAL_SWAP_SIGMOID_tau2.0/2026-08-06/16-49-26/checkpoints/periodic-epoch=epoch=89.ckpt"
if [ ! -f "${CKPT_PATH}" ]; then
    echo "[ckpt] ERROR: checkpoint not found: ${CKPT_PATH}" >&2
    echo "[ckpt] Edit CKPT_PATH in this script." >&2
    exit 1
fi
echo "[ckpt] using: ${CKPT_PATH}"

# --- node-local scratch ----------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_NPZ_DIR="${SCRATCH_ROOT}/PushT_Task_npz"                 # staged inputs
DEST_PRED_DIR="${SCRATCH_ROOT}/PushT_GMM_PRED_SIGMOID_tau2.0"        # generator outputs

# --- (1) stage npz source to /local ----------------------------------------
THREADS="${RSYNC_THREADS:-32}"
echo "[stage] source : ${SRC_NPZ_DIR}"
echo "[stage] dest   : ${DEST_NPZ_DIR}"
mkdir -p "${DEST_NPZ_DIR}"
stage_start=$(date +%s)

copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export SRC_DIR_ENV="${SRC_NPZ_DIR}"
export DEST_DIR_ENV="${DEST_NPZ_DIR}"

seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DIR_ENV}/$1" "${DEST_DIR_ENV}/"' _ {}

stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. $(du -sh "${DEST_NPZ_DIR}" | cut -f1) staged."

# --- (2) run the prediction generator, writing npz to /local ----------------
mkdir -p "${DEST_PRED_DIR}"
cd "${REPO_DIR}"
gen_start=$(date +%s)

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python scripts/run_gmm_pred_to_npz.py \
    --dataset_dir "${DEST_NPZ_DIR}/" \
    --ckpt_path "${CKPT_PATH}" \
    --output_dir "${DEST_PRED_DIR}" \
    --key_suffix "${GOAL_SOURCE}" \
    --start_demo 0 \
    --max_files "${NUM_DEMOS}" \
    --batch_size 164

gen_elapsed=$(( $(date +%s) - gen_start ))
echo "[gen] done in ${gen_elapsed}s. $(find "${DEST_PRED_DIR}" -name '*.npz' | wc -l) npz files written ($(du -sh "${DEST_PRED_DIR}" | cut -f1))."

# --- (3) ship back to /ocean, with a free-space guard -----------------------
need_kb=$(du -sk "${DEST_PRED_DIR}" | cut -f1)
avail_kb=$(df -k --output=avail "$(dirname "${FINAL_OCEAN_DIR}")" | tail -1 | tr -d ' ')
buffer_kb=$(( 20 * 1024 * 1024 ))  # keep >=20GB headroom on /ocean after shipping
if [ "$(( need_kb + buffer_kb ))" -gt "${avail_kb}" ]; then
    echo "[ship] ERROR: need ${need_kb}KB + 20GB buffer but only ${avail_kb}KB free on /ocean." >&2
    echo "[ship] Prediction tree is preserved on ${DEST_PRED_DIR} for THIS job only — free space and re-run (generation resumes instantly via per-demo skip)." >&2
    exit 1
fi

echo "[ship] dest : ${FINAL_OCEAN_DIR}"
mkdir -p "${FINAL_OCEAN_DIR}"
ship_start=$(date +%s)
export SRC_DIR_ENV="${DEST_PRED_DIR}"
export DEST_DIR_ENV="${FINAL_OCEAN_DIR}"
find "${DEST_PRED_DIR}" -mindepth 1 -maxdepth 1 -printf '%f\n' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DIR_ENV}/$1" "${DEST_DIR_ENV}/"' _ {}
ship_elapsed=$(( $(date +%s) - ship_start ))
echo "[ship] done in ${ship_elapsed}s. $(find "${FINAL_OCEAN_DIR}" -name '*.npz' | wc -l) npz files now in ${FINAL_OCEAN_DIR}."
