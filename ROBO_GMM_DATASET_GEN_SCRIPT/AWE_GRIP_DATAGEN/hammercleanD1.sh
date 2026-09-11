#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12    # 12 CPU cores for npz load + npz write workers
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100 (needed by the high-level GMM forward pass)
#SBATCH -t 2:00:00 # RDP run of this job took ~8 min end to end (32s stage, 406s gen, 34s ship)
#SBATCH --job-name hammer-cleanup-d1-awe-grip-pred-gen
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning/ROBO_GMM_DATASET_GEN_SCRIPT/AWE_GRIP_DATAGEN/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning/ROBO_GMM_DATASET_GEN_SCRIPT/AWE_GRIP_DATAGEN/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# Generate per-frame GMM prediction npz files for HAMMER_CLEANUP_D1 by running the
# AWE-goal-trained high-level articubot model (AWE greedy, th=0.35, gripper-aware)
# on every demo's per-step npz. Writes ONLY the model's outputs
# (gmm_pred_goal_awe / gmm_all_goals_awe / gmm_all_weights_awe) as a parallel
# npz tree — no h5 files are touched. The generator never reads goal keys, so
# the goal source only decides which checkpoint runs and how the keys are named.
#
# Pipeline (identical to RDP_DATAGEN/hammercleanD1.sh):
#   1. Stage the source npz tree onto the node's /local SSD.
#   2. Run scripts/run_gmm_pred_to_npz.py writing to a /local npz dir.
#   3. rsync the prediction tree back to the durable /ocean location — the
#      AWE keypoint folder, as <TASK>_GMM_PRED next to the <TASK> keypoints
#      (same layout the RDP trees use under EXTRA_KEYPOINTS/). First-100 tree
#      is ~6.4 GB for this task (~290 frames/demo, ~235 KB/frame).
#
# Checkpoint: CKPT_EPOCH selects the periodic checkpoint of the 2026-09-08
# AWE high-level run. Default 74 (user-selected 2026-09-08; same epoch the
# RDP prediction trees were generated from). Override at submission time:
#   CKPT_EPOCH=99 sbatch this_script.sh
# The script fails fast if the checkpoint file does not exist yet.

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

GOAL_SOURCE="awe"

# --- demo selection ------------------------------------------------------
# Only the first NUM_DEMOS demos are staged + processed: the low-level WCA
# experiments train on the first 100 demos only. The AWE tree currently holds
# demo_0..demo_99 only. Lift later with e.g.:
#   NUM_DEMOS=1000 sbatch this_script.sh
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[demo_limit] first NUM_DEMOS=${NUM_DEMOS} demos (demo_0 .. demo_$((NUM_DEMOS-1)))"

# --- paths ---------------------------------------------------------------
SRC_NPZ_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/HAMMER_CLEANUP_D1"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/2d_Representation_Hierarchical_Policy_Learning"
AWE_ROOT="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/EXTRA_KEYPOINTS/AWE_EXTRA_KEYPOINTS/EXTRA_KEYPOINTS_awe-greedy-th0.35-grip"
FINAL_OCEAN_DIR="${AWE_ROOT}/HAMMER_CLEANUP_D1_GMM_PRED"

# --- checkpoint (CKPT_EPOCH env var, default 74) ----------------------------
CKPT_EPOCH="${CKPT_EPOCH:-74}"
RUN_DIR="${REPO_DIR}/logs/train_HammerCleanup_D1_GOAL_SWAP_AWE_GRIP_100demo/2026-09-08/09-54-45"
CKPT_PATH="${RUN_DIR}/checkpoints/periodic-epoch=epoch=${CKPT_EPOCH}.ckpt"
if [ ! -f "${CKPT_PATH}" ]; then
    echo "[ckpt] ERROR: checkpoint not found: ${CKPT_PATH}" >&2
    echo "[ckpt] Available: $(ls "${RUN_DIR}/checkpoints" 2>/dev/null | grep -o 'epoch=[0-9]*\.ckpt' | tr '\n' ' ')" >&2
    exit 1
fi
echo "[ckpt] using: ${CKPT_PATH}"

# Refuse to ship over an existing /ocean tree generated from a DIFFERENT
# checkpoint — rsync would overwrite file by file and a partial failure could
# leave a mixed tree. Remove the old tree (or set FORCE=1) to regenerate.
META="${FINAL_OCEAN_DIR}/_generation_meta.json"
if [ -f "${META}" ] && [ "${FORCE:-0}" != "1" ]; then
    old_ckpt=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['ckpt_path'])" "${META}" 2>/dev/null || echo "?")
    if [ "${old_ckpt}" != "${CKPT_PATH}" ]; then
        echo "[guard] ERROR: ${FINAL_OCEAN_DIR} already holds predictions from" >&2
        echo "[guard]        ${old_ckpt}" >&2
        echo "[guard]        Delete that tree or re-submit with FORCE=1 to overwrite." >&2
        exit 1
    fi
fi

# --- node-local scratch ----------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_NPZ_DIR="${SCRATCH_ROOT}/HAMMER_CLEANUP_D1_npz"            # staged inputs
DEST_PRED_DIR="${SCRATCH_ROOT}/HAMMER_CLEANUP_D1_GMM_PRED_AWE"  # generator outputs

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

# Sanity check: every requested demo dir must be staged with .npz frames,
# otherwise the generator would silently produce a smaller tree.
staged_nonempty=$(
    find "${DEST_NPZ_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' \
        -exec sh -c 'ls "$1"/*.npz >/dev/null 2>&1' _ {} \; -print | wc -l
)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_nonempty} demo dirs with npz, $(du -sh "${DEST_NPZ_DIR}" | cut -f1) staged."
if [ "${staged_nonempty}" -ne "${NUM_DEMOS}" ]; then
    echo "[stage] ERROR: expected ${NUM_DEMOS} staged demo dirs with npz, got ${staged_nonempty}." >&2
    exit 1
fi

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
n_pred=$(find "${DEST_PRED_DIR}" -name '*.npz' | wc -l)
n_src=$(find "${DEST_NPZ_DIR}" -name '*.npz' | wc -l)
echo "[gen] done in ${gen_elapsed}s. ${n_pred} npz files written ($(du -sh "${DEST_PRED_DIR}" | cut -f1)); source had ${n_src}."
if [ "${n_pred}" -ne "${n_src}" ]; then
    echo "[gen] ERROR: prediction count ${n_pred} != source frame count ${n_src}." >&2
    exit 1
fi

# --- (3) ship back to /ocean, with a free-space guard -----------------------
need_kb=$(du -sk "${DEST_PRED_DIR}" | cut -f1)
avail_kb=$(df -k --output=avail "$(dirname "${FINAL_OCEAN_DIR}")" | tail -1 | tr -d ' ')
buffer_kb=$(( 20 * 1024 * 1024 ))  # keep >=20GB headroom on /ocean after shipping
if [ "$(( need_kb + buffer_kb ))" -gt "${avail_kb}" ]; then
    echo "[ship] ERROR: need ${need_kb}KB + 20GB buffer but only ${avail_kb}KB free on /ocean." >&2
    echo "[ship] Prediction tree is preserved on ${DEST_PRED_DIR} for THIS job only — free space and re-run." >&2
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
