#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 20:00:00 # sized from measured 20hz-equivalent Approach2 runs (hammer 9.8h,
                    # kitchen 21h, coffee 22.8h for 100 epochs), scaled by frame count,
                    # with ~1.5-2x margin. Resume with RESUME_CKPT if it ever hits the wall.
#SBATCH --job-name kitchen-d1-10hz-fixed-interval-c1-0.1
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# ===========================================================================
# APPROACH 2 RATE ABLATION — kitchen_d1_10hz, FIXED_INTERVAL GOALS, c1 = 0.1
# ===========================================================================
# Sibling of Approach2/*/FIXED_INTERVAL_FOLDER/*_fixed_interval_c1_0.1.sh, but
# on the RATE-ABLATION datasets (eswaramo's rate_ablations/h5 tree): the same
# task re-recorded at 5/10/20/50/100 Hz control rate. The policy hyperparams
# (horizon=16, n_obs_steps=2, n_action_steps=8 — all in STEPS) are DELIBERATELY
# kept identical across rates: the ablation asks how the fixed-interval goal
# definition behaves as the wall-clock meaning of a step changes.
#
# The low-level policy is NOT given the goal. obs/goal_gripper_pts_fixed_interval
# supervises an auxiliary GMM head on the 3D-grounded visual tokens:
#
#     total_loss = flow_loss + c1 * gmm_loss          (c1 = 0.1, matches the
#                                                      non-rate FIXED_INTERVAL runs)
#
# Goal selection is by NAME (task config + policy.aux_goal_key), NOT by
# task.dataset.goal_source — goal_source would blanket-remap present_gripper_pts
# too (typed goal_gripper for the mixture anchors) and look for a key that
# deliberately does not exist. See the non-rate sibling script for the full note.
#
# The fixed_interval goals were injected into these h5 files from the
# NPZ_FIXED_KEYPOINTS tree by scripts/inject_fixed_keypoints_rate_ablations.sh;
# the pre-flight check below fails loudly if any staged file is missing the key.
#
# DEMO SELECTION: demo numbering in the rate dirs is NON-contiguous (e.g.
# coffee starts at demo_3), so this script stages the NUM_DEMOS
# SMALLEST-NUMBERED demo_*.h5 files, not demo_0..demo_{N-1}.
#
# Resume (workspace treats num_epochs=100 as an ABSOLUTE stop):
#   RESUME_CKPT=/path/to/epoch_N.ckpt sbatch this_script.sh

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- auxiliary loss weight (kept equal to the non-rate FIXED_INTERVAL runs) ---
C1=0.1

# --- goal source -----------------------------------------------------------
GOAL_KEY=goal_gripper_pts_fixed_interval
TASK_CFG=MimicGen_Tasks/kitchen_goal_gmm_aux_fixed_interval

# --- demo selection --------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[config] c1=${C1}, goal_key=${GOAL_KEY}, NUM_DEMOS=${NUM_DEMOS} (smallest-numbered demos)"

# --- paths -----------------------------------------------------------------
SRC_DATA_DIR="/ocean/projects/cis240052p/eswaramo/data/rate_ablations/h5/kitchen_d1_10hz"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

RESUME_CKPT="${RESUME_CKPT:-}"

# --- pick the NUM_DEMOS smallest-numbered demos ------------------------------
mapfile -t DEMO_FILES < <(
    ls "${SRC_DATA_DIR}"/demo_*.h5 2>/dev/null \
        | xargs -n1 basename \
        | sed -E 's/^demo_([0-9]+)\.h5$/\1/' \
        | sort -n | head -n "${NUM_DEMOS}" \
        | awk '{print "demo_" $1 ".h5"}'
)
if [ "${#DEMO_FILES[@]}" -lt "${NUM_DEMOS}" ]; then
    echo "[data] ERROR: need ${NUM_DEMOS} h5 files in ${SRC_DATA_DIR}, found ${#DEMO_FILES[@]}." >&2
    exit 1
fi
echo "[data] staging ${#DEMO_FILES[@]} demos: ${DEMO_FILES[0]} .. ${DEMO_FILES[-1]}"

# --- node-local scratch ------------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/local/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/kitchen_d1_10hz_FIXED_INTERVAL_${NUM_DEMOS}demo"

# --- stage dataset (read-only from the shared source) ------------------------
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

printf '%s\n' "${DEMO_FILES[@]}" \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DATA_DIR}/$1" "${DEST_DATA_DIR}/"' _ {}

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
echo "[stage] done in $(( $(date +%s) - stage_start ))s. ${staged_count} files, $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."
if [ "${staged_count}" -ne "${NUM_DEMOS}" ]; then
    echo "[stage] ERROR: expected ${NUM_DEMOS} files staged, got ${staged_count}." >&2
    exit 1
fi

# --- train -------------------------------------------------------------------
cd "${REPO_DIR}"

# --- verify the FIXED_INTERVAL goal key is actually present -------------------
# Fail loudly here rather than deep inside the dataset loader: a missing key
# (e.g. injection not yet run on this pair) would otherwise surface as an
# opaque h5 KeyError mid-training.
pixi run python - "${DEST_DATA_DIR}" "${GOAL_KEY}" <<'PYEOF'
import sys, glob, h5py
data_dir, key = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{data_dir}/*.h5"))
missing = []
for f in files:
    with h5py.File(f, "r") as h:
        if key not in h["obs"]:
            missing.append(f.split("/")[-1])
if missing:
    print(f"[verify] ERROR: obs/{key} missing from {len(missing)}/{len(files)} h5 files, "
          f"e.g. {missing[:5]}. Run scripts/inject_fixed_keypoints_rate_ablations.sh first.",
          file=sys.stderr)
    sys.exit(1)
print(f"[verify] obs/{key} present in all {len(files)} staged h5 files.")
PYEOF

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

RUN_NAME="kitchen_d1_10hz_approach2_FIXED_INTERVAL_c1_${C1}_${NUM_DEMOS}demo"
if [ -n "${RESUME_CKPT}" ]; then
    RUN_NAME="${RUN_NAME}_resumeE$(basename "${RESUME_CKPT}" .ckpt | grep -oE '[0-9]+' || echo X)"
fi

# Every knob below matches the non-rate FIXED_INTERVAL c1=0.1 scripts (batch
# 128, checkpoint_every=5, workspace defaults for horizon/n_obs/n_action and
# num_epochs=100) — only the dataset differs, so differences across the 15
# runs are attributable to control rate alone.
USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/wandb_cache \
WANDB_DATA_DIR=/ocean/projects/cis240052p/pbhowal/wandb_data \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_goal_gmm_workspace.yaml \
    task=${TASK_CFG} \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    policy.aux_gmm_loss_weight=${C1} \
    policy.aux_goal_key=${GOAL_KEY} \
    logging.project=mimicgen_rate_ablations \
    logging.name=${RUN_NAME} \
    name=${RUN_NAME} \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
