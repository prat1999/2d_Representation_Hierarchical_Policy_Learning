#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 48:00:00 # 48-hour budget
#SBATCH --job-name kitchen-d1-action-rope-gripper-c1-0.1
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# ===========================================================================
# ACTION-ROPE EXPERIMENT on KITCHEN_D1, 10-FRAME FIXED_INTERVAL GOALS
#   action_pos_mode = gripper   aux head = dense GMM, c1 = 0.1
# ===========================================================================
# Approach 2 with 3D grounding AND action grounding inside the DiT, MINO-style
# (MINO_4D_Rope_Understanding/.../rope4d_dit.py), while KEEPING our grounded
# visual encoder (RoPE4D trunk over patches + gripper keypoints) and our
# interleaved block layout (even = cross-attention, odd = self-attention).
#
# What differs from the dense-GMM control
# (…/REGRESSION_HEAD_EXPERIMENT/KITCHEN_D1_K_10_FIXED_INTERVAL/kitchen_d1_iv10_gmm_c1_0.1.sh):
#   * every DiT attention rotates q and k by continuous (x, y, z, t) positions
#       visual keys   : the trunk's per-patch world anchors, t = obs step
#       state tokens  : grasp centre (keypoint 3) at that obs step
#       action step k : grasp centre at the last obs step + displacement, t = To+k
#   * the fixed sinusoidal slot embedding on the 18-token stream is removed
#   * depth-less patches are masked out of the cross-attention keys
# Everything else — data, trunk, aux head, c1, batch, epochs, seed — is identical.
# Policy: diffusion_policy/policy/flow_matching_rope4d_dit_goal_gmm_policy.py
# DiT:    diffusion_policy/model/flow_matching/rope4d_interleaved_dit.py
# Workspace: train_flow_matching_rope4d_dit_goal_gmm_workspace.yaml
#
# Two independent RoPE modules: the trunk keeps xyz_scale 5 / time_scale 1 /
# base 100 (weak time, 2 obs steps); the DiT uses dit_xyz_scale 5 (same metric
# convention) / dit_time_scale 18 (one radian per step, orders the 16 action
# steps) / base 100.
#
# Three sibling scripts in this folder differ ONLY in action_pos_mode:
#   gripper : action tokens all at the current grasp centre (differ only by time)
#   cumsum  : grasp centre + cumsum of the UNNORMALISED predicted position
#             deltas of the current noisy chunk (MINO's workspace setting;
#             noisy early in denoising)
#   blended : grasp centre + t * cumsum(...): gripper at t=0, estimated path
#             at t=1 (MINO's policy default; fails gracefully)
# Positions are rebuilt at every Euler step at inference from the current chunk.
#
# ACTION_POS_GAIN: hybrid_delta actions are OSC position COMMANDS (clipped at
# ±0.05 m), not realised motion. Measured on KITCHEN_D1 (20 demos, 763 windows
# of 16 steps): realised displacement = 0.236 × summed command per axis,
# residual 2.2 cm; summed commands reach 1.2 m where the gripper moves 0.3 m.
# cumsum/blended multiply the summed commands by this gain so action tokens
# land on the realised path. 1.0 = MINO's raw cumsum. Ignored by "gripper".
#
# c1 stays 0.1: the main loss is still flow matching, so the measured balance
# from the Approach 2 scripts carries over.
#
# --- goal target ----------------------------------------------------------
# obs/goal_gripper_pts_fixed_interval_iv10: the gripper pose at the next frame
# on a 10-frame grid (10, 20, 30, ... , T-1). Source npz tree:
#   
# That tree stores the goal under the SAME npz key as the T/20 grid already in
# the h5 (goal_gripper_pcd_fixed_interval), so it is injected under an ALIASED
# h5 name by the step below. Frame alignment verified for demos 0..99 (counts
# match the h5, incl. gapped kitchen demos; goal == present pose at every grid
# frame when matched by position).
#
# Selected by task-config + aux_goal_key, NOT task.dataset.goal_source (that
# would also remap present_gripper_pts, which has no _iv10 variant).
#
# NUM_DEMOS defaults to 100. Override at submission time:
#   NUM_DEMOS=200 sbatch this_script.sh

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- the knob this folder varies -------------------------------------------
ACTION_POS=gripper
ACTION_POS_GAIN="${ACTION_POS_GAIN:-0.236}"   # realised/commanded delta gain (see header)
AUX_HEAD=gmm
C1=0.1

# --- goal target (shared by both siblings) -------------------------------
GOAL_KEY=goal_gripper_pts_fixed_interval_iv10
TASK_CFG=MimicGen_Tasks/kitchen_goal_gmm_aux_fixed_interval_iv10
IV10_NPZ_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/EXTRA_KEYPOINTS/FIXED_INTERVAL_KEYPOINTS/KITCHEN_D1/iv10"

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[config] action-rope DiT, action_pos_mode=${ACTION_POS}, delta_gain=${ACTION_POS_GAIN}, aux_head=${AUX_HEAD}, c1=${C1}, goal_key=${GOAL_KEY}, NUM_DEMOS=${NUM_DEMOS} (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

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

# --- inject the iv10 goals into the source h5 (idempotent, lock-guarded) --
# Append-only: files already carrying obs/${GOAL_KEY} are opened read-only and
# skipped, so only the very first job ever writes; every later launch is a
# no-op. The mkdir lock serialises the two sibling jobs if they are submitted
# together, so no two processes append to the same h5 file at once. The lock
# is removed on any normal exit or error; after a hard kill mid-inject, remove
# it by hand:  rmdir "${NO_GMM_H5_DIR}/.inject_${GOAL_KEY}.lock"
INJECT_LOCK="${NO_GMM_H5_DIR}/.inject_${GOAL_KEY}.lock"
while ! mkdir "${INJECT_LOCK}" 2>/dev/null; do
    echo "[inject] another job holds ${INJECT_LOCK}; waiting 30s"
    sleep 30
done
trap 'rmdir "${INJECT_LOCK}" 2>/dev/null || true' EXIT
(
    cd "${REPO_DIR}"
    USE_TF=0 \
    GIT_LFS_SKIP_SMUDGE=1 \
    PYTHONNOUSERSITE=1 \
    PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
    pixi run python generate_non_gmm_goals_for_low_level.py \
        --dataset_dir "${NO_GMM_H5_DIR}" \
        --inject_extra_goals \
        --extra_goals_dir "${IV10_NPZ_DIR}" \
        --extra_goal_sources fixed_interval \
        --extra_goal_key_alias fixed_interval=fixed_interval_iv10 \
        --max_files "${NUM_DEMOS}"
)
rmdir "${INJECT_LOCK}"
trap - EXIT

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
DEST_DATA_DIR="${SCRATCH_ROOT}/Kitchen_D1_ActionRoPE_IV10_${ACTION_POS}_${NUM_DEMOS}demo"

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

# --- verify the iv10 goal key is present in every staged file ------------
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
          f"e.g. {missing[:5]}", file=sys.stderr)
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

RUN_NAME="kitchen_D1_ACTION_ROPE_IV10_${ACTION_POS}_GMM_c1_${C1}_${NUM_DEMOS}demo"

# batch_size=128 matches every other 100-demo Approach 2 run. Keep every knob
# below identical across the three siblings; only ACTION_POS may differ.
USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/wandb_cache \
WANDB_DATA_DIR=/ocean/projects/cis240052p/pbhowal/wandb_data \
PYTHONNOUSERSITE=1 \
PIXI_CACHE_DIR=/ocean/projects/cis240052p/pbhowal/pixi_cache \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_rope4d_dit_goal_gmm_workspace.yaml \
    task=${TASK_CFG} \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    policy.action_pos_mode=${ACTION_POS} \
    policy.action_pos_delta_gain=${ACTION_POS_GAIN} \
    policy.aux_head_type=${AUX_HEAD} \
    policy.aux_gmm_loss_weight=${C1} \
    policy.aux_goal_key=${GOAL_KEY} \
    logging.project=mimicgen_tasks \
    logging.name=${RUN_NAME} \
    name=${RUN_NAME} \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
