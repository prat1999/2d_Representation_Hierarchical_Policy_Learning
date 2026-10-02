#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 48:00:00 # non-RoPE kitchen gt-mix took 12.6h on H100; the RoPE trunk ~1.7-2x -> ~24h expected
#SBATCH --job-name kitchen-d1-wca-ropegoals-100demo-dinov2-awe-grip-gtmix
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/100_trajectory_trainings/KITCHEN_D1_TASK/AWE_GRIP_FOLDER/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/100_trajectory_trainings/KITCHEN_D1_TASK/AWE_GRIP_FOLDER/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# AWE-GRIP GT/predicted goal-MIX run on KITCHEN_D1 with 3D-GROUNDED VISION
# (RoPE4D trunk after DINOv2) and the top-6 GMM goal candidates as grounded
# tokens in the trunk.
#
# This is the COFFEE_PREPERATION_D1 ROPE_GOALS recipe
# (coffee_preperation_wca_dinov2_gt_predicted_mix_ROPE_GOALS.sh) applied to
# kitchen and re-pointed at the AWE goal source, the same way the RDP_FOLDER
# gt_predicted_mix scripts re-point the plain WCA run at RDP:
#   * GT goals   : AWE keypoint tree  <AWE_ROOT>/KITCHEN_D1
#                  (key goal_gripper_pcd_awe) -> gt_goal_npz_dir / gt_goal_npz_key
#   * predictions: AWE high-level GMM tree <AWE_ROOT>/KITCHEN_D1_GMM_PRED
#                  (keys gmm_all_goals_awe / gmm_all_weights_awe)
#                  -> gmm_pred_npz_dir / gmm_pred_key_suffix
#   * h5 files   : LOW_LEVEL_WITH_GMM tree KITCHEN_D1, READ-ONLY for images /
#                  depth / intrinsics / extrinsics / state / present_gripper_pts /
#                  actions. Its default gmm_* and goal_gripper_pts keys are never read.
# Per TRAIN sample, Bernoulli(gt_mix_p) picks the ground-truth AWE goal set
# (present goal, plus the neighbor keyframe goal inside +-5 frames of an AWE
# transition with triangular weights); otherwise the AWE high-level prediction.
# VALIDATION always uses the predicted GMM from the AWE prediction tree.
#
# KITCHEN_D1 frame gaps: source demos miss some frame numbers; the h5 files
# were built from the sorted existing frames, the prediction tree keeps the
# source numbering, and the AWE keypoint tree is contiguous 0..T-1. Both npz
# trees are consumed by sorted position with length guards, so all three stay
# aligned (verified for demo_0..demo_99).
#
# Task config kitchen_D1_gmm_goal_gt_mix_rope.yaml was added 2026-09-09 as a
# verbatim copy of the coffee rope config (name + data_dir changed).
#
# GT_MIX_P defaults to 0.5. Override: GT_MIX_P=0.3 sbatch this_script.sh
#   NUM_DEMOS=200 sbatch this_script.sh

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# npz key suffix of the AWE trees (gmm_*_awe / goal_gripper_pcd_awe).
GOAL_SOURCE="awe"
RUN_TAG="AWE_GRIP"

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[demo_limit] using first NUM_DEMOS=${NUM_DEMOS} demos (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

# --- gt-mix probability (overridable) -------------------------------------
GT_MIX_P="${GT_MIX_P:-0.5}"
echo "[gt_mix] gt_mix_p=${GT_MIX_P}  (P of using ground-truth ${GOAL_SOURCE} goals per sample)"

# --- paths ---------------------------------------------------------------
SRC_DATA_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/LOW_LEVEL_WITH_GMM_DATASET_GROOT_STYLE_DATASET/D2/KITCHEN_D1"
AWE_ROOT="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/D2/EXTRA_KEYPOINTS/AWE_EXTRA_KEYPOINTS/EXTRA_KEYPOINTS_awe-greedy-th0.35-grip"
PRED_SRC_DIR="${AWE_ROOT}/KITCHEN_D1_GMM_PRED"
GT_SRC_DIR="${AWE_ROOT}/KITCHEN_D1"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# The prediction tree must have been generated with the matching key suffix
# (AWE_GRIP_DATAGEN/kitchenD1.sh writes _generation_meta.json).
pred_sfx=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['key_suffix'])" "${PRED_SRC_DIR}/_generation_meta.json" 2>/dev/null || echo "?")
if [ "${pred_sfx}" != "${GOAL_SOURCE}" ]; then
    echo "[pred] ERROR: ${PRED_SRC_DIR} key_suffix='${pred_sfx}', expected '${GOAL_SOURCE}' (missing tree or wrong goal source)." >&2
    exit 1
fi
echo "[pred] using $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['ckpt_path'])" "${PRED_SRC_DIR}/_generation_meta.json")"

# --- resume from checkpoint ----------------------------------------------
# Empty by default (fresh run). To resume a previous run OF THIS SAME VARIANT:
#   RESUME_CKPT=/path/to/epoch_X.ckpt sbatch this_script.sh
RESUME_CKPT="${RESUME_CKPT:-}"
if [ -n "${RESUME_CKPT}" ]; then
    RESUME_TAG="_resumeE$(basename "${RESUME_CKPT}" .ckpt | grep -oE '[0-9]+' || echo X)"
else
    RESUME_TAG=""
fi

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
DEST_DATA_DIR="${SCRATCH_ROOT}/Kitchen_D1_Low_Level_${NUM_DEMOS}demo"
DEST_PRED_DIR="${SCRATCH_ROOT}/Kitchen_D1_GMM_PRED_${GOAL_SOURCE}_${NUM_DEMOS}demo"
DEST_GT_DIR="${SCRATCH_ROOT}/Kitchen_D1_${GOAL_SOURCE}_goals_${NUM_DEMOS}demo"

# --- stage dataset (only NUM_DEMOS entries per tree) ----------------------
THREADS="${RSYNC_THREADS:-32}"

echo "[stage] h5 source  : ${SRC_DATA_DIR}"
echo "[stage] pred source: ${PRED_SRC_DIR}"
echo "[stage] gt source  : ${GT_SRC_DIR}"
mkdir -p "${DEST_DATA_DIR}" "${DEST_PRED_DIR}" "${DEST_GT_DIR}"

stage_start=$(date +%s)

# Per-entry rsync wrapper. Exit code 24 (vanished files) is benign.
copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export SRC_DATA_DIR DEST_DATA_DIR PRED_SRC_DIR DEST_PRED_DIR GT_SRC_DIR DEST_GT_DIR

# h5 episode files.
seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1 ".h5"}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${SRC_DATA_DIR}/$1" "${DEST_DATA_DIR}/"' _ {}

# AWE prediction npz tree (val + any predicted-GMM samples).
seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${PRED_SRC_DIR}/$1" "${DEST_PRED_DIR}/"' _ {}

# AWE GT keypoint npz tree (GT modes for the gt-mix precompute).
seq 0 $((NUM_DEMOS - 1)) \
    | awk '{print "demo_" $1}' \
    | xargs -P "${THREADS}" -I {} \
        bash -c 'copy_one "${GT_SRC_DIR}/$1" "${DEST_GT_DIR}/"' _ {}

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
pred_count=$(
    find "${DEST_PRED_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' \
        -exec sh -c 'ls "$1"/*.npz >/dev/null 2>&1' _ {} \; -print | wc -l
)
gt_count=$(
    find "${DEST_GT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' \
        -exec sh -c 'ls "$1"/*.npz >/dev/null 2>&1' _ {} \; -print | wc -l
)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} h5, ${pred_count} pred dirs, ${gt_count} gt dirs."
if [ "${staged_count}" -ne "${NUM_DEMOS}" ] || [ "${pred_count}" -ne "${NUM_DEMOS}" ] || [ "${gt_count}" -ne "${NUM_DEMOS}" ]; then
    echo "[stage] ERROR: expected ${NUM_DEMOS} of each (h5/pred/gt); got ${staged_count}/${pred_count}/${gt_count}." >&2
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
    task=MimicGen_Tasks/kitchen_D1_gmm_goal_gt_mix_rope \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    task.dataset.gt_mix_p="${GT_MIX_P}" \
    +task.dataset.gmm_pred_npz_dir="${DEST_PRED_DIR}" \
    +task.dataset.gmm_pred_key_suffix=${GOAL_SOURCE} \
    +task.dataset.gt_goal_npz_dir="${DEST_GT_DIR}" \
    +task.dataset.gt_goal_npz_key=goal_gripper_pcd_${GOAL_SOURCE} \
    visual_encoder=dinov2_rope4d_grounded_goals \
    policy.use_goal_cross_attention=true \
    policy.use_weighted_cross_attention=true \
    policy.gmm_top_k=6 \
    logging.project=MimicGen_GMM_Low_Level_Policy \
    logging.name=groot_GMM_WCA_ROPE_GOALS_${NUM_DEMOS}demo_dinov2_Kitchen_D1_${RUN_TAG}_GTMIX_p${GT_MIX_P}${RESUME_TAG} \
    name=groot_GMM_WCA_ROPE_GOALS_${NUM_DEMOS}demo_dinov2_Kitchen_D1_${RUN_TAG}_GTMIX_p${GT_MIX_P}${RESUME_TAG} \
    training.checkpoint_every=10 \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
