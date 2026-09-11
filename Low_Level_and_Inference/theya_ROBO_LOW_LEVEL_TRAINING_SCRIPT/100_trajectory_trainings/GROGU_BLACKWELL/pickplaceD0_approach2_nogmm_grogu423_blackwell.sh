#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=16    # matches dataloader.num_workers=16 below
#SBATCH --mem=80G             # explicit floor above DefMemPerCPU(3.5G)*cpus, so worker/page-cache spikes don't hit the cgroup limit
#SBATCH -p dheld
#SBATCH -w grogu-4-23         # pin to the RTX 6000 Blackwell node
#SBATCH --gres=gpu:1
#SBATCH -C 6000Blackwell      # grogu-4-23 is an 8x RTX 6000 Blackwell node, not A6000 -- match the node's AVAIL_FEATURES
#SBATCH -t 24:00:00           # use RESUME_CKPT to continue in a follow-up job if more time is needed
#SBATCH --job-name pickplace-d0-approach2-nogmm-grogu423-blackwell
#SBATCH -o /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/pickplace-d0-approach2-nogmm-grogu423-blackwell_job_%j.out
#SBATCH -e /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/pickplace-d0-approach2-nogmm-grogu423-blackwell_job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# ===========================================================================
# APPROACH 2 *CONTROL ARM* (arm B) on PICK_PLACE_D0
#   -> NO auxiliary loss at all; the auxiliary head is NOT constructed.
# grogu cluster / dheld partition, pinned to grogu-4-23 (8x RTX 6000 Blackwell).
#
# Architecture-matched control for the two treatment arms in this directory
# (pickplaceD0_approach2_awe_greedy_th0.35_grip_grogu423_blackwell.sh, GMM c1=0.1,
# and pickplaceD0_approach2_auxregression_relgripper_c1_10_grogu423_blackwell.sh,
# regression c1=10): goal removed from the DiT input, same DINOv2 RoPE4D
# grounded trunk. The ONLY difference is the auxiliary head:
#
#     treatment : total_loss = flow_loss + c1 * aux_loss
#     this arm  : total_loss = flow_loss
#
# Driven by `policy.aux_gmm_loss_weight=null`, which leaves the head
# uninstantiated in FlowMatchingDiTGoalGMMPolicy.__init__ (aux_head_type is
# forced to None), so there are no auxiliary parameters in the model at all and
# compute_loss() returns fm_loss without ever reading the goal. That is what
# makes the treatment arms interpretable.
#
# Because the head does not exist, NO goal keypoints are needed: unlike its
# siblings this script references no EXTRA_KEYPOINTS tree, never runs
# --inject_extra_goals, and passes no goal_source. The dataset still serves
# obs/goal_gripper_pts (the native field the --no_gmm conversion writes,
# declared in the task yaml's shape_meta so n_keypoints can be read from it) --
# the policy simply never touches it.
#
# Demo count matches the treatment arms so the control is comparable: staging
# caps at NUM_DEMOS by numeric demo index, which is the same demo_0..demo_99
# subset the treatment arms get from the keypoints-tree intersection.
#
#   NUM_DEMOS=50 sbatch this_script.sh
# ===========================================================================

set -euo pipefail
set -x

export PIXI_HOME="/project_data/held/teswaram/pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# --- control arm: no auxiliary loss, no goal source ------------------------
ARM_TAG="NOGMM"

NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[config] arm=${ARM_TAG} (aux_gmm_loss_weight=null, no aux head), NUM_DEMOS cap=${NUM_DEMOS}"

# --- paths -----------------------------------------------------------------
# npz sources live in the NEW_TASKS tree under $HOME; the converted h5 pool does
# NOT -- $HOME has ~34G free and one pool runs ~19G, so pools go to /project_data.
SRC_NPZ_DIR="${SRC_NPZ_DIR:-/home/teswaram/data/NEW_TASKS/pick_place_d0}"
LOCAL_NO_GMM_H5_DIR="${LOCAL_NO_GMM_H5_DIR:-/project_data/held/teswaram/data/NEW_TASKS/NO_GMM_preds/pick_place_d0}"
REPO_DIR="/home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

# --- resume from checkpoint --------------------------------------------------
# Empty by default: a fresh variant must NOT resume from a sibling arm's
# checkpoint. Override only when resuming THIS SAME variant:
#   RESUME_CKPT=/path/to/epoch_N.ckpt sbatch this_script.sh
RESUME_CKPT="${RESUME_CKPT:-}"


# --- generate local NO_GMM h5 pool if not already cached --------------------
# flock-guarded and SHARED by all three pick_place_d0 Approach-2 arms
# (gmm c1=0.1 / regression c1=10 / no-aux control): the demo_N/ npz -> h5
# conversion reads only SRC_NPZ_DIR and depends on neither the goal source nor
# the auxiliary head, so the three may be submitted together and the first one
# in does the work.
mkdir -p "${LOCAL_NO_GMM_H5_DIR}"
(
    flock -x 200
    existing_h5_count=$(find "${LOCAL_NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" 2>/dev/null | wc -l)
    echo "[gen] existing local *.h5 in ${LOCAL_NO_GMM_H5_DIR}: ${existing_h5_count}"
    if [ "${existing_h5_count}" -ge "${NUM_DEMOS}" ]; then
        echo "[gen] already have ${existing_h5_count} local h5 files (>= NUM_DEMOS=${NUM_DEMOS}) -> skipping conversion"
    else
        echo "[gen] converting demo_*/ -> demo_*.h5 in ${LOCAL_NO_GMM_H5_DIR}"
        cd "${REPO_DIR}"
        USE_TF=0 \
        GIT_LFS_SKIP_SMUDGE=1 \
        PYTHONNOUSERSITE=1 \
        pixi run python generate_non_gmm_goals_for_low_level.py \
            --dataset_dir "${SRC_NPZ_DIR}" \
            --no_gmm \
            --no_gmm_output_dir "${LOCAL_NO_GMM_H5_DIR}" \
            --max_files "${NUM_DEMOS}"
    fi
) 200>"${LOCAL_NO_GMM_H5_DIR}.genlock"
NO_GMM_H5_DIR="${LOCAL_NO_GMM_H5_DIR}"

# --- node-local scratch ------------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRATCH_ROOT="/scratch/teswaram/slurm-${SLURM_JOB_ID}/local"
    mkdir -p "${SCRATCH_ROOT}"
elif [ -n "${LOCAL:-}" ]; then
    SCRATCH_ROOT="${LOCAL}"
else
    SCRATCH_ROOT="${TMPDIR:-/tmp}"
fi
DEST_DATA_DIR="${SCRATCH_ROOT}/PickPlace_D0_Approach2_${ARM_TAG}"

# --- stage demos from the h5 pool (no goal tree to intersect with) ---------
THREADS="${RSYNC_THREADS:-32}"
mkdir -p "${DEST_DATA_DIR}"

demos_common=$(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name 'demo_*.h5' -printf '%f\n' | sed 's/\.h5$//' | sort)
n_common=$(echo -n "${demos_common}" | grep -c . || true)
echo "[stage] h5 demos available: ${n_common}"
if [ "${n_common}" -eq 0 ]; then
    echo "[stage] ERROR: no demo_*.h5 found in ${NO_GMM_H5_DIR}." >&2
    exit 1
fi

# --- cap the staged set to NUM_DEMOS ---------------------------------------
# Numeric-index cap (demo_0..demo_$((NUM_DEMOS-1))), NOT a lexicographic slice,
# so the control trains on the same demo subset the treatment arms get from
# their intersection with the EXTRA_KEYPOINTS tree.
if [ "${n_common}" -gt "${NUM_DEMOS}" ]; then
    demos_common=$(echo "${demos_common}" | sort -t_ -k2 -n | head -n "${NUM_DEMOS}")
    n_common=$(echo -n "${demos_common}" | grep -c . || true)
    echo "[stage] capped to NUM_DEMOS=${NUM_DEMOS} -> ${n_common} demos (demo_0..demo_$((NUM_DEMOS-1)) by numeric index)"
fi


echo "[stage] source : ${NO_GMM_H5_DIR}"
echo "[stage] dest   : ${DEST_DATA_DIR}"
echo "[stage] threads: ${THREADS}"

stage_start=$(date +%s)

copy_one() {
    rsync -a --exclude='.*.??????' "$1" "$2"
    local rc=$?
    [ "$rc" -eq 24 ] && return 0
    return "$rc"
}
export -f copy_one
export NO_GMM_H5_DIR DEST_DATA_DIR

echo "${demos_common}" | xargs -P "${THREADS}" -I {} \
    bash -c 'copy_one "${NO_GMM_H5_DIR}/{}.h5" "${DEST_DATA_DIR}/"'

staged_count=$(find "${DEST_DATA_DIR}" -maxdepth 1 -name '*.h5' | wc -l)
stage_elapsed=$(( $(date +%s) - stage_start ))
echo "[stage] done in ${stage_elapsed}s. ${staged_count} files, $(du -sh "${DEST_DATA_DIR}" | cut -f1) staged."
if [ "${staged_count}" -ne "${n_common}" ]; then
    echo "[stage] ERROR: expected ${n_common} files staged, got ${staged_count}." >&2
    exit 1
fi


# --- train -------------------------------------------------------------------
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

RUN_NAME="PickPlace_D0_APPROACH2_${ARM_TAG}_${staged_count}demo_dinov2_DIT_grogu423_blackwell"

USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/project_data/held/teswaram/logs/wandb_cache \
WANDB_DATA_DIR=/project_data/held/teswaram/logs/wandb_data \
PYTHONNOUSERSITE=1 \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_goal_gmm_workspace.yaml \
    task=MimicGen_Tasks/pick_place_d0_goal_gmm_aux \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    policy.aux_gmm_loss_weight=null \
    logging.project=mimicgen_tasks \
    logging.name=${RUN_NAME} \
    name=${RUN_NAME} \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
