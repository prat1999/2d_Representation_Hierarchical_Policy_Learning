#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=16    # matches dataloader.num_workers=16 below
#SBATCH --mem=80G             # explicit floor above DefMemPerCPU(3.5G)*cpus, so worker/page-cache spikes don't hit the cgroup limit
#SBATCH -p dheld
#SBATCH -w grogu-4-23         # pin to the RTX 6000 Blackwell node
#SBATCH --gres=gpu:1
#SBATCH -C 6000Blackwell      # grogu-4-23 is an 8x RTX 6000 Blackwell node, not A6000 -- match the node's AVAIL_FEATURES
#SBATCH -t 48:00:00
#SBATCH --job-name kitchen-d1-approach2-fixed-interval-iv5-grogu423-blackwell
#SBATCH -o /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/kitchen-d1-approach2-fixed-interval-iv5-grogu423-blackwell_job_%j.out
#SBATCH -e /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/kitchen-d1-approach2-fixed-interval-iv5-grogu423-blackwell_job_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# ===========================================================================
# APPROACH 2 (GMM-as-auxiliary-loss) on KITCHEN_D1  —  goal_source=fixed_interval, interval=5
# grogu cluster / dheld partition, pinned to the Blackwell node grogu-4-23.
# Sibling of kitchenD1_approach2_awe_greedy_th0.35_grip_grogu423_blackwell.sh
# -- see that script for the full GMM-as-aux-loss rationale.
# ===========================================================================
# The low-level policy is NOT given the goal. goal_gripper_pts supervises a
# GMM head that reads the same 3D-grounded visual tokens the DiT cross-attends
# to, so the goal shapes the visual representation rather than being an input:
#
#     total_loss = flow_loss + c1 * gmm_loss
#
# This variant supervises the GMM head with fixed_interval keypoints -- goals
# placed every 5 frames along the trajectory -- read from the iv5 tree of
# the FIXED_INTERVAL_SWEEP:
#
#   /project_data/held/teswaram/data/D1/FIXED_INTERVAL_SWEEP/iv5/EXTRA_KEYPOINTS_fixed_interval/KITCHEN_D1
#
# The npz key is goal_gripper_pcd_fixed_interval regardless of interval (the
# interval only selects which on-disk tree it is read from), so GOAL_SOURCE
# stays "fixed_interval" for hydra -- only EXTRA_GOALS_DIR, SOURCE_TAG and the
# run/log naming vary across the iv5/iv10/iv30/iv50 sibling scripts.
#
# c1=0.1 is the established "start here" auxiliary-loss weight for Approach 2.
#
#   - The raw demo_N/ npz -> h5 conversion is goal_source-independent, so this
#     script reuses the SAME LOCAL_NO_GMM_H5_DIR as every other KITCHEN_D1
#     grogu script (skips reconversion if that pool already exists).
#   - Injection writes into the node-local staged copy only, so the four
#     interval variants never fight over the same h5 files.
#   - Override the demo cap at submission time:
#       NUM_DEMOS=50 sbatch this_script.sh

set -euo pipefail
set -x

export PIXI_HOME="/project_data/held/teswaram/pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# --- the one knob these sibling scripts vary ------------------------------
GOAL_SOURCE="fixed_interval"
IV=5
SOURCE_TAG="fixed_interval_iv${IV}"
C1=0.1

NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[config] goal_source=${GOAL_SOURCE} (${SOURCE_TAG}), c1=${C1}, NUM_DEMOS cap=${NUM_DEMOS}"

# --- paths -----------------------------------------------------------------
SRC_NPZ_DIR="/project_data/held/teswaram/data/D1/KITCHEN_D1"
SWEEP_TAR="/project_data/held/teswaram/data/D1/FIXED_INTERVAL.tar.gz"
LOCAL_NO_GMM_H5_DIR="/project_data/held/teswaram/data/D1/NO_GMM_preds/KITCHEN_D1"
REPO_DIR="/home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

echo "[config] SWEEP_TAR=${SWEEP_TAR} (interval subtree iv${IV})"

# --- resume from checkpoint --------------------------------------------------
# Empty by default: a fresh interval variant should NOT resume from a different
# variant's checkpoint. Override only when resuming THIS variant:
#   RESUME_CKPT=/path/to/epoch_N.ckpt sbatch this_script.sh
RESUME_CKPT="${RESUME_CKPT:-}"

if [ ! -f "${SWEEP_TAR}" ]; then
    echo "[error] SWEEP_TAR not found: ${SWEEP_TAR}" >&2
    exit 1
fi

# --- generate local NO_GMM h5 pool if not already cached --------------------
existing_h5_count=0
if [ -d "${LOCAL_NO_GMM_H5_DIR}" ]; then
    existing_h5_count=$(find "${LOCAL_NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" 2>/dev/null | wc -l)
fi
echo "[gen] existing local *.h5 in ${LOCAL_NO_GMM_H5_DIR}: ${existing_h5_count}"
if [ "${existing_h5_count}" -eq 0 ]; then
    echo "[gen] converting demo_*/ -> demo_*.h5 in ${LOCAL_NO_GMM_H5_DIR}"
    mkdir -p "${LOCAL_NO_GMM_H5_DIR}"
    (
        cd "${REPO_DIR}"
        USE_TF=0 \
        GIT_LFS_SKIP_SMUDGE=1 \
        PYTHONNOUSERSITE=1 \
        pixi run python generate_non_gmm_goals_for_low_level.py \
            --dataset_dir "${SRC_NPZ_DIR}" \
            --no_gmm \
            --no_gmm_output_dir "${LOCAL_NO_GMM_H5_DIR}" \
            --max_files "${NUM_DEMOS}"
    )
else
    echo "[gen] already have ${existing_h5_count} local h5 files -> skipping conversion"
fi
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
DEST_DATA_DIR="${SCRATCH_ROOT}/Kitchen_D1_Approach2_${SOURCE_TAG}"

# --- unpack this interval's keypoint tree to node-local scratch -----------
# The sweep is 66k ~311-byte npz per interval. Creating those on NFS is
# metadata-bound (~40 files/s, hours); on node-local NVMe the whole archive
# extracts in ~3s. So the tarball is the durable artifact on /project_data and
# each job unpacks only what it needs, locally.
#
# --wildcards restricts extraction to iv${IV}: a job can only ever see its own
# interval's keypoints, so a copy/paste slip between the four sibling scripts
# fails loudly (missing dir) instead of silently training on the wrong goals.
EXTRA_GOALS_ROOT="${SCRATCH_ROOT}/FIXED_INTERVAL_SWEEP_iv${IV}"
EXTRA_GOALS_DIR="${EXTRA_GOALS_ROOT}/FIXED_INTERVAL_SWEEP/iv${IV}/EXTRA_KEYPOINTS_fixed_interval/KITCHEN_D1"
mkdir -p "${EXTRA_GOALS_ROOT}"
echo "[unpack] ${SWEEP_TAR}  ->  ${EXTRA_GOALS_ROOT}  (iv${IV} only)"
unpack_start=$(date +%s)
tar -xzf "${SWEEP_TAR}" -C "${EXTRA_GOALS_ROOT}" --wildcards "FIXED_INTERVAL_SWEEP/iv${IV}/*"
echo "[unpack] done in $(( $(date +%s) - unpack_start ))s"

if [ ! -d "${EXTRA_GOALS_DIR}" ]; then
    echo "[error] iv${IV} subtree missing after unpack: ${EXTRA_GOALS_DIR}" >&2
    exit 1
fi
# guard against a wrong-interval tree sneaking in
stray=$(find "${EXTRA_GOALS_ROOT}/FIXED_INTERVAL_SWEEP" -mindepth 1 -maxdepth 1 -type d ! -name "iv${IV}" | wc -l)
if [ "${stray}" -ne 0 ]; then
    echo "[error] unpacked tree contains intervals other than iv${IV}" >&2
    exit 1
fi
n_goal_demos=$(find "${EXTRA_GOALS_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' | wc -l)
echo "[unpack] EXTRA_GOALS_DIR=${EXTRA_GOALS_DIR} (${n_goal_demos} demos)"

# --- stage demos present in BOTH the h5 dir and the EXTRA_KEYPOINTS tree ----
THREADS="${RSYNC_THREADS:-32}"
mkdir -p "${DEST_DATA_DIR}"

demos_h5=$(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name 'demo_*.h5' -printf '%f\n' | sed 's/\.h5$//' | sort)
demos_goals=$(find "${EXTRA_GOALS_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' -printf '%f\n' | sort)
demos_common=$(comm -12 <(echo "${demos_h5}") <(echo "${demos_goals}"))
n_common=$(echo -n "${demos_common}" | grep -c . || true)
echo "[stage] h5 demos: $(echo -n "${demos_h5}" | grep -c . || true), goal demos: $(echo -n "${demos_goals}" | grep -c . || true), common: ${n_common}"
if [ "${n_common}" -eq 0 ]; then
    echo "[stage] ERROR: no demo present in both NO_GMM_H5_DIR and EXTRA_GOALS_DIR." >&2
    exit 1
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

# --- inject extra goal keys into the staged, node-local copy ---------------
echo "[inject] ensuring obs/goal_gripper_pts_fixed_interval exists in staged demos (from ${SOURCE_TAG})"
(
    cd "${REPO_DIR}"
    USE_TF=0 \
    GIT_LFS_SKIP_SMUDGE=1 \
    PYTHONNOUSERSITE=1 \
    pixi run python generate_non_gmm_goals_for_low_level.py \
        --dataset_dir "${DEST_DATA_DIR}" \
        --inject_extra_goals \
        --extra_goals_dir "${EXTRA_GOALS_DIR}"
)

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

RUN_NAME="kitchen_D1_APPROACH2_${SOURCE_TAG}_c1_${C1}_${staged_count}demo_dinov2_DIT_grogu423_blackwell"

# batch_size=128: measured ~0.24 GiB/sample over a ~2.75 GiB floor, i.e. ~34 GiB
# at 128 -- comfortable on the Blackwell card. Drop it (e.g. 64) if you see OOM.
USE_TF=0 \
GIT_LFS_SKIP_SMUDGE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_CACHE_DIR=/project_data/held/teswaram/logs/wandb_cache \
WANDB_DATA_DIR=/project_data/held/teswaram/logs/wandb_data \
PYTHONNOUSERSITE=1 \
pixi run python diffusion_policy/train.py \
    --config-name=train_flow_matching_dit_goal_gmm_workspace.yaml \
    task=MimicGen_Tasks/kitchen_goal_gmm_aux \
    task.dataset.data_dir="${DEST_DATA_DIR}" \
    +task.dataset.goal_source=${GOAL_SOURCE} \
    policy.aux_gmm_loss_weight=${C1} \
    logging.project=mimicgen_tasks \
    logging.name=${RUN_NAME} \
    name=${RUN_NAME} \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
