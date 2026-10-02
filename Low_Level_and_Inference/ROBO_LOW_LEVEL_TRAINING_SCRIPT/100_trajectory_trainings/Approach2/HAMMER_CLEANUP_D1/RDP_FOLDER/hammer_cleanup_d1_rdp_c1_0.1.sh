#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1   # 1 python process per node (PyTorch Lightning rejects -n / --ntasks)
#SBATCH --cpus-per-task=12    # 12 CPU cores for the python process (dataloader workers etc.)
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 48:00:00 # 48-hour budget
#SBATCH --job-name hammer-cleanup-d1-rdp-c1-0.1
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# ===========================================================================
# APPROACH 2 on HAMMER_CLEANUP_D1, RDP GOALS  —  c1 = 0.1
# ===========================================================================
# Sibling of ../BAYESIAN_FOLDER/hammer_cleanup_d1_c1_0.1.sh. The ONLY difference
# is which subgoal supervises the auxiliary GMM head:
#
#     Bayesian : obs/goal_gripper_pts
#     RDP      : obs/goal_gripper_pts_rdp      <-- this script
#
# Everything else — dataset, staging, c1, encoder, batch size, epochs, seed —
# is byte-identical, so a difference in results is attributable to the goal
# definition alone.
#
# The low-level policy is NOT given the goal. The RDP subgoal supervises a GMM
# head that reads the same 3D-grounded visual tokens the DiT cross-attends to,
# so the goal shapes the visual representation rather than being an input.
#
#     total_loss = flow_loss + c1 * gmm_loss
#
# c1 = policy.aux_gmm_loss_weight. Measured at init on this dataset (Bayesian
# goals; the RDP target has the same shape and comparable scale):
#     fm_loss  ~ 1.37     |grad_fm  -> shared trunk| ~ 1.47
#     gmm_loss ~ 10.05    |grad_gmm -> shared trunk| ~ 3.78  (per unit c1)
#   c1 = 0.136 equalises the two loss magnitudes
#   c1 = 0.389 equalises the gradient each delivers to the shared trunk
#   c1 = 0.1   -> auxiliary gradient is ~26% of the flow gradient: a real
#                 signal, but subordinate to the actual task. Matches the
#                 Bayesian run this is being compared against.
# Both terms are logged separately as train_fm_loss / train_goal_gmm_loss.
#
# ---------------------------------------------------------------------------
# HOW THE RDP GOAL IS SELECTED  (differs from the Approach 1 goal-source scripts)
# ---------------------------------------------------------------------------
# The Approach 1 goal-source scripts select the goal with
# +task.dataset.goal_source=<source>. That does NOT
# work here. goal_source remaps EVERY 'goal_gripper'-typed key to obs/<key>_rdp:
#
#     for key in self.goal_gripper_keys:
#         key_to_h5path[key] = f'obs/{key}_{self.goal_source}'
#
# Approach 1 configs type only goal_gripper_pts that way, so the blanket remap
# is harmless. Approach 2 ALSO types present_gripper_pts as goal_gripper (it
# supplies the 4 gripper anchors for the mixture), and obs/present_gripper_pts_rdp
# does not exist in the h5 — nor should it, since the present gripper pose is
# independent of how goals are defined. goal_source=rdp would therefore look for
# a key that deliberately does not exist.
#
# So instead this run selects the target by NAME:
#     task=MimicGen_Tasks/hammercleanup_D1_goal_gmm_aux_rdp   (shape_meta
#         declares goal_gripper_pts_rdp in place of goal_gripper_pts)
#     policy.aux_goal_key=goal_gripper_pts_rdp                  (the policy
#         reads shape_meta["obs"][aux_goal_key] for n_keypoints)
# present_gripper_pts is untouched.
#
# Uses the SAME NO_GMM h5 dataset as the Bayesian Approach 2 run — the goals
# are already inside those files (verified: all 100 demos carry
# obs/goal_gripper_pts_rdp with shape (T, 4, 3)). No extra npz tree, no
# EXTRA_KEYPOINTS staging, and no gt/predicted mix: in Approach 2 the goal is a
# supervision target, never an input, so there is no train/inference mismatch to
# hedge against.
#
# Trains from scratch. At ~550 steps/epoch (batch 128, 100 demos) the full 100
# epochs is expected to fit inside the 48-hour budget in a single job.
# checkpoint_every=5, so if it does not, resume with:
#   RESUME_CKPT=/path/to/epoch_N.ckpt sbatch this_script.sh
#
# NUM_DEMOS defaults to 100. Override at submission time:
#   NUM_DEMOS=200 sbatch this_script.sh

set -euo pipefail
set -x

export PATH="$HOME/.pixi/bin:$PATH"

# --- auxiliary loss weight (kept equal to the Bayesian sibling) -----------
C1=0.1

# --- goal source ---------------------------------------------------------
# Selected by task-config + aux_goal_key (see the note above), NOT by
# task.dataset.goal_source.
GOAL_KEY=goal_gripper_pts_rdp
TASK_CFG=MimicGen_Tasks/hammercleanup_D1_goal_gmm_aux_rdp

# --- demo selection ------------------------------------------------------
NUM_DEMOS="${NUM_DEMOS:-100}"
echo "[config] c1=${C1}, goal_key=${GOAL_KEY}, NUM_DEMOS=${NUM_DEMOS} (demo_0.h5 .. demo_$((NUM_DEMOS-1)).h5)"

# --- paths ---------------------------------------------------------------
NO_GMM_H5_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Dataset/LOW_LEVEL_GROOT_TRAINING_DATASET/NO_GMM_DATASET/HAMMER_CLEANUP_D1"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

RESUME_CKPT="${RESUME_CKPT:-}"

existing_h5_count=$(find "${NO_GMM_H5_DIR}" -maxdepth 1 -name "*.h5" 2>/dev/null | wc -l)
echo "[data] existing *.h5 in ${NO_GMM_H5_DIR}: ${existing_h5_count}"
if [ "${existing_h5_count}" -lt "${NUM_DEMOS}" ]; then
    echo "[data] ERROR: need ${NUM_DEMOS} h5 files, found ${existing_h5_count}." >&2
    echo "[data] Run generate_non_gmm_goals_for_low_level.py --no_gmm first." >&2
    exit 1
fi

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
DEST_DATA_DIR="${SCRATCH_ROOT}/Hammer_Cleanup_D1_Approach2_RDP_${NUM_DEMOS}demo"

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

# --- verify the RDP goal key is actually present -------------------------
# Fail loudly here rather than deep inside the dataset loader: a missing key
# would otherwise surface as an opaque h5 KeyError mid-training.
# Must run AFTER `cd ${REPO_DIR}` and via `pixi run` — bare `python` on a
# compute node is the system interpreter and has no h5py, which under `set -e`
# would kill the job before training ever starts.
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

RUN_NAME="hammercleanup_D1_APPROACH2_RDP_c1_${C1}_${NUM_DEMOS}demo"

# batch_size=128 matches the Bayesian Approach 2 sibling. Measured scaling for
# this architecture is ~0.24 GiB/sample over a ~2.75 GiB floor, i.e. ~34 GiB at
# 128 — comfortable on an 80 GB H100.
#
# Keep every knob below identical to ../BAYESIAN_FOLDER/hammer_cleanup_d1_c1_0.1.sh;
# a comparison across differing batch size / epochs / seed says nothing about
# the goal definition.
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
    logging.project=mimicgen_tasks \
    logging.name=${RUN_NAME} \
    name=${RUN_NAME} \
    dataloader.batch_size=128 \
    dataloader.num_workers=16 \
    training.checkpoint_every=5 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
