#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p all
#SBATCH -t 00:10:00
#SBATCH --job-name resume-driver
#SBATCH -o /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/resume-driver_%x_job_%j.out
#SBATCH -e /home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/theya_ROBO_LOW_LEVEL_TRAINING_SCRIPT/logs/resume-driver_%x_job_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=teswaram@andrew.cmu.edu

# ===========================================================================
# Self-chaining resume driver. Runs (no GPU needed) after a training job
# finishes -- successfully, by walltime, or by failure -- via
# --dependency=afterany. It:
#   1. Finds the newest checkpoints/latest.ckpt under outputs/ for TASK_GLOB.
#   2. Reads the saved epoch from that checkpoint.
#   3. If epoch >= TARGET_EPOCHS: stops, nothing more to do.
#   4. Otherwise: checks whether the Blackwell node (grogu-4-23, constraint
#      6000Blackwell) has any free GPU right now; picks that if so, else
#      falls back to A6000. Submits TRAIN_SCRIPT with RESUME_CKPT set to the
#      found checkpoint, and resubmits ITSELF with --dependency=afterany on
#      that new training job, so the chain continues until TARGET_EPOCHS is
#      reached (or MAX_LINKS safety cap is hit).
#
# Required env vars (passed via `sbatch --export=` at submission time):
#   TASK_NAME        e.g. kitchen_goal_gmm_aux        (hydra task.name, used to find outputs/)
#   RUN_NAME_GLOB    e.g. "kitchen_D1_APPROACH2_awe_greedy_th0.35_grip_c1_0.1_*demo_dinov2_DIT_grogu"
#   TRAIN_SCRIPT     absolute path to the *_grip_grogu.sh training script
#   DRIVER_SCRIPT    absolute path to this script (for self-resubmission)
#   TARGET_EPOCHS    default 100
#   LINK             current chain-link number (default 1), safety cap MAX_LINKS
# ===========================================================================

set -uo pipefail
set -x

REPO_DIR="/home/teswaram/code/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"
TARGET_EPOCHS="${TARGET_EPOCHS:-100}"
LINK="${LINK:-1}"
MAX_LINKS="${MAX_LINKS:-15}"

echo "[driver] TASK_NAME=${TASK_NAME} RUN_NAME_GLOB=${RUN_NAME_GLOB} link=${LINK}/${MAX_LINKS}"

if [ "${LINK}" -gt "${MAX_LINKS}" ]; then
    echo "[driver] MAX_LINKS (${MAX_LINKS}) reached without hitting TARGET_EPOCHS -- stopping chain, needs manual look." >&2
    exit 1
fi

# --- find newest checkpoint for this task -----------------------------------
cd "${REPO_DIR}"
LATEST_CKPT=""
LATEST_CKPT=$(find outputs -maxdepth 5 -type f -path "*${RUN_NAME_GLOB}*/checkpoints/latest.ckpt" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -n1 | cut -d' ' -f2-)

if [ -z "${LATEST_CKPT}" ]; then
    echo "[driver] no checkpoint found matching outputs/*/*${RUN_NAME_GLOB}*/checkpoints/latest.ckpt -- previous job may have died before epoch ${TARGET_EPOCHS}/checkpoint_every. Resubmitting training from scratch is NOT automatic; investigate manually." >&2
    exit 1
fi
echo "[driver] latest checkpoint: ${LATEST_CKPT}"

# --- read epoch reached from the checkpoint ----------------------------------
CKPT_EPOCH=$(PYTHONNOUSERSITE=1 USE_TF=0 GIT_LFS_SKIP_SMUDGE=1 pixi run python - "${LATEST_CKPT}" <<'PYEOF'
import sys, dill, torch
payload = torch.load(sys.argv[1], map_location="cpu", pickle_module=dill)
print(dill.loads(payload["pickles"]["epoch"]))
PYEOF
)
echo "[driver] checkpoint epoch: ${CKPT_EPOCH}"

if [ "${CKPT_EPOCH}" -ge "${TARGET_EPOCHS}" ]; then
    echo "[driver] epoch ${CKPT_EPOCH} >= target ${TARGET_EPOCHS} -- training complete, chain stops."
    exit 0
fi

# --- pick GPU: Blackwell if a GPU is actually free on it, else A6000 --------
BLACKWELL_NODE="grogu-4-23"
free_on_blackwell=$(scontrol show node "${BLACKWELL_NODE}" 2>/dev/null \
    | grep -oP 'gres/gpu=\K[0-9]+' )
alloc_gpu=$(scontrol show node "${BLACKWELL_NODE}" 2>/dev/null \
    | grep -oP 'AllocTRES=.*gres/gpu=\K[0-9]+' )
total_gpu=$(scontrol show node "${BLACKWELL_NODE}" 2>/dev/null \
    | grep -oP 'CfgTRES=.*gres/gpu=\K[0-9]+' )
alloc_gpu="${alloc_gpu:-0}"
total_gpu="${total_gpu:-8}"

if [ "${alloc_gpu}" -lt "${total_gpu}" ]; then
    CONSTRAINT="6000Blackwell"
    PARTITION="all"
    echo "[driver] Blackwell has a free GPU (${alloc_gpu}/${total_gpu} allocated) -> submitting there."
else
    CONSTRAINT="A6000"
    PARTITION="all"
    echo "[driver] Blackwell fully allocated (${alloc_gpu}/${total_gpu}) -> falling back to A6000."
fi

# --- submit the continuation training job -----------------------------------
train_submit_out=$(sbatch --parsable \
    -C "${CONSTRAINT}" \
    -p "${PARTITION}" \
    --export=ALL,RESUME_CKPT="${LATEST_CKPT}" \
    "${TRAIN_SCRIPT}")
new_train_jobid="${train_submit_out}"
echo "[driver] submitted continuation training job ${new_train_jobid} (constraint=${CONSTRAINT}) resuming from ${LATEST_CKPT}"

# --- resubmit this driver to run again after the new training job ends ------
next_link=$(( LINK + 1 ))
driver_submit_out=$(sbatch --parsable \
    --dependency=afterany:${new_train_jobid} \
    --export=ALL,TASK_NAME="${TASK_NAME}",RUN_NAME_GLOB="${RUN_NAME_GLOB}",TRAIN_SCRIPT="${TRAIN_SCRIPT}",DRIVER_SCRIPT="${DRIVER_SCRIPT}",TARGET_EPOCHS="${TARGET_EPOCHS}",LINK="${next_link}",MAX_LINKS="${MAX_LINKS}" \
    "${DRIVER_SCRIPT}")
echo "[driver] chained next driver as job ${driver_submit_out}, dependent on afterany:${new_train_jobid}"
