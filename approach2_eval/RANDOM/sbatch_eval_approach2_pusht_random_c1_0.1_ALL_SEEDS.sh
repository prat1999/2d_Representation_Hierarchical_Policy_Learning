#!/bin/bash
#SBATCH -N 1                  # Number of nodes
#SBATCH --ntasks-per-node=1   # single python process (one rollout at a time)
#SBATCH --cpus-per-task=5     # per-GPU share on a v100-32 node (40 cores / 8 GPUs)
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1      # needed for the DiT forward. Rendering is pygame
                              # on CPU -- no EGL, unlike the MimicGen evals.
#SBATCH -t 8:00:00            # 3 seeds x 50 episodes x <=300 steps, sequential;
                              # PushT steps are far cheaper than MuJoCo ones.
#SBATCH --job-name a2-eval-pusht-random-c1-0.1
#SBATCH -o /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/PushT_Task/diffusion_policy/approach2_eval/RANDOM/logs/job_%j.out
#SBATCH -e /ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/PushT_Task/diffusion_policy/approach2_eval/RANDOM/logs/job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=pbhowal@andrew.cmu.edu

# ===========================================================================
# APPROACH 2 eval on PushT (c1 = 0.1) -- all three seeds.
# ===========================================================================
# THIN WRAPPER around the interactive script in this same folder:
#
#   eval_approach2_pusht_c1_0.1_ALL_SEEDS.sh
#
# All eval logic -- the 3 seeds, the per-seed output tree, and the resume+merge
# bookkeeping -- lives there and is NOT duplicated here. This file only supplies
# the SLURM allocation. Keeping it a wrapper means the batch and interactive
# paths can never drift apart.
#
# Submit:
#   sbatch sbatch_eval_approach2_pusht_c1_0.1_ALL_SEEDS.sh
#
# The eval is RESUMABLE: if the walltime runs out mid-run, resubmit -- the
# inner script counts completed episodes in results.jsonl and continues.
# ===========================================================================

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="${SCRIPT_DIR}/eval_approach2_pusht_random_c1_0.1_ALL_SEEDS.sh"

mkdir -p "${SCRIPT_DIR}/logs"

if [[ ! -e "${INNER_SCRIPT}" ]]; then
    echo "[ERROR] inner eval script not found: ${INNER_SCRIPT}" >&2
    exit 1
fi

# --- headless pygame ------------------------------------------------------
export SDL_VIDEODRIVER=dummy

# --- keep scratch off $HOME ----------------------------------------------
# $HOME is quota-tight (25 GB) and the ffmpeg encoder writes temp files.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export TMPDIR="/local/slurm-${SLURM_JOB_ID}/local/a2_eval_pusht"
else
    export TMPDIR="${TMPDIR:-/tmp}/a2_eval_pusht_$$"
fi
mkdir -p "${TMPDIR}"

# --- interpreter isolation -----------------------------------------------
# PYTHONNOUSERSITE stops ~/.local/lib packages from shadowing the pixi env.
# The inner script invokes the Mimicgen_Inference pixi python directly, so no
# pixi activation is required here.
export PYTHONNOUSERSITE=1

echo "[job]   ${SLURM_JOB_ID:-<interactive>} on $(hostname)"
echo "[gpu]   ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[tmp]   ${TMPDIR}"
echo "[inner] ${INNER_SCRIPT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

exec bash "${INNER_SCRIPT}"
