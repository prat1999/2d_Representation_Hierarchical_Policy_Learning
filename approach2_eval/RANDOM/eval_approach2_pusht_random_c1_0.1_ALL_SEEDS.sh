#!/usr/bin/env bash
# APPROACH 2 eval on PushT: FlowMatchingDiTGoalGMM low-level policy ALONE.
#
# Sim (PushTImageEnv @256, legacy physics) counterpart of the Hammer_Cleanup_D1
# Approach 2 eval. There is NO high-level policy and no goal at rollout.
#
# Single script, runs seeds 100000 / 150000 / 250000 SEQUENTIALLY.
# Per-seed outputs land in:
#   ${SCRIPT_DIR}/APPROACH2_2D_DIT_PUSHT_50_SAMPLES_c1_0.1_<NTH>_SEED/
# Each seed has its own auto-resume bookkeeping (do_merge below); an interrupted
# run can be re-launched and continues from where it left off.
#
# Layout written per seed (identical to the MimicGen evals, so downstream
# analysis and the merge logic are unchanged):
#   args.json  results.jsonl  summary.json  media/*.mp4  media_with_goal_overlay/
# media_with_goal_overlay/ stays empty -- Approach 2 has no goal to overlay.
#
# Metrics per episode: max_coverage (normalized covered area, exact) and
# success (= coverage > 0.95). summary.json carries mean/std coverage and the
# success rate.

set -euo pipefail

# --------------------------------------------------------------------------- #
# Paths (all on PSC)
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Interpreter + policy deps come from the Mimicgen_Inference pixi env (it has
# torch/hydra/transformers/diffusers AND pygame/pymunk/shapely; its site-packages
# diffusion_policy is renamed _diffusion_policy_SHADOWED_BAK so the Low_Level
# namespace package resolves).
INFERENCE_ROOT="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Mimicgen_Inference/2d_Representation_Hierarchical_Policy_Learning"
LL_REPO="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"

ENV_PY="${INFERENCE_ROOT}/.pixi/envs/default/bin/python"

# LL: 206-demo Approach 2 RANDOM keypoint-variant run on PushT, c1 = 0.1,
# 200-epoch training (2026-08-27 batch, same budget as the WCA baseline).

LL_EXP_DIR="${LL_REPO}/outputs/2026.08.27/11.56.53_push_t_task_APPROACH2_RANDOM_c1_0.1_206demo_push_t_task_goal_gmm_aux_random"
LL_CKPT="epoch_199.ckpt"

# --------------------------------------------------------------------------- #
# Eval knobs  (mirroring the Hammer Approach 2 eval; max_steps from the
# training config's env_runner block)
# --------------------------------------------------------------------------- #
N_EPISODES=50
MAX_STEPS=300
N_OBS_STEPS=2
N_ACTION_STEPS=8
RENDER_SIZE=256
# rolling:        real t-1/t history (training's pad_before=1 convention).
# repeat_current: both obs steps are frame t (no t-1) -- tried, worse.
OBS_HISTORY=rolling

SAVE_VIDEOS=1
VIDEO_FPS=10

# _200EP: distinct from the 100-epoch model's result dirs, so the resume logic
# never mistakes the old runs for completed episodes of this checkpoint.
OUTPUT_BASE="APPROACH2_2D_DIT_PUSHT_50_SAMPLES_RANDOM_c1_0.1_200EP"

# --------------------------------------------------------------------------- #
# do_merge SRC DST ORIG_SEED  -- folds a sibling _RESUME_<N> dir into the main per-seed dir.
# --------------------------------------------------------------------------- #
do_merge() {
  local SRC="$1"
  local DST="$2"
  local THE_ORIG_SEED="$3"
  if [[ ! -f "${SRC}/results.jsonl" ]]; then
    echo "[merge] ${SRC} has no results.jsonl, skipping"
    return 0
  fi
  mkdir -p "${DST}/media" "${DST}/media_with_goal_overlay"
  "${ENV_PY}" - "${SRC}" "${DST}" "${THE_ORIG_SEED}" <<'PYEOF'
import json, sys, shutil
from pathlib import Path
src_dir = Path(sys.argv[1])
dst_dir = Path(sys.argv[2])
orig_seed = int(sys.argv[3])
src_jsonl = src_dir / "results.jsonl"
dst_jsonl = dst_dir / "results.jsonl"
n = 0
with open(src_jsonl) as fi, open(dst_jsonl, "a") as fo:
    for line in fi:
        d = json.loads(line)
        n += 1
        seed = d["seed"]
        new_ep = seed - orig_seed + 1
        outcome = "success" if d["success"] else "failure"
        d["episode"] = new_ep
        for key, sub in (("video", "media"),
                         ("video_with_goal_overlay", "media_with_goal_overlay")):
            old = d.get(key)
            if not old:
                continue
            candidates = [Path(old), src_dir / sub / Path(old).name]
            old_path = next((p for p in candidates if p.exists()), None)
            new_name = f"episode_{new_ep:03d}_seed_{seed}_{outcome}.mp4"
            dst_path = dst_dir / sub / new_name
            if old_path is not None:
                shutil.move(str(old_path), str(dst_path))
            else:
                print(f"[merge][WARN] missing source video for line {n}: {old}")
            d[key] = str(dst_dir / sub / new_name)
        fo.write(json.dumps(d) + "\n")
print(f"[merge] appended {n} rows from {src_jsonl} -> {dst_jsonl}")
PYEOF
  touch "${SRC}/.merged"
  echo "[merge] marked ${SRC}/.merged"
}

# --------------------------------------------------------------------------- #
# Sanity checks -- including the dependency whose silent absence bit the
# MimicGen evals (the Low_Level diffusion_policy namespace package) and the
# pusht env module.
# --------------------------------------------------------------------------- #
for f in "${LL_EXP_DIR}/.hydra/config.yaml" \
         "${LL_EXP_DIR}/checkpoints/${LL_CKPT}" \
         "${LL_REPO}/diffusion_policy/diffusion_policy/policy/flow_matching_dit_goal_gmm_policy.py" \
         "${LL_REPO}/diffusion_policy/diffusion_policy/env/pusht/pusht_image_env.py" \
         "${SCRIPT_DIR}/eval_approach2_pusht.py" \
         "${ENV_PY}"; do
  if [[ ! -e "${f}" ]]; then
    echo "[ERROR] missing required path: ${f}" >&2
    exit 1
  fi
done
# The regular diffusion_policy package in site-packages must stay renamed, or
# it shadows the Low_Level namespace package and the policy class "disappears".
SITE_DP="${INFERENCE_ROOT}/.pixi/envs/default/lib/python3.10/site-packages/diffusion_policy"
if [[ -e "${SITE_DP}" ]]; then
  echo "[ERROR] ${SITE_DP} exists -- it would shadow the Low_Level namespace" >&2
  echo "        package. Rename it back to _diffusion_policy_SHADOWED_BAK." >&2
  exit 1
fi

# --------------------------------------------------------------------------- #
# Launch env
# --------------------------------------------------------------------------- #
cd "${SCRIPT_DIR}"
export PYTHONNOUSERSITE=1
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}   # headless pygame
# $HOME is quota-tight; keep ffmpeg temp writes off it.
export TMPDIR="${TMPDIR:-/tmp/approach2_pusht_eval_$$}"
mkdir -p "${TMPDIR}"
# NOTE: no PYTHONPATH here on purpose -- eval_approach2_pusht.py does its own
# sys.path bootstrap (Low_Level repo only; PushT repo root must stay off it).

run_one_seed() {
  local sfx="$1" seed="$2"
  local ORIG_SEED=${seed}
  local MAIN_OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_BASE}_${sfx}_SEED"

  echo
  echo "==========================================================================="
  echo "[seed ${sfx}] SEED=${seed}  OUTPUT=${MAIN_OUTPUT_DIR}"
  echo "==========================================================================="

  shopt -s nullglob
  for resume_dir in "${MAIN_OUTPUT_DIR}_RESUME_"*; do
    if [[ -d "${resume_dir}" && ! -f "${resume_dir}/.merged" ]]; then
      echo "[merge] folding $(basename "${resume_dir}") into ${MAIN_OUTPUT_DIR}"
      do_merge "${resume_dir}" "${MAIN_OUTPUT_DIR}" "${ORIG_SEED}"
    fi
  done
  shopt -u nullglob

  local COMPLETED=0
  if [[ -f "${MAIN_OUTPUT_DIR}/results.jsonl" ]]; then
    COMPLETED=$(wc -l < "${MAIN_OUTPUT_DIR}/results.jsonl")
  fi

  if (( COMPLETED >= N_EPISODES )); then
    echo "[resume] all ${N_EPISODES} episodes already in ${MAIN_OUTPUT_DIR}. Skipping."
    return 0
  fi

  local PY_OUTPUT_DIR="${MAIN_OUTPUT_DIR}"
  local CUR_SEED=${seed}
  local CUR_N_EP=${N_EPISODES}
  if (( COMPLETED > 0 )); then
    CUR_SEED=$(( seed + COMPLETED ))
    CUR_N_EP=$(( N_EPISODES - COMPLETED ))
    PY_OUTPUT_DIR="${MAIN_OUTPUT_DIR}_RESUME_${COMPLETED}"
    echo "[resume] ${COMPLETED} episodes already in ${MAIN_OUTPUT_DIR}."
    echo "[resume] running ${CUR_N_EP} more (seeds ${CUR_SEED}..$(( CUR_SEED + CUR_N_EP - 1 )))."
    echo "[resume] Python writes to ${PY_OUTPUT_DIR}; will fold into ${MAIN_OUTPUT_DIR} on success."
  fi

  local VIDEO_FLAG
  if [[ "${SAVE_VIDEOS}" == "0" ]]; then
    VIDEO_FLAG=(--no-save-videos)
  else
    VIDEO_FLAG=(--save_videos --video_fps "${VIDEO_FPS}")
  fi

  "${ENV_PY}" "${SCRIPT_DIR}/eval_approach2_pusht.py" \
      --low_level_exp_dir    "${LL_EXP_DIR}"     \
      --low_level_checkpoint "${LL_CKPT}"        \
      --low_level_repo       "${LL_REPO}"        \
      --n_episodes           "${CUR_N_EP}"       \
      --max_steps            "${MAX_STEPS}"      \
      --seed                 "${CUR_SEED}"       \
      --n_obs_steps          "${N_OBS_STEPS}"    \
      --n_action_steps       "${N_ACTION_STEPS}" \
      --obs_history          "${OBS_HISTORY}"    \
      --render_size          "${RENDER_SIZE}"    \
      "${VIDEO_FLAG[@]}"                         \
      --output_dir           "${PY_OUTPUT_DIR}"

  if [[ "${PY_OUTPUT_DIR}" != "${MAIN_OUTPUT_DIR}" ]]; then
    echo "[merge] folding ${PY_OUTPUT_DIR} into ${MAIN_OUTPUT_DIR}"
    do_merge "${PY_OUTPUT_DIR}" "${MAIN_OUTPUT_DIR}" "${ORIG_SEED}"
  fi
}

# --------------------------------------------------------------------------- #
# Run all three seeds sequentially
# --------------------------------------------------------------------------- #
run_one_seed 1ST 100000
run_one_seed 2ND 150000
run_one_seed 3RD 250000

echo
echo "All seeds done. Outputs:"
for s in 1ST 2ND 3RD; do
  echo "  ${SCRIPT_DIR}/${OUTPUT_BASE}_${s}_SEED"
done
