#!/bin/bash
# Inject fixed_interval keypoint goals into the rate-ablation h5 datasets.
#
# For each of the 15 <task>_<rate>hz dirs under RATE_ABLATIONS_ROOT (3 tasks x
# 5 control rates), runs generate_non_gmm_goals_for_low_level.py
# --inject_extra_goals with the matching NPZ_FIXED_KEYPOINTS/<pair> npz tree.
# The npz frames carry only goal_gripper_pcd_fixed_interval, so this backfills
# exactly one new dataset per demo file: obs/goal_gripper_pts_fixed_interval
# (T, 4, 3) float32. The injector is append-only and idempotent — already-
# complete files are skipped — so this script is safe to rerun after a partial
# failure; only the pairs/files that are missing keys get touched.
#
# NOTE: this WRITES IN PLACE into the h5 files under RATE_ABLATIONS_ROOT
# (eswaramo's project space; files are group-writable). Each file grows by a
# few KB only.
#
# Usage (login or interactive node, no GPU needed):
#   bash scripts/inject_fixed_keypoints_rate_ablations.sh
#
# Env overrides:
#   PARALLEL=5     max concurrent pairs (each pair is one python process;
#                  the 15 pairs touch disjoint files so any value is safe —
#                  Lustre IO is the bottleneck, not CPU)
#   RATE_ABLATIONS_ROOT=...   alternate dataset root
#   ONLY=kitchen_d1_10hz      run a single named pair (repeatable via glob,
#                             e.g. ONLY='kitchen_*' for all kitchen rates)

set -euo pipefail

export PATH="$HOME/.pixi/bin:$PATH"

RATE_ABLATIONS_ROOT="${RATE_ABLATIONS_ROOT:-/ocean/projects/cis240052p/eswaramo/data/rate_ablations/h5}"
NPZ_ROOT="${RATE_ABLATIONS_ROOT}/NPZ_FIXED_KEYPOINTS"
REPO_DIR="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference"
PARALLEL="${PARALLEL:-5}"
ONLY="${ONLY:-*}"

LOG_DIR="${REPO_DIR}/logs/inject_fixed_keypoints_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# --- collect pairs ---------------------------------------------------------
# A pair is valid when both the h5 dir and its npz twin exist.
pairs=()
for d in "${RATE_ABLATIONS_ROOT}"/${ONLY}; do
    name=$(basename "${d}")
    [ "${name}" = "NPZ_FIXED_KEYPOINTS" ] && continue
    [ -d "${d}" ] || continue
    if [ ! -d "${NPZ_ROOT}/${name}" ]; then
        echo "[skip] ${name}: no matching dir in NPZ_FIXED_KEYPOINTS" >&2
        continue
    fi
    pairs+=("${name}")
done
if [ "${#pairs[@]}" -eq 0 ]; then
    echo "ERROR: no dataset pairs found under ${RATE_ABLATIONS_ROOT} (ONLY=${ONLY})" >&2
    exit 1
fi

echo "[inject] ${#pairs[@]} pair(s), up to ${PARALLEL} in parallel"
echo "[inject] logs: ${LOG_DIR}"
printf '  %s\n' "${pairs[@]}"

# --- worker ----------------------------------------------------------------
# One pair = one python process; pairs touch disjoint h5 files so concurrent
# workers never collide.
inject_one() {
    local name="$1"
    local log="${LOG_DIR}/${name}.log"
    if (
        cd "${REPO_DIR}" &&
        PYTHONNOUSERSITE=1 pixi run python generate_non_gmm_goals_for_low_level.py \
            --dataset_dir     "${RATE_ABLATIONS_ROOT}/${name}" \
            --inject_extra_goals \
            --extra_goals_dir "${NPZ_ROOT}/${name}"
    ) > "${log}" 2>&1; then
        echo "[done] ${name}: $(grep -o '\[inject\] done.*' "${log}" | tail -1)"
    else
        echo "[FAIL] ${name} — see ${log}" >&2
        return 1
    fi
}
export -f inject_one
export RATE_ABLATIONS_ROOT NPZ_ROOT REPO_DIR LOG_DIR

# --- run -------------------------------------------------------------------
# xargs -P fans the pairs out; exit status is nonzero if any worker failed.
set +e
printf '%s\n' "${pairs[@]}" | xargs -P "${PARALLEL}" -I {} bash -c 'inject_one "$1"' _ {}
status=$?
set -e

echo
if [ "${status}" -ne 0 ]; then
    echo "[inject] FINISHED WITH FAILURES — rerun this script after fixing; completed pairs are skipped automatically." >&2
    exit 1
fi

# --- summary ---------------------------------------------------------------
echo "[inject] all pairs completed. Per-pair results:"
for name in "${pairs[@]}"; do
    tail_line=$(grep -o '\[inject\] done.*' "${LOG_DIR}/${name}.log" | tail -1)
    echo "  ${name}: ${tail_line:-'(no summary line — inspect log)'}"
done
echo "[inject] logs kept in ${LOG_DIR}"
