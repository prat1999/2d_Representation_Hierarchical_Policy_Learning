#!/bin/bash
# Submit the KITCHEN_D1 Approach2 fixed-interval sweep (iv5/iv10/iv30/iv50).
# Skips any interval whose EXTRA_KEYPOINTS tree isn't populated yet -- the
# sweep trees are still being copied in, so re-run this as they land.
#   ./submit_fixed_interval_sweep.sh            # all four
#   ./submit_fixed_interval_sweep.sh 10 30      # just those
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_ROOT="/project_data/held/teswaram/data/D1/FIXED_INTERVAL_SWEEP"
IVS=("${@:-}")
[ -z "${IVS[0]}" ] && IVS=(5 10 30 50)

for IV in "${IVS[@]}"; do
    goals="${SWEEP_ROOT}/iv${IV}/EXTRA_KEYPOINTS_fixed_interval/KITCHEN_D1"
    n=$(find "${goals}" -mindepth 1 -maxdepth 1 -type d -name 'demo_*' 2>/dev/null | wc -l)
    if [ "${n}" -eq 0 ]; then
        echo "[skip] iv${IV}: no demo_* under ${goals} (not copied in yet)"
        continue
    fi
    echo "[submit] iv${IV} (${n} demos)"
    sbatch "${HERE}/kitchenD1_approach2_fixed_interval_iv${IV}_grogu423_blackwell.sh"
done
