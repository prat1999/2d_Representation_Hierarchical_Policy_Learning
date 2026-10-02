#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# RUN THIS ON YOUR LOCAL MACHINE, NOT ON PSC. Do NOT sbatch it.
#
# Hard-coded downloader for the 6 AWE_GRIP low-level runs (100 demo,
# {hammer_cleanup, kitchen, coffee_preparation}) in outputs/2026.09.08 and
# outputs/2026.09.09:
#
#   goal_gripper   (2026.09.08)  DiT conditioned on the single AWE GT goal
#   wca_rope_goals (2026.09.09)  WCA + RoPE4D grounded trunk with goal tokens,
#                                AWE GT/predicted gt-mix p=0.5
#
# Pulls the epochs in EPOCHS (default 20/40/60/80/99) through the `psc-data`
# SSH alias, one local folder per run:
#
#   <DEST>/<task>_d1_awe_grip_<variant>/
#       checkpoints/epoch_{20,40,60,80,99}.ckpt   (merged across legs)
#       .hydra/  logs.json.txt  *.yaml            (from leg 1)
#       resume_legs/<leg-2 dir name>/             (leg-2 .hydra + logs, if any)
#       .psc-source                               (remote dirs this came from)
#
# None of the six runs has been resumed, so every leg-2 slot in RUNS is empty.
# If a run is ever resumed into a new output dir, put that dir in the third
# field: leg 1 holds epochs <= the resume epoch, leg 2 the rest, and both rsync
# into the same checkpoints/ folder.
#
# Checkpoints are ~3.5 GB (goal_gripper) / ~4.4 GB (wca_rope_goals) each.
# 6 runs x 5 epochs = ~120 GB total.
#
# As of 2026-09-10 two runs are STILL TRAINING (48h jobs, ~16h in):
#   coffee_preparation_d1_awe_grip_wca_rope_goals   at epoch_65
#   kitchen_d1_awe_grip_wca_rope_goals              at epoch_70
# Their missing epochs are simply skipped; re-run this script later and only
# the new files are fetched.
#
# Resumable and non-destructive: nothing is ever deleted, on PSC or locally.
# Complete local files are skipped by rsync; partial ones are resumed.
#
# Usage:
#   ./download_awe_grip_ckpts_to_local.sh [-n] [-v] [-j N] [-d DEST] [-H HOST] [-e LIST]
#     -n, --dry-run   print the plan + which remote checkpoints exist; copy nothing
#     -v, --verbose   echo every rsync command before running it
#     -j, --jobs N    runs transferred concurrently (default 2)
#     -d, --dest DIR  local root; run folders are created directly inside it
#                     (default: the current directory)
#     -H, --host H    SSH alias for the PSC data node (default psc-data)
#     -e, --epochs L  comma-separated epochs to fetch (default 20,40,60,80,99);
#                     e.g. -e 99 for final checkpoints only
#
# The remote listing is informational only. rsync always asks every leg for
# every epoch in EPOCHS; a file that is not there is simply not matched. A
# local check at the end reports exactly which checkpoints arrived.
# ---------------------------------------------------------------------------

set -euo pipefail

DATA_HOST="psc-data"
LOCAL_DEST="$(pwd)"
JOBS=2
DRY_RUN=0
VERBOSE=0

# Epochs to fetch from every run. Override with -e / --epochs.
EPOCHS=(20 40 60 80 99)

PSC_OUTPUTS="/ocean/projects/cis240052p/pbhowal/2d_Representation_Hierarchical_Policy_Learning/MimicGen_Uncertainty_Code/Low_Level_Policy/2d_Representation_Hierarchical_Policy_Learning/Low_Level_and_Inference/outputs"

# local_name|leg1_dir|leg2_dir      (leg dirs are relative to PSC_OUTPUTS; leg2 may be empty)
RUNS=(
  # ---- goal_gripper (single AWE GT goal, no GMM) -- 2026.09.08 ---------------
  "hammer_cleanup_d1_awe_grip_goal_gripper|2026.09.08/09.55.57_hammercleanup_D1_goal_gripper_awe_grip_100demo_dinov2_DIT_hammercleanup_D1_goal_gripper|"
  "kitchen_d1_awe_grip_goal_gripper|2026.09.08/09.56.48_kitchen_D1_goal_gripper_awe_grip_100demo_dinov2_DIT_kitchen_goal_gripper|"
  "coffee_preparation_d1_awe_grip_goal_gripper|2026.09.08/09.57.05_coffee_preperation_D1_goal_gripper_awe_grip_100demo_dinov2_DIT_coffee_preperation_goal_gripper|"
  # ---- WCA + RoPE4D goal tokens, AWE gt-mix p=0.5 -- 2026.09.09 -------------
  "hammer_cleanup_d1_awe_grip_wca_rope_goals|2026.09.09/08.19.06_groot_GMM_WCA_ROPE_GOALS_100demo_dinov2_HammerCleanup_D1_AWE_GRIP_GTMIX_p0.5_hammercleanup_D1_gmm_goal_gt_mix_rope|"
  "kitchen_d1_awe_grip_wca_rope_goals|2026.09.09/08.20.30_groot_GMM_WCA_ROPE_GOALS_100demo_dinov2_Kitchen_D1_AWE_GRIP_GTMIX_p0.5_kitchen_D1_gmm_goal_gt_mix_rope|"
  "coffee_preparation_d1_awe_grip_wca_rope_goals|2026.09.09/08.20.36_groot_GMM_WCA_ROPE_GOALS_100demo_dinov2_Coffee_Preperation_D1_AWE_GRIP_GTMIX_p0.5_coffee_preperation_gmm_goal_gt_mix_rope|"
)

# ---------------------------------------------------------------------------
die() { echo "error: $*" >&2; exit 2; }

while (($#)); do
  case "$1" in
    -n|--dry-run) DRY_RUN=1; shift ;;
    -v|--verbose) VERBOSE=1; shift ;;
    -j|--jobs)    (($# >= 2)) || die "$1 requires a value"; JOBS="$2"; shift 2 ;;
    -d|--dest)    (($# >= 2)) || die "$1 requires a value"; LOCAL_DEST="$2"; shift 2 ;;
    -H|--host)    (($# >= 2)) || die "$1 requires a value"; DATA_HOST="$2"; shift 2 ;;
    -e|--epochs)  (($# >= 2)) || die "$1 requires a value"; IFS=',' read -r -a EPOCHS <<< "$2"; shift 2 ;;
    -h|--help)    sed -n '2,47p' "$0"; exit 0 ;;
    *)            die "unknown argument: $1" ;;
  esac
done

[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
((${#EPOCHS[@]})) || die "--epochs must list at least one epoch"
for ep in "${EPOCHS[@]}"; do [[ "$ep" =~ ^[0-9]+$ ]] || die "bad epoch '$ep' (integers only)"; done
command -v ssh   >/dev/null 2>&1 || die "ssh is not installed"
command -v rsync >/dev/null 2>&1 || die "rsync is not installed"

SSH_CMD="ssh -T -o BatchMode=yes -o Compression=no"

declare -a PLAN_NAME=() PLAN_LEG1=() PLAN_LEG2=()
for entry in "${RUNS[@]}"; do
  IFS='|' read -r name leg1 leg2 <<< "$entry"
  PLAN_NAME+=("$name"); PLAN_LEG1+=("$leg1"); PLAN_LEG2+=("$leg2")
done

# ---------------------------------------------------------------------------
# Pre-flight (INFORMATIONAL ONLY). One short SSH round trip: one glob per leg.
# If it fails, the transfer still runs and simply asks rsync for every epoch.
# ---------------------------------------------------------------------------
echo "==> Listing checkpoints on $DATA_HOST"
preflight_ls="ls -1d"
for ((i = 0; i < ${#PLAN_NAME[@]}; i++)); do
  for leg in "${PLAN_LEG1[$i]}" "${PLAN_LEG2[$i]}"; do
    [[ -n "$leg" ]] || continue
    preflight_ls+=" $PSC_OUTPUTS/$leg/checkpoints/epoch_*.ckpt"
  done
done
preflight_err="$(mktemp)"
present_output="$(ssh -o BatchMode=yes "$DATA_HOST" "$preflight_ls" 2>"$preflight_err" || true)"

PREFLIGHT_OK=1
declare -A PRESENT=()
while IFS= read -r p; do
  [[ -n "$p" ]] && PRESENT["$p"]=1
done <<< "$present_output"
if ((${#PRESENT[@]} == 0)); then
  PREFLIGHT_OK=0
  echo "warning: remote listing returned nothing; the transfer will still request every epoch." >&2
  if [[ -s "$preflight_err" ]]; then
    echo "warning: ssh/ls stderr was:" >&2
    sed 's/^/    /' "$preflight_err" >&2
  fi
fi
rm -f "$preflight_err"

# ---------------------------------------------------------------------------
# Plan display. rsync will request all EPOCHS from every leg no matter what;
# this only shows where each epoch is expected to come from.
# ---------------------------------------------------------------------------
total_files=0
missing_report=""
echo
echo "Transfer plan (epochs: ${EPOCHS[*]}; ~3.5-4.4 GB per checkpoint):"
for ((i = 0; i < ${#PLAN_NAME[@]}; i++)); do
  name="${PLAN_NAME[$i]}"; leg1="${PLAN_LEG1[$i]}"; leg2="${PLAN_LEG2[$i]}"
  ep1=""; ep2=""
  if ((PREFLIGHT_OK)); then
    for ep in "${EPOCHS[@]}"; do
      if [[ -n "${PRESENT[$PSC_OUTPUTS/$leg1/checkpoints/epoch_${ep}.ckpt]:-}" ]]; then
        ep1+="$ep "; ((total_files += 1))
      elif [[ -n "$leg2" && -n "${PRESENT[$PSC_OUTPUTS/$leg2/checkpoints/epoch_${ep}.ckpt]:-}" ]]; then
        ep2+="$ep "; ((total_files += 1))
      else
        missing_report+="  $name: epoch_${ep}.ckpt not found in any leg"$'\n'
      fi
    done
  else
    ep1="? "; ep2="? "
  fi
  echo "  $name"
  echo "    -> $LOCAL_DEST/$name"
  echo "       leg1 [${ep1% }]  $leg1"
  [[ -n "$leg2" ]] && echo "       leg2 [${ep2% }]  $leg2"
done
if ((PREFLIGHT_OK)); then
  echo "  $total_files checkpoint(s) expected"
  if [[ -n "$missing_report" ]]; then
    echo
    echo "Not on PSC (will be skipped by rsync; still-training runs fill in later):"
    printf '%s' "$missing_report"
  fi
fi

if ((DRY_RUN)); then
  echo
  echo "==> Dry run complete; no local files were changed"
  exit 0
fi

mkdir -p "$LOCAL_DEST"

# ---------------------------------------------------------------------------
# rsync helpers. Checkpoints are incompressible, so SSH compression stays off.
# --append-verify resumes an interrupted copy and verifies the finished file.
# ---------------------------------------------------------------------------
RSYNC_BASE=(-a --append-verify --human-readable --info=progress2,stats1 -e "$SSH_CMD")

run_rsync() {
  if ((VERBOSE)); then
    printf '   $ rsync'; printf ' %q' "$@"; printf '\n'
  fi
  rsync "$@"
}

# Checkpoint filters are the same for every leg: ask for every epoch in
# EPOCHS, let rsync skip the ones that do not exist on that leg.
CKPT_FILTERS=(--include='/checkpoints/')
for ep in "${EPOCHS[@]}"; do
  CKPT_FILTERS+=(--include="/checkpoints/epoch_${ep}.ckpt")
done
CKPT_FILTERS+=(--exclude='*')

# sync_ckpts REMOTE_DIR LOCAL_DIR
sync_ckpts() {
  run_rsync "${RSYNC_BASE[@]}" "${CKPT_FILTERS[@]}" "$DATA_HOST:$1/" "$2/"
}

# sync_meta REMOTE_DIR LOCAL_DIR
sync_meta() {
  mkdir -p "$2"
  run_rsync "${RSYNC_BASE[@]}" \
    --include='/logs.json.txt' --include='/*.yaml' --include='/.hydra/***' \
    --exclude='*' \
    "$DATA_HOST:$1/" "$2/"
}

sync_run() {
  local i="$1"
  local name="${PLAN_NAME[$i]}" leg1="${PLAN_LEG1[$i]}" leg2="${PLAN_LEG2[$i]}"
  local dest="$LOCAL_DEST/$name"
  mkdir -p "$dest/checkpoints"

  echo "==> [$name] leg 1 checkpoints + metadata"
  sync_ckpts "$PSC_OUTPUTS/$leg1" "$dest"
  sync_meta  "$PSC_OUTPUTS/$leg1" "$dest"

  if [[ -n "$leg2" ]]; then
    echo "==> [$name] leg 2 checkpoints + metadata"
    sync_ckpts "$PSC_OUTPUTS/$leg2" "$dest"
    sync_meta  "$PSC_OUTPUTS/$leg2" "$dest/resume_legs/${leg2##*/}"
  fi

  {
    echo "$DATA_HOST:$PSC_OUTPUTS/$leg1"
    [[ -n "$leg2" ]] && echo "$DATA_HOST:$PSC_OUTPUTS/$leg2"
  } > "$dest/.psc-source"
  echo "==> [$name] complete"
}

worker() {
  local wid="$1" i rc=0
  for ((i = wid; i < ${#PLAN_NAME[@]}; i += JOBS)); do
    sync_run "$i" || { echo "!! [${PLAN_NAME[$i]}] transfer failed (rerun to resume)" >&2; rc=1; }
  done
  return "$rc"
}

declare -a pids=()
nworkers="$JOBS"; ((nworkers > ${#PLAN_NAME[@]})) && nworkers="${#PLAN_NAME[@]}"
for ((w = 0; w < nworkers; w++)); do
  worker "$w" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done

# ---------------------------------------------------------------------------
# Local verification: what actually landed in each checkpoints/ folder.
# ---------------------------------------------------------------------------
echo
echo "Local result ($LOCAL_DEST):"
got_total=0
for ((i = 0; i < ${#PLAN_NAME[@]}; i++)); do
  name="${PLAN_NAME[$i]}"; have=""; lack=""
  for ep in "${EPOCHS[@]}"; do
    f="$LOCAL_DEST/$name/checkpoints/epoch_${ep}.ckpt"
    if [[ -s "$f" ]]; then have+="$ep "; ((got_total += 1)); else lack+="$ep "; fi
  done
  printf '  %-50s have [%s]' "$name" "${have% }"
  [[ -n "$lack" ]] && printf '  missing [%s]' "${lack% }"
  printf '\n'
done
echo "  $got_total checkpoint file(s) present locally"

((failed)) && die "one or more transfers failed; rerun the same command to resume"
if ((got_total == 0)); then
  die "no checkpoints were transferred. Rerun with -v to see the rsync commands, and check that '$DATA_HOST' can read $PSC_OUTPUTS"
fi

echo
echo "==> Done. Synced ${#PLAN_NAME[@]} run(s) into $LOCAL_DEST"
