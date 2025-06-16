#!/usr/bin/env bash
# ------------------------------------------------------------
# Train DRAS over all Theta jobsets.
# Run from inside  src/  directory:
#     cd src && bash train.bash
# ------------------------------------------------------------
set -euo pipefail

# --- Paths relative to the PROJECT ROOT ---
# This script assumes it's being run from the `src` directory.
# All paths passed to cqsim.py must be relative to the project root.
PROJECT_ROOT=".."
CFG="config_sys.set"
JOBSET_DIR="$PROJECT_ROOT/data/jobsets"
RESULT_DIR="$PROJECT_ROOT/results/theta"
DEBUG_DIR="$PROJECT_ROOT/debug/theta"

mkdir -p "$RESULT_DIR" "$DEBUG_DIR"

EP=0
for JOBSET_PATH in "$JOBSET_DIR"/{sampled,real,synthetic}/*.swf ; do
  [[ -e "$JOBSET_PATH" ]] || { echo "No SWF files found in $JOBSET_DIR, have you run create_jobsets.py?"; exit 1; }

  EP=$((EP+1))
  BASENAME=$(basename "$JOBSET_PATH" .swf)

  echo "=== episode $EP  ( $BASENAME ) ==="

  # IMPORTANT: We are in `src`, so cqsim.py is local, but all paths
  # passed to it must be relative to the project root (`../`)
  python cqsim.py \
    --config_sys  "$CFG" \
    --job         "${JOBSET_PATH#"$PROJECT_ROOT/"}" \
    --node        "${JOBSET_PATH#"$PROJECT_ROOT/"}" \
    --weight_num  "${EP}" \
    --is_training 1 \
    --output      "$BASENAME" \
    --debug       "debug_$BASENAME" \
    --path_in     "$PROJECT_ROOT/" \
    --path_fmt    "$PROJECT_ROOT/" \
    --path_out    "${RESULT_DIR#"$PROJECT_ROOT/"}/" \
    --path_debug  "${DEBUG_DIR#"$PROJECT_ROOT/"}/" \
    --debug_lvl   10 
done