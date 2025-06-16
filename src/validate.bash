#!/usr/bin/env bash
# ------------------------------------------------------------
#  Evaluate a list of weights on the validation set
# ------------------------------------------------------------
set -euo pipefail

# --- Paths relative to the PROJECT ROOT ---
# This script assumes it's being run from the `src` directory.
# All paths passed to cqsim.py must be relative to the project root.
PROJECT_ROOT=".."
CFG="config_sys.set"
VALIDATION_JOBSET="$PROJECT_ROOT/data/jobsets/validation_2023_jan.swf"
RESULT_DIR="$PROJECT_ROOT/results/theta"
DEBUG_DIR="$PROJECT_ROOT/debug/theta"

mkdir -p "$RESULT_DIR" "$DEBUG_DIR"

# Check if validation set exists
[[ -e "$VALIDATION_JOBSET" ]] || { echo "Validation set not found: $VALIDATION_JOBSET"; exit 1; }

BASENAME=$(basename "$VALIDATION_JOBSET" .swf)

# Validate all weights (1-15) on the validation set
for WEIGHT_NUM in {1..100}; do
  echo "=== Validating weight $WEIGHT_NUM on $BASENAME ==="

  # IMPORTANT: We are in `src`, so cqsim.py is local, but all paths
  # passed to it must be relative to the project root (`../`)
  python cqsim.py \
    --config_sys  "$CFG" \
    --job         "${VALIDATION_JOBSET#"$PROJECT_ROOT/"}" \
    --node        "${VALIDATION_JOBSET#"$PROJECT_ROOT/"}" \
    --weight_num  "${WEIGHT_NUM}" \
    --is_training 0 \
    --output      "${BASENAME}_weight_${WEIGHT_NUM}" \
    --debug       "debug_${BASENAME}_weight_${WEIGHT_NUM}" \
    --path_in     "$PROJECT_ROOT/" \
    --path_fmt    "$PROJECT_ROOT/" \
    --path_out    "${RESULT_DIR#"$PROJECT_ROOT/"}/" \
    --path_debug  "${DEBUG_DIR#"$PROJECT_ROOT/"}/" \
    --debug_lvl   10
done