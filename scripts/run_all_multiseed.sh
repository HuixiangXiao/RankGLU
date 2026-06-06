#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
UNIVERSE="${UNIVERSE:-csi300}"
GPU="${GPU:-0}"
SECTIONS="${SECTIONS:-main,ablation,diagnostic}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="$REPO_ROOT/results/${UNIVERSE}_multiseed_${TIMESTAMP}"

mkdir -p "$OUT_DIR"

echo "Repository root: $REPO_ROOT"
echo "Output dir: $OUT_DIR"
echo "GPU: $GPU"
echo "Universe: $UNIVERSE"
echo "Sections: $SECTIONS"

cd "$REPO_ROOT"
"$PYTHON_BIN" scripts/run_multiseed_protocol.py \
  --universe "$UNIVERSE" \
  --prefix "opensource" \
  --gpu "$GPU" \
  --sections "$SECTIONS" \
  --out-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/console_output.txt"

echo ""
echo "All requested runs finished."
echo "Please send back this folder:"
echo "$OUT_DIR"
echo ""
echo "Most important summary file:"
echo "$OUT_DIR/final_results.txt"
