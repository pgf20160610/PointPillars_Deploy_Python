#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: tools/simplify_onnx.sh [options]

Options:
  --pfn-input PATH          Input PFN ONNX (default: models/pfn.onnx)
  --pfn-output PATH         Output simplified PFN ONNX (default: models/pfn_sim.onnx)
  --backbone-input PATH     Input Backbone/Head ONNX (default: models/backbone_head.onnx)
  --backbone-output PATH    Output simplified Backbone/Head ONNX (default: models/backbone_head_sim.onnx)
  --pfn-shape STRING        onnxsim input-shape for PFN
  --backbone-shape STRING   onnxsim input-shape for Backbone/Head
  --skip-pfn                Do not simplify PFN
  --skip-backbone           Do not simplify Backbone/Head
  --dry-run                 Print commands only
  -h, --help                Show help
USAGE
}

PFN_IN=models/pfn.onnx
PFN_OUT=models/pfn_sim.onnx
BACKBONE_IN=models/backbone_head.onnx
BACKBONE_OUT=models/backbone_head_sim.onnx
PFN_SHAPE='pillar_features:1,12000,32,10 pillar_mask:1,12000,32,1'
BACKBONE_SHAPE='bev_feature:1,64,496,432'
SKIP_PFN=0
SKIP_BACKBONE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pfn-input) PFN_IN="$2"; shift 2 ;;
    --pfn-output) PFN_OUT="$2"; shift 2 ;;
    --backbone-input) BACKBONE_IN="$2"; shift 2 ;;
    --backbone-output) BACKBONE_OUT="$2"; shift 2 ;;
    --pfn-shape) PFN_SHAPE="$2"; shift 2 ;;
    --backbone-shape) BACKBONE_SHAPE="$2"; shift 2 ;;
    --skip-pfn) SKIP_PFN=1; shift ;;
    --skip-backbone) SKIP_BACKBONE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! python3 -c 'import onnxsim' >/dev/null 2>&1; then
  echo "onnxsim is not installed. Install with: python3 -m pip install onnxsim" >&2
  exit 3
fi

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

if [[ "$SKIP_PFN" -eq 0 ]]; then
  [[ -f "$PFN_IN" ]] || { echo "PFN ONNX not found: $PFN_IN" >&2; exit 4; }
  mkdir -p "$(dirname "$PFN_OUT")"
  run_cmd python3 -m onnxsim "$PFN_IN" "$PFN_OUT" --input-shape $PFN_SHAPE
  [[ "$DRY_RUN" -eq 1 || -f "$PFN_OUT" ]] || { echo "PFN simplify output missing: $PFN_OUT" >&2; exit 5; }
fi

if [[ "$SKIP_BACKBONE" -eq 0 ]]; then
  [[ -f "$BACKBONE_IN" ]] || { echo "Backbone ONNX not found: $BACKBONE_IN" >&2; exit 6; }
  mkdir -p "$(dirname "$BACKBONE_OUT")"
  run_cmd python3 -m onnxsim "$BACKBONE_IN" "$BACKBONE_OUT" --input-shape $BACKBONE_SHAPE
  [[ "$DRY_RUN" -eq 1 || -f "$BACKBONE_OUT" ]] || { echo "Backbone simplify output missing: $BACKBONE_OUT" >&2; exit 7; }
fi
