#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: tools/convert_mnn.sh [options]

Options:
  --mnnconvert PATH         MNNConvert executable (default: $MNNConvert or MNNConvert in PATH)
  --pfn-input PATH          Input simplified PFN ONNX (default: models/pfn_sim.onnx)
  --pfn-output PATH         Output PFN MNN (default: models/pfn.mnn)
  --backbone-input PATH     Input simplified Backbone/Head ONNX (default: models/backbone_head_sim.onnx)
  --backbone-output PATH    Output Backbone/Head MNN (default: models/backbone_head.mnn)
  --fp16                    Add --fp16 to MNNConvert
  --static                  Add --saveStaticModel and generated input config files
  --skip-pfn                Do not convert PFN
  --skip-backbone           Do not convert Backbone/Head
  --dry-run                 Print commands only
  -h, --help                Show help
USAGE
}

MNN_BIN=${MNNConvert:-MNNConvert}
PFN_IN=models/pfn_sim.onnx
PFN_OUT=models/pfn.mnn
BACKBONE_IN=models/backbone_head_sim.onnx
BACKBONE_OUT=models/backbone_head.mnn
FP16=0
STATIC=0
SKIP_PFN=0
SKIP_BACKBONE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mnnconvert) MNN_BIN="$2"; shift 2 ;;
    --pfn-input) PFN_IN="$2"; shift 2 ;;
    --pfn-output) PFN_OUT="$2"; shift 2 ;;
    --backbone-input) BACKBONE_IN="$2"; shift 2 ;;
    --backbone-output) BACKBONE_OUT="$2"; shift 2 ;;
    --fp16) FP16=1; shift ;;
    --static) STATIC=1; shift ;;
    --skip-pfn) SKIP_PFN=1; shift ;;
    --skip-backbone) SKIP_BACKBONE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v "$MNN_BIN" >/dev/null 2>&1 && [[ ! -x "$MNN_BIN" ]]; then
  echo "MNNConvert not found: $MNN_BIN" >&2
  echo "Set MNNConvert=MNNConvert or pass --mnnconvert." >&2
  exit 3
fi

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

extra=()
[[ "$FP16" -eq 1 ]] && extra+=(--fp16)

convert_one() {
  local in="$1" out="$2" code="$3" input_name="$4" input_dims="$5"
  [[ -f "$in" ]] || { echo "Input ONNX not found: $in" >&2; exit 4; }
  mkdir -p "$(dirname "$out")"
  local args=("$MNN_BIN" -f ONNX --modelFile "$in" --MNNModel "$out" --bizCode "$code" "${extra[@]}")
  if [[ "$STATIC" -eq 1 ]]; then
    local cfg="${out}.input_config.txt"
    printf 'input_names = %s\ninput_dims = %s\n' "$input_name" "$input_dims" > "$cfg"
    args+=(--saveStaticModel --inputConfigFile "$cfg")
  fi
  run_cmd "${args[@]}"
  [[ "$DRY_RUN" -eq 1 || -f "$out" ]] || { echo "MNN output missing: $out" >&2; exit 5; }
}

[[ "$SKIP_PFN" -eq 1 ]] || convert_one "$PFN_IN" "$PFN_OUT" pointpillars_pfn pillar_features 1x12000x32x10
[[ "$SKIP_BACKBONE" -eq 1 ]] || convert_one "$BACKBONE_IN" "$BACKBONE_OUT" pointpillars_backbone_head bev_feature 1x64x496x432
