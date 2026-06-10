#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: tools/prepare_models.sh [options]

Runs checkpoint-first model pipeline:
  download checkpoint -> export ONNX -> verify ONNX -> simplify ONNX -> convert MNN

Options:
  --manifest PATH           Manifest JSON (default: models/model_manifest.json)
  --skip-download           Skip checkpoint/model download stage
  --skip-export             Skip PyTorch checkpoint -> ONNX export stage
  --skip-verify             Skip ONNXRuntime Python verification stage
  --skip-simplify           Skip ONNX simplification stage
  --skip-mnn                Skip MNN conversion stage
  --dummy-export            Use built-in dummy PyTorch export smoke test instead of external exporter
  --fallback-dummy-export   If real export fails, retry once with --dummy-export
  --allow-example-url       Allow example.com placeholder URLs during download (debug only)
  --force                   Force re-download existing assets
  --dry-run                 Print actions only where supported
  --mnnconvert PATH         MNNConvert executable
  --fp16                    Convert MNN with fp16
  --static                  Convert MNN static model
  -h, --help                Show help
USAGE
}

MANIFEST=models/model_manifest.json
SKIP_DOWNLOAD=0
SKIP_EXPORT=0
SKIP_VERIFY=0
SKIP_SIMPLIFY=0
SKIP_MNN=0
DUMMY_EXPORT=0
FALLBACK_DUMMY_EXPORT=0
FORCE=0
DRY_RUN=0
ALLOW_EXAMPLE_URL=0
MNN_BIN=""
FP16=0
STATIC=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --skip-export) SKIP_EXPORT=1; shift ;;
    --skip-verify) SKIP_VERIFY=1; shift ;;
    --skip-simplify) SKIP_SIMPLIFY=1; shift ;;
    --skip-mnn) SKIP_MNN=1; shift ;;
    --dummy-export) DUMMY_EXPORT=1; shift ;;
    --fallback-dummy-export) FALLBACK_DUMMY_EXPORT=1; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --allow-example-url) ALLOW_EXAMPLE_URL=1; shift ;;
    --mnnconvert) MNN_BIN="$2"; shift 2 ;;
    --fp16) FP16=1; shift ;;
    --static) STATIC=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  echo "Copy models/model_manifest.example.json to models/model_manifest.json and edit checkpoint URL/export command." >&2
  exit 3
fi

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  if [[ "$DUMMY_EXPORT" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
    echo "== Download checkpoint/model assets =="
    echo "dummy-export enabled: skip checkpoint download because built-in dummy export does not need a .pth file"
  else
    args=(python tools/download_models.py --manifest "$MANIFEST")
    [[ "$FORCE" -eq 1 ]] && args+=(--force)
    [[ "$DRY_RUN" -eq 1 ]] && args+=(--dry-run --allow-example-url)
    [[ "$ALLOW_EXAMPLE_URL" -eq 1 ]] && args+=(--allow-example-url)
    echo "== Download checkpoint/model assets =="
    "${args[@]}"
  fi
fi

if [[ "$SKIP_EXPORT" -eq 0 ]]; then
  echo "== Export PyTorch checkpoint to split ONNX =="
  args=(python tools/export_pointpillars_onnx.py --manifest "$MANIFEST")
  [[ "$DUMMY_EXPORT" -eq 1 ]] && args+=(--dummy)
  [[ "$DRY_RUN" -eq 1 ]] && args+=(--dry-run)
  "${args[@]}"
fi

if [[ "$SKIP_VERIFY" -eq 0 ]]; then
  echo "== Verify ONNX with ONNXRuntime Python =="
  args=(python tools/verify_onnx.py --manifest "$MANIFEST")
  "${args[@]}"
fi

if [[ "$SKIP_SIMPLIFY" -eq 0 ]]; then
  echo "== Simplify ONNX =="
  args=(tools/simplify_onnx.sh)
  [[ "$DRY_RUN" -eq 1 ]] && args+=(--dry-run)
  "${args[@]}"
fi

if [[ "$SKIP_MNN" -eq 0 ]]; then
  echo "== Convert MNN =="
  args=(tools/convert_mnn.sh)
  [[ -n "$MNN_BIN" ]] && args+=(--mnnconvert "$MNN_BIN")
  [[ "$FP16" -eq 1 ]] && args+=(--fp16)
  [[ "$STATIC" -eq 1 ]] && args+=(--static)
  [[ "$DRY_RUN" -eq 1 ]] && args+=(--dry-run)
  "${args[@]}"
fi
