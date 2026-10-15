#!/usr/bin/env bash
# GC10-DET full complexity re-measurement (FP32, batch 1, 800x800).
#
# Runs ALL ablation configurations in ONE session on the SAME GPU so the
# latency numbers are mutually comparable. Seed-0 checkpoints only, following
# the established protocol (latency stats are within-benchmark, not
# across-seed).
#
# Prerequisite: the two missing factorial configs must be trained first:
#   models/rt-detr/rtdetr-SlimNeck-MSAIFI.yaml
#     -> runs/train/rtdetr-r18-gc10-fixed42-slimneck-msaifi-800/weights/best.pt
#   models/rt-detr/rtdetr-r18-MSAIFI-InnerGIoU.yaml
#     -> runs/train/rtdetr-r18-gc10-fixed42-msaifi-innergiou-800/weights/best.pt
# If they are not trained yet, comment out the two lines marked [NEW].
#
# Usage (on AutoDL, from the repo root):
#   bash run_gc10_complexity_full_fp32.sh

set -euo pipefail
cd "$(dirname "$0")"

OUT=runs/complexity/gc10_full_complexity_fp32.csv
ENV_OUT=runs/complexity/gc10_full_complexity_fp32_env.txt
mkdir -p runs/complexity

# --- Record hardware / runtime provenance (needed for the paper's table) ---
{
  echo "date:      $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "hostname:  $(hostname)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.sm,clocks.max.sm --format=csv
  python - <<'PY'
import sys, torch
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())
print("device:", torch.cuda.get_device_name(0))
PY
} > "$ENV_OUT" 2>&1
echo "Environment saved: $ENV_OUT"

python complexity_test.py \
  runs/train/rtdetr-r18-gc10-fixed42-baseline-800/weights/best.pt \
  runs/train/rtdetr-r18-gc10-fixed42-slimneck-800/weights/best.pt \
  runs/train/rtdetr-r18-gc10-fixed42-innergiou-800-seed0/weights/best.pt \
  runs/train/rtdetr-r18-gc10-fixed42-msaifi-only-800/weights/best.pt \
  runs/train/rtdetr-r18-gc10-fixed42-slimneck-innergiou-800/weights/best.pt \
  runs/train/rtdetr-r18-gc10-fixed42-slimneck-msaifi-800/weights/best.pt \
  runs/train/rtdetr-r18-gc10-fixed42-msaifi-innergiou-800/weights/best.pt \
  runs/train/rtdetr-r18-gc10-fixed42-slimneck-innergiou-msaifi-800/weights/best.pt \
  runs/train/rtdetr-r50-gc10-fixed42-baseline/weights/best.pt \
  --kind auto \
  --imgsz 800 \
  --device 0 \
  --warmup 100 \
  --iterations 500 \
  --output "$OUT"
