#!/usr/bin/env bash
# 创建并验证 YOLOv8 conda 环境
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-yolov8}"

echo "==> 创建 conda 环境: ${ENV_NAME}"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "环境 ${ENV_NAME} 已存在，补装 pip 依赖..."
else
  conda env create -f "${SCRIPT_DIR}/environment-yolov8.yml"
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
pip install -q "ultralytics>=8.0.0" "open3d>=0.17.0" "pyrealsense2>=2.54.0" torch torchvision

echo "==> 验证依赖"

python - <<'PY'
import sys
print("Python:", sys.version)

checks = []
try:
    import cv2
    checks.append(f"opencv: {cv2.__version__}")
except ImportError as e:
    checks.append(f"opencv: FAIL ({e})")

try:
    import numpy as np
    checks.append(f"numpy: {np.__version__}")
except ImportError as e:
    checks.append(f"numpy: FAIL ({e})")

try:
    import open3d as o3d
    checks.append(f"open3d: {o3d.__version__}")
except ImportError as e:
    checks.append(f"open3d: FAIL ({e})")

try:
    import pyrealsense2 as rs
    checks.append(f"pyrealsense2: {rs.__version__}")
except ImportError as e:
    checks.append(f"pyrealsense2: FAIL ({e})")

try:
    import torch
    cuda = torch.cuda.is_available()
    checks.append(f"torch: {torch.__version__}  CUDA={cuda}")
except ImportError as e:
    checks.append(f"torch: FAIL ({e})")

try:
    from ultralytics import YOLO
    checks.append("ultralytics: OK")
except ImportError as e:
    checks.append(f"ultralytics: FAIL ({e})")

for line in checks:
    print(" ", line)
PY

echo ""
echo "完成。使用方式:"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${SCRIPT_DIR}"
echo "  python realtime_yolo_pointcloud.py"
