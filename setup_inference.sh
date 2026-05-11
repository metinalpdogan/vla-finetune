#!/bin/bash
# setup_inference.sh
#
# Installs the runtime dependencies needed by experiments/robot/xarm/run_xarm_inference.py
# into the root uv project venv (the same venv used for VLA training).
#
# Camera + math deps come from PyPI; the xArm SDK is the vendored copy at ril-env/xarm/
# (the inference script prepends ril-env/ to sys.path before importing it).
#
# Run from the repo root:
#   bash setup_inference.sh

set -euo pipefail

cd "$(dirname "$0")"  # cd to repo root

echo "==> Adding inference dependencies to the root uv project..."

# Camera + math libs (no torch — already pinned in root pyproject.toml)
uv add \
    "scipy>=1.10" \
    "opencv-python>=4.5" \
    "pyrealsense2>=2.55"

echo ""
echo "==> Verifying imports..."
uv run python - <<'PY'
import sys
import importlib

mods = ["torch", "cv2", "scipy", "PIL", "numpy"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"  ok  {m:12s} {getattr(mod, '__version__', '?')}")

# pyrealsense2 is optional at runtime — only required when --use_realsense is set
try:
    import pyrealsense2  # noqa: F401
    print("  ok  pyrealsense2  (RealSense available)")
except ImportError:
    print("  --  pyrealsense2 not importable — RealSense path will be unavailable")

# Vendored xArm SDK lives at ril-env/xarm/ — confirm it imports when ril-env is on sys.path
sys.path.insert(0, "ril-env")
from xarm.wrapper import XArmAPI  # noqa: F401
print(f"  ok  xarm.wrapper (vendored at ril-env/xarm/)")

import torch
print(f"\nCUDA available: {torch.cuda.is_available()}  | device count: {torch.cuda.device_count()}")
PY

echo ""
echo "==> Done."
echo ""
echo "Run inference (dry run, no robot) with:"
echo ""
echo "  uv run python experiments/robot/xarm/run_xarm_inference.py \\"
echo "      --checkpoint runs/prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-xarm-pick-red-block+n0+b8+x7--image_aug/checkpoints/step-015000-epoch-36-loss=0.0938.pt \\"
echo "      --dry_run --max_steps 5"
echo ""
