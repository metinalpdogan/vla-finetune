#!/bin/bash
# setup_inference.sh
#
# Installs the full training + inference environment at the repo root.
#
# Two-step install because flash-attn has an implicit torch build-dep that uv
# can't satisfy during a single sync (it tries to build flash-attn for metadata
# before torch is installed). Pattern:
#   1. uv sync             # everything except flash-attn
#   2. uv pip install      # flash-attn against the now-populated venv
#
# After this, the same .venv handles BOTH training (`vla-scripts/train.py`,
# `run_training_4gpu.sh`) AND real-robot inference (`run_xarm_inference.py`).
#
# Camera / robot deps (scipy, opencv-python, pyrealsense2) are in pyproject.toml
# already. The xArm SDK is the vendored copy at ril-env/xarm/; the inference
# script prepends ril-env/ to sys.path before importing it.
#
# Run from repo root:
#   bash setup_inference.sh

set -euo pipefail

cd "$(dirname "$0")"

# Defend against the data_collection venv being active — uv pip respects
# VIRTUAL_ENV and would install into the wrong venv.
unset VIRTUAL_ENV

echo "==> Step 1/2: uv sync (installs torch, transformers, tf, cameras, ...)"
# Force the system gcc — conda's bundled gcc has stale linux/input-event-codes.h
# which breaks evdev (pulled in transitively by pynput in some configurations).
CC=/usr/bin/gcc CXX=/usr/bin/g++ uv sync

echo ""
echo "==> Step 2/2: uv pip install flash-attn (compiles against venv torch)"
CC=/usr/bin/gcc CXX=/usr/bin/g++ uv pip install \
    --python .venv/bin/python \
    "flash-attn==2.5.5" --no-build-isolation

echo ""
echo "==> Verifying imports..."
.venv/bin/python - <<'PY'
import sys
import importlib

sys.path.insert(0, "ril-env")  # so vendored xarm package resolves

mods = [
    "torch", "transformers", "tensorflow", "tensorflow_datasets",
    "cv2", "scipy", "PIL", "numpy", "flash_attn", "pyrealsense2",
]
for m in mods:
    mod = importlib.import_module(m)
    print(f"  ok  {m:25s} {getattr(mod, '__version__', '?')}")

from xarm.wrapper import XArmAPI  # noqa: F401
print(f"  ok  xarm.wrapper            (vendored at ril-env/xarm/)")

from prismatic.models import load_vla  # noqa: F401
print(f"  ok  prismatic.models.load_vla")

import torch
print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"Device count:   {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device 0:       {torch.cuda.get_device_name(0)}")
PY

echo ""
echo "==> Done."
echo ""
echo "Activate the env with:"
echo "  source .venv/bin/activate"
echo ""
echo "Dry-run inference (no robot motion):"
echo "  python experiments/robot/xarm/run_xarm_inference.py \\"
echo "      --checkpoint step-015000-epoch-36-loss=0.0938.pt \\"
echo "      --dry_run --max_steps 5"
echo ""
echo "Real rollout on the xArm:"
echo "  python experiments/robot/xarm/run_xarm_inference.py \\"
echo "      --checkpoint step-015000-epoch-36-loss=0.0938.pt \\"
echo "      --xarm_ip 192.168.1.223"
