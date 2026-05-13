#!/usr/bin/env bash
# convert.sh — phone zarr -> LIBERO HDF5 -> RLDS, ready for VLA training.
#
# Usage:
#   bash convert.sh                                     # use defaults
#   bash convert.sh "pick up the green block"          # override language
#   ZARR=/path/to/teleop_data.zarr bash convert.sh    # override zarr path
#
# Idempotent: wipes prior HDF5 + RLDS before rebuilding.

set -euo pipefail
cd "$(dirname "$0")"

ZARR="${ZARR:-source/teleop_data.zarr}"
LANG_INSTR="${1:-pick up the red block}"
HDF5="recordings/_libero_hdf5/x_arm_phone_teleop_demo.hdf5"
RLDS_DIR="$HOME/edward/vla-finetune/data/rlds"

echo "==> Stage A: zarr -> LIBERO HDF5"
echo "    zarr:     $ZARR"
echo "    language: $LANG_INSTR"
rm -f "$HDF5"
.venv/bin/python zarr_to_libero_hdf5.py \
    --zarr "$ZARR" \
    --out  "$HDF5" \
    --language "$LANG_INSTR"

echo ""
echo "==> Stage B: LIBERO HDF5 -> RLDS"
rm -rf "$RLDS_DIR/x_arm_phone_teleop"
cd rlds_builder
CC=/usr/bin/gcc CXX=/usr/bin/g++ .venv/bin/tfds build XArmPhoneTeleop --data_dir "$RLDS_DIR" 2>&1 | tail -3
cd ..

echo ""
echo "==> Done."
echo "    HDF5: $(pwd)/$HDF5"
echo "    RLDS: $RLDS_DIR/x_arm_phone_teleop/1.0.0/"
echo ""
echo "Train with:"
echo "  torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py \\"
echo "      --vla.type 'prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-xarm-phone-teleop' \\"
echo "      --pretrained_checkpoint 'checkpoints/minivla-libero90-prismatic/checkpoints/latest-checkpoint.pt' \\"
echo "      --data_root_dir '$RLDS_DIR' \\"
echo "      --run_root_dir runs/ --is_resume False --image_aug True --save_interval 1000"
