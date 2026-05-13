# phone_data_bridge/

Bridges the **phone_data_collection** zarr (iPhone teledex → xArm) to the LIBERO HDF5 + RLDS format the VLA training pipeline consumes.

Two stages, each its own uv env (kept small / no torch):

```
~/edward/phone_data_collection/teleop_data.zarr
          │
          ▼
   stage A: zarr_to_libero_hdf5.py       (in phone_data_bridge/)
          │
          ▼
phone_data_bridge/recordings/_libero_hdf5/x_arm_phone_teleop_demo.hdf5
          │
          ▼
   stage B: rlds_builder/  +  tfds build
          │
          ▼
data/rlds/x_arm_phone_teleop/1.0.0/      (RLDS / TFRecord shards)
          │
          ▼
   stage C: torchrun vla-scripts/train.py --vla.type prism-qwen25-...-xarm-phone-teleop
```

The schema mappings (phone zarr → LIBERO HDF5 → RLDS):

| Phone zarr key | LIBERO HDF5 path | Notes |
|---|---|---|
| `data/state[:, :3]` (xyz mm) | `obs/ee_pos` (xyz m), `obs/ee_states[:, :3]` | divide by 1000 |
| `data/state[:, 3:6]` (rpy deg) | `obs/ee_ori` (axis-angle rad), `obs/ee_states[:, 3:6]` | Euler→axis-angle |
| `data/state[:, 6]` ∈ {0=open, 1=close} | `actions[:, 6]` ∈ {−1=open, +1=close} | `2*g − 1` (LIBERO HDF5 raw) |
| `data/img_0` float32 [0,1] 224×224 | `obs/agentview_rgb` uint8 224×224 | `*255` then cast |
| `data/img_1` float32 [0,1] 224×224 | `obs/eye_in_hand_rgb` uint8 224×224 | same |
| `meta/episode_ends` | one HDF5 `demo_<i>/` per episode | |
| (synthesized) | `actions[:, :6]` = `tcp[t+1] − tcp[t]` | mm→m, deg→rad axis-angle |
| (zeros) | `obs/joint_states` (T, 7) | phone doesn't record joints; downstream `libero_dataset_transform` ignores them |

## Stage A — zarr → LIBERO HDF5

```bash
cd phone_data_bridge
uv sync                                           # ~200 MB, slim env
source .venv/bin/activate
python zarr_to_libero_hdf5.py \
    --zarr ~/edward/phone_data_collection/teleop_data.zarr \
    --out  recordings/_libero_hdf5/x_arm_phone_teleop_demo.hdf5 \
    --language "pick up the red block"
```

What the converter does automatically:

- **Leading-noop trim** — drops any leading frame where consecutive TCP delta is < 0.5 mm xyz / 0.5° rpy. Operator-pause frames teach the model a noop-attractor at deployment; trimming kills it.
- **No-grasp filter** — skips episodes where the gripper never toggled (failed recording).
- **Action labels** = `tcp[t+1] − tcp[t]` in (m, rad axis-angle), not the commanded-vs-current lag. This matches how `libero_dataset_transform` expects to see motion.
- **Image cast** — phone's `[0,1] float32` → `[0,255] uint8` via `*255` and clip.

## Stage B — LIBERO HDF5 → RLDS

```bash
cd rlds_builder
uv sync                                           # ~600 MB, includes tensorflow
source .venv/bin/activate
tfds build XArmPhoneTeleop --data_dir "$HOME/edward/vla-finetune/data/rlds"
```

Output: `data/rlds/x_arm_phone_teleop/1.0.0/`.

## Stage C — train

Dataset is registered in the prismatic pipeline (same recipe as the ril-env path):

- `prismatic/vla/datasets/rlds/oxe/configs.py` → `x_arm_phone_teleop`
- `prismatic/vla/datasets/rlds/oxe/transforms.py` → reuses `libero_dataset_transform`
- `prismatic/vla/datasets/rlds/oxe/mixtures.py` → `x_arm_phone_teleop`
- `prismatic/conf/vla.py` → `Exp_Qwen25_DinoSigLIP_224px_wrist_0_5B_XArm_PhoneTeleop` registered as `QWEN25_DINOSIGLIP_224PX_WRIST_0_5B_XARM_PHONE_TELEOP`

Train:

```bash
cd /home/u-ril/edward/vla-finetune
source .venv/bin/activate
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py \
  --vla.type "prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-xarm-phone-teleop" \
  --pretrained_checkpoint "checkpoints/minivla-libero90-prismatic/checkpoints/latest-checkpoint.pt" \
  --data_root_dir "$HOME/edward/vla-finetune/data/rlds" \
  --run_root_dir runs/ \
  --is_resume False \
  --image_aug True \
  --save_interval 1000 \
  --wandb_project "xarm-phone-teleop"
```

The 4-GPU launcher works too — just edit the `--vla.type` line in `run_training_4gpu.sh`.

## Notes / things to watch

- **Image size**: phone collects 224×224. We keep it (no upsample), and the RLDS builder declares `(224, 224, 3)`. Prismatic's vision backbone reads 224 internally — no further resize at training. The previous ril-env path used 256 in the HDF5; both work as long as the RLDS builder's declared shape matches.
- **Joint state column is zeros**. `libero_dataset_transform` reads only `ee_state` + `gripper_state`, so this is structurally inert. If you later swap to a transform that reads joint state, you'll need to record real joint angles in the phone collection script (`obs["pose"]` currently only has TCP, not joints).
- **62 demos was not enough** for the ril-env path even after fixing every preprocessing pipeline detail. Plan for at least 80–100 phone demos before expecting a working policy.
