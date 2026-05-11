# Real-robot data collection on the xArm

End-to-end guide for collecting MiniVLA finetuning data on a real **xArm** using [ril-env](https://github.com/UCLA-Robot-Intelligence-Lab/ril-env) (RealSense cameras + 3Dconnexion SpaceMouse), and converting it to the format the training scripts expect.

> Big picture: see [`../README.md`](../README.md) for what to do *after* you have HDF5s — RLDS conversion, dataset registration, training.

---

## Table of contents

1. [Pipeline overview](#pipeline-overview)
2. [Camera placement](#camera-placement-read-this-first)
3. [Environment setup](#environment-setup)
4. [Stage 1 — collect demos](#stage-1--collect-demos)
5. [Stage 2 — convert to LIBERO HDF5](#stage-2--convert-to-libero-hdf5)
6. [Stage 3 — convert HDF5 to RLDS](#stage-3--convert-hdf5-to-rlds)
7. [Stage 4 — train (→ top-level README)](#stage-4--train)
8. [How many demos? What tasks?](#how-many-demos-what-tasks)
9. [Troubleshooting](#troubleshooting)

---

## Pipeline overview

```
[ xArm + 2× RealSense + SpaceMouse ]
          │  ril-env: MultiRealsense + XArmController + ReplayBuffer
          ▼
   Stage 1: collect_demos.py
          │
          ▼
recordings/<task>/replay_buffer.zarr   # robot state + 7-D action @ 30 Hz
recordings/<task>/videos/<ep>/{0,1}.mp4 # primary, wrist H.264 @ 30 fps
recordings/<task>/task_meta.json
          │
          ▼
   Stage 2: zarr_to_libero_hdf5.py    (downsamples 30 → 10 Hz)
          │
          ▼
recordings/_libero_hdf5/<task>_demo.hdf5
          │
          ▼
   Stage 3: rlds_builder + tfds build
          │
          ▼
data/rlds/<dataset>/1.0.0/             # RLDS / TFRecord shards
          │
          ▼
   Stage 4: torchrun vla-scripts/train.py  (top-level README)
```

---

## Camera placement (read this first)

The MiniVLA configs in `prismatic/conf/vla.py` consume either an agentview camera alone OR agentview + wrist. To keep both options open, **collect both streams**.

- **Primary / agentview / 3rd-person**: fixed tripod or rigid mount. Frame the workspace so the gripper, every task-relevant object, and the goal region stay in view at every step of every demo. ~45° above and to the side works well. Mark the camera position with tape — if you bump it between collection and deployment, the policy degrades.
- **Wrist / eye-in-hand**: rigidly mounted on the last link of the arm. Useful for fine alignment (precise grasps, peg-in-hole). Bad as the only view because it loses spatial context.
- **Lock exposure, white balance, focus.** `collect_demos.py` already pins `exposure=120, gain=0, white_balance=5900`.
- **Two-camera diversity across demos vs sessions**: do NOT move cameras between demos of the same dataset; DO vary the block / target positions within the workspace across demos.

---

## Environment setup

This directory is its own [uv](https://docs.astral.sh/uv/) project, separate from the heavy training environment at the repo root. It pulls only ril-env runtime deps + h5py — no torch / tensorflow / transformers. Install footprint is ~200 MB.

### System packages (one-time, per machine)

These are dynamically loaded by the Python wrappers; the wheels alone aren't enough:

```bash
sudo apt install libspnav-dev libhidapi-dev linux-libc-dev
# RealSense — follow Intel's instructions:
#   https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
```

### Python environment

```bash
cd data_collection
uv sync
source .venv/bin/activate
```

`.python-version` pins Python 3.10. `pyproject.toml` declares the deps; `uv.lock` is committed for reproducibility. The SpaceMouse Python wrapper is installed from [cheng-chi's GitHub fork](https://github.com/cheng-chi/spnav) (PyPI's `spnav==0.9` is a 2010 Python-2 release and crashes on import on Python ≥ 3.2).

> **Conda build gotcha**: if you have conda on `PATH`, uv's build step for `evdev` (transitive dep of `pynput`) may fail with `'ABS_PROFILE' undeclared` — conda's bundled GCC has stale kernel headers. Force the system compiler:
>
> ```bash
> CC=/usr/bin/gcc CXX=/usr/bin/g++ uv sync
> ```

### Networking to the xArm (one-time)

If the arm is unreachable:

```bash
sudo ip addr add 192.168.1.100/24 dev enp2s0
sudo ip link set enp2s0 up
sudo ip route add 192.168.1.223 dev enp2s0
```

(From `RIL_ENV_NOTES.md`; adapt `enp2s0` to your interface and `192.168.1.223` to your arm's IP.)

---

## Stage 1 — collect demos

### Configure the run

Edit the CONFIG block at the top of `collect_demos.py` for the **task you're recording right now**:

```python
PRIMARY_CAM_SERIAL = "317222072157"        # 3rd-person agentview, fixed tripod
WRIST_CAM_SERIAL   = "332322072612"        # eye-in-hand, on gripper

TASK_NAME = "pick_red_block"               # output dir + HDF5 filename + RLDS dataset name
LANGUAGE  = "pick up the red block"        # what the model sees as the prompt

FREQUENCY = 30                             # control loop rate (do NOT lower; xArm servo needs ≥30)
```

> **Don't lower `FREQUENCY`** — at 10 Hz the xArm servo controller staircases between targets and teleop becomes unusable. We collect at 30 Hz and let `zarr_to_libero_hdf5.py` downsample to 10 Hz (the OpenVLA-recommended training rate).

To find your RealSense serial numbers:

```bash
python ../ril-env/camera_calibration/check_realsense_serial_number.py
```

### Home the arm (optional)

```bash
python home.py
```

Moves the arm to the home pose (joint angles `[0,0,0,70,0,70,0]°`) and opens the gripper. Useful at the start of a session.

### Run collection

```bash
python collect_demos.py
```

Keep the terminal focused — `KeystrokeCounter` is an X11 listener.

| Key | Action |
|---|---|
| `c` | **start** a new episode |
| `s` | **save** the current episode |
| `backspace` | drop the most-recent episode (asks confirmation, works mid-recording too) |
| `space` | stage marker (held; optional) |
| `q` | quit |

**One demo cycle**:

1. Place the block at a new position in the workspace.
2. Press `c`. Recording started.
3. SpaceMouse-drive the xArm: approach, grasp, lift ~5 cm, hold for ~0.5 s.
4. Press `s`. Episode saved. **End with the gripper still closed** (block held) — releasing here trains the model to drop the block, the opposite of "pick up." (The converter also auto-trims any post-release tail as a safety net.)
5. Move the arm somewhere neutral, open the gripper, reposition the block.
6. Goto step 1.

When done, press `q`. The script handles cleanup via `finally` (saves an in-progress episode if you forgot `s`).

### Output

Per task:

```
recordings/<TASK_NAME>/
    replay_buffer.zarr/        # robot state + 7-D action per step, 30 Hz
    videos/<episode_id>/0.mp4  # primary (agentview) H.264, 30 fps
    videos/<episode_id>/1.mp4  # wrist H.264, 30 fps
    task_meta.json             # language, control_hz, camera serials, action layout
```

Re-running with the same `TASK_NAME` **appends** episodes (ril-env uses `ReplayBuffer.create_from_path(mode="a")`). Different task? New `TASK_NAME`.

### What's stored in the zarr (gotcha)

ril-env's `RealEnv` applies its `DEFAULT_OBS_KEY_MAP` rename only on `get_obs()` *return values*, **not** when persisting to zarr. On-disk keys are the raw xArm names:

- `TCPPose` (T, 6) — mm + degrees
- `JointAngles` (T, 7) — degrees
- `Grasp` (T,) — `0=open, 1=close`
- `action` (T, 7) — absolute commanded pose + grasp, what `RealEnv.exec_actions` received
- `TCPSpeed`, `JointSpeeds`, `robot_receive_timestamp`, `timestamp`, `stage`

**Camera frames are NOT in the zarr** — the camera-accumulator loop in `ril-env/ril_env/real_env.py:304-308` is commented out. Frames live in the MP4s and are decoded by stage 2 with PTS-based timestamp resampling (RealSense capture is bursty; nearest-frame-by-PTS is correct, frame-index pairing is not).

---

## Stage 2 — convert to LIBERO HDF5

```bash
python zarr_to_libero_hdf5.py                    # all tasks under recordings/
python zarr_to_libero_hdf5.py --task pick_red_block   # one task only
```

Output: `recordings/_libero_hdf5/<TASK_NAME>_demo.hdf5`.

The converter does several things automatically:

- **30 → 10 Hz downsample** (`DOWNSAMPLE_STRIDE = 3` at the top of the script). Action labels become `tcp[t+3] − tcp[t]` — the actual robot displacement over one 100 ms control step, which is the right target for a policy that runs at 10 Hz.
- **Drops leading zero-action rows**. ril-env's action accumulator pre-fills with zeros and the first `exec_actions()` call typically lands one tick after `start_episode()`, so index 0 would otherwise produce a giant garbage delta.
- **Drops the post-release tail**. Truncates each demo at the last frame where grasp was commanded closed (`+1`) so the model isn't taught to drop the block at the end.
- **Filters failed demos**. Episodes with zero gripper transitions (you forgot to grasp) are skipped with a warning.
- **Timestamp-based image resampling**. Reads PTS from each MP4 frame via PyAV; for each obs step at time `i × dt`, picks the frame whose PTS is nearest. RealSense MP4s have ±100 ms of bursty capture jitter at 30 fps nominal; index-based pairing would silently misalign images with action labels.
- **Joints converted to radians** (xArm reports degrees; LIBERO HDF5 stores radians).
- **Grasp remapped** from ril-env `{0=open, 1=close}` to LIBERO HDF5 `{−1=open, +1=close}`.

The output HDF5 matches the schema in `experiments/robot/xarm/collect_xarm_demos.py`. Drop-in for the LIBERO RLDS builder.

---

## Stage 3 — convert HDF5 to RLDS

Stage 3 has its own uv environment (it needs TensorFlow + TFDS, which we don't want in the lightweight collection env):

```bash
cd rlds_builder
uv sync                   # ~600 MB; pulls tensorflow 2.15 + tfds 4.9
source .venv/bin/activate
```

Then run the builder:

```bash
tfds build XArmPickRedBlock --data_dir "$HOME/edward/vla-finetune/data/rlds"
```

Output: `data/rlds/x_arm_pick_red_block/1.0.0/` — TFRecord shards + `dataset_info.json`.

> The builder at `rlds_builder/XArmPickRedBlock/` reads from `recordings/_libero_hdf5/*.hdf5`. Edit the `DATA_HDF5_GLOB` constant at the top if your HDF5 lives elsewhere.

### Adding more tasks

Each new task gets its own builder. Copy and edit:

```bash
cp -r rlds_builder/XArmPickRedBlock rlds_builder/YourNewTask
# In rlds_builder/YourNewTask/YourNewTask_dataset_builder.py:
#   - rename `class XArmPickRedBlock` to `class YourNewTask`
#   - rename the import `from XArmPickRedBlock.conversion_utils` to `from YourNewTask.conversion_utils`
#   - update DATA_HDF5_GLOB if needed

tfds build YourNewTask --data_dir "$HOME/edward/vla-finetune/data/rlds"
```

Then register the dataset in four prismatic files — see [Adding a new task / dataset](../README.md#adding-a-new-task--dataset) in the top-level README.

---

## Stage 4 — train

Follow the **[top-level README](../README.md#end-to-end-workflow)** from step 3 ("install training env") onward. The `vla_id` `prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-xarm-pick-red-block` is already registered for `x_arm_pick_red_block` and tuned for a single 24 GB GPU.

---

## How many demos? What tasks?

| Setup | Demos per task |
|---|---|
| Finetune from MiniVLA LIBERO checkpoint, single task | **~50–100** |
| LIBERO benchmark fine-tunes (reference) | 50 |
| Multi-task across N tasks | ~100 × N total |
| Hard task (long horizon, precise grasps, distractors) | 200+ |

Sample tasks (concrete, runnable on an xArm with a 2-cam rig):

| Difficulty | TASK_NAME | LANGUAGE | Suggested demos |
|---|---|---|---|
| Easy | `pick_red_block` | "pick up the red block" | 50 |
| Easy | `push_button` | "push the green button" | 50 |
| Medium | `pick_place_plate` | "pick up the red block and place it on the plate" | 100 |
| Medium | `open_drawer` | "open the drawer" | 100 |
| Hard | `stack_blocks` | "stack the blocks from largest to smallest" | 200 |
| Hard | `peg_insertion` | "insert the peg into the hole" | 200 |

### Quality > quantity

From the upstream OpenVLA "VLA Performance Troubleshooting" section:

- **Move continuously**, no pauses. The model learns "when uncertain, freeze" otherwise.
- **Vary initial conditions** across demos — block position, target position, mild lighting. Identical-start demos teach a policy that only works from that one start.
- **Stay consistent within a strategy** — always approach from the same side, always perform sub-steps in the same order. Multi-modal demos cause the model to average between strategies and do neither well.
- **Don't release the block as part of the task** if the task is "pick up X." (The stage-2 trim handles this defensively, but cleaner data = cleaner training.)

---

## Troubleshooting

- **"Edit PRIMARY_CAM_SERIAL ..."** — you forgot to fill in the CONFIG block at the top of `collect_demos.py`.
- **`RuntimeError: Couldn't resolve requests` (from `SingleRealsense`)** — `VIDEO_CAPTURE_FPS` is not a value RealSense natively supports. Valid values: 6, 15, 30, 60, 90. Default is 30. Don't change unless you know what you're doing.
- **Robot jittery / staircases between SpaceMouse motions** — your `FREQUENCY` is too low. xArm servo control needs ≥ 30 Hz. Don't lower it.
- **`KeyError: 'TCPPose'` in stage 2** — your zarr came from a ril-env version that renames keys on write. Update the zarr-key strings in `convert_episode()` accordingly.
- **xArm errors on init** — check IP, ethernet route, the controller is powered, no error/warn state. Use xArm Studio to clear errors first if needed.
- **No camera frames** — verify both serial numbers with `check_realsense_serial_number.py`. Typos give an opaque init failure.
- **SpaceMouse does nothing** — check `libspnav.so` is on the loader path (`ldconfig -p | grep spnav`); confirm the device shows up under `/dev/input/`; you may need to start `spacenavd` (`sudo systemctl start spacenavd`).
- **Keys do nothing** — wrong terminal focused, no `DISPLAY` set, or X server unreachable. KeystrokeCounter is a `pynput` X11 listener.
- **HDF5 conversion warns "skipped — no grasp transitions"** — that demo had zero gripper presses (failed recording). Safe to ignore; the demo is dropped from the HDF5 automatically.
- **Demos all show end-state grasp = +1 (good) and exactly 1 transition** — the converter trimmed the release tail; you're good.

If you hit something not covered here, the relevant code is short (~400 lines across stage 1 + stage 2). Read it.
