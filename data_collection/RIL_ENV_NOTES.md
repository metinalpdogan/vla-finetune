# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment setup

This repo targets Python 3.8 on Ubuntu Linux and runs against real hardware (xArm 7 at `192.168.1.223`, Intel RealSense cameras, 3Dconnexion SpaceMouse).

```bash
conda env create -n rilenv -f environment.yml
conda activate rilenv
pip install -e .
```

The `xarm/` directory at the repo root is the vendored xArm-Python-SDK. If it goes missing or breaks, reinstall by following xArm-Developer/xArm-Python-SDK's instructions and copy the `xarm` folder back to the repo root (see README.md).

Pre-commit runs `black` only (`.pre-commit-config.yaml`). There is no test suite or CI; `pytest`/`flake8`/`mypy` are listed as dev extras but no tests exist.

### Connecting to the arm
If the arm is unreachable, the README's recipe (run on the host that talks to the robot) is:
```bash
sudo ip addr add 192.168.1.100/24 dev enp2s0
sudo ip link set enp2s0 up
sudo ip route add 192.168.1.223 dev enp2s0
```

## Running things

All entry-point scripts live at the repo root and are invoked directly with `python <script>.py`. There are no console_scripts.

- `demo_real_robot.py` — full teleop + record loop (SpaceMouse + xArm + RealSense → zarr + mp4). Keys inside the loop: `c` start episode, `s` stop, `backspace` drop last, `space` stage marker, `q` quit.
- `try.py` — replay the first episode from `recordings/replay_buffer.zarr` against the real arm via `XArmController`. Read carefully before running; it does not stub out motion.
- `demo.py` — legacy synchronous teleop using the older `XArm` class (no recording into the zarr buffer).
- `home.py` — minimal: enter the `XArm` context (which homes on `__enter__`) and exit.
- `test.py` — open the replay buffer and plot TCP poses for the latest episode.
- `camera_calibration/check_realsense_serial_number.py` — list connected RealSense serials.
- `camera_calibration/multicam.py`, `test_calib.py` — extrinsic calibration utilities (depend on a sibling `xArm-Python-SDK` checkout at `/home/u-ril/URIL/xArm-Python-SDK` — see the `sys.path.append` in those files).

Outputs go to `recordings/` (`replay_buffer.zarr` + `videos/`). `*.zarr/` and `*.mp4` are gitignored.

## Architecture

Two parallel control stacks live side-by-side. Pick the right one before changing code.

### Legacy synchronous stack
- `ril_env.xarm_controller.XArm` — single-process, blocking. Entered via `with XArm(cfg) as arm:` which calls `initialize()` (connect, `set_mode(1)` servo mode, enable gripper, home) on enter and `shutdown()` on exit. `arm.step(dpos, drot, grasp)` integrates a *relative* delta into `current_position`/`current_orientation` and pushes via `set_servo_cartesian`.
- `ril_env.realsense.SingleRealsense` and `ril_env.multi_realsense.MultiRealsense` — used by `demo.py`.

### Multiprocessing stack (preferred for new work; this is what `nightly.md` describes)
Every long-running component is an `mp.Process` (or thread) with shared-memory IPC. Top-level orchestration goes through `ril_env.real_env.RealEnv`, which composes the pieces and is what `demo_real_robot.py` uses.

Key classes:
- `XArmController(mp.Process)` in `ril_env/xarm_controller.py` — owns the `XArmAPI` connection inside its child process. The parent talks to it via:
  - **input_queue** (`SharedMemoryQueue`): commands `STOP` / `STEP` / `HOME` plus a target pose, grasp, and target_time. Submitted by `step()` / `schedule_waypoint()`.
  - **ring_buffer** (`SharedMemoryRingBuffer`): the child publishes `TCPPose`, `TCPSpeed`, `JointAngles`, `JointSpeeds`, `Grasp`, plus timestamps. Parent reads with `get_state()` / `get_all_state()`.
  Note: `XArmController.step()` takes an *absolute* TCP pose (unlike the legacy `XArm.step()`, which takes deltas). `demo_real_robot.py` integrates SpaceMouse deltas into `target_pose` itself before calling `env.exec_actions`.
- `SingleRealsense(mp.Process)` / `MultiRealsense` in `ril_env/realsense.py`, `ril_env/multi_realsense.py` — same pattern: input queue for commands (`SET_COLOR_OPTION`, `START_RECORDING`, etc.), ring buffer for frames + timestamps. `MultiRealsense` fans out to N `SingleRealsense` workers, one per camera serial.
- `VideoRecorder` in `ril_env/video_recorder.py` — invoked by the camera process to write H.264 mp4s.
- `MultiCameraVisualizer` in `ril_env/multi_camera_visualizer.py` — optional separate process for live preview. **Currently broken**; `demo_real_robot.py` passes `enable_multi_cam_vis=False` and the OpenCV preview block is commented out.
- `RealEnv` in `ril_env/real_env.py` — composes `XArmController` + `MultiRealsense` + `ReplayBuffer` + `TimestampObsAccumulator` / `TimestampActionAccumulator`. `start_episode()` / `end_episode()` / `drop_episode()` manage zarr episodes; `get_obs()` aligns camera frames + robot state to the most recent N obs steps; `exec_actions(actions, timestamps, stages)` schedules waypoints with the `XArmController`.
- `ReplayBuffer` in `ril_env/replay_buffer.py` — zarr-backed append-only store. Layout: `meta/episode_ends` + `data/<key>` arrays; `DEFAULT_OBS_KEY_MAP` in `real_env.py` is the source of truth for which fields are persisted and how they're renamed (e.g. `TCPPose` → `robot_eef_pose`).

### Shared-memory primitives
The top-level `shared_memory/` package (NOT inside `ril_env`) holds the IPC building blocks: `SharedMemoryQueue`, `SharedMemoryRingBuffer`, `SharedNDArray`. Imports use absolute paths like `from shared_memory.shared_memory_queue import SharedMemoryQueue, Empty`. When adding new processes, follow the same pattern: a queue for commands in, a ring buffer for state out, and an `mp.Event` for ready/stop signaling. All of these need a `SharedMemoryManager` from `multiprocessing.managers`, which is created at the top of every entry-point script.

### Other utilities
- `ril_env/spacemouse.py` — `Spacemouse` is a `Thread` (not a `Process`) wrapping `spnav`. It applies a fixed translation/rotation transform to map the device frame into the robot frame; `get_motion_state_transformed()` returns `[dpos(3), drot(3)]` and `grasp` is updated on button press.
- `ril_env/keystroke_counter.py` — captures keypresses for the teleop loop without blocking.
- `ril_env/precise_sleep.py` — `precise_wait(t_target)` for the fixed-rate control loop in `demo_real_robot.py`.
- `ril_env/timestamp_accumulator.py` — aligns variable-rate observations and actions onto the shared `frequency` grid using `get_accumulate_timestamp_idxs`.

## Conventions worth knowing
- Every long-lived component is a context manager. Always use `with ... as x:`; bare construction will leave processes / shared memory dangling. `demo_real_robot.py` shows the correct nesting (SharedMemoryManager → KeystrokeCounter → Spacemouse → RealEnv).
- Configs are dataclasses (e.g. `XArmConfig`). Override fields at construction (`XArmConfig(frequency=20)`); don't mutate after the controller has started.
- Robot state keys use the xArm's "TCPPose / JointAngles / ..." names internally and are renamed via `DEFAULT_OBS_KEY_MAP` only when written to the replay buffer / returned from `get_obs()`. If you're inside `XArmController` you'll see `TCPPose`; from `RealEnv.get_obs()` you'll see `robot_eef_pose`.
- The legacy `XArm.step` takes *deltas*; `XArmController.step` takes *absolute* poses. Don't conflate them.
