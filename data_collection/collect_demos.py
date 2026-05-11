"""
data_collection/collect_demos.py

Real-robot teleop demo collection on the xArm using the ril-env stack
(MultiRealsense + XArmController + ReplayBuffer + SpaceMouse).

Output (per task):
    <OUTPUT_BASE_DIR>/<TASK_NAME>/
        replay_buffer.zarr/   # ril-env zarr buffer; 7-D `action`, robot state per step
        videos/<episode>/0.mp4, 1.mp4   # primary, wrist H.264
        task_meta.json        # language, control_hz, camera serials, action layout

Edit the CONFIG block below before running. Then:
    python data_collection/collect_demos.py

Keys (focus the terminal that started the script — KeystrokeCounter is local):
    c          start a new episode
    s          stop and save current episode
    backspace  drop the most-recently-recorded episode (with confirmation)
    space      stage marker (held)
    q          quit
"""

import json
import logging
import pathlib
import sys
import time
import traceback
from multiprocessing.managers import SharedMemoryManager

import click
import numpy as np
import scipy.spatial.transform as st

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RIL_ENV_DIR = REPO_ROOT / "ril-env"
sys.path.insert(0, str(RIL_ENV_DIR))

from ril_env.keystroke_counter import Key, KeyCode, KeystrokeCounter  # noqa: E402
from ril_env.precise_sleep import precise_wait  # noqa: E402
from ril_env.real_env import RealEnv  # noqa: E402
from ril_env.spacemouse import Spacemouse  # noqa: E402
from ril_env.xarm_controller import XArmConfig  # noqa: E402

# ============================================================================
# CONFIG — edit before running
# ============================================================================

# RealSense serial numbers (12-digit strings). Find them with:
#   python ril-env/camera_calibration/check_realsense_serial_number.py
# Order matters: the list below becomes camera_0, camera_1 in the obs.
PRIMARY_CAM_SERIAL = "317222072157"   # 3rd-person agentview, fixed mount
WRIST_CAM_SERIAL   = "332322072612"   # eye-in-hand, mounted on the gripper

# Task identity. Becomes the output dir name + sidecar metadata + LIBERO HDF5 filename.
TASK_NAME = "pick_red_block"
LANGUAGE  = "pick up the red block"

# Where the zarr + videos go. Per-task subdir is created automatically.
OUTPUT_BASE_DIR = REPO_ROOT / "data_collection" / "recordings"

# Control loop. xArm servo mode (set_servo_cartesian) needs ≥ 30 Hz commands
# for smooth motion — at lower rates teleop staircases between targets and
# the SpaceMouse feels laggy/jittery. So we collect at 30 Hz here and
# downsample to the OpenVLA-recommended 10 Hz in stage 2 (zarr → HDF5).
# Don't drop this to 10 unless you want shaky teleop.
FREQUENCY           = 30

# Camera capture FPS. Must be a value RealSense natively supports (6, 15, 30,
# 60, 90); librealsense rejects pipeline.start() with "Couldn't resolve
# requests" otherwise.
VIDEO_CAPTURE_FPS   = 30

COMMAND_LATENCY     = 0.01
RECORD_RES          = (1280, 720)
SPACEMOUSE_DEADZONE = 0.05

# xArm IP (matches XArmConfig default).
XARM_IP = "192.168.1.223"

# ============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("collect_demos")


def _validate_config() -> None:
    if "REPLACE_ME" in PRIMARY_CAM_SERIAL or "REPLACE_ME" in WRIST_CAM_SERIAL:
        raise SystemExit(
            f"Edit PRIMARY_CAM_SERIAL and WRIST_CAM_SERIAL at the top of {__file__}.\n"
            "Find serials with: python ril-env/camera_calibration/check_realsense_serial_number.py"
        )


def _write_task_meta(task_dir: pathlib.Path) -> None:
    # Stage 2 reads this to recover language + camera roles + action units.
    meta = {
        "task_name": TASK_NAME,
        "language_instruction": LANGUAGE,
        "control_hz": FREQUENCY,
        "primary_camera_serial": PRIMARY_CAM_SERIAL,
        "wrist_camera_serial": WRIST_CAM_SERIAL,
        "primary_camera_idx": 0,
        "wrist_camera_idx": 1,
        "record_resolution": list(RECORD_RES),
        "xarm_ip": XARM_IP,
        # Zarr `action` is what RealEnv.exec_actions received: 7-D absolute pose + grasp.
        "action_layout": ["x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg", "grasp"],
        "grasp_convention": "ril_env",  # 0=open, 1=close
    }
    (task_dir / "task_meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    _validate_config()

    task_dir = pathlib.Path(OUTPUT_BASE_DIR) / TASK_NAME
    task_dir.mkdir(parents=True, exist_ok=True)
    _write_task_meta(task_dir)
    logger.info(f"Task dir: {task_dir}")

    dt = 1.0 / FREQUENCY
    xarm_config = XArmConfig(robot_ip=XARM_IP, frequency=FREQUENCY)

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, Spacemouse(
            deadzone=SPACEMOUSE_DEADZONE, shm_manager=shm_manager
        ) as sm, RealEnv(
            output_dir=task_dir,
            xarm_config=xarm_config,
            frequency=FREQUENCY,
            num_obs_steps=2,
            obs_image_resolution=RECORD_RES,
            max_obs_buffer_size=30,
            camera_serial_numbers=[PRIMARY_CAM_SERIAL, WRIST_CAM_SERIAL],
            obs_float32=True,
            init_joints=True,
            video_capture_fps=VIDEO_CAPTURE_FPS,
            video_capture_resolution=RECORD_RES,
            record_raw_video=True,
            thread_per_video=3,
            video_crf=21,
            enable_multi_cam_vis=False,
            multi_cam_vis_resolution=(1280, 720),
            shm_manager=shm_manager,
        ) as env:
            logger.info("Configuring camera settings...")
            env.realsense.set_exposure(exposure=120, gain=0)
            env.realsense.set_white_balance(white_balance=5900)
            time.sleep(1)

            state = env.get_robot_state()
            target_pose = np.array(state["TCPPose"], dtype=np.float32)
            logger.info(f"Initial TCP pose (mm + deg): {target_pose}")
            logger.info(
                "Ready. Keys: c=start | s=save | backspace=drop | space=stage | q=quit"
            )

            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False

            try:
                while not stop:
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_command_target = t_cycle_end + dt
                    t_sample = t_cycle_end - COMMAND_LATENCY

                    # Pump observations every tick — this is what advances the obs accumulator
                    # while a demo is recording.
                    env.get_obs()

                    for key_stroke in key_counter.get_press_events():
                        if key_stroke == KeyCode(char="q"):
                            stop = True
                        elif key_stroke == KeyCode(char="c"):
                            if not is_recording:
                                env.start_episode()
                                is_recording = True
                                logger.info(f"[REC] episode {env.replay_buffer.n_episodes} started")
                        elif key_stroke == KeyCode(char="s"):
                            if is_recording:
                                env.end_episode()
                                is_recording = False
                                logger.info(f"[SAVE] total episodes: {env.replay_buffer.n_episodes}")
                        elif key_stroke == Key.backspace:
                            if click.confirm("Drop the most recently recorded episode?"):
                                env.drop_episode()
                                is_recording = False
                                logger.info("Episode dropped.")

                    stage_val = key_counter[Key.space]
                    precise_wait(t_sample)

                    sm_state = sm.get_motion_state_transformed()
                    dpos, drot = sm_state[:3], sm_state[3:]
                    grasp = sm.grasp

                    # Integrate SpaceMouse deltas into the absolute target pose only when the
                    # input is meaningful — but pump exec_actions every tick regardless so the
                    # obs/action streams stay dense and aligned for BC training.
                    if np.linalg.norm(dpos) + np.linalg.norm(drot) > SPACEMOUSE_DEADZONE * 8.0:
                        dpos = dpos * xarm_config.position_gain
                        drot = drot * xarm_config.orientation_gain
                        curr_rot = st.Rotation.from_euler("xyz", target_pose[3:], degrees=True)
                        delta_rot = st.Rotation.from_euler("xyz", drot, degrees=True)
                        target_pose[:3] += dpos
                        target_pose[3:] = (delta_rot * curr_rot).as_euler("xyz", degrees=True)

                    action = np.concatenate([target_pose, [grasp]])
                    exec_timestamp = t_command_target - time.monotonic() + time.time()
                    env.exec_actions(actions=[action], timestamps=[exec_timestamp], stages=[stage_val])

                    precise_wait(t_cycle_end)
                    iter_idx += 1
            except KeyboardInterrupt:
                logger.info("Interrupted.")
            except Exception:
                logger.error("Exception in main loop:")
                traceback.print_exc()
            finally:
                if is_recording:
                    logger.info("Saving in-progress episode before exit...")
                    try:
                        env.end_episode()
                    except Exception:
                        logger.error("Failed to flush episode on exit:")
                        traceback.print_exc()
                logger.info(f"Done. {env.replay_buffer.n_episodes} episodes in {task_dir}")


if __name__ == "__main__":
    main()
