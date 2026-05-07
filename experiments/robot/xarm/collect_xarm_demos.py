"""
collect_xarm_demos.py

Real-robot teleop data collection for an xArm, saving demonstrations in the same
LIBERO HDF5 schema that `experiments/robot/libero/regenerate_libero_dataset.py`
emits. The output is a drop-in for the OpenVLA-mini RLDS conversion pipeline used
to fine-tune `prism-qwen25-extra-dinosiglip-224px+0_5b` (and friends).

Per-task layout:
    <out_dir>/<task_name>_demo.hdf5
        attrs: language_instruction, control_hz
        data/
            demo_0/
                attrs: language_instruction, num_steps, success
                obs/
                    gripper_states  [T, 2]   gripper qpos (mirrored width, meters)
                    joint_states    [T, J]   joint angles (rad)
                    ee_states       [T, 6]   EEF xyz (m) + axis-angle (rad)
                    ee_pos          [T, 3]
                    ee_ori          [T, 3]
                    agentview_rgb   [T, 256, 256, 3]   uint8
                    eye_in_hand_rgb [T, 256, 256, 3]   uint8
                actions      [T, 7]   dxyz (m), drpy (rad), gripper in [-1=open, +1=close]
                states       [T, 9]   placeholder (= robot_states; no sim state on real hw)
                robot_states [T, 9]   [grip_qpos(2), ee_pos(3), ee_quat(4)]  (libero layout)
                rewards      [T]      zeros, 1 at last
                dones        [T]      zeros, 1 at last
            demo_1/ ...

Hardware assumptions
--------------------
- xArm 6 or 7 reachable over Ethernet, via `xArm-Python-SDK` (`pip install xarm-python-sdk`).
- Two RGB cameras visible to OpenCV (USB / RealSense in RGB mode are both fine).
- Optional 3D mouse (3Dconnexion SpaceMouse) for teleop input
  (`pip install pyspacemouse`). A keyboard fallback is provided.

Usage
-----
    python experiments/robot/xarm/collect_xarm_demos.py \
        --xarm_ip 192.168.1.220 \
        --task_name pick_red_block \
        --language "pick up the red block and place it on the plate" \
        --agent_cam 0 --wrist_cam 2 \
        --out_dir ./xarm_data/pick_red_block

Controls
--------
SpaceMouse:
    - puck XYZ      -> Cartesian translation command
    - puck RPY      -> rotation command
    - LEFT  button  -> toggle gripper close/open
    - RIGHT button  -> end current demo (then prompt save/discard at terminal)

Keyboard fallback (focused terminal needed):
    WASD + RF for translation; IJKL + UO for rotation; SPACE toggles gripper;
    ENTER ends a demo.
"""

import argparse
import os
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Optional, Tuple

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as Rot

try:
    from xarm.wrapper import XArmAPI
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing xArm SDK. Install with: pip install xArm-Python-SDK"
    ) from e

try:
    import pyspacemouse  # type: ignore

    HAS_SPACEMOUSE = True
except ImportError:
    HAS_SPACEMOUSE = False


# === Constants =================================================================
CONTROL_HZ = 20
DT = 1.0 / CONTROL_HZ
IMAGE_SIZE = 256

MAX_DPOS_PER_TICK = 0.03  # meters; clip teleop translation commands
MAX_DROT_PER_TICK = 0.15  # radians; clip teleop rotation commands

# xArm gripper position range (xArm parallel gripper). Override via --gripper_max if different.
GRIPPER_OPEN_POS_DEFAULT = 850
GRIPPER_CLOSE_POS_DEFAULT = 0


# === Teleop dataclass ==========================================================
@dataclass
class TeleopCommand:
    dxyz: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    drpy: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    gripper: float = -1.0  # -1 open, +1 close (libero raw convention)
    end_demo: bool = False
    quit: bool = False


# === Camera ====================================================================
class ThreadedCamera:
    """OpenCV VideoCapture in a background thread; always returns the latest frame.

    Falls back to repeating the last frame if a read fails. Frames are converted
    to RGB uint8 and resized to (IMAGE_SIZE, IMAGE_SIZE).
    """

    def __init__(self, source, image_size: int = IMAGE_SIZE):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source!r}")
        # Try to grab native 480p; minimize buffer so we always get the freshest frame.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.image_size = image_size
        self._latest: Optional[np.ndarray] = None
        self._stop = Event()
        self._th = Thread(target=self._loop, daemon=True)
        self._th.start()
        # Warm up
        for _ in range(20):
            if self._latest is not None:
                break
            time.sleep(0.05)
        if self._latest is None:
            raise RuntimeError(f"Camera {source!r} did not produce frames after warm-up")

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame_bgr = self.cap.read()
            if not ok or frame_bgr is None:
                time.sleep(0.005)
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(
                frame_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
            )
            self._latest = frame_rgb

    def read(self) -> np.ndarray:
        return self._latest.copy()

    def close(self) -> None:
        self._stop.set()
        self._th.join(timeout=1.0)
        self.cap.release()


# === Teleop sources ============================================================
class SpaceMouseTeleop:
    def __init__(self) -> None:
        if not HAS_SPACEMOUSE:
            raise RuntimeError("pyspacemouse not installed; pip install pyspacemouse")
        ok = pyspacemouse.open()
        if not ok:
            raise RuntimeError("pyspacemouse.open() failed; check device + permissions")
        self.gripper_state = -1.0  # start open
        self._left_prev = False
        self._right_prev = False

    def step(self) -> TeleopCommand:
        s = pyspacemouse.read()
        # Puck axes are in [-1, 1]; scale to per-tick deltas.
        dxyz = np.array([s.x, s.y, s.z], dtype=np.float32) * MAX_DPOS_PER_TICK
        drpy = np.array([s.roll, s.pitch, s.yaw], dtype=np.float32) * MAX_DROT_PER_TICK

        left = bool(s.buttons[0]) if len(s.buttons) > 0 else False
        right = bool(s.buttons[1]) if len(s.buttons) > 1 else False
        toggle_grip = left and not self._left_prev
        end_demo = right and not self._right_prev
        self._left_prev = left
        self._right_prev = right
        if toggle_grip:
            self.gripper_state = -self.gripper_state

        return TeleopCommand(
            dxyz=dxyz, drpy=drpy, gripper=self.gripper_state, end_demo=end_demo, quit=False
        )

    def close(self) -> None:
        try:
            pyspacemouse.close()
        except Exception:
            pass


class KeyboardTeleop:
    """Polled raw-stdin teleop. Translation: w/s/a/d/r/f. Rotation: i/k/j/l/u/o.
    Toggle gripper: SPACE. End demo: ENTER. Quit: q.
    """

    KEY_MAP = {
        "w": ("dxyz", 0, +1.0),
        "s": ("dxyz", 0, -1.0),
        "d": ("dxyz", 1, +1.0),
        "a": ("dxyz", 1, -1.0),
        "r": ("dxyz", 2, +1.0),
        "f": ("dxyz", 2, -1.0),
        "i": ("drpy", 0, +1.0),
        "k": ("drpy", 0, -1.0),
        "j": ("drpy", 1, +1.0),
        "l": ("drpy", 1, -1.0),
        "u": ("drpy", 2, +1.0),
        "o": ("drpy", 2, -1.0),
    }

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._key_q: Queue = Queue()
        self._stop = Event()
        self._th = Thread(target=self._loop, daemon=True)
        self._th.start()
        self.gripper_state = -1.0

    def _loop(self) -> None:
        import select

        while not self._stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.01)
            if r:
                ch = sys.stdin.read(1)
                self._key_q.put(ch)

    def step(self) -> TeleopCommand:
        cmd = TeleopCommand()
        # Drain all keys queued during the past tick (so motion stays responsive).
        while not self._key_q.empty():
            ch = self._key_q.get_nowait()
            if ch == "\n":
                cmd.end_demo = True
            elif ch == "q":
                cmd.quit = True
            elif ch == " ":
                self.gripper_state = -self.gripper_state
            elif ch in self.KEY_MAP:
                kind, axis, sign = self.KEY_MAP[ch]
                if kind == "dxyz":
                    cmd.dxyz[axis] += sign * MAX_DPOS_PER_TICK
                else:
                    cmd.drpy[axis] += sign * MAX_DROT_PER_TICK
        cmd.gripper = self.gripper_state
        return cmd

    def close(self) -> None:
        self._stop.set()
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        except Exception:
            pass


# === xArm wrapper ==============================================================
class XArmInterface:
    """Thin wrapper around xArm-Python-SDK for streaming Cartesian commands at 20 Hz."""

    def __init__(
        self,
        ip: str,
        gripper_open_pos: int = GRIPPER_OPEN_POS_DEFAULT,
        gripper_close_pos: int = GRIPPER_CLOSE_POS_DEFAULT,
    ) -> None:
        self.arm = XArmAPI(ip, is_radian=True)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.clean_warn()
        self.arm.clean_error()
        # Detect joint count (axis attribute is the configured DoF for xArm 5/6/7)
        self.num_joints = int(getattr(self.arm, "axis", 7))

        # Switch to streaming Cartesian servo mode
        self.arm.set_mode(1)
        self.arm.set_state(0)

        # Gripper init
        self.grip_open = gripper_open_pos
        self.grip_close = gripper_close_pos
        try:
            self.arm.set_gripper_mode(0)
            self.arm.set_gripper_enable(True)
            self.arm.set_gripper_speed(3000)
            self.arm.set_gripper_position(self.grip_open, wait=False)
            self._last_grip_cmd = -1.0
        except Exception:
            print("[warn] gripper init failed; continuing without gripper control")
            self._last_grip_cmd = -1.0

    def get_state(self) -> dict:
        # xArm SDK returns positions in mm and radians when is_radian=True.
        code, pose = self.arm.get_position(is_radian=True)
        if code != 0:
            raise RuntimeError(f"get_position failed code={code}")
        x, y, z, roll, pitch, yaw = pose
        ee_pos_m = np.array([x, y, z], dtype=np.float32) / 1000.0
        rpy = np.array([roll, pitch, yaw], dtype=np.float32)
        ee_quat_xyzw = Rot.from_euler("xyz", rpy).as_quat().astype(np.float32)
        ee_axis_angle = Rot.from_euler("xyz", rpy).as_rotvec().astype(np.float32)

        code, joints = self.arm.get_servo_angle(is_radian=True)
        if code != 0:
            raise RuntimeError(f"get_servo_angle failed code={code}")
        joints = np.asarray(joints[: self.num_joints], dtype=np.float32)

        # Gripper width in [0, 1] -> approximate finger qpos in meters (~0..0.04 m for parallel).
        try:
            code, grip_pos = self.arm.get_gripper_position()
            grip_norm = float(grip_pos) / max(1, self.grip_open - self.grip_close)
        except Exception:
            grip_norm = 0.5
        # Mirror to a 2-element vector to match libero layout (Franka two-finger).
        finger_half = 0.04 * grip_norm  # rough; actual scale is set by Q99 normalization later
        gripper_qpos = np.array([finger_half, finger_half], dtype=np.float32)

        return {
            "ee_pos": ee_pos_m,
            "rpy": rpy,
            "ee_quat_xyzw": ee_quat_xyzw,
            "ee_axis_angle": ee_axis_angle,
            "joints": joints,
            "gripper_qpos": gripper_qpos,
        }

    def step(self, target_xyz_m: np.ndarray, target_rpy: np.ndarray, gripper_cmd: float) -> None:
        target_mm = (np.asarray(target_xyz_m, dtype=np.float64) * 1000.0).tolist()
        target_pose = target_mm + np.asarray(target_rpy, dtype=np.float64).tolist()
        # Streaming Cartesian servo (radians on RPY because is_radian=True at init)
        self.arm.set_servo_cartesian(target_pose, is_radian=True)
        # Latch gripper only on commanded sign change to avoid hammering the controller.
        if gripper_cmd > 0.5 and self._last_grip_cmd <= 0.5:
            try:
                self.arm.set_gripper_position(self.grip_close, wait=False)
            except Exception:
                pass
            self._last_grip_cmd = 1.0
        elif gripper_cmd < -0.5 and self._last_grip_cmd >= -0.5:
            try:
                self.arm.set_gripper_position(self.grip_open, wait=False)
            except Exception:
                pass
            self._last_grip_cmd = -1.0

    def close(self) -> None:
        try:
            self.arm.set_mode(0)
            self.arm.set_state(0)
            self.arm.disconnect()
        except Exception:
            pass


# === Demo collection loop ======================================================
def collect_one_demo(
    arm: XArmInterface,
    agent_cam: ThreadedCamera,
    wrist_cam: ThreadedCamera,
    teleop,
    language: str,
) -> Optional[dict]:
    """Run a single 20 Hz teleop episode. Returns None if user requests quit."""

    print(f"\n>>> Recording demo. Task language: {language!r}")
    print("    (SpaceMouse: right button to end | Keyboard: ENTER to end, q to quit)")

    obs_gripper, obs_joints, obs_ee, obs_ee_pos, obs_ee_ori = [], [], [], [], []
    obs_agent, obs_wrist = [], []
    actions, robot_states = [], []

    t_next = time.perf_counter()
    n_overruns = 0

    while True:
        t_next += DT
        # Read robot + cameras
        st = arm.get_state()
        agent_img = agent_cam.read()
        wrist_img = wrist_cam.read()

        # Read teleop
        cmd = teleop.step()
        if cmd.quit:
            print("[teleop] quit requested.")
            return None
        if cmd.end_demo:
            break

        # Clip per-tick deltas defensively
        dxyz = np.clip(cmd.dxyz, -MAX_DPOS_PER_TICK, MAX_DPOS_PER_TICK).astype(np.float32)
        drpy = np.clip(cmd.drpy, -MAX_DROT_PER_TICK, MAX_DROT_PER_TICK).astype(np.float32)
        gripper_cmd = float(np.clip(cmd.gripper, -1.0, 1.0))

        # Compute target pose in base frame and command robot
        target_xyz = st["ee_pos"] + dxyz
        target_rpy = st["rpy"] + drpy
        arm.step(target_xyz, target_rpy, gripper_cmd)

        # Record (the *commanded* delta is what becomes the action label).
        action = np.concatenate([dxyz, drpy, np.array([gripper_cmd], dtype=np.float32)]).astype(
            np.float32
        )
        robot_state = np.concatenate(
            [st["gripper_qpos"], st["ee_pos"], st["ee_quat_xyzw"]]
        ).astype(np.float32)

        obs_gripper.append(st["gripper_qpos"])
        obs_joints.append(st["joints"])
        ee_state = np.concatenate([st["ee_pos"], st["ee_axis_angle"]]).astype(np.float32)
        obs_ee.append(ee_state)
        obs_ee_pos.append(ee_state[:3])
        obs_ee_ori.append(ee_state[3:])
        obs_agent.append(agent_img)
        obs_wrist.append(wrist_img)
        actions.append(action)
        robot_states.append(robot_state)

        # Real-time pacing
        sleep = t_next - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            n_overruns += 1
            t_next = time.perf_counter()  # resync to avoid runaway lag

    T = len(actions)
    if T < 5:
        print(f"[demo] only {T} steps recorded; discarding (too short).")
        return None

    print(f"[demo] {T} steps recorded ({T / CONTROL_HZ:.1f} s).  overruns={n_overruns}")

    rewards = np.zeros(T, dtype=np.uint8)
    rewards[-1] = 1
    dones = np.zeros(T, dtype=np.uint8)
    dones[-1] = 1

    return {
        "language": language,
        "obs": {
            "gripper_states": np.stack(obs_gripper, axis=0),
            "joint_states": np.stack(obs_joints, axis=0),
            "ee_states": np.stack(obs_ee, axis=0),
            "ee_pos": np.stack(obs_ee_pos, axis=0),
            "ee_ori": np.stack(obs_ee_ori, axis=0),
            "agentview_rgb": np.stack(obs_agent, axis=0),
            "eye_in_hand_rgb": np.stack(obs_wrist, axis=0),
        },
        "actions": np.stack(actions, axis=0),
        "robot_states": np.stack(robot_states, axis=0),
        "rewards": rewards,
        "dones": dones,
    }


# === Saving ====================================================================
def append_demo_to_hdf5(out_path: Path, demo: dict, success: bool, language: str) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if out_path.exists() else "w"
    with h5py.File(out_path, mode) as f:
        f.attrs.setdefault("language_instruction", language)
        f.attrs.setdefault("control_hz", CONTROL_HZ)
        grp = f.require_group("data")
        # Find next demo index
        idx = len([k for k in grp.keys() if k.startswith("demo_")])
        ep = grp.create_group(f"demo_{idx}")
        ep.attrs["language_instruction"] = language
        ep.attrs["num_steps"] = int(demo["actions"].shape[0])
        ep.attrs["success"] = bool(success)

        obs = ep.create_group("obs")
        obs.create_dataset("gripper_states", data=demo["obs"]["gripper_states"], compression="gzip")
        obs.create_dataset("joint_states", data=demo["obs"]["joint_states"], compression="gzip")
        obs.create_dataset("ee_states", data=demo["obs"]["ee_states"], compression="gzip")
        obs.create_dataset("ee_pos", data=demo["obs"]["ee_pos"], compression="gzip")
        obs.create_dataset("ee_ori", data=demo["obs"]["ee_ori"], compression="gzip")
        obs.create_dataset("agentview_rgb", data=demo["obs"]["agentview_rgb"], compression="gzip")
        obs.create_dataset("eye_in_hand_rgb", data=demo["obs"]["eye_in_hand_rgb"], compression="gzip")

        ep.create_dataset("actions", data=demo["actions"], compression="gzip")
        # `states` is a sim-state placeholder on real hardware; mirror robot_states for shape compatibility.
        ep.create_dataset("states", data=demo["robot_states"], compression="gzip")
        ep.create_dataset("robot_states", data=demo["robot_states"], compression="gzip")
        ep.create_dataset("rewards", data=demo["rewards"])
        ep.create_dataset("dones", data=demo["dones"])

    return idx


# === Main ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xarm_ip", required=True, help="xArm controller IP, e.g. 192.168.1.220")
    parser.add_argument("--task_name", required=True, help="Used for output filename.")
    parser.add_argument(
        "--language",
        required=True,
        help="Language instruction for the task (one prompt per task).",
    )
    parser.add_argument("--agent_cam", default="0", help="OpenCV index or video path for 3rd-person camera.")
    parser.add_argument("--wrist_cam", default="2", help="OpenCV index or video path for wrist camera.")
    parser.add_argument("--out_dir", default="./xarm_data", help="Output directory for HDF5 files.")
    parser.add_argument(
        "--teleop",
        choices=["spacemouse", "keyboard"],
        default="spacemouse" if HAS_SPACEMOUSE else "keyboard",
    )
    parser.add_argument("--gripper_max", type=int, default=GRIPPER_OPEN_POS_DEFAULT)
    parser.add_argument("--gripper_min", type=int, default=GRIPPER_CLOSE_POS_DEFAULT)
    args = parser.parse_args()

    out_path = Path(args.out_dir) / f"{args.task_name}_demo.hdf5"

    def _maybe_int(s: str):
        return int(s) if s.isdigit() else s

    print(f"[init] Connecting to xArm at {args.xarm_ip} ...")
    arm = XArmInterface(
        args.xarm_ip,
        gripper_open_pos=args.gripper_max,
        gripper_close_pos=args.gripper_min,
    )
    print(f"[init] xArm connected. DoF={arm.num_joints}")

    print(f"[init] Opening cameras: agent={args.agent_cam!r}, wrist={args.wrist_cam!r}")
    agent_cam = ThreadedCamera(_maybe_int(args.agent_cam))
    wrist_cam = ThreadedCamera(_maybe_int(args.wrist_cam))

    teleop = SpaceMouseTeleop() if args.teleop == "spacemouse" else KeyboardTeleop()

    print(f"\n[task] {args.task_name}")
    print(f"[task] language: {args.language!r}")
    print(f"[task] writing to: {out_path}")
    if not HAS_SPACEMOUSE and args.teleop == "spacemouse":
        print("[warn] spacemouse selected but pyspacemouse not installed; falling back to keyboard.")

    n_saved = 0
    try:
        while True:
            demo = collect_one_demo(arm, agent_cam, wrist_cam, teleop, args.language)
            if demo is None:
                break
            ans = input("Save (y), discard (n), quit (q)? ").strip().lower()
            if ans == "y":
                idx = append_demo_to_hdf5(out_path, demo, success=True, language=args.language)
                n_saved += 1
                print(f"[save] demo_{idx} appended. total saved this session: {n_saved}")
            elif ans == "q":
                break
            else:
                print("[discard] demo dropped.")
    finally:
        try:
            agent_cam.close()
        except Exception:
            pass
        try:
            wrist_cam.close()
        except Exception:
            pass
        try:
            teleop.close()
        except Exception:
            pass
        try:
            arm.close()
        except Exception:
            pass
    print(f"[done] saved {n_saved} demos to {out_path}")


if __name__ == "__main__":
    main()
