"""
data_collection/zarr_to_libero_hdf5.py

Convert ril-env zarr replay buffers + per-episode MP4s into the LIBERO HDF5 schema
documented at the top of experiments/robot/xarm/collect_xarm_demos.py.

Output:
    <out_dir>/<task_name>_demo.hdf5

The result is a drop-in for the LIBERO -> RLDS converter at
https://github.com/moojink/rlds_dataset_builder. After running RLDS conversion,
register the new dataset in:
    prismatic/vla/datasets/rlds/oxe/configs.py     (OXE_DATASET_CONFIGS)
    prismatic/vla/datasets/rlds/oxe/transforms.py  (use libero_dataset_transform)
    prismatic/vla/datasets/rlds/oxe/mixtures.py    (OXE_NAMED_MIXTURES)
and add an Exp_* / VLARegistry entry in prismatic/conf/vla.py.

Usage:
    python data_collection/zarr_to_libero_hdf5.py                # all tasks
    python data_collection/zarr_to_libero_hdf5.py --task pick_red_block

Notes on zarr keys (this is the gotcha): ril-env's RealEnv applies its
DEFAULT_OBS_KEY_MAP rename only on the get_obs() return value, NOT on what's
stored in the zarr. So the on-disk keys are the raw xArm names: TCPPose,
TCPSpeed, JointAngles, JointSpeeds, Grasp, robot_receive_timestamp.
Camera frames are NOT in the zarr; they live in videos/<episode>/<idx>.mp4.
"""

import argparse
import json
import logging
import pathlib
import sys

import av
import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as Rot

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RIL_ENV_DIR = REPO_ROOT / "ril-env"
sys.path.insert(0, str(RIL_ENV_DIR))

from ril_env.replay_buffer import ReplayBuffer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("zarr2hdf5")

DEFAULT_RECORDINGS_DIR = REPO_ROOT / "data_collection" / "recordings"
DEFAULT_OUT_DIR = DEFAULT_RECORDINGS_DIR / "_libero_hdf5"
TARGET_IMAGE_SIZE = 256

# Downsample stride: collect at 30 Hz (smooth xArm teleop), train at 30 // stride Hz.
# stride=3 → 10 Hz training data, the OpenVLA recommendation.
DOWNSAMPLE_STRIDE = 3


def _decode_video_with_pts(video_path: pathlib.Path, target_size: int):
    """Decode an MP4 to (frames_uint8, pts_seconds_relative_to_first).

    PTS values come from the container's stream time_base. They are absolute
    capture times in seconds relative to the first frame — important because
    RealSense MP4s have noticeably variable instantaneous frame rate (median
    50ms intervals on this rig even at 30fps nominal). Pairing frame index ↔
    obs index is wrong; pairing by PTS is correct.
    """
    if not video_path.exists():
        raise RuntimeError(f"Missing video file: {video_path}")
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    time_base = float(stream.time_base)
    frames, pts = [], []
    for packet in container.demux(stream):
        for frame in packet.decode():
            if frame.pts is None:
                continue
            img = frame.to_ndarray(format="rgb24")
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
            frames.append(img)
            pts.append(frame.pts * time_base)
    container.close()
    if not frames:
        raise RuntimeError(f"Empty video: {video_path}")
    arr = np.stack(frames, axis=0).astype(np.uint8)
    pts_arr = np.asarray(pts, dtype=np.float64)
    pts_arr = pts_arr - pts_arr[0]  # relative to first frame
    return arr, pts_arr


def _sample_at_times(frames: np.ndarray, pts: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """For each `t` in target_times, pick the frame whose PTS is nearest to t.
    PTS is assumed monotonically non-decreasing (true for valid MP4s)."""
    # Use searchsorted to find insertion points, then compare to neighbors.
    insert = np.searchsorted(pts, target_times)
    insert = np.clip(insert, 1, len(pts) - 1)
    left = insert - 1
    use_left = (target_times - pts[left]) <= (pts[insert] - target_times)
    idx = np.where(use_left, left, insert)
    return frames[idx]


def _tcp_to_delta_actions(tcp_pose_mm_deg: np.ndarray) -> np.ndarray:
    """
    Convert (T, 6) absolute TCP poses [mm + deg] into per-step LIBERO-style
    deltas [m + rad axis-angle] computed from *actual physical motion*:
        delta[t] = tcp[t+1] - tcp[t]   for t in [0, T-2]
        delta[T-1] = 0                 (no "next" pose)

    We deliberately do NOT use commanded_pose - tcp_pose: with streaming
    SpaceMouse teleop the commanded target runs ahead of the actual robot,
    and that gap (often >10 cm in a single tick) is teleop lag, not motion.
    LIBERO sim actions are policy commands that get executed within one tick
    in MuJoCo — to match that semantics on real hardware, we use the actual
    pose change between consecutive obs steps.
    """
    xyz_m = tcp_pose_mm_deg[:, :3] / 1000.0
    rot = Rot.from_euler("xyz", tcp_pose_mm_deg[:, 3:6], degrees=True)

    delta_xyz = np.zeros_like(xyz_m, dtype=np.float32)
    delta_xyz[:-1] = (xyz_m[1:] - xyz_m[:-1]).astype(np.float32)

    delta_rot = np.zeros((len(tcp_pose_mm_deg), 3), dtype=np.float32)
    delta_rot[:-1] = (rot[1:] * rot[:-1].inv()).as_rotvec().astype(np.float32)
    return np.concatenate([delta_xyz, delta_rot], axis=1)


def _euler_deg_to_axis_angle(rpy_deg: np.ndarray) -> np.ndarray:
    return Rot.from_euler("xyz", rpy_deg, degrees=True).as_rotvec().astype(np.float32)


def _euler_deg_to_quat_xyzw(rpy_deg: np.ndarray) -> np.ndarray:
    return Rot.from_euler("xyz", rpy_deg, degrees=True).as_quat().astype(np.float32)


def convert_episode(
    rb: ReplayBuffer, episode_idx: int, video_dir: pathlib.Path, language: str,
    control_hz: int = 30,
) -> dict:
    ep = rb.get_episode(episode_idx)

    tcp_pose = np.asarray(ep["TCPPose"], dtype=np.float32)        # (T, 6) mm + deg
    # xArm SDK is queried with is_radian=False, so stored joints are in degrees.
    # LIBERO HDF5 uses radians; convert here.
    joints = np.deg2rad(np.asarray(ep["JointAngles"], dtype=np.float32))  # (T, J) rad
    grasp_ril = np.asarray(ep["Grasp"], dtype=np.float32).reshape(-1)  # (T,) {0=open, 1=close}
    abs_action = np.asarray(ep["action"], dtype=np.float32)       # (T, 7)

    # Trim leading all-zero action rows. ril-env's TimestampActionAccumulator
    # pre-fills its buffer with np.zeros_like(...) and only writes the indices
    # whose timestamps it actually saw. exec_actions() typically lands one
    # tick *after* start_episode(), leaving index 0 (and occasionally a couple
    # more) as the zero-fill sentinel — those rows would produce a giant
    # bogus delta against the real TCP pose. We trim the contiguous all-zero
    # prefix only; zeros that appear after real data are legitimate noops.
    is_zero = (abs_action == 0).all(axis=1)
    n_lead = int(np.argmax(~is_zero)) if not is_zero.all() else len(is_zero)
    if n_lead > 0:
        tcp_pose = tcp_pose[n_lead:]
        joints = joints[n_lead:]
        grasp_ril = grasp_ril[n_lead:]
        abs_action = abs_action[n_lead:]

    # Downsample by DOWNSAMPLE_STRIDE so the trained model runs at a lower
    # rate than collection. We slice with [::stride] BEFORE computing deltas
    # so the action is "what happens over one downsampled control step",
    # i.e. tcp[t+stride] - tcp[t] — exactly the change the policy should
    # produce when given obs at t. control_hz passed downstream is also
    # divided (e.g. 30 // 3 = 10).
    if DOWNSAMPLE_STRIDE > 1:
        tcp_pose = tcp_pose[::DOWNSAMPLE_STRIDE]
        joints = joints[::DOWNSAMPLE_STRIDE]
        grasp_ril = grasp_ril[::DOWNSAMPLE_STRIDE]
        abs_action = abs_action[::DOWNSAMPLE_STRIDE]

    n_steps = tcp_pose.shape[0]
    if n_steps < 5:
        raise RuntimeError(f"episode {episode_idx} has only {n_steps} steps after trim; skipping")

    delta_pose = _tcp_to_delta_actions(tcp_pose)
    # ril-env grasp {0=open, 1=close} -> LIBERO HDF5 raw {-1=open, +1=close}.
    # libero_dataset_transform clips to [0,1] then 1-x, so signed input is fine
    # (matches collect_xarm_demos.py convention). Grasp comes from the
    # commanded action (user intent), not from the state — TCP state never
    # reflects gripper commands, only the gripper width feedback.
    grasp_libero = (2.0 * abs_action[:, 6] - 1.0).astype(np.float32).reshape(-1, 1)
    actions = np.concatenate([delta_pose, grasp_libero], axis=1).astype(np.float32)

    # Resample each camera by *timestamp*, not by frame index. RealSense MP4s
    # have noticeably bursty capture timing — pairing frame i with obs i can
    # produce misalignments of up to ±100 ms (3+ frames). Stage-1 obs were
    # bucketed at start_time + i / collect_hz; after trimming n_lead leading
    # zero-action rows and then downsampling by DOWNSAMPLE_STRIDE, the
    # remaining obs steps correspond to collect-time indices
    #   (n_lead + i*stride)   for i in [0, n_steps).
    dt_collect = 1.0 / control_hz
    target_times = (n_lead + np.arange(n_steps) * DOWNSAMPLE_STRIDE) * dt_collect

    primary_frames, primary_pts = _decode_video_with_pts(
        video_dir / str(episode_idx) / "0.mp4", TARGET_IMAGE_SIZE
    )
    wrist_frames, wrist_pts = _decode_video_with_pts(
        video_dir / str(episode_idx) / "1.mp4", TARGET_IMAGE_SIZE
    )
    primary = _sample_at_times(primary_frames, primary_pts, target_times)
    wrist = _sample_at_times(wrist_frames, wrist_pts, target_times)

    ee_pos_m = (tcp_pose[:, :3] / 1000.0).astype(np.float32)
    ee_quat = np.stack([_euler_deg_to_quat_xyzw(rpy) for rpy in tcp_pose[:, 3:6]], axis=0)
    ee_axis_angle = np.stack([_euler_deg_to_axis_angle(rpy) for rpy in tcp_pose[:, 3:6]], axis=0)
    ee_states = np.concatenate([ee_pos_m, ee_axis_angle], axis=1).astype(np.float32)

    # LIBERO's gripper_states is a 2-D vector (mirrored fingers). We don't have real
    # finger qpos from the xArm gripper SDK here; mirror the binary state. The Q99
    # normalization at training time will rescale anyway.
    grip_qpos_2d = np.stack([grasp_ril, grasp_ril], axis=1).astype(np.float32)

    # robot_states (LIBERO layout): [grip_qpos(2), ee_pos(3), ee_quat_xyzw(4)]
    robot_states = np.concatenate([grip_qpos_2d, ee_pos_m, ee_quat], axis=1).astype(np.float32)

    # Trim the post-grasp "release" tail. For "pick up X" tasks the demo
    # should end while the block is still held — but operators often release
    # the block before pressing `s`, which trains the model to drop the
    # block after lifting (the opposite of the task). Keep everything
    # through the *last* frame where grasp was commanded closed (+1).
    held = np.where(actions[:, 6] > 0)[0]
    if len(held) > 0 and held[-1] + 1 < n_steps:
        cut = held[-1] + 1
        actions = actions[:cut]
        primary = primary[:cut]
        wrist = wrist[:cut]
        ee_pos_m = ee_pos_m[:cut]
        ee_axis_angle = ee_axis_angle[:cut]
        ee_states = ee_states[:cut]
        grip_qpos_2d = grip_qpos_2d[:cut]
        joints = joints[:cut]
        robot_states = robot_states[:cut]
        n_steps = cut

    rewards = np.zeros(n_steps, dtype=np.uint8)
    rewards[-1] = 1
    dones = np.zeros(n_steps, dtype=np.uint8)
    dones[-1] = 1

    return {
        "language": language,
        "obs": {
            "gripper_states": grip_qpos_2d,
            "joint_states": joints,
            "ee_states": ee_states,
            "ee_pos": ee_pos_m,
            "ee_ori": ee_axis_angle,
            "agentview_rgb": primary,
            "eye_in_hand_rgb": wrist,
        },
        "actions": actions,
        "robot_states": robot_states,
        "rewards": rewards,
        "dones": dones,
        "num_steps": n_steps,
    }


def append_demo_to_hdf5(
    out_path: pathlib.Path, demo: dict, language: str, control_hz: int
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if out_path.exists() else "w"
    # control_hz here is the downsampled (training) rate, not the collection rate.
    effective_hz = control_hz // DOWNSAMPLE_STRIDE
    with h5py.File(out_path, mode) as f:
        f.attrs.setdefault("language_instruction", language)
        f.attrs.setdefault("control_hz", effective_hz)
        grp = f.require_group("data")
        idx = len([k for k in grp.keys() if k.startswith("demo_")])
        ep = grp.create_group(f"demo_{idx}")
        ep.attrs["language_instruction"] = language
        ep.attrs["num_steps"] = int(demo["num_steps"])
        ep.attrs["success"] = True

        obs = ep.create_group("obs")
        for k, v in demo["obs"].items():
            obs.create_dataset(k, data=v, compression="gzip")
        ep.create_dataset("actions", data=demo["actions"], compression="gzip")
        # `states` is a sim-state placeholder in real-hw datasets; mirror robot_states.
        ep.create_dataset("states", data=demo["robot_states"], compression="gzip")
        ep.create_dataset("robot_states", data=demo["robot_states"], compression="gzip")
        ep.create_dataset("rewards", data=demo["rewards"])
        ep.create_dataset("dones", data=demo["dones"])
    return idx


def convert_task(task_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    meta_path = task_dir / "task_meta.json"
    zarr_path = task_dir / "replay_buffer.zarr"
    video_dir = task_dir / "videos"
    if not meta_path.exists():
        LOG.warning(f"skip {task_dir} (missing task_meta.json)")
        return
    if not zarr_path.exists():
        LOG.warning(f"skip {task_dir} (missing replay_buffer.zarr)")
        return

    meta = json.loads(meta_path.read_text())
    language = meta["language_instruction"]
    control_hz = int(meta["control_hz"])

    rb = ReplayBuffer.create_from_path(str(zarr_path), mode="r")
    n_eps = rb.n_episodes
    if n_eps == 0:
        LOG.warning(f"skip {task_dir} (zarr has 0 episodes)")
        return

    out_path = out_dir / f"{meta['task_name']}_demo.hdf5"
    if out_path.exists():
        out_path.unlink()
    LOG.info(f"[{meta['task_name']}] {n_eps} episodes -> {out_path}")

    n_written = 0
    n_skipped_nograsp = 0
    for i in range(n_eps):
        try:
            demo = convert_episode(rb, i, video_dir, language, control_hz=control_hz)
        except Exception as e:
            LOG.error(f"  episode {i}: failed ({e})")
            continue
        # Drop demos where the gripper never closed — these are failed
        # recordings (operator forgot to grasp, hit `s` too early, etc.) and
        # would teach the model to ignore the grasp signal.
        if int((np.diff(demo["actions"][:, 6]) != 0).sum()) == 0:
            LOG.warning(f"  episode {i}: SKIPPED — no grasp transitions (failed demo)")
            n_skipped_nograsp += 1
            continue
        idx = append_demo_to_hdf5(out_path, demo, language, control_hz)
        LOG.info(f"  episode {i} -> demo_{idx} (T={demo['num_steps']})")
        n_written += 1
    LOG.info(f"  wrote {n_written}/{n_eps} demos  (skipped {n_skipped_nograsp} no-grasp)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings-dir", type=pathlib.Path, default=DEFAULT_RECORDINGS_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--task", type=str, default=None,
                        help="Convert a single task subdirectory (default: convert all).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.task is not None:
        task_dirs = [args.recordings_dir / args.task]
    else:
        task_dirs = [
            p for p in sorted(args.recordings_dir.iterdir())
            if p.is_dir() and not p.name.startswith("_")
        ]
    if not task_dirs:
        raise SystemExit(f"No task directories under {args.recordings_dir}")

    for td in task_dirs:
        convert_task(td, args.out_dir)


if __name__ == "__main__":
    main()
