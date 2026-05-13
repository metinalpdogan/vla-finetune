"""
phone_data_bridge/zarr_to_libero_hdf5.py

Convert the phone_data_collection teleop_data.zarr into LIBERO-format HDF5
that the existing rlds_builder + VLA training pipeline can consume.

Input  (phone_data_collection/recorder.py schema):
    /data
      state       (N, 7)  float32   [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg, grasp{0,1}]
      n_contacts  (N, 1)  float32   placeholder (always 0)
      img_0       (N,H,W,3) float32 agentview, in [0, 1], H=W=224
      img_1       (N,H,W,3) float32 wrist,     in [0, 1], H=W=224
    /meta
      episode_ends (E,)   int64     cumulative end indices

Output  (LIBERO HDF5, drop-in for rlds_builder/XArmPhoneTeleop):
    <out_hdf5>
      attrs: language_instruction, control_hz
      data/demo_<i>/
        attrs: language_instruction, num_steps, success
        obs/
          gripper_states  (T, 2)   float32   mirrored binary grasp
          joint_states    (T, 7)   float32   zeros (phone doesn't record joints)
          ee_states       (T, 6)   float32   xyz_m + axis_angle_rad
          ee_pos          (T, 3)   float32
          ee_ori          (T, 3)   float32
          agentview_rgb   (T, 224, 224, 3) uint8
          eye_in_hand_rgb (T, 224, 224, 3) uint8
        actions      (T, 7)        float32   delta xyz_m + delta axis-angle_rad + grasp{-1,+1}
        states       (T, 9)        float32   == robot_states
        robot_states (T, 9)        float32   [grip_qpos(2), ee_pos(3), ee_quat_xyzw(4)]
        rewards      (T,)          uint8     1 at last step
        dones        (T,)          uint8     1 at last step

Action labels are `tcp[t+1] - tcp[t]` (actual physical motion) — same convention
the previous ril-env converter settled on after we found that commanded-vs-tcp
deltas store teleop lag, not motion. Grasp is from the commanded state stream
(phone button); we map {0=open, 1=close} (phone convention, matches ril-env)
into LIBERO HDF5 raw {-1=open, +1=close}. Downstream libero_dataset_transform
clips to [0,1] then inverts, so signed input works.

Usage:
    python zarr_to_libero_hdf5.py \\
        --zarr /home/u-ril/edward/phone_data_collection/teleop_data.zarr \\
        --out  /home/u-ril/edward/vla-finetune/phone_data_bridge/recordings/_libero_hdf5/x_arm_phone_teleop_demo.hdf5 \\
        --language "pick up the red block"
"""

import argparse
import logging
import pathlib

import cv2
import h5py
import numpy as np
import zarr
from scipy.spatial.transform import Rotation as Rot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("phone_zarr_to_libero")

CONTROL_HZ = 10                    # phone_data_collection RecordingThread default
TARGET_IMAGE_SIZE = 224            # phone collects at 224x224 already; no resize
PAUSE_XYZ_MM_THRESHOLD = 0.5       # leading-frame trim: consecutive TCP delta < this = paused
PAUSE_RPY_DEG_THRESHOLD = 0.5


def _tcp_to_delta_actions(tcp_pose_mm_deg: np.ndarray) -> np.ndarray:
    """(T, 6) absolute TCP [mm + deg] -> (T, 6) per-step delta [m + rad axis-angle].
    Last step's delta is zero (no t+1 to subtract from)."""
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


def _trim_leading_paused(tcp: np.ndarray) -> int:
    """Count of leading frames where the operator was holding still. We trim
    these because they teach the policy 'this scene -> predict noop', which
    becomes a closed-loop attractor at deployment."""
    if len(tcp) < 2:
        return 0
    dxyz = np.linalg.norm(np.diff(tcp[:, :3], axis=0), axis=1)
    drpy = np.linalg.norm(np.diff(tcp[:, 3:6], axis=0), axis=1)
    is_paused = np.concatenate(
        [[True], (dxyz < PAUSE_XYZ_MM_THRESHOLD) & (drpy < PAUSE_RPY_DEG_THRESHOLD)]
    )
    return int(np.argmax(~is_paused)) if not is_paused.all() else len(is_paused)


def _float01_to_uint8(img_f: np.ndarray) -> np.ndarray:
    """Phone stores images as float32 in [0,1]; LIBERO HDF5 wants uint8."""
    return np.clip(img_f * 255.0, 0, 255).astype(np.uint8)


def _pick_image_key(g: zarr.Group, cam_idx: int) -> str:
    """Prefer img_{i} (tactile-overlay-painted frames) over img_{i}_raw (clean).

    The overlay-painted stream is the deliberate training input — the tactile
    force visualization drawn on top of the camera frame is information we
    WANT the model to see and learn from. At deployment, the same overlay must
    be re-rendered onto the live camera feed before handing it to the model;
    otherwise the model sees an out-of-distribution input.

    Fall back to img_{i}_raw only for legacy zarrs that didn't write the
    non-raw stream (none of the current phone_data_collection ones).
    """
    overlay_key = f"data/img_{cam_idx}"
    raw_key = f"data/img_{cam_idx}_raw"
    if overlay_key in g:
        return overlay_key
    if raw_key in g:
        return raw_key
    raise KeyError(f"Neither {overlay_key} nor {raw_key} present in zarr")


def convert_episode(g: zarr.Group, episode_idx: int, ep_start: int, ep_end: int,
                    language: str) -> dict:
    state = np.asarray(g["data/state"][ep_start:ep_end], dtype=np.float32)  # (T, 7)
    img_a = np.asarray(g[_pick_image_key(g, 0)][ep_start:ep_end])           # (T, 224, 224, 3) f32
    img_w = np.asarray(g[_pick_image_key(g, 1)][ep_start:ep_end])

    tcp = state[:, :6]      # (T, 6) mm + deg
    grasp = state[:, 6]     # (T,)   {0=open, 1=close}

    # Trim leading "user was paused" frames (operator holding the phone still).
    n_lead = _trim_leading_paused(tcp)
    if n_lead > 0:
        tcp = tcp[n_lead:]
        grasp = grasp[n_lead:]
        img_a = img_a[n_lead:]
        img_w = img_w[n_lead:]

    n_steps = tcp.shape[0]
    if n_steps < 5:
        raise RuntimeError(f"episode {episode_idx}: only {n_steps} steps after trim")

    # Action: tcp[t+1] - tcp[t] (m + rad), grasp mapped to LIBERO raw {-1=open, +1=close}.
    delta_pose = _tcp_to_delta_actions(tcp)
    grasp_libero = (2.0 * grasp - 1.0).astype(np.float32).reshape(-1, 1)
    actions = np.concatenate([delta_pose, grasp_libero], axis=1).astype(np.float32)

    # Observation tensors in LIBERO layout
    ee_pos_m = (tcp[:, :3] / 1000.0).astype(np.float32)
    ee_axis_angle = np.stack([_euler_deg_to_axis_angle(rpy) for rpy in tcp[:, 3:6]], axis=0)
    ee_quat = np.stack([_euler_deg_to_quat_xyzw(rpy) for rpy in tcp[:, 3:6]], axis=0)
    ee_states = np.concatenate([ee_pos_m, ee_axis_angle], axis=1).astype(np.float32)
    grip_qpos_2d = np.stack([grasp, grasp], axis=1).astype(np.float32)
    robot_states = np.concatenate([grip_qpos_2d, ee_pos_m, ee_quat], axis=1).astype(np.float32)
    # Phone doesn't record joint angles. Write zeros; libero_dataset_transform
    # downstream only reads ee_state + gripper_state, so this column is unused
    # by training but is required by the LIBERO HDF5 schema.
    joint_states = np.zeros((n_steps, 7), dtype=np.float32)

    # Phone stored 224x224 already; keep size. If a different vision backbone
    # ever needs 256, resize here.
    agentview_rgb = np.stack([cv2.resize(_float01_to_uint8(f),
                                         (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE),
                                         interpolation=cv2.INTER_AREA) for f in img_a], axis=0)
    wrist_rgb = np.stack([cv2.resize(_float01_to_uint8(f),
                                      (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE),
                                      interpolation=cv2.INTER_AREA) for f in img_w], axis=0)

    rewards = np.zeros(n_steps, dtype=np.uint8)
    rewards[-1] = 1
    dones = np.zeros(n_steps, dtype=np.uint8)
    dones[-1] = 1

    return {
        "language": language,
        "obs": {
            "gripper_states": grip_qpos_2d,
            "joint_states": joint_states,
            "ee_states": ee_states,
            "ee_pos": ee_pos_m,
            "ee_ori": ee_axis_angle,
            "agentview_rgb": agentview_rgb,
            "eye_in_hand_rgb": wrist_rgb,
        },
        "actions": actions,
        "robot_states": robot_states,
        "rewards": rewards,
        "dones": dones,
        "num_steps": n_steps,
    }


def append_demo_to_hdf5(out_path: pathlib.Path, demo: dict, language: str, control_hz: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if out_path.exists() else "w"
    with h5py.File(out_path, mode) as f:
        f.attrs.setdefault("language_instruction", language)
        f.attrs.setdefault("control_hz", control_hz)
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
        ep.create_dataset("states", data=demo["robot_states"], compression="gzip")
        ep.create_dataset("robot_states", data=demo["robot_states"], compression="gzip")
        ep.create_dataset("rewards", data=demo["rewards"])
        ep.create_dataset("dones", data=demo["dones"])
    return idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent / "source/teleop_data.zarr",
                        help="Path to phone-collected teleop_data.zarr. Default is the local "
                             "copy under phone_data_bridge/source/.")
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent
                        / "recordings/_libero_hdf5/x_arm_phone_teleop_demo.hdf5",
                        help="Output LIBERO HDF5 path.")
    parser.add_argument("--language", type=str, default="pick up the red block",
                        help="Task instruction baked into every demo.")
    parser.add_argument("--control-hz", type=int, default=CONTROL_HZ,
                        help="Control rate of the recorded data (phone default is 10 Hz).")
    args = parser.parse_args()

    g = zarr.open(str(args.zarr), mode="r")
    ep_ends = np.asarray(g["meta/episode_ends"][:])
    n_eps = len(ep_ends)
    if n_eps == 0:
        raise SystemExit(f"No episodes in {args.zarr}")
    LOG.info(f"Source: {args.zarr}  ({n_eps} episodes, {int(ep_ends[-1])} total steps)")

    if args.out.exists():
        args.out.unlink()

    n_written = n_skipped = 0
    ep_start = 0
    for i, ep_end in enumerate(ep_ends.tolist()):
        try:
            demo = convert_episode(g, i, ep_start, int(ep_end), args.language)
        except Exception as e:
            LOG.error(f"  episode {i}: failed ({e})")
            ep_start = int(ep_end)
            n_skipped += 1
            continue
        # Drop demos with no grasp transition — operator forgot to grasp.
        if int((np.diff(demo["actions"][:, 6]) != 0).sum()) == 0:
            LOG.warning(f"  episode {i}: SKIPPED (no grasp transitions)")
            ep_start = int(ep_end)
            n_skipped += 1
            continue
        idx = append_demo_to_hdf5(args.out, demo, args.language, int(args.control_hz))
        LOG.info(f"  episode {i} -> demo_{idx}  (T={demo['num_steps']})")
        n_written += 1
        ep_start = int(ep_end)

    LOG.info(f"Done. Wrote {n_written}/{n_eps} demos to {args.out} (skipped {n_skipped})")


if __name__ == "__main__":
    main()
