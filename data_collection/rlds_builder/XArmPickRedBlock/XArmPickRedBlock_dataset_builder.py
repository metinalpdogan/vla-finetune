"""Custom RLDS dataset builder for xArm pick-up data collected via ril-env.

Adapted from rlds_dataset_builder/LIBERO_Object. Differences:
  - Language instruction comes from the HDF5 attrs (not parsed from filename).
  - Images are NOT rotated 180° (LIBERO sim renders upside-down; our real
    RealSense cameras render right-side up).
  - Single HDF5 file containing many demos (one per task), not a directory.
"""

import glob
import os
from typing import Any, Iterator, Tuple

import h5py
import numpy as np
import tensorflow_datasets as tfds

from XArmPickRedBlock.conversion_utils import MultiThreadedDatasetBuilder


# Single source HDF5 produced by data_collection/zarr_to_libero_hdf5.py.
# Edit this if you add more task HDF5s and want to bundle them into one RLDS dataset.
DATA_HDF5_GLOB = "/home/u-ril/edward/vla-finetune/data_collection/recordings/_libero_hdf5/*.hdf5"


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    def _parse_example(episode_path, demo_id):
        with h5py.File(episode_path, "r") as F:
            if f"demo_{demo_id}" not in F["data"].keys():
                return None
            demo = F["data"][f"demo_{demo_id}"]
            actions = demo["actions"][()]
            states = demo["obs"]["ee_states"][()]               # (T, 6) xyz_m + axis_angle_rad
            gripper_states = demo["obs"]["gripper_states"][()]  # (T, 2)
            joint_states = demo["obs"]["joint_states"][()]      # (T, 7) rad
            images = demo["obs"]["agentview_rgb"][()]
            wrist_images = demo["obs"]["eye_in_hand_rgb"][()]
            # Prefer the per-demo language attr; fall back to file-level attr.
            if "language_instruction" in demo.attrs:
                command = str(demo.attrs["language_instruction"])
            else:
                command = str(F.attrs.get("language_instruction", ""))

        episode = []
        T = actions.shape[0]
        for i in range(T):
            episode.append({
                "observation": {
                    "image": images[i],          # NO 180° rotation (real cameras)
                    "wrist_image": wrist_images[i],
                    "state": np.asarray(np.concatenate((states[i], gripper_states[i]), axis=-1), dtype=np.float32),
                    "joint_state": np.asarray(joint_states[i], dtype=np.float32),
                },
                "action": np.asarray(actions[i], dtype=np.float32),
                "discount": 1.0,
                "reward": float(i == (T - 1)),
                "is_first": i == 0,
                "is_last": i == (T - 1),
                "is_terminal": i == (T - 1),
                "language_instruction": command,
            })

        sample = {
            "steps": episode,
            "episode_metadata": {"file_path": episode_path},
        }
        return f"{episode_path}_{demo_id}", sample

    for hdf5_path in paths:
        with h5py.File(hdf5_path, "r") as F:
            n_demos = len(F["data"])
        idx = 0
        cnt = 0
        while cnt < n_demos:
            ret = _parse_example(hdf5_path, idx)
            if ret is not None:
                cnt += 1
            idx += 1
            if ret is not None:
                yield ret


class XArmPickRedBlock(MultiThreadedDatasetBuilder):
    """RLDS builder for xArm pick-up data."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}
    N_WORKERS = 8
    MAX_PATHS_IN_MEMORY = 4
    PARSE_FCN = _generate_examples

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg",
                            doc="Agentview RGB observation.",
                        ),
                        "wrist_image": tfds.features.Image(
                            shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg",
                            doc="Wrist camera RGB observation.",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(8,), dtype=np.float32,
                            doc="EEF state: xyz_m(3) + axis_angle_rad(3) + gripper_qpos(2).",
                        ),
                        "joint_state": tfds.features.Tensor(
                            shape=(7,), dtype=np.float32, doc="Joint angles (rad).",
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(7,), dtype=np.float32,
                        doc="EEF delta action: dxyz_m(3) + d_axis_angle_rad(3) + grasp{-1,+1}.",
                    ),
                    "discount": tfds.features.Scalar(dtype=np.float32, doc="Discount."),
                    "reward": tfds.features.Scalar(dtype=np.float32, doc="1 on final step."),
                    "is_first": tfds.features.Scalar(dtype=np.bool_, doc="First step flag."),
                    "is_last": tfds.features.Scalar(dtype=np.bool_, doc="Last step flag."),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_, doc="Terminal flag."),
                    "language_instruction": tfds.features.Text(doc="Task instruction."),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(doc="Source HDF5 path."),
                }),
            }))

    def _split_paths(self):
        paths = sorted(glob.glob(DATA_HDF5_GLOB))
        if not paths:
            raise RuntimeError(f"No HDF5s found at {DATA_HDF5_GLOB}")
        return {"train": paths}
