"""RLDS dataset builder for phone-teleoperated xArm data.

Reads LIBERO-format HDF5 files produced by
    phone_data_bridge/zarr_to_libero_hdf5.py
and emits a TFDS-format RLDS dataset that the existing prismatic training
stack consumes via `libero_dataset_transform`. The dataset registration
hooks in prismatic/conf/vla.py + prismatic/vla/datasets/rlds/oxe/{configs,
transforms,mixtures}.py reference the snake_case form `x_arm_phone_teleop`
(TFDS converts the class name automatically).

Differences vs the LIBERO sim builder upstream:
  - No 180° image flip (real cameras render right-side-up; LIBERO sim is
    upside-down because of how the renderer is set up).
  - Per-demo language pulled from the HDF5 attrs, not parsed from filename.
  - 224x224 images instead of 256x256 (phone data is captured at 224).
"""

import glob
import os
from typing import Any, Iterator, Tuple

import h5py
import numpy as np
import tensorflow_datasets as tfds

# HDF5s produced by phone_data_bridge/zarr_to_libero_hdf5.py.
DATA_HDF5_GLOB = "/home/u-ril/edward/vla-finetune/phone_data_bridge/recordings/_libero_hdf5/*.hdf5"


class XArmPhoneTeleop(tfds.core.GeneratorBasedBuilder):
    """RLDS builder for phone-teleoperated xArm data."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(224, 224, 3), dtype=np.uint8, encoding_format="jpeg",
                            doc="Agentview RGB (phone img_0)."),
                        "wrist_image": tfds.features.Image(
                            shape=(224, 224, 3), dtype=np.uint8, encoding_format="jpeg",
                            doc="Wrist RGB (phone img_1)."),
                        "state": tfds.features.Tensor(
                            shape=(8,), dtype=np.float32,
                            doc="EEF state: xyz_m(3) + axis_angle_rad(3) + gripper_qpos(2)."),
                        "joint_state": tfds.features.Tensor(
                            shape=(7,), dtype=np.float32,
                            doc="Joint angles (rad). Phone records none -> zeros."),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(7,), dtype=np.float32,
                        doc="EEF delta action: dxyz_m(3) + d_axis_angle_rad(3) + grasp{-1,+1}."),
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

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        paths = sorted(glob.glob(DATA_HDF5_GLOB))
        if not paths:
            raise RuntimeError(f"No HDF5s found at {DATA_HDF5_GLOB}")
        return {"train": self._generate_examples(paths)}

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        for hdf5_path in paths:
            with h5py.File(hdf5_path, "r") as F:
                file_lang = str(F.attrs.get("language_instruction", ""))
                for demo_key in sorted(F["data"].keys(),
                                       key=lambda k: int(k.split("_")[1])):
                    demo = F["data"][demo_key]
                    actions = demo["actions"][()]
                    states = demo["obs"]["ee_states"][()]
                    gripper_states = demo["obs"]["gripper_states"][()]
                    joint_states = demo["obs"]["joint_states"][()]
                    images = demo["obs"]["agentview_rgb"][()]
                    wrist_images = demo["obs"]["eye_in_hand_rgb"][()]
                    language = str(demo.attrs.get("language_instruction", file_lang))

                    T = actions.shape[0]
                    steps = []
                    for i in range(T):
                        steps.append({
                            "observation": {
                                "image": images[i],
                                "wrist_image": wrist_images[i],
                                "state": np.asarray(
                                    np.concatenate((states[i], gripper_states[i]), axis=-1),
                                    dtype=np.float32),
                                "joint_state": np.asarray(joint_states[i], dtype=np.float32),
                            },
                            "action": np.asarray(actions[i], dtype=np.float32),
                            "discount": 1.0,
                            "reward": float(i == (T - 1)),
                            "is_first": i == 0,
                            "is_last": i == (T - 1),
                            "is_terminal": i == (T - 1),
                            "language_instruction": language,
                        })

                    yield f"{hdf5_path}::{demo_key}", {
                        "steps": steps,
                        "episode_metadata": {"file_path": hdf5_path},
                    }
