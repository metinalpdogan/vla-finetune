"""
run_xarm_inference.py

Real-robot inference script for running a trained MiniVLA policy on the xArm robot.
Loads a Prismatic checkpoint, captures images from cameras, predicts actions via the VLA model,
and executes them on the real robot in a closed loop.

Hardware Requirements:
    - xArm 6/7 robot (reachable via IP)
    - 2x RealSense cameras (agent view + wrist view) OR USB cameras
    - CUDA-capable GPU for inference

Dependencies:
    - Main VLA environment (see repo root pyproject.toml)
    - xarm-python-sdk (for robot control)
    - pyrealsense2 (for RealSense cameras) OR opencv-python (for USB cameras)

Usage:
    # Run with default settings (pick red block task)
    python experiments/robot/xarm/run_xarm_inference.py \
        --checkpoint runs/prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-xarm-pick-red-block+<timestamp>/checkpoints/step-002000-epoch-04-loss=0.5817.pt \
        --xarm_ip 192.168.1.223

    # Custom task and cameras
    python experiments/robot/xarm/run_xarm_inference.py \
        --checkpoint path/to/checkpoint.pt \
        --xarm_ip 192.168.1.223 \
        --instruction "pick up the red block and place it on the plate" \
        --agent_cam_id 0 \
        --wrist_cam_id 2 \
        --control_hz 10 \
        --max_steps 200

Safety:
    - Press Ctrl+C to emergency stop at any time
    - Robot will return to home position on exit
    - Set --dry_run to test without robot execution
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# Prepend the vendored xArm SDK at ril-env/xarm/ to sys.path so `from xarm.wrapper import XArmAPI`
# resolves to the in-repo copy (matches what data_collection/collect_demos.py does). This avoids
# requiring `pip install xarm-python-sdk`, which would shadow the vendored copy.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_RIL_ENV_DIR = _REPO_ROOT / "ril-env"
if _RIL_ENV_DIR.exists() and str(_RIL_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(_RIL_ENV_DIR))

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

# VLA model imports
from prismatic.models import load_vla
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer

# Robot control imports — uses the vendored SDK at ril-env/xarm/ (see sys.path insert above)
try:
    from xarm.wrapper import XArmAPI
except ImportError:
    raise SystemExit(
        f"xArm SDK not found. Expected vendored copy at {_RIL_ENV_DIR}/xarm/. "
        "If the ril-env/ directory is missing, run `bash setup_inference.sh` "
        "or `pip install xarm-python-sdk` as a fallback."
    )


# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# === Configuration ===
@dataclass
class XArmInferenceConfig:
    # Model paths
    checkpoint: Path                             # Path to trained .pt checkpoint
    base_vlm: str = "prism-qwen25-extra-dinosiglip-224px+0_5b"  # Base VLM architecture

    # Robot configuration
    xarm_ip: str = "192.168.1.223"              # xArm robot IP address
    control_hz: int = 10                         # Control frequency (Hz)
    home_joints: list = None                     # Home joint positions [7D]; None = use default

    # Camera configuration
    agent_cam_id: int = 0                        # Agent view camera ID (USB cam index or RS serial)
    wrist_cam_id: int = 2                        # Wrist view camera ID
    image_size: int = 224                        # Match training-time `resize_resolution` (the
                                                  # vision backbone's default_image_resolution).
                                                  # CameraManager does stretch-resize 1280x720 -> 224
                                                  # then center-crops 0.9 area, mirroring the training
                                                  # RLDS pipeline (resize_resolution=(224,224) +
                                                  # random_resized_crop(scale=[0.9,0.9])).
    use_realsense: bool = True                   # Use RealSense; False = USB cameras

    # Task configuration
    instruction: str = "pick up the red block and place it on the plate"
    max_steps: int = 200                         # Maximum steps per rollout
    unnorm_key: str = "x_arm_pick_red_block"    # Dataset key for action denormalization

    # Execution parameters
    action_scale: float = 1.0                    # Scale predicted actions (for safety)
    dry_run: bool = False                        # If True, predict but don't execute actions
    save_video: bool = True                      # Save rollout video
    video_path: Optional[Path] = None            # Output video path; None = auto-generate

    # Model inference
    device: str = "cuda:0"                       # Device for inference
    torch_dtype: torch.dtype = torch.bfloat16    # Model dtype

    # Safety
    max_action_norm: float = 0.15                # Clip action magnitude for safety (m/rad)
    enable_collision_check: bool = True          # Enable xArm collision detection

    # Warm-up / "kickstart" override (band-aid for the closed-loop noop
    # attractor — model predicts noop on start-of-demo-looking scenes, arm
    # never moves, scene never changes). For the first `warmup_steps` ticks,
    # override the predicted xyz/rpy with a fixed downward motion of magnitude
    # `warmup_dz` per step. The model's grasp prediction is kept (so we don't
    # accidentally fight a CLOSED command). After warmup, control reverts to
    # the model. If after warmup the model still predicts noop, the issue is
    # the trained model itself — re-collect/re-train. If it starts predicting
    # real actions, the model is fine and the production fix is to retrain
    # with leading paused frames trimmed (already wired into stage 2).
    warmup_steps: int = 0                        # 0 disables; ~10 is a sane test value
    warmup_dz: float = -0.005                    # per-step delta in meters (negative = down)

    def __post_init__(self):
        if self.home_joints is None:
            # Median joint config the operator manually positioned the arm to
            # at the start of every successful pick_red_block demo (computed
            # from the 62-episode replay buffer: median of JointAngles[:5]
            # across all demos). Homing here means the agentview at step 0
            # of the rollout matches what the model saw at frame 0 of every
            # training demo (gripper pointing down, ~47 cm forward, ~23 cm
            # up, visible in the agent camera frame). The vanilla ril-env
            # default [0, 0, 0, 70, 0, 70, 0] parks the arm slightly higher
            # / further back and out of the model's training distribution,
            # which makes it stuck in a noop loop on rollout start.
            self.home_joints = [-1.4, 0.0, -1.3, 67.4, 2.1, 65.9, -3.7]
        if self.video_path is None and self.save_video:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.video_path = Path(f"rollouts/xarm_inference_{timestamp}.mp4")


# === Camera Manager ===
class CameraManager:
    """Manages multiple cameras (RealSense or USB) for robot observation."""

    def __init__(self, agent_cam_id: int, wrist_cam_id: int,
                 image_size: int = 224, use_realsense: bool = True):
        self.agent_cam_id = agent_cam_id
        self.wrist_cam_id = wrist_cam_id
        self.image_size = image_size
        self.use_realsense = use_realsense

        if use_realsense:
            self._init_realsense()
        else:
            self._init_usb_cameras()

    def _init_realsense(self):
        """Initialize RealSense cameras to match the collection-time pipeline EXACTLY.

        Training-time pipeline (data_collection/collect_demos.py):
            - capture resolution 1280x720 @ 30 fps
            - exposure = 120, gain = 0   (manual; not auto)
            - white_balance = 5900 K     (manual; produces the warm/peach tint)
            - frames recorded to MP4, then stage-2 cv2.resize(1280x720 -> 256x256)
              which *stretches* 16:9 into 1:1 (no crop)

        If we capture at a different resolution, set auto exposure, or
        center-crop the frame, the deployment-time pixel distribution is
        different enough that the model falls back to predicting the median
        action bin (i.e. zero motion). We saw this firsthand on this rig.
        """
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise SystemExit("pyrealsense2 not found. Install: pip install pyrealsense2")

        self.agent_pipeline = rs.pipeline()
        self.wrist_pipeline = rs.pipeline()

        # MUST match collect_demos.py's RECORD_RES (1280x720) and fps (30)
        config_agent = rs.config()
        config_agent.enable_device(str(self.agent_cam_id))
        config_agent.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        config_wrist = rs.config()
        config_wrist.enable_device(str(self.wrist_cam_id))
        config_wrist.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        agent_profile = self.agent_pipeline.start(config_agent)
        wrist_profile = self.wrist_pipeline.start(config_wrist)

        # Apply the same exposure / white-balance overrides as collection.
        # Without these, RealSense auto-WB produces a cyan tint instead of
        # the warm 5900K tint the model trained on.
        for profile in (agent_profile, wrist_profile):
            sensor = profile.get_device().query_sensors()[1]  # color sensor
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.exposure, 120)
            sensor.set_option(rs.option.gain, 0)
            sensor.set_option(rs.option.enable_auto_white_balance, 0)
            sensor.set_option(rs.option.white_balance, 5900)

        # Warm up cameras (settings need a moment to take effect)
        for _ in range(30):
            self.agent_pipeline.wait_for_frames()
            self.wrist_pipeline.wait_for_frames()

        logger.info(
            f"RealSense cameras initialized: agent={self.agent_cam_id}, wrist={self.wrist_cam_id} "
            f"(1280x720 @ 30 fps, exposure=120, wb=5900K — matches collect_demos.py)"
        )

    def _init_usb_cameras(self):
        """Initialize USB cameras."""
        self.agent_cap = cv2.VideoCapture(self.agent_cam_id)
        self.wrist_cap = cv2.VideoCapture(self.wrist_cam_id)

        if not self.agent_cap.isOpened() or not self.wrist_cap.isOpened():
            raise RuntimeError("Failed to open USB cameras")

        # Set resolution
        for cap in [self.agent_cap, self.wrist_cap]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Warm up
        for _ in range(10):
            self.agent_cap.read()
            self.wrist_cap.read()

        logger.info(f"USB cameras initialized: agent={self.agent_cam_id}, wrist={self.wrist_cam_id}")

    def get_observation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture images from both cameras.

        Returns:
            agent_image: (224, 224, 3) RGB uint8
            wrist_image: (224, 224, 3) RGB uint8
        """
        if self.use_realsense:
            import pyrealsense2 as rs

            agent_frames = self.agent_pipeline.wait_for_frames()
            agent_color = agent_frames.get_color_frame()
            agent_img = np.asanyarray(agent_color.get_data())  # BGR

            wrist_frames = self.wrist_pipeline.wait_for_frames()
            wrist_color = wrist_frames.get_color_frame()
            wrist_img = np.asanyarray(wrist_color.get_data())  # BGR
        else:
            ret1, agent_img = self.agent_cap.read()
            ret2, wrist_img = self.wrist_cap.read()

            if not ret1 or not ret2:
                raise RuntimeError("Failed to capture from USB cameras")

        # Convert BGR -> RGB
        agent_rgb = cv2.cvtColor(agent_img, cv2.COLOR_BGR2RGB)
        wrist_rgb = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)

        # === Training pipeline replication (CRITICAL) ===
        # 1. Stage 2 converter: cv2.resize(1280x720 -> 256x256), stretches 16:9.
        # 2. RLDS dataloader at training time: cv2.resize(256x256 -> 224x224)
        #    to match the vision backbone's `default_image_resolution`.
        # 3. image_aug=True applied random_resized_crop(scale=[0.9, 0.9]) to
        #    every training sample — a ~95%-side / 90%-area square crop placed
        #    randomly, then resized back to 224x224.
        # 4. The vision backbone's image_transform then normalizes the 224x224
        #    PIL image.
        #
        # At deployment we should reproduce all of steps 1-3 deterministically,
        # picking the *center* 90%-area crop (since LIBERO's eval convention).
        # Skipping step 3 (as the script originally did) means the model is
        # being shown a strict superset of every training pixel — the framing
        # is wrong by a few percent — and it collapses to a noop prediction.
        # See `experiments/robot/openvla_utils.py:crop_and_resize` (the LIBERO
        # eval helper) and the upstream OpenVLA README's "--center_crop True"
        # note for context.

        def _stretch_then_center_crop_0p9(img: np.ndarray) -> np.ndarray:
            # Step 1+2: stretch-resize to 224 (matches training resize_resolution).
            img224 = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            # Step 3: 0.9 area => sqrt(0.9) ≈ 0.9487 side. Center-crop, then
            # resize back to the model's input size.
            side_frac = np.sqrt(0.9)
            new_side = int(round(self.image_size * side_frac))
            off = (self.image_size - new_side) // 2
            cropped = img224[off : off + new_side, off : off + new_side]
            return cv2.resize(cropped, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        agent_resized = _stretch_then_center_crop_0p9(agent_rgb)
        wrist_resized = _stretch_then_center_crop_0p9(wrist_rgb)

        return agent_resized, wrist_resized

    def close(self):
        """Release camera resources."""
        if self.use_realsense:
            self.agent_pipeline.stop()
            self.wrist_pipeline.stop()
        else:
            self.agent_cap.release()
            self.wrist_cap.release()
        logger.info("Cameras closed")


# === xArm Robot Controller ===
class XArmRobotController:
    """Wrapper for xArm robot control with safety features."""

    def __init__(self, robot_ip: str, control_hz: int = 10, home_joints: list = None,
                 enable_collision_check: bool = True):
        self.robot_ip = robot_ip
        self.control_hz = control_hz
        self.home_joints = home_joints or [0, 0, 0, 70, 0, 70, 0]
        self.dt = 1.0 / control_hz

        # Initialize xArm
        self.arm = XArmAPI(robot_ip)
        self.arm.connect()

        # Clear errors
        self.arm.clean_error()
        self.arm.clean_warn()

        # Enable motion
        code = self.arm.motion_enable(enable=True)
        if code != 0:
            raise RuntimeError(f"Failed to enable motion: code {code}")

        # Set servo mode (real-time control)
        self.arm.set_mode(1)
        self.arm.set_state(0)

        # Configure collision detection
        if enable_collision_check:
            self.arm.set_collision_sensitivity(3)  # Medium sensitivity

        # Get current state
        self.current_pose = None
        self.current_gripper = 0.0
        self.update_state()

        logger.info(f"xArm initialized at {robot_ip}, control Hz: {control_hz}")

    def update_state(self):
        """Read current robot state."""
        code, pose = self.arm.get_position()
        if code == 0:
            # pose = [x, y, z, roll, pitch, yaw] in mm and degrees
            self.current_pose = np.array(pose[:6], dtype=np.float32)
            self.current_pose[:3] /= 1000.0  # mm to meters
            self.current_pose[3:] = np.deg2rad(self.current_pose[3:])  # deg to rad

        code, gripper_pos = self.arm.get_gripper_position()
        if code == 0:
            # Normalize gripper: 0=open (850), 1=closed (0)
            self.current_gripper = 1.0 - (gripper_pos / 850.0)

    def move_to_home(self, speed: float = 50.0):
        """Move robot to home position."""
        logger.info("Moving to home position...")
        self.arm.set_mode(0)  # Position mode
        self.arm.set_state(0)
        self.arm.set_servo_angle(angle=self.home_joints, speed=speed, wait=True)
        self.arm.set_mode(1)  # Back to servo mode
        self.arm.set_state(0)
        logger.info("Reached home position")

    def execute_action(self, action: np.ndarray, action_scale: float = 1.0):
        """
        Execute one VLA action on the robot.

        The action vector is the *unnormalized* output of MiniVLA after the LIBERO
        RLDS transform's gripper inversion. Layout (matches training data
        normalization stats in dataset_statistics.json):

            action[0:3]  delta xyz in meters         (action_scale applies)
            action[3:6]  delta axis-angle in radians (action_scale applies)
            action[6]    gripper target in [0, 1]    where 1 = OPEN, 0 = CLOSED
                          (this is the libero_dataset_transform's inverted
                           convention; the raw user-side data was {-1=open,
                           +1=closed} but RLDS clipped to [0,1] then did 1-x,
                           so the model emits 1.0 for open / 0.0 for closed.)

        Only the 6-D pose component is scaled by action_scale; gripper is a
        target state, not a delta, so scaling it makes no physical sense.
        """
        # Pose delta (scaled for safety)
        delta_pos = action[:3] * action_scale
        delta_rot = action[3:6] * action_scale
        target_pos = self.current_pose[:3] + delta_pos
        target_rot_euler = self.current_pose[3:] + delta_rot

        target_xarm = np.concatenate([
            target_pos * 1000.0,            # m -> mm
            np.rad2deg(target_rot_euler),    # rad -> deg
        ])
        code = self.arm.set_servo_cartesian(target_xarm.tolist(), speed=100, mvacc=2000)
        if code != 0:
            logger.warning(f"Servo command failed: code {code}")

        # Gripper: model emits target state in [0, 1] (1=open, 0=closed).
        # xArm parallel gripper: position 0 = closed, 850 = open. So map open
        # probability directly: pos = round(open_prob * 850). Threshold to
        # discrete {open, closed} to avoid jittery half-closed commands.
        open_prob = float(np.clip(action[6], 0.0, 1.0))
        gripper_pos = 850 if open_prob > 0.5 else 0
        self.arm.set_gripper_position(gripper_pos, wait=False, speed=5000)

        self.update_state()

    def emergency_stop(self):
        """Emergency stop the robot."""
        logger.warning("EMERGENCY STOP!")
        self.arm.set_state(4)  # Stop state
        time.sleep(0.1)
        self.arm.set_state(0)  # Ready state

    def close(self):
        """Disconnect from robot."""
        self.move_to_home()
        self.arm.disconnect()
        logger.info("Robot disconnected")


# === VLA Model Wrapper ===
class VLAModel:
    """Wrapper around the Prismatic-format MiniVLA for closed-loop inference.

    Notes:
      - Uses prismatic.models.load_vla. That function expects the checkpoint path to be
        `<RUN_DIR>/checkpoints/<file>.pt` with sibling `config.json` and
        `dataset_statistics.json`. Don't pass a bare .pt sitting at the repo root.
      - Norm stats are loaded by load_vla from dataset_statistics.json and attached to
        the returned OpenVLA model; this class doesn't manage them separately.
      - `predict_action(image=..., instruction=..., unnorm_key=...)` is the Prismatic
        OpenVLA API. It handles tokenization + image transform + generation + action
        de-normalization internally. We just hand it PIL images and a string.
    """

    def __init__(self, checkpoint_path: Path, unnorm_key: str,
                 device: str = "cuda:0", torch_dtype: torch.dtype = torch.bfloat16):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.unnorm_key = unnorm_key

        # Sanity-check the run-dir layout before load_vla asserts on it (its message
        # is cryptic).
        ckpt = Path(checkpoint_path)
        if not (ckpt.suffix == ".pt" and ckpt.parent.name == "checkpoints"):
            raise RuntimeError(
                f"Checkpoint must live at <RUN_DIR>/checkpoints/<file>.pt. "
                f"Got {ckpt}. Move the .pt into a checkpoints/ subdir of the run dir "
                f"that also contains config.json and dataset_statistics.json."
            )
        run_dir = ckpt.parents[1]
        for f in ("config.json", "dataset_statistics.json"):
            if not (run_dir / f).exists():
                raise RuntimeError(f"Missing {run_dir / f}; required by prismatic.models.load_vla.")

        logger.info(f"Loading VLA from {checkpoint_path} (run_dir={run_dir})")
        self.vla = load_vla(str(checkpoint_path))
        self.vla = self.vla.to(self.device).eval()
        logger.info(f"VLA model ready on {device}")

    def predict_action(self, agent_image: np.ndarray, wrist_image: np.ndarray,
                       instruction: str) -> np.ndarray:
        """
        Predict an action from a (agentview, wrist) image pair + instruction.

        Args:
            agent_image: (H, W, 3) RGB uint8 — primary / agentview camera.
            wrist_image: (H, W, 3) RGB uint8 — eye-in-hand camera.
            instruction: language instruction string (e.g. "pick up the red block").

        Returns:
            action: (7,) float32 — [dx_m, dy_m, dz_m, drx_rad, dry_rad, drz_rad, grasp].
                    Already denormalized via the model's stored Q01/Q99 stats for
                    `unnorm_key` (no further post-processing needed).
        """
        # Order must match training: agentview first, then wrist (see the t-2 / wrist
        # MiniVLA configs in prismatic/conf/vla.py — image_sequence_len=2 with
        # use_wrist_image=True interleaves [agent, wrist]).
        images = [Image.fromarray(agent_image), Image.fromarray(wrist_image)]
        with torch.no_grad():
            action = self.vla.predict_action(image=images, instruction=instruction, unnorm_key=self.unnorm_key)
        return action  # (7,) numpy array, denormalized


# === Video Recorder ===
class VideoRecorder:
    """Simple video recorder for rollouts."""

    def __init__(self, video_path: Path, fps: int = 10, frame_size: Tuple[int, int] = (448, 224)):
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
        self.video_path = video_path
        logger.info(f"Video recorder initialized: {video_path}")

    def write_frame(self, agent_img: np.ndarray, wrist_img: np.ndarray):
        """Write concatenated frame to video."""
        # Concatenate side-by-side
        frame = np.concatenate([agent_img, wrist_img], axis=1)  # (224, 448, 3)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame_bgr)

    def close(self):
        self.writer.release()
        logger.info(f"Video saved: {self.video_path}")


# === Main Inference Loop ===
def run_inference(config: XArmInferenceConfig):
    """Main inference loop: load model, capture observations, predict actions, execute."""

    # Initialize components
    logger.info("=" * 80)
    logger.info("Starting xArm VLA Inference")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {config.checkpoint}")
    logger.info(f"Instruction: {config.instruction}")
    logger.info(f"Control Hz: {config.control_hz}")
    logger.info(f"Max steps: {config.max_steps}")
    logger.info(f"Dry run: {config.dry_run}")
    logger.info("=" * 80)

    # Load VLA model
    vla_model = VLAModel(
        checkpoint_path=config.checkpoint,
        unnorm_key=config.unnorm_key,
        device=config.device,
        torch_dtype=config.torch_dtype
    )

    # Initialize cameras
    camera_manager = CameraManager(
        agent_cam_id=config.agent_cam_id,
        wrist_cam_id=config.wrist_cam_id,
        image_size=config.image_size,
        use_realsense=config.use_realsense
    )

    # Initialize robot (only if not dry run)
    robot = None
    if not config.dry_run:
        robot = XArmRobotController(
            robot_ip=config.xarm_ip,
            control_hz=config.control_hz,
            home_joints=config.home_joints,
            enable_collision_check=config.enable_collision_check
        )
        robot.move_to_home()
        input("Press Enter to start rollout...")

    # Video recorder
    video_recorder = None
    if config.save_video:
        video_recorder = VideoRecorder(
            video_path=config.video_path,
            fps=config.control_hz,
            frame_size=(config.image_size * 2, config.image_size)  # side-by-side
        )

    # Rollout loop
    try:
        logger.info("Starting rollout...")
        for step in range(config.max_steps):
            step_start = time.time()

            # Capture observations
            agent_img, wrist_img = camera_manager.get_observation()

            # Predict action
            action = vla_model.predict_action(agent_img, wrist_img, config.instruction)
            model_xyz = action[:3].copy()  # remember model output for logging during warmup

            # === Warm-up override (band-aid for noop attractor) ===
            in_warmup = step < config.warmup_steps
            if in_warmup:
                # Force a fixed downward delta; keep model's grasp prediction.
                action[0] = 0.0
                action[1] = 0.0
                action[2] = config.warmup_dz
                action[3] = 0.0
                action[4] = 0.0
                action[5] = 0.0

            # Clip action magnitude for safety
            action_norm = np.linalg.norm(action[:6])
            if action_norm > config.max_action_norm:
                action[:6] *= config.max_action_norm / action_norm
                logger.warning(f"Step {step}: Clipped action (norm={action_norm:.3f})")

            # Execute action
            if not config.dry_run:
                robot.execute_action(action, action_scale=config.action_scale)

            # Record video
            if video_recorder:
                video_recorder.write_frame(agent_img, wrist_img)

            # Logging — show full action precision + the model's underlying
            # prediction during warmup so we can spot when the model starts
            # producing real motion of its own.
            tag = "WARMUP " if in_warmup else "MODEL  "
            extra = (
                f"  model_xyz=[{model_xyz[0]:+.4f}, {model_xyz[1]:+.4f}, {model_xyz[2]:+.4f}]"
                if in_warmup else ""
            )
            logger.info(
                f"Step {step:3d}/{config.max_steps} [{tag}]: "
                f"xyz=[{action[0]:+.4f}, {action[1]:+.4f}, {action[2]:+.4f}] m  "
                f"rpy=[{action[3]:+.4f}, {action[4]:+.4f}, {action[5]:+.4f}] rad  "
                f"grasp={action[6]:.2f} ({'OPEN' if action[6] > 0.5 else 'CLOSED'})"
                f"{extra}"
            )

            # Maintain control frequency
            elapsed = time.time() - step_start
            if elapsed < 1.0 / config.control_hz:
                time.sleep(1.0 / config.control_hz - elapsed)

        logger.info("Rollout completed successfully!")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C)")
        if robot:
            robot.emergency_stop()

    except Exception as e:
        logger.error(f"Error during rollout: {e}", exc_info=True)
        if robot:
            robot.emergency_stop()
        raise

    finally:
        # Cleanup
        logger.info("Cleaning up...")
        camera_manager.close()
        if robot:
            robot.close()
        if video_recorder:
            video_recorder.close()
        logger.info("Shutdown complete")


# === CLI Entry Point ===
def main():
    parser = argparse.ArgumentParser(description="Run MiniVLA inference on xArm robot")

    # Model
    parser.add_argument("--checkpoint", type=Path, required=True,
                       help="Path to trained .pt checkpoint")
    parser.add_argument("--base_vlm", type=str,
                       default="prism-qwen25-extra-dinosiglip-224px+0_5b",
                       help="Base VLM architecture")
    parser.add_argument("--unnorm_key", type=str, default="x_arm_pick_red_block",
                       help="Dataset key for action denormalization")

    # Robot
    parser.add_argument("--xarm_ip", type=str, default="192.168.1.223",
                       help="xArm robot IP address")
    parser.add_argument("--control_hz", type=int, default=10,
                       help="Control frequency (Hz)")

    # Cameras
    parser.add_argument("--agent_cam_id", type=int, default=0,
                       help="Agent camera ID")
    parser.add_argument("--wrist_cam_id", type=int, default=2,
                       help="Wrist camera ID")
    parser.add_argument("--use_usb_cameras", action="store_true",
                       help="Use USB cameras instead of RealSense")

    # Task
    parser.add_argument("--instruction", type=str,
                       default="pick up the red block and place it on the plate",
                       help="Language instruction")
    parser.add_argument("--max_steps", type=int, default=200,
                       help="Maximum steps per rollout")

    # Safety & execution
    parser.add_argument("--action_scale", type=float, default=1.0,
                       help="Scale actions for safety")
    parser.add_argument("--max_action_norm", type=float, default=0.15,
                       help="Clip action magnitude (m/rad)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Predict actions but don't execute on robot")

    # Warm-up override (band-aid for noop attractor; see XArmInferenceConfig docs)
    parser.add_argument("--warmup_steps", type=int, default=0,
                       help="Override model with fixed descent for this many initial steps. 0 disables.")
    parser.add_argument("--warmup_dz", type=float, default=-0.005,
                       help="Per-step downward delta in meters during warmup (default -5mm).")

    # Video
    parser.add_argument("--save_video", action="store_true", default=True,
                       help="Save rollout video")
    parser.add_argument("--video_path", type=Path, default=None,
                       help="Output video path")

    args = parser.parse_args()

    # Build config
    config = XArmInferenceConfig(
        checkpoint=args.checkpoint,
        base_vlm=args.base_vlm,
        xarm_ip=args.xarm_ip,
        control_hz=args.control_hz,
        agent_cam_id=args.agent_cam_id,
        wrist_cam_id=args.wrist_cam_id,
        use_realsense=not args.use_usb_cameras,
        instruction=args.instruction,
        warmup_steps=args.warmup_steps,
        warmup_dz=args.warmup_dz,
        max_steps=args.max_steps,
        unnorm_key=args.unnorm_key,
        action_scale=args.action_scale,
        max_action_norm=args.max_action_norm,
        dry_run=args.dry_run,
        save_video=args.save_video,
        video_path=args.video_path
    )

    # Run inference
    run_inference(config)


if __name__ == "__main__":
    main()
