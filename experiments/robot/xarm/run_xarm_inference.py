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

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

# VLA model imports
from prismatic.models import load_vla
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer

# Robot control imports
try:
    from xarm.wrapper import XArmAPI
except ImportError:
    raise SystemExit("xArm SDK not found. Install: pip install xarm-python-sdk")


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
    image_size: int = 224                        # Model input image size (224px for MiniVLA)
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

    def __post_init__(self):
        if self.home_joints is None:
            # Default home position for xArm7 (adjust as needed)
            self.home_joints = [0, 0, 0, 70, 0, 70, 0]  # degrees
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
        """Initialize RealSense cameras."""
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise SystemExit("pyrealsense2 not found. Install: pip install pyrealsense2")

        self.agent_pipeline = rs.pipeline()
        self.wrist_pipeline = rs.pipeline()

        config_agent = rs.config()
        config_agent.enable_device(str(self.agent_cam_id))
        config_agent.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        config_wrist = rs.config()
        config_wrist.enable_device(str(self.wrist_cam_id))
        config_wrist.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.agent_pipeline.start(config_agent)
        self.wrist_pipeline.start(config_wrist)

        # Warm up cameras
        for _ in range(30):
            self.agent_pipeline.wait_for_frames()
            self.wrist_pipeline.wait_for_frames()

        logger.info(f"RealSense cameras initialized: agent={self.agent_cam_id}, wrist={self.wrist_cam_id}")

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

        # Convert BGR to RGB and resize
        agent_rgb = cv2.cvtColor(agent_img, cv2.COLOR_BGR2RGB)
        wrist_rgb = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)

        agent_resized = cv2.resize(agent_rgb, (self.image_size, self.image_size))
        wrist_resized = cv2.resize(wrist_rgb, (self.image_size, self.image_size))

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
        Execute VLA action on robot.

        Args:
            action: (7,) array [dx, dy, dz, droll, dpitch, dyaw, gripper_delta]
                    dx, dy, dz in meters
                    droll, dpitch, dyaw in radians
                    gripper_delta in [-1, 1]
            action_scale: Scaling factor for actions (safety)
        """
        # Scale action
        action = action * action_scale

        # Extract components
        delta_pos = action[:3]  # meters
        delta_rot = action[3:6]  # radians
        gripper_cmd = action[6]  # -1=open, +1=close

        # Compute target pose
        target_pos = self.current_pose[:3] + delta_pos
        target_rot_euler = self.current_pose[3:] + delta_rot

        # Convert to xArm format (mm, degrees)
        target_xarm = np.concatenate([
            target_pos * 1000.0,  # meters to mm
            np.rad2deg(target_rot_euler)  # rad to deg
        ])

        # Send servo command
        code = self.arm.set_servo_cartesian(target_xarm.tolist(), speed=100, mvacc=2000)
        if code != 0:
            logger.warning(f"Servo command failed: code {code}")

        # Execute gripper command
        target_gripper = np.clip(self.current_gripper + gripper_cmd * 0.1, 0.0, 1.0)
        gripper_pos = int((1.0 - target_gripper) * 850)  # 0-850 range
        self.arm.set_gripper_position(gripper_pos, wait=False, speed=5000)

        # Update state
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
    """Wrapper for VLA model inference."""

    def __init__(self, checkpoint_path: Path, base_vlm: str, unnorm_key: str,
                 device: str = "cuda:0", torch_dtype: torch.dtype = torch.bfloat16):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.unnorm_key = unnorm_key

        logger.info(f"Loading VLA model from {checkpoint_path}...")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract norm stats
        if "norm_stats" in checkpoint:
            self.norm_stats = checkpoint["norm_stats"]
        else:
            raise ValueError("Checkpoint missing norm_stats. Ensure checkpoint is from training.")

        # Load VLA model
        self.vla = load_vla(
            base_vlm=base_vlm,
            checkpoint=checkpoint_path,
            device=self.device
        )
        self.vla.eval()

        # Get processor (for image + text preprocessing)
        self.processor = self.vla.get_processor()

        logger.info(f"VLA model loaded on {device}")

    def predict_action(self, agent_image: np.ndarray, wrist_image: np.ndarray,
                      instruction: str) -> np.ndarray:
        """
        Predict action from observations.

        Args:
            agent_image: (224, 224, 3) RGB uint8
            wrist_image: (224, 224, 3) RGB uint8
            instruction: Language instruction string

        Returns:
            action: (7,) float32 array [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        # Prepare prompt
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

        # Stack images (agent, wrist order matches training)
        # For multi-image: concatenate along batch dimension, then model processes
        images = [Image.fromarray(agent_image), Image.fromarray(wrist_image)]

        # Process inputs
        inputs = self.processor(prompt, images)
        inputs = {k: v.to(self.device, dtype=self.torch_dtype) if torch.is_tensor(v) else v
                  for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            action = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)

        return action  # Already denormalized, (7,) numpy array


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
        base_vlm=config.base_vlm,
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

            # Logging
            if step % 10 == 0:
                logger.info(
                    f"Step {step}/{config.max_steps}: "
                    f"action={action[:3].round(3).tolist()} (xyz), "
                    f"gripper={action[6]:.2f}"
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
