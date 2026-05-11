"""Home the xArm to its default position with gripper open.

Run before / between recording sessions to reset the robot.

    python home.py
"""

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ril-env"))

from ril_env.xarm_controller import XArm, XArmConfig  # noqa: E402

XARM_IP = "192.168.1.223"


def main() -> None:
    with XArm(XArmConfig(robot_ip=XARM_IP)) as arm:
        try:
            print("Homing robot...")
            arm.home()
            print("Robot homed successfully.")
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
