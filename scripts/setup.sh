#!/usr/bin/env bash
# One-time setup on Ubuntu 22.04 + ROS 2 Humble.
#
#   ./scripts/setup.sh            # apt + rosdep + uv venv + colcon build
#   ./scripts/setup.sh --no-apt   # skip sudo apt/rosdep (dependencies already installed)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INSTALL_APT=1
for arg in "$@"; do
  case "$arg" in
    --no-apt) INSTALL_APT=0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS 2 Humble is not installed at /opt/ros/humble." >&2
  echo "Install ros-humble-desktop first: https://docs.ros.org/en/humble/Installation.html" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "==> Fetching the TM driver submodule"
git submodule update --init --recursive

# ROS setup scripts reference unset variables; relax -u while sourcing.
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u

# Only the TM packages this workspace builds; the rest of tmr_ros2 (demos,
# GUI tools, other robot models) is skipped by colcon --packages-up-to.
TM_PATHS=(
  src/tmr_ros2/tm_msgs
  src/tmr_ros2/tm_driver
  src/tmr_ros2/tm_description
  src/tmr_ros2/tm_moveit/tm5-700_moveit_config
)

if [[ "$INSTALL_APT" == 1 ]]; then
  echo "==> Installing apt dependencies (sudo)"
  sudo apt-get update
  sudo apt-get install -y \
    python3-colcon-common-extensions python3-rosdep python3-vcstool \
    ros-humble-moveit \
    ros-humble-moveit-planners-ompl \
    ros-humble-pilz-industrial-motion-planner \
    ros-humble-moveit-simple-controller-manager \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-description \
    ros-humble-cv-bridge \
    ros-humble-xacro \
    ros-humble-tf2-ros
  if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    sudo rosdep init
  fi
  rosdep update
  echo "==> Resolving remaining ROS dependencies with rosdep"
  rosdep install --from-paths src/tm5_manual_book_grasp "${TM_PATHS[@]}" \
    --ignore-src --rosdistro humble -y
fi

echo "==> Creating the uv environment on the ROS system Python"
# rclpy/cv_bridge are compiled for /usr/bin/python3.10, so the venv must use
# that interpreter and see the system/ROS site-packages.
if [[ ! -f .venv/pyvenv.cfg ]] || ! grep -q "include-system-site-packages = true" .venv/pyvenv.cfg; then
  rm -rf .venv
  uv venv --python /usr/bin/python3.10 --system-site-packages .venv
fi
uv sync --frozen

echo "==> Building the colcon workspace"
colcon build --symlink-install --packages-up-to tm5_manual_book_grasp

echo
echo "Setup complete. In every new terminal run:"
echo "  source $ROOT/scripts/env.sh"
