# Source in every terminal:  source scripts/env.sh
# Order matters: ROS first, then this workspace, then the uv venv so that
# `python3` (and every `ros2 run` Python node) resolves to .venv.

_TM5_GRASP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$_TM5_GRASP_ROOT/install/setup.bash" ]]; then
  echo "Workspace is not built. Run: $_TM5_GRASP_ROOT/scripts/setup.sh" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -f "$_TM5_GRASP_ROOT/.venv/bin/activate" ]]; then
  echo "uv environment is missing. Run: $_TM5_GRASP_ROOT/scripts/setup.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$_TM5_GRASP_ROOT/install/setup.bash"
# shellcheck disable=SC1091
source "$_TM5_GRASP_ROOT/.venv/bin/activate"
unset _TM5_GRASP_ROOT
