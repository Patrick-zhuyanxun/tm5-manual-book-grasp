#!/usr/bin/env bash
# One-time setup on Ubuntu 22.04 + ROS 2 Humble.
#
#   ./scripts/setup.sh            # check, install only missing apt packages (asks first), uv venv, colcon build
#   ./scripts/setup.sh --check    # only run the read-only check (same as scripts/check_env.sh)
#   ./scripts/setup.sh --no-apt   # never call apt/rosdep; stop if something is missing
#   ./scripts/setup.sh --yes      # do not ask before apt changes (still never upgrades installed packages)
#
# Dependencies provided by another sourced workspace (e.g. a source-built
# RealSense or TM driver) count as installed; source that workspace before
# running this script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE=install
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --check) MODE=check ;;
    --no-apt) MODE=no-apt ;;
    --yes) ASSUME_YES=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

confirm() {
  [[ "$ASSUME_YES" == 1 ]] && return 0
  local reply
  read -r -p "$1 [y/N] " reply
  [[ "$reply" == [yY] || "$reply" == [yY][eE][sS] ]]
}

MISSING_FILE="$(mktemp)"
trap 'rm -f "$MISSING_FILE"' EXIT

run_check() {
  local code=0
  "$ROOT/scripts/check_env.sh" --missing-apt "$MISSING_FILE" || code=$?
  return "$code"
}

if [[ "$MODE" == check ]]; then
  run_check
  exit $?
fi

# Fetching the pinned TM driver is part of this repo, not a system change.
if [[ ! -f src/tmr_ros2/tm_driver/package.xml ]]; then
  echo "==> Fetching the TM driver submodule"
  git submodule update --init --recursive
fi

echo "==> Checking the environment (read-only)"
CHECK=0
run_check || CHECK=$?
if [[ "$CHECK" == 2 ]]; then
  exit 2
fi

if [[ "$MODE" == no-apt ]]; then
  if [[ "$CHECK" != 0 ]]; then
    echo "--no-apt: required packages are missing (see the report above). Install them, or source the workspace that provides them, then retry." >&2
    exit 1
  fi
else
  # rosdep must be ready before it can list every package.xml dependency.
  if ! command -v rosdep >/dev/null 2>&1 || ! command -v colcon >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
    tools=()
    command -v rosdep >/dev/null 2>&1 || tools+=(python3-rosdep)
    command -v colcon >/dev/null 2>&1 || tools+=(python3-colcon-common-extensions)
    command -v git >/dev/null 2>&1 || tools+=(git)
    echo
    echo "Missing build tools: ${tools[*]}"
    confirm "Install them with sudo apt-get (installed packages are not upgraded)?" || { echo "Aborted; nothing was installed."; exit 1; }
    sudo apt-get update
    sudo apt-get install -y --no-upgrade "${tools[@]}"
  fi
  if ! compgen -G "/etc/ros/rosdep/sources.list.d/*.list" >/dev/null; then
    confirm "rosdep is not initialized. Run 'sudo rosdep init'?" || { echo "Aborted."; exit 1; }
    sudo rosdep init
  fi
  if [[ ! -d "$HOME/.ros/rosdep/sources.cache" ]]; then
    echo "==> rosdep update (downloads the rosdep database into ~/.ros)"
    rosdep update --rosdistro humble
    echo "==> Re-checking with rosdep"
    CHECK=0
    run_check || CHECK=$?
    [[ "$CHECK" == 2 ]] && exit 2
  fi

  mapfile -t PACKAGES < "$MISSING_FILE"
  if [[ "${#PACKAGES[@]}" -gt 0 ]]; then
    echo
    echo "The following apt packages are missing:"
    printf '  %s\n' "${PACKAGES[@]}"
    confirm "Refresh apt lists (sudo apt-get update) and preview the installation?" || { echo "Aborted; nothing was installed."; exit 1; }
    sudo apt-get update
    # --no-upgrade keeps every already-installed package at its version unless
    # a new package strictly requires a newer one; the preview shows that case.
    preview="$(apt-get -s install --no-upgrade "${PACKAGES[@]}")"
    new_count="$(grep -c '^Inst [^ ]* (' <<< "$preview" || true)"
    upgrades="$(grep '^Inst [^ ]* \[' <<< "$preview" | awk '{print "  " $2 " " $3 " -> " $4}' | tr -d '[](' || true)"
    removals="$(grep '^Remv ' <<< "$preview" | awk '{print "  " $2}' || true)"
    echo
    echo "Preview: $new_count new package(s)."
    if [[ -n "$upgrades" ]]; then
      echo "WARNING: these already-installed packages would be UPGRADED as dependencies:"
      echo "$upgrades"
    fi
    if [[ -n "$removals" ]]; then
      echo "WARNING: these packages would be REMOVED:"
      echo "$removals"
    fi
    confirm "Proceed with this installation?" || { echo "Aborted; nothing was installed."; exit 1; }
    sudo apt-get install -y --no-upgrade "${PACKAGES[@]}"
    echo "==> Re-checking after installation"
    CHECK=0
    run_check || CHECK=$?
    if [[ "$CHECK" != 0 ]]; then
      echo "Dependencies are still missing after installation; see the report above." >&2
      exit 1
    fi
  fi
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
# ROS setup scripts reference unset variables; relax -u while sourcing.
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u
colcon build --symlink-install --packages-up-to tm5_manual_book_grasp

echo
echo "Setup complete. In every new terminal run:"
echo "  source $ROOT/scripts/env.sh"
