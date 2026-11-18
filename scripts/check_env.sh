#!/usr/bin/env bash
# Read-only environment check. Never installs, removes or modifies anything.
#
#   ./scripts/check_env.sh                       # print the report
#   ./scripts/check_env.sh --missing-apt FILE    # also write missing apt packages to FILE
#
# Exit codes: 0 = ready to build, 1 = apt packages missing, 2 = blocker that
# setup.sh cannot fix (no ROS 2 Humble, no Python 3.10, no uv, ...).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MISSING_OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --missing-apt) MISSING_OUT="${2:?--missing-apt needs a file}"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

ROS_PREFIX=/opt/ros/humble
# rosdep keys declared by tm5-700_moveit_config that are not used here and
# cannot be installed on Humble (ROS 1 Gazebo plugin, no apt package).
SKIP_KEYS="gazebo_ros_control warehouse_ros_mongo"
TM_PATHS=(
  "$ROOT/src/tmr_ros2/tm_msgs"
  "$ROOT/src/tmr_ros2/tm_driver"
  "$ROOT/src/tmr_ros2/tm_description"
  "$ROOT/src/tmr_ros2/tm_moveit/tm5-700_moveit_config"
)

BLOCKERS=0
MISSING_APT=()
ok()      { printf '  [OK]      %s\n' "$*"; }
info()    { printf '  [INFO]    %s\n' "$*"; }
warn()    { printf '  [WARN]    %s\n' "$*"; }
missing() { printf '  [MISSING] %s\n' "$*"; }
blocker() { printf '  [BLOCKER] %s\n' "$*"; BLOCKERS=$((BLOCKERS + 1)); }
add_apt() {
  local pkg
  for pkg in "$@"; do
    [[ " ${MISSING_APT[*]} " == *" $pkg "* ]] || MISSING_APT+=("$pkg")
  done
}

# Prefixes searched for ROS packages: everything already sourced in this shell
# (underlay workspaces count as installed) plus the system ROS install. This
# repo's own install/ is excluded so the report reflects the machine.
PREFIXES=()
IFS=: read -r -a _current <<< "${AMENT_PREFIX_PATH:-}"
for p in "${_current[@]}"; do
  [[ -z "$p" || "$p" == "$ROOT/install"* ]] && continue
  PREFIXES+=("$p")
done
[[ " ${PREFIXES[*]} " == *" $ROS_PREFIX "* ]] || PREFIXES+=("$ROS_PREFIX")

find_pkg() {
  local p
  for p in "${PREFIXES[@]}"; do
    if [[ -f "$p/share/$1/package.xml" ]]; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

echo "== 系統"
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  if [[ "${VERSION_ID:-}" == "22.04" ]]; then ok "${PRETTY_NAME:-Ubuntu 22.04}"; else warn "${PRETTY_NAME:-unknown OS}（只在 Ubuntu 22.04 驗證過）"; fi
fi
if [[ -f "$ROS_PREFIX/setup.bash" ]]; then
  ok "ROS 2 Humble：$ROS_PREFIX"
else
  blocker "找不到 ROS 2 Humble（$ROS_PREFIX）。請先依官方說明安裝 ros-humble-desktop。"
fi
if [[ -n "${ROS_DISTRO:-}" && "${ROS_DISTRO}" != "humble" ]]; then
  blocker "目前 shell 已 source ROS ${ROS_DISTRO}；請開新的 terminal，不要 source 其他版本的 ROS。"
fi
if [[ -x /usr/bin/python3.10 ]]; then ok "系統 Python：/usr/bin/python3.10"; else blocker "找不到 /usr/bin/python3.10（ROS Humble 的 Python）。"; fi

echo "== 工具"
if command -v git >/dev/null 2>&1; then ok "git"; else missing "git"; add_apt git; fi
if command -v uv >/dev/null 2>&1; then
  ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"
else
  blocker "找不到 uv。安裝：curl -LsSf https://astral.sh/uv/install.sh | sh，然後 export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
if command -v colcon >/dev/null 2>&1; then ok "colcon"; else missing "colcon"; add_apt python3-colcon-common-extensions; fi
ROSDEP_READY=0
if command -v rosdep >/dev/null 2>&1; then
  ok "rosdep $(rosdep --version 2>/dev/null)"
  if ! compgen -G "/etc/ros/rosdep/sources.list.d/*.list" >/dev/null; then
    missing "rosdep 尚未初始化（需要 sudo rosdep init；setup.sh 會詢問）"
  elif [[ ! -d "$HOME/.ros/rosdep/sources.cache" ]]; then
    missing "rosdep 資料庫尚未下載（需要 rosdep update；setup.sh 會執行）"
  else
    ROSDEP_READY=1
  fi
else
  missing "rosdep"; add_apt python3-rosdep
fi

echo "== 目前 shell 環境"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  warn "Conda 環境啟用中（$CONDA_PREFIX）。請先 conda deactivate，否則 Python 套件會互相覆蓋。"
fi
if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" != "$ROOT/.venv" ]]; then
  warn "其他 Python venv 啟用中（$VIRTUAL_ENV）。請先 deactivate。"
fi
WORKSPACES=()
for p in "${PREFIXES[@]}"; do
  [[ "$p" == "$ROS_PREFIX" ]] && continue
  # colcon isolated installs add one prefix per package; report the workspace.
  [[ -f "$(dirname "$p")/local_setup.bash" ]] && p="$(dirname "$p")"
  [[ " ${WORKSPACES[*]} " == *" $p "* ]] || WORKSPACES+=("$p")
done
for ws in "${WORKSPACES[@]}"; do
  info "已 source 的其他 workspace（裡面的套件視為已安裝）：$ws"
done
[[ "${#WORKSPACES[@]}" == 0 ]] && ok "只有系統 ROS，沒有其他 workspace"

echo "== TM driver"
if [[ -f "$ROOT/src/tmr_ros2/tm_driver/package.xml" ]]; then
  status="$(git -C "$ROOT" submodule status src/tmr_ros2 2>/dev/null)"
  case "${status:0:1}" in
    " ") ok "submodule src/tmr_ros2 @ ${status:1:7}（含 patch 的 fork）" ;;
    "+") warn "submodule src/tmr_ros2 不是 repo 指定的版本；執行 git submodule update --init --recursive" ;;
    *)   ok "submodule src/tmr_ros2" ;;
  esac
else
  missing "submodule src/tmr_ros2 尚未下載（setup.sh 會執行 git submodule update --init --recursive）"
fi
for pkg in tm_msgs tm_driver tm_description tm5-700_moveit_config; do
  if where="$(find_pkg "$pkg")"; then
    info "本機已有 $pkg（$where）。本 repo 會另外建置自己的 fork 並覆蓋（overlay）它，不會修改原本的安裝。"
  fi
done

echo "== ROS 套件"
KEY_PACKAGES=(
  moveit_ros_move_group moveit_planners_ompl pilz_industrial_motion_planner
  moveit_simple_controller_manager moveit_ros_visualization moveit_kinematics moveit_configs_utils
  realsense2_camera realsense2_description
  cv_bridge xacro tf2_ros robot_state_publisher joint_state_publisher rviz2
)
for pkg in "${KEY_PACKAGES[@]}"; do
  if where="$(find_pkg "$pkg")"; then
    if [[ "$where" == "$ROS_PREFIX" ]]; then ok "$pkg"; else ok "$pkg（來自 $where）"; fi
  else
    apt_name="ros-humble-${pkg//_/-}"
    missing "$pkg → $apt_name"
    add_apt "$apt_name"
  fi
done

echo "== rosdep 完整相依檢查"
if [[ "$ROSDEP_READY" == 1 && -f "$ROOT/src/tmr_ros2/tm_driver/package.xml" ]]; then
  search_path="$(IFS=:; echo "${PREFIXES[*]}")"
  output="$(AMENT_PREFIX_PATH="$search_path" ROS_DISTRO=humble ROS_PYTHON_VERSION=3 \
    rosdep check --from-paths "$ROOT/src/tm5_manual_book_grasp" "${TM_PATHS[@]}" \
    --ignore-src --rosdistro humble --skip-keys "$SKIP_KEYS" \
    -t build -t buildtool -t build_export -t exec 2>&1)"
  if grep -q "All system dependencies have been satisfied" <<< "$output"; then
    ok "package.xml 宣告的所有相依都已具備"
  else
    found_unknown=0
    while IFS=$'\t' read -r installer pkg; do
      if [[ "$installer" == "apt" && -n "$pkg" ]]; then
        missing "$pkg"
        add_apt "$pkg"
      fi
    done < <(grep -P '^\w+\t' <<< "$output")
    if grep -q -i "error" <<< "$output"; then
      found_unknown=1
    fi
    if grep -P '^\w+\t' <<< "$output" | grep -v -q -P '^apt\t'; then
      found_unknown=1
    fi
    if [[ "$found_unknown" == 1 ]]; then
      warn "rosdep 回報了無法自動處理的項目："
      sed 's/^/            /' <<< "$output"
    fi
  fi
else
  info "略過（rosdep 未就緒或 submodule 未下載）；setup.sh 準備好後會再檢查一次"
fi

echo "== 硬體權限與桌面（僅提示，不影響安裝）"
if id -nG | tr ' ' '\n' | grep -qx dialout; then ok "使用者在 dialout 群組（夾爪 serial）"; else warn "使用者不在 dialout 群組：sudo usermod -aG dialout \$USER，然後重新登入"; fi
if compgen -G "/etc/udev/rules.d/*realsense*" >/dev/null || compgen -G "/lib/udev/rules.d/*realsense*" >/dev/null; then
  ok "RealSense udev rules"
else
  warn "找不到 RealSense udev rules；若 D435i 無法開啟，請依 librealsense 文件安裝 udev rules"
fi
if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then ok "有桌面顯示（框選視窗可用）"; else warn "沒有 DISPLAY；安裝可以進行，但操作時需在本機桌面 terminal 執行"; fi

echo "== uv Python 環境"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  if versions="$("$ROOT/.venv/bin/python" -c 'import numpy, cv2; print("numpy", numpy.__version__, "/ opencv", cv2.__version__)' 2>/dev/null)"; then
    ok ".venv 已存在：$versions"
  else
    info ".venv 已存在但套件不完整；setup.sh 會用 uv sync 補齊"
  fi
else
  info ".venv 尚未建立；setup.sh 會建立"
fi

echo
if [[ -n "$MISSING_OUT" ]]; then
  printf '%s\n' "${MISSING_APT[@]}" | sed '/^$/d' > "$MISSING_OUT"
fi
if [[ "$BLOCKERS" -gt 0 ]]; then
  echo "結果：有 $BLOCKERS 個 BLOCKER，請先處理後再執行 setup.sh。"
  exit 2
fi
if [[ "${#MISSING_APT[@]}" -gt 0 ]]; then
  echo "結果：缺少 ${#MISSING_APT[@]} 個 apt 套件：${MISSING_APT[*]}"
  exit 1
fi
echo "結果：相依套件都已具備。"
exit 0
