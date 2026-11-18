# tm5-manual-book-grasp

以 **手動四角框選** 指定目標書本、以 **ArUco marker** 決定插入方向，讓 Techman **TM5-700** 搭配腕上 Intel RealSense **D435i** 與自製平行夾爪抓取書架上的書。

本 repo 是 `book_grasp_ws` 最新路徑規劃版本（`main_ros_simple_row_consensus.py`，2026-07-23）的獨立精簡版：**移除語言／書名查詢與書本辨識**（perception worker、adapter、Conda 環境），只把「目標 mask 的來源」改成操作員手動框選。幾何計算、MoveIt 規劃、夾爪指令與所有安全閘門的程式碼與原版相同。

> [!WARNING]
> **尚未完成 TM5 實機驗收。** 任何可能移動手臂的操作都必須有人在現場、急停可及、TMflow 處於 remote/Listen、speed override 5–10%，並先在 RViz 確認當次完整軌跡。異常時立即使用 TMflow Stop 或實體急停，不要依賴 Python 例外或逾時。

## 流程

1. 擷取一張同步的 RGB／aligned depth／CameraInfo／capture-time TF 畫面。
2. 視窗顯示該畫面（若偵測到 ArUco ID 0 會畫出外框）；**操作員點選目標書的四個角** → Enter。
3. 以框內深度 RANSAC 擬合書面平面，把四角投影到該平面並存成 robot `base` 座標。
4. 目標書右側書縫決定插入位置（往右側縫隙偏 1.5 mm、沿書脊上移 15 mm），ArUco（`DICT_4X4_50`、ID 0、50 mm）marker `-Z` 決定插入方向 tool `+Z`，書脊決定 tool `+Y`。marker 只提供方向，**不提供位置**。
5. ArUco 單張觀測失敗時，最多用三張更新的畫面重試；重試畫面的 mask 由 base 座標框經 TF 重投影產生，不需再框。
6. 碰撞感知 IK 多 seed 搜尋 → 最多 4 個不同關節解送 Pilz PTP → 以 J1–J6 累計行程最短者移動到 70 mm stand-off（只有這一步要求按 Enter）。
7. 等待 1.5 s 後擷取新畫面，**把同一個 base 座標框重投影**進新影像做二次觀測（見下方限制），只允許鎖定 tool `+X` 的有界修正。
8. 張開夾爪 → 單一 LIN 沿鎖定 tool `+Z` 插入（stand-off + 102 mm）→ 送 `H <厚度>` 再送 `T`。
9. 鎖定 quaternion 的單一 LIN 向外 232 mm 並沿書脊上抬 40 mm。
10. Pilz PTP 回到流程開始前記錄的六軸關節。
11. H/T 仍保持；確認書本已被支撐後**再按一次 Enter** 才送 `Y`、`O` 釋放。Ctrl-C 會保持夾持。

### 已知限制：stand-off 二次修正量預期接近 0

手動框固定在 base 座標，重投影後新影像中的書右緣仍由同一個框決定，TF／定位誤差也會在同一個轉換中抵銷。因此第 7 步的 tool `+X` 修正量預期接近 0，實際作用是「新鮮 depth／平面／厚度仍有效」的驗證閘門，而不是新的書縫偵測。**插入精度完全取決於初始框選的右緣準確度與 hand-eye 標定。** 框選時請把右側兩個角點精準點在目標書右緣上。

## 系統需求

- Ubuntu 22.04、ROS 2 Humble（`ros-humble-desktop`）
- [uv](https://docs.astral.sh/uv/)：`curl -LsSf https://astral.sh/uv/install.sh | sh`
- 有桌面環境（框選視窗需要 `DISPLAY`）
- 硬體：TM5-700（預設 `192.168.1.33`）、D435i、夾爪 Arduino（預設 `/dev/ttyUSB0`，使用者需在 `dialout` 群組）

## 安裝

完整的逐步安裝與操作指令（含驗證與故障排查、每段可一鍵複製）在 [`Manual.html`](Manual.html)：clone 後以瀏覽器開啟 `xdg-open Manual.html`。

```bash
git clone --recursive https://github.com/Patrick-zhuyanxun/tm5-manual-book-grasp.git
cd tm5-manual-book-grasp
./scripts/setup.sh
```

`setup.sh` 會：

1. 拉取 `src/tmr_ros2` submodule（[fork](https://github.com/Patrick-zhuyanxun/tmr_ros2/tree/tm5-manual-book-grasp)，見下方說明）。
2. 以 apt 安裝 MoveIt、Pilz、RealSense 等 ROS 套件並執行 `rosdep install`（需要 sudo）。
3. 用 uv 在 **ROS 的系統 Python 3.10** 上建立 `.venv`（`--system-site-packages`，才能使用 `rclpy`／`cv_bridge`），並依 `uv.lock` 安裝 `numpy<2`、`opencv-contrib-python`、`scipy`、`pyserial`。
4. `colcon build --symlink-install --packages-up-to tm5_manual_book_grasp`。

若電腦已自行從原始碼安裝 RealSense／MoveIt 等相依套件，可用 `./scripts/setup.sh --no-apt` 跳過 apt 與 rosdep。

之後**每個 terminal** 都先執行：

```bash
source scripts/env.sh   # ROS → 本 workspace → uv .venv（順序固定）
```

> 不要另外 activate Conda；`python3` 必須是 `.venv/bin/python3`。

## 操作

每次操作前確認：銘牌為 TM5-700、工作區淨空、急停可及、TMflow 無錯誤；TCP、payload、hand-eye、夾爪零點與碰撞模型已驗證；ArUco ID 0 已貼在書架上且在相機視野內。

**Terminal 1 — TM5 driver、MoveIt、RViz**（連接真實手臂）

```bash
source scripts/env.sh
ros2 launch tm5_manual_book_grasp tm5-700_with_gripper.launch.py robot_ip:=192.168.1.33
```

不給 `robot_ip` 時為離線模式：只啟動 MoveIt 與 RViz、以固定關節值代替手臂，不會執行軌跡，可用來確認規劃。

**Terminal 2 — D435i**

```bash
source scripts/env.sh
ros2 launch realsense2_camera rs_launch.py camera_name:=camera camera_namespace:=camera \
  depth_module.depth_profile:=1280x720x30 rgb_camera.color_profile:=1280x720x30 \
  pointcloud.enable:=true align_depth.enable:=true enable_sync:=true clip_distance:=0.8
```

**Terminal 3 — 夾爪**

```bash
source scripts/env.sh
ros2 run tm5_manual_book_grasp gripper_node.py --ros-args -p serial_port:=/dev/ttyUSB0 -p baud_rate:=115200
```

Arduino 指令只代表「已送出角度命令」，不是力或位置回授。

**Terminal 4 — 抓取 coordinator**

先以預設 `motion_enabled:=false` 乾跑：只框選、計算、在 RViz 顯示 approach PTP，不會移動。

```bash
source scripts/env.sh
ros2 run tm5_manual_book_grasp manual_book_grasp.py --ros-args \
  --params-file install/tm5_manual_book_grasp/share/tm5_manual_book_grasp/config/manual_grasp.yaml
```

確認 RViz 軌跡後才加上 `-p motion_enabled:=true` 執行完整序列。

**框選視窗操作：** 左鍵依序點 4 個角（順序不限）、右鍵或 Backspace 復原、Enter／Space 確認、C／Esc 取消。只框**目標書**，書的左右兩側都必須完整在畫面內（框碰到左右邊界會被拒絕）。

**偵錯輸出：** `/tmp/book_grasp_initial.png`、`/tmp/book_grasp_fine.png`，以及 RViz 中的 `/manual_book_grasp/debug_overlay` 影像（框選範圍、mask、ArUco、插入點）。

## 主要參數（`config/manual_grasp.yaml`）

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `motion_enabled` | `false` | `true` 才會執行整個抓取序列 |
| `manual_selection_timeout_s` | 300 | 框選逾時；`tf_buffer_cache_s` 需至少再大 30 s |
| `outline_border_margin_px` | 2 | 框的左右範圍距影像邊界的最小像素 |
| `approach_standoff_m` | 0.070 | 書面前的 stand-off 距離 |
| `insert_depth_m` | 0.102 | 從書面再插入的深度 |
| `extract_extra_m` / `final_lift_m` | 0.060 / 0.040 | 抽出時額外外移 / 沿書脊上抬 |
| `target_spine_up_offset_m` | 0.015 | 插入點相對書縫中點上移量 |
| `aruco_*` | DICT_4X4_50 / ID 0 / 0.050 m | 程式會拒絕其他字典、ID 或尺寸 |
| `planning_retry_count` | 3 | 同一端點額外嘗試的不同 IK 解數 |

速度、加速度、J5/J6 防護、夾爪間隙等其餘參數與預設值見 `scripts/manual_book_grasp.py` 的 `declare_parameter`。

## 專案結構

```
src/tm5_manual_book_grasp/
  scripts/manual_book_grasp.py   coordinator（手動框選 + ArUco + MoveIt 序列）
  scripts/manual_target.py       四角框選 UI、平面錨定、重投影（純函式）
  scripts/aruco_geometry.py      ArUco 插入軸估計（與原 workspace 相同）
  scripts/gripper_node.py        Arduino 夾爪 serial bridge
  scripts/tm_joint_state_adapter.py、offline_joint_state_publisher.py、tm_moveit_graph_preflight.py
  launch/ urdf/ gripper_urdf/ config/   TM5-700 + 夾爪 + D435i 模型、SRDF、關節限制、RViz
src/tmr_ros2/                    TM driver submodule（fork）
tests/                           pytest（不需硬體）
```

## TM driver fork

`src/tmr_ros2` 指向 [Patrick-zhuyanxun/tmr_ros2@tm5-manual-book-grasp](https://github.com/Patrick-zhuyanxun/tmr_ros2/tree/tm5-manual-book-grasp)，以官方 `ac4c6c2`（1.3.2）為基礎，只加入原 workspace 的一個 3 行修改：`TmRos2SctMoveit::has_points()` 允許沒有 velocities 的軌跡點。注意該修改的註解說 `set_pvt_traj` 會補算速度，但實際上 `set_pvt_traj` 只是原樣複製（空的）velocities；Pilz 規劃的軌跡一般都帶有速度，因此本流程預期不會走到這條路徑，但未經實機驗證。

## 測試

```bash
source scripts/env.sh
python -m pytest
```

涵蓋 ArUco 方向選擇、框選正規化、平面錨定與跨相機姿態重投影、邊界裁切拒絕、無語言／辨識介面、YAML 與節點參數一致、安裝檔案完整性等。測試不需要相機、手臂或 MoveIt。

## 授權

Apache-2.0，見 [LICENSE](LICENSE)。`src/tmr_ros2` 依其上游授權。
