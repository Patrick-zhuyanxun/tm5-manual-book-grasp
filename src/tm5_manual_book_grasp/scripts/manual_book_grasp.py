#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manual-outline ROS2 book-grasping coordinator with an ArUco insertion reference.

This is the language/recognition-free variant of the book_grasp_ws
``main_ros_simple_row_consensus.py`` coordinator. Only the target-mask source
differs; geometry, planning, gripper, and safety logic are unchanged.

Execution path
--------------
1. Capture one synchronized RGB-D/TF frame. The operator outlines the target
   book with four corners. The outline is anchored on the fitted book plane and
   stored in the robot base frame.
2. Fit only the target book for its right seam, spine, grasp position, and
   thickness. Detect DICT_4X4_50 marker ID 0 at 50 mm in the same RGB frame.
   Project directed marker -Z perpendicular to the book spine for tool +Z;
   never use the marker centre as the insertion position. ArUco retries use
   newer frames with the base-frame outline re-projected through TF.
3. Search collision-aware IK, plan up to four distinct explicit-joint-goal
   Pilz PTP candidates, and select the legal trajectory with the least total
   J1--J6 travel before moving to stand-off.
4. Wait for camera settling, re-project the base-frame outline into one fresh
   frame, and apply one bounded locked-tool-+X PTP correction while keeping
   the initial ArUco-derived orientation locked. Because the outline is fixed
   in the base frame, this correction is expected to be near zero; the step is
   a fresh depth/plane/thickness validation gate. The insertion point is 15 mm
   above the seam midpoint.
5. Open, insert 102 mm into the gap, then immediately send H followed by T.
   There is no intermediate extraction, G command, downward compensation, or
   second insertion/grip.
6. Extract along one locked-quaternion LIN; planning may retry without changing
   its straight geometry or safety guards.
7. Return to the exact recorded six arm joints with Pilz PTP.
8. Require a separate release confirmation before sending final Y and O.

The marker supplies direction while the outlined right book seam supplies
position. Arm motion is disabled by default.
"""

from __future__ import annotations

import copy
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    Constraints,
    DisplayTrajectory,
    JointConstraint,
    OrientationConstraint,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA, Float32, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker

from aruco_geometry import (
    ArucoObservationError,
    compute_locked_side_correction,
    estimate_aruco_insertion_axis,
    select_aruco_ground_truth_axes,
)
from manual_target import (
    pixel_rays_to_plane,
    project_points_to_pixels,
    projected_quad_mask,
    quad_mask,
    select_manual_book_quad,
)


@dataclass
class CameraBundle:
    color_bgr: np.ndarray
    depth: np.ndarray
    color_info: CameraInfo
    color_msg: Image


@dataclass
class BookGeometry:
    target_pose_camera: PoseStamped
    center_camera: np.ndarray
    normal_camera: np.ndarray
    target_normal_camera: np.ndarray
    spine_camera: np.ndarray
    right_camera: np.ndarray
    width_m: float
    plane_inlier_ratio: float
    plane_residual_m: float
    mask_bbox: tuple[int, int, int, int]
    outline_px: Optional[np.ndarray] = None
    aruco_corners_px: Optional[np.ndarray] = None
    aruco_axis_camera: Optional[np.ndarray] = None
    aruco_reprojection_error_px: Optional[float] = None


class ManualBookGraspNode(Node):
    def __init__(self) -> None:
        super().__init__('manual_book_grasp')

        # Frames and MoveIt.
        self.group_name = str(self.declare_parameter('group_name', 'tmr_arm').value)
        self.base_frame = str(self.declare_parameter('base_frame', 'base').value)
        self.eef_link = str(self.declare_parameter('eef_link', 'gripper_tip').value)
        self.robot_model_id = str(
            self.declare_parameter('robot_model_id', 'tm5-700_with_gripper').value
        )
        self.pilz_pipeline_id = str(
            self.declare_parameter(
                'pilz_pipeline_id', 'pilz_industrial_motion_planner'
            ).value
        )
        self.ptp_planner_id = str(self.declare_parameter('ptp_planner_id', 'PTP').value)
        self.lin_planner_id = str(self.declare_parameter('lin_planner_id', 'LIN').value)
        self.planning_time_s = float(
            self.declare_parameter('planning_time_s', 5.0).value
        )
        self.ptp_velocity_scaling = float(
            self.declare_parameter('ptp_velocity_scaling', 0.10).value
        )
        self.ptp_acceleration_scaling = float(
            self.declare_parameter('ptp_acceleration_scaling', 0.05).value
        )
        self.lin_velocity_scaling = float(
            self.declare_parameter('lin_velocity_scaling', 0.03).value
        )
        self.lin_acceleration_scaling = float(
            self.declare_parameter('lin_acceleration_scaling', 0.02).value
        )
        self.ptp_position_tolerance_m = float(
            self.declare_parameter('ptp_position_tolerance_m', 0.002).value
        )
        self.ptp_orientation_tolerance_rad = float(
            self.declare_parameter('ptp_orientation_tolerance_rad', 0.02).value
        )
        self.lin_position_tolerance_m = float(
            self.declare_parameter('lin_position_tolerance_m', 0.0008).value
        )
        self.lin_orientation_tolerance_rad = float(
            self.declare_parameter('lin_orientation_tolerance_rad', 0.01).value
        )
        self.max_ptp_joint_5_delta_rad = math.radians(float(
            self.declare_parameter('max_ptp_joint_5_delta_deg', 60.0).value
        ))
        self.max_ptp_joint_6_delta_rad = math.radians(float(
            self.declare_parameter('max_ptp_joint_6_delta_deg', 90.0).value
        ))
        self.max_lin_joint_6_delta_rad = math.radians(float(
            self.declare_parameter('max_lin_joint_6_delta_deg', 30.0).value
        ))
        self.planning_retry_count = int(
            self.declare_parameter('planning_retry_count', 3).value
        )
        self.ptp_ik_max_seed_attempts = int(
            self.declare_parameter('ptp_ik_max_seed_attempts', 20).value
        )
        self.ptp_ik_seed_step_rad = math.radians(float(
            self.declare_parameter('ptp_ik_seed_step_deg', 45.0).value
        ))
        self.ptp_ik_timeout_s = float(
            self.declare_parameter('ptp_ik_timeout_s', 0.20).value
        )
        self.ptp_unique_solution_tolerance_rad = math.radians(float(
            self.declare_parameter('ptp_unique_solution_tolerance_deg', 0.05).value
        ))
        self.trajectory_start_tolerance_rad = float(
            self.declare_parameter('trajectory_start_tolerance_rad', 0.05).value
        )
        self.joint_state_max_age_s = float(
            self.declare_parameter('joint_state_max_age_s', 0.5).value
        )

        # Manual target outline and geometry. The capture-time TF must stay in
        # the buffer while the operator outlines the target.
        self.manual_selection_timeout_s = float(
            self.declare_parameter('manual_selection_timeout_s', 300.0).value
        )
        self.debug_output_dir = str(
            self.declare_parameter('debug_output_dir', '/tmp').value
        )
        self.fine_perception_settle_s = float(
            self.declare_parameter('fine_perception_settle_s', 1.5).value
        )
        self.tf_buffer_cache_s = float(
            self.declare_parameter('tf_buffer_cache_s', 360.0).value
        )
        self.camera_sync_slop_s = float(
            self.declare_parameter('camera_sync_slop_s', 0.05).value
        )
        self.depth_unit_scale_m = float(
            self.declare_parameter('depth_unit_scale_m', 0.001).value
        )
        self.minimum_mask_pixels = int(
            self.declare_parameter('minimum_mask_pixels', 500).value
        )
        self.outline_border_margin_px = int(
            self.declare_parameter('outline_border_margin_px', 2).value
        )
        self.minimum_geometry_points = int(
            self.declare_parameter('minimum_geometry_points', 500).value
        )
        self.ransac_iterations = int(
            self.declare_parameter('ransac_iterations', 300).value
        )
        self.ransac_distance_m = float(
            self.declare_parameter('ransac_distance_m', 0.004).value
        )
        self.minimum_plane_inlier_ratio = float(
            self.declare_parameter('minimum_plane_inlier_ratio', 0.50).value
        )
        self.maximum_plane_residual_m = float(
            self.declare_parameter('maximum_plane_residual_m', 0.004).value
        )
        self.minimum_book_width_m = float(
            self.declare_parameter('minimum_book_width_m', 0.005).value
        )
        self.maximum_book_width_m = float(
            self.declare_parameter('maximum_book_width_m', 0.080).value
        )
        self.right_gap_tip_offset_m = float(
            self.declare_parameter('right_gap_tip_offset_m', 0.0015).value
        )
        self.target_spine_up_offset_m = float(
            self.declare_parameter('target_spine_up_offset_m', 0.015).value
        )

        # The marker supplies only the directed insertion axis. The selected
        # book mask/depth still supplies the physical right-seam target and
        # book-spine direction. OpenCV marker -Z is the requested tool +Z.
        self.use_aruco_orientation = bool(
            self.declare_parameter('use_aruco_orientation', True).value
        )
        self.aruco_dictionary = str(
            self.declare_parameter('aruco_dictionary', 'DICT_4X4_50').value
        )
        self.aruco_target_id = int(
            self.declare_parameter('aruco_target_id', 0).value
        )
        self.aruco_marker_length_m = float(
            self.declare_parameter('aruco_marker_length_m', 0.050).value
        )
        self.aruco_max_reprojection_error_px = float(
            self.declare_parameter('aruco_max_reprojection_error_px', 2.0).value
        )
        self.aruco_min_side_px = float(
            self.declare_parameter('aruco_min_side_px', 20.0).value
        )
        self.aruco_book_spine_trust_max_error_deg = float(
            self.declare_parameter(
                'aruco_book_spine_trust_max_error_deg', 10.0
            ).value
        )
        self.aruco_retry_count = int(
            self.declare_parameter('aruco_retry_count', 3).value
        )
        self.aruco_retry_frame_timeout_s = float(
            self.declare_parameter('aruco_retry_frame_timeout_s', 3.0).value
        )

        # Sequence geometry.
        self.approach_standoff_m = float(
            self.declare_parameter('approach_standoff_m', 0.070).value
        )
        self.insert_depth_m = float(
            self.declare_parameter('insert_depth_m', 0.102).value
        )
        self.extract_extra_m = float(
            self.declare_parameter('extract_extra_m', 0.060).value
        )
        self.final_lift_m = float(
            self.declare_parameter('final_lift_m', 0.040).value
        )
        self.minimum_spine_up_component = float(
            self.declare_parameter('minimum_spine_up_component', 0.20).value
        )
        self.minimum_measured_standoff_m = float(
            self.declare_parameter('minimum_measured_standoff_m', 0.030).value
        )
        self.maximum_measured_standoff_m = float(
            self.declare_parameter('maximum_measured_standoff_m', 0.120).value
        )
        self.fine_correction_deadband_m = float(
            self.declare_parameter('fine_correction_deadband_m', 0.0008).value
        )
        self.maximum_fine_correction_m = float(
            self.declare_parameter('maximum_fine_correction_m', 0.030).value
        )
        self.pose_verification_tolerance_m = float(
            self.declare_parameter('pose_verification_tolerance_m', 0.006).value
        )

        # Gripper.
        self.gripper_min_gap_mm = float(
            self.declare_parameter('gripper_min_gap_mm', 0.0).value
        )
        self.gripper_max_gap_mm = float(
            self.declare_parameter('gripper_max_gap_mm', 36.4).value
        )
        self.gripper_open_clearance_mm = float(
            self.declare_parameter('gripper_open_clearance_mm', 3.0).value
        )
        self.gripper_min_clearance_mm = float(
            self.declare_parameter('gripper_min_clearance_mm', 1.0).value
        )
        self.use_high_voltage_hold = bool(
            self.declare_parameter('use_high_voltage_hold', True).value
        )
        self.gripper_settle_s = float(
            self.declare_parameter('gripper_settle_s', 0.7).value
        )
        self.return_velocity_scaling = float(
            self.declare_parameter('return_velocity_scaling', 0.20).value
        )
        self.return_acceleration_scaling = float(
            self.declare_parameter('return_acceleration_scaling', 0.10).value
        )
        self.return_joint_tolerance_rad = float(
            self.declare_parameter('return_joint_tolerance_rad', 0.02).value
        )
        self.release_settle_s = float(
            self.declare_parameter('release_settle_s', 0.7).value
        )

        # Operator gates.
        self.motion_enabled = bool(
            self.declare_parameter('motion_enabled', False).value
        )
        self.confirm_before_motion = bool(
            self.declare_parameter('confirm_before_motion', True).value
        )
        self.confirm_before_release = bool(
            self.declare_parameter('confirm_before_release', True).value
        )

        self._validate_parameters()

        self.bridge = CvBridge()
        self.tf_buffer = Buffer(
            cache_time=Duration(seconds=self.tf_buffer_cache_s)
        )
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.move_group_client = ActionClient(self, MoveGroup, 'move_action')
        self.ik_client = self.create_client(GetPositionIK, 'compute_ik')
        self.execute_client = ActionClient(
            self, ExecuteTrajectory, 'execute_trajectory'
        )

        self.color_info: Optional[CameraInfo] = None
        self.color_image: Optional[np.ndarray] = None
        self.color_msg: Optional[Image] = None
        self.color_stamp_ns: Optional[int] = None
        self.depth_image: Optional[np.ndarray] = None
        self.depth_stamp_ns: Optional[int] = None
        self.color_lock = threading.Lock()
        self.depth_lock = threading.Lock()

        self.joint_positions: dict[str, float] = {}
        self.joint_state_received_monotonic: Optional[float] = None
        self.joint_lock = threading.Lock()

        # Base-frame corners of the operator's initial target outline.
        self.target_outline_base: Optional[np.ndarray] = None

        self.gripper_status: Optional[str] = None
        self.gripper_hold_expected = False
        self.gripper_hold_lost = False
        self.gripper_lock = threading.Lock()

        color_info_topic = str(self.declare_parameter(
            'color_info_topic', '/camera/camera/color/camera_info'
        ).value)
        color_topic = str(self.declare_parameter(
            'color_image_topic', '/camera/camera/color/image_raw'
        ).value)
        depth_topic = str(self.declare_parameter(
            'depth_image_topic',
            '/camera/camera/aligned_depth_to_color/image_raw',
        ).value)

        self.create_subscription(CameraInfo, color_info_topic, self._color_info_cb, 10)
        self.create_subscription(Image, color_topic, self._color_cb, 10)
        self.create_subscription(Image, depth_topic, self._depth_cb, 10)
        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)

        debug_overlay_topic = str(self.declare_parameter(
            'debug_overlay_topic', '/manual_book_grasp/debug_overlay'
        ).value)
        overlay_qos = QoSProfile(depth=1)
        overlay_qos.reliability = ReliabilityPolicy.RELIABLE
        overlay_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.debug_overlay_pub = self.create_publisher(
            Image, debug_overlay_topic, overlay_qos
        )

        self.gripper_gap_pub = self.create_publisher(Float32, '/gripper/gap', 10)
        self.gripper_cmd_pub = self.create_publisher(String, '/gripper/command', 10)
        self.create_subscription(
            String, '/gripper/status', self._gripper_status_cb, 10
        )

        marker_qos = QoSProfile(depth=20)
        marker_qos.reliability = ReliabilityPolicy.RELIABLE
        marker_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.marker_pub = self.create_publisher(
            Marker, '/grasp_target_marker', marker_qos
        )
        display_qos = QoSProfile(depth=1)
        display_qos.reliability = ReliabilityPolicy.RELIABLE
        display_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.display_pub = self.create_publisher(
            DisplayTrajectory, '/display_planned_path', display_qos
        )

        self.get_logger().info('Waiting for MoveIt move_action...')
        self.move_group_client.wait_for_server()
        self.get_logger().info('Connected to move_action.')
        if self.motion_enabled:
            self.get_logger().warning('ARM MOTION ENABLED')
            self.execute_client.wait_for_server()
        else:
            self.get_logger().warning(
                'Arm motion is disabled. The node will only detect, plan, and publish RViz output.'
            )

    def _validate_parameters(self) -> None:
        scalings = {
            'ptp_velocity_scaling': self.ptp_velocity_scaling,
            'ptp_acceleration_scaling': self.ptp_acceleration_scaling,
            'lin_velocity_scaling': self.lin_velocity_scaling,
            'lin_acceleration_scaling': self.lin_acceleration_scaling,
        }
        for name, value in scalings.items():
            if not 0.0 < value <= 1.0:
                raise ValueError(f'{name} must be within (0, 1]')
        if not 10.0 <= self.manual_selection_timeout_s <= 1800.0:
            raise ValueError('manual_selection_timeout_s must be within [10, 1800] s')
        if self.tf_buffer_cache_s < self.manual_selection_timeout_s + 30.0:
            raise ValueError(
                'tf_buffer_cache_s must exceed manual_selection_timeout_s by at least 30 seconds'
            )
        if not 0.001 <= self.approach_standoff_m <= 0.20:
            raise ValueError('approach_standoff_m must be within [0.001, 0.20] m')
        if not 0.001 <= self.insert_depth_m <= 0.12:
            raise ValueError('insert_depth_m must be within [0.001, 0.12] m')
        if not 0 <= self.planning_retry_count <= 10:
            raise ValueError('planning_retry_count must be within [0, 10]')
        if not self.planning_retry_count + 1 <= self.ptp_ik_max_seed_attempts <= 153:
            raise ValueError(
                'ptp_ik_max_seed_attempts must cover all planning attempts and be <= 153'
            )
        if not math.radians(1.0) <= self.ptp_ik_seed_step_rad <= math.radians(90.0):
            raise ValueError('ptp_ik_seed_step_deg must be within [1, 90]')
        if not 0.01 <= self.ptp_ik_timeout_s <= 2.0:
            raise ValueError('ptp_ik_timeout_s must be within [0.01, 2.0]')
        if not math.radians(0.001) <= self.ptp_unique_solution_tolerance_rad <= math.radians(1.0):
            raise ValueError('ptp_unique_solution_tolerance_deg must be within [0.001, 1.0]')
        if not 1.0 <= self.fine_perception_settle_s <= 2.0:
            raise ValueError('fine_perception_settle_s must be within [1.0, 2.0] s')
        if not 0.0 <= self.target_spine_up_offset_m <= 0.10:
            raise ValueError('target_spine_up_offset_m must be within [0, 0.10] m')
        if not 0.0 <= self.extract_extra_m <= 0.30:
            raise ValueError('extract_extra_m must be within [0, 0.30] m')
        if not 0.0 <= self.final_lift_m <= 0.150:
            raise ValueError('final_lift_m must be within [0, 0.150] m')
        if not 0.20 <= self.minimum_spine_up_component <= 1.0:
            raise ValueError('minimum_spine_up_component must be within [0.20, 1.0]')
        if self.minimum_measured_standoff_m >= self.maximum_measured_standoff_m:
            raise ValueError('minimum_measured_standoff_m must be below maximum')
        if not 0 <= self.outline_border_margin_px <= 100:
            raise ValueError('outline_border_margin_px must be within [0, 100]')
        if self.minimum_book_width_m >= self.maximum_book_width_m:
            raise ValueError('minimum_book_width_m must be below maximum_book_width_m')
        if self.aruco_dictionary != 'DICT_4X4_50':
            raise ValueError('aruco_dictionary must remain DICT_4X4_50')
        if self.aruco_target_id != 0:
            raise ValueError('aruco_target_id must remain 0')
        if not math.isclose(self.aruco_marker_length_m, 0.050, abs_tol=1.0e-9):
            raise ValueError('aruco_marker_length_m must remain 0.050 m')
        if not 0.1 <= self.aruco_max_reprojection_error_px <= 5.0:
            raise ValueError('aruco_max_reprojection_error_px must be within [0.1, 5.0]')
        if not 8.0 <= self.aruco_min_side_px <= 200.0:
            raise ValueError('aruco_min_side_px must be within [8, 200]')
        if not 0.0 <= self.aruco_book_spine_trust_max_error_deg <= 45.0:
            raise ValueError(
                'aruco_book_spine_trust_max_error_deg must be within [0, 45]'
            )
        if not 0 <= self.aruco_retry_count <= 10:
            raise ValueError('aruco_retry_count must be within [0, 10]')
        if not 0.1 <= self.aruco_retry_frame_timeout_s <= 10.0:
            raise ValueError('aruco_retry_frame_timeout_s must be within [0.1, 10.0] s')
        if self.gripper_min_gap_mm >= self.gripper_max_gap_mm:
            raise ValueError('gripper_min_gap_mm must be below gripper_max_gap_mm')
        if not 0.0 < self.return_velocity_scaling <= 1.0:
            raise ValueError('return_velocity_scaling must be within (0, 1]')
        if not 0.0 < self.return_acceleration_scaling <= 1.0:
            raise ValueError('return_acceleration_scaling must be within (0, 1]')
        if self.return_joint_tolerance_rad <= 0.0:
            raise ValueError('return_joint_tolerance_rad must be positive')

    # ------------------------------------------------------------------
    # ROS callbacks and synchronized observations
    # ------------------------------------------------------------------
    @staticmethod
    def _stamp_to_ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _query_time(stamp):
        if int(stamp.sec) == 0 and int(stamp.nanosec) == 0:
            return rclpy.time.Time()
        return rclpy.time.Time.from_msg(stamp)

    def _color_info_cb(self, msg: CameraInfo) -> None:
        self.color_info = msg

    def _color_cb(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.color_lock:
                self.color_image = np.asarray(image).copy()
                self.color_msg = msg
                self.color_stamp_ns = self._stamp_to_ns(msg.header.stamp)
        except Exception as exc:
            self.get_logger().error(f'Color conversion failed: {exc}')

    def _depth_cb(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.depth_lock:
                self.depth_image = np.asarray(image).copy()
                self.depth_stamp_ns = self._stamp_to_ns(msg.header.stamp)
        except Exception as exc:
            self.get_logger().error(f'Depth conversion failed: {exc}')

    def _joint_state_cb(self, msg: JointState) -> None:
        with self.joint_lock:
            self.joint_positions = {
                name: float(position)
                for name, position in zip(msg.name, msg.position)
            }
            self.joint_state_received_monotonic = time.monotonic()

    def _gripper_status_cb(self, msg: String) -> None:
        status = str(msg.data).strip()
        with self.gripper_lock:
            self.gripper_status = status
            if self.gripper_hold_expected and status == 'V_control = LOW':
                self.gripper_hold_lost = True
        if self.gripper_hold_lost:
            self.get_logger().error(
                'Gripper high-voltage hold was lost. Automatic retreat is blocked.'
            )

    def _wait_future(self, future, timeout_s: Optional[float] = None) -> bool:
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        return bool(future.done())

    def _has_timestamped_transform(self, source_frame: str, stamp) -> bool:
        try:
            self.tf_buffer.lookup_transform(
                self.base_frame, source_frame, self._query_time(stamp)
            )
            return True
        except Exception:
            return False

    def wait_for_camera_bundle(
        self,
        *,
        not_before_stamp_ns: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> Optional[CameraBundle]:
        started = time.monotonic()
        last_log = 0.0
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            with self.color_lock:
                color = None if self.color_image is None else self.color_image.copy()
                color_msg = self.color_msg
                color_stamp = self.color_stamp_ns
            with self.depth_lock:
                depth = None if self.depth_image is None else self.depth_image.copy()
                depth_stamp = self.depth_stamp_ns
            info = self.color_info

            synchronized = (
                color_stamp is not None
                and depth_stamp is not None
                and abs(color_stamp - depth_stamp)
                <= int(self.camera_sync_slop_s * 1_000_000_000)
            )
            new_enough = (
                not_before_stamp_ns is None
                or (
                    color_stamp is not None
                    and depth_stamp is not None
                    and color_stamp >= int(not_before_stamp_ns)
                    and depth_stamp >= int(not_before_stamp_ns)
                )
            )
            tf_ready = (
                color_msg is not None
                and bool(color_msg.header.frame_id)
                and self._has_timestamped_transform(
                    color_msg.header.frame_id, color_msg.header.stamp
                )
            )
            if (
                color is not None
                and depth is not None
                and info is not None
                and color_msg is not None
                and synchronized
                and new_enough
                and tf_ready
            ):
                return CameraBundle(color, depth, info, color_msg)

            now = time.monotonic()
            if now - last_log >= 1.0:
                missing = []
                if color is None:
                    missing.append('color')
                if depth is None:
                    missing.append('depth')
                if info is None:
                    missing.append('camera_info')
                if color is not None and depth is not None and not synchronized:
                    missing.append('RGB-D sync')
                if not new_enough:
                    missing.append('post-motion frame')
                if color_msg is not None and not tf_ready:
                    missing.append('capture-time TF')
                self.get_logger().info('Waiting for ' + ', '.join(missing or ['camera bundle']))
                last_log = now

            if timeout_s is not None and now - started >= timeout_s:
                return None
        return None

    def _lookup_rigid_transform(self, target_frame: str, source_frame: str, stamp):
        transform = self.tf_buffer.lookup_transform(
            target_frame, source_frame, self._query_time(stamp)
        )
        rotation = Rotation.from_quat([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ])
        translation = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ])
        return rotation, translation

    def _check_target_mask(self, mask: np.ndarray, *, context: str) -> bool:
        mask_pixels = int(np.count_nonzero(mask))
        if mask_pixels < self.minimum_mask_pixels:
            self.get_logger().error(
                f'{context} target outline covers only {mask_pixels} pixels '
                f'(< {self.minimum_mask_pixels}); too small for stable 3D geometry'
            )
            return False
        # The book width/right seam come from the left/right mask extent. A
        # clipped side would silently turn the image border into a book edge.
        margin = self.outline_border_margin_px
        columns = np.flatnonzero(np.any(mask > 0, axis=0))
        if columns[0] < margin or columns[-1] >= mask.shape[1] - margin:
            self.get_logger().error(
                f'{context} target outline touches the left/right image border; '
                'the seam would be clipped. Re-position the camera so the whole '
                'book width is visible.'
            )
            return False
        return True

    def select_initial_target_mask(
        self, bundle: CameraBundle
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Ask the operator for the target outline and anchor it in base."""
        if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            self.get_logger().error(
                'No DISPLAY/WAYLAND_DISPLAY: the manual target outline needs a GUI session'
            )
            return None
        preview = bundle.color_bgr.copy()
        overlay_lines = []
        if self.use_aruco_orientation:
            try:
                preview_estimate = estimate_aruco_insertion_axis(
                    bundle.color_bgr,
                    bundle.color_info.k,
                    bundle.color_info.d,
                    dictionary_name=self.aruco_dictionary,
                    marker_id=self.aruco_target_id,
                    marker_length_m=self.aruco_marker_length_m,
                    maximum_reprojection_error_px=self.aruco_max_reprojection_error_px,
                    minimum_side_px=self.aruco_min_side_px,
                )
                corners = np.rint(preview_estimate.corners_px).astype(np.int32)
                cv2.polylines(preview, [corners], True, (0, 255, 255), 2)
                overlay_lines.append(
                    f'ArUco id {self.aruco_target_id}: OK '
                    f'({preview_estimate.reprojection_error_px:.2f} px)'
                )
            except (ValueError, cv2.error) as exc:
                overlay_lines.append(f'ArUco id {self.aruco_target_id}: {exc}')
        quad = select_manual_book_quad(
            'Manual book target (outline the TARGET book only)',
            preview,
            timeout_s=self.manual_selection_timeout_s,
            overlay_lines=overlay_lines,
        )
        if quad is None:
            self.get_logger().error('Target outline canceled or timed out. No motion sent.')
            return None
        mask = quad_mask(bundle.color_bgr.shape, quad)
        if not self._check_target_mask(mask, context='Initial'):
            return None
        outline_px = np.asarray(quad, dtype=float)
        try:
            map_3d = self.build_3d_map(bundle)
            _, center, normal, _, _, _ = self._fit_plane_ransac(map_3d[mask > 0])
            corners_camera = pixel_rays_to_plane(
                outline_px, bundle.color_info.k, center, normal
            )
            rotation, translation = self._lookup_rigid_transform(
                self.base_frame,
                bundle.color_msg.header.frame_id,
                bundle.color_msg.header.stamp,
            )
        except Exception as exc:
            self.get_logger().error(f'Could not anchor the target outline: {exc}')
            return None
        self.target_outline_base = rotation.apply(corners_camera) + translation
        self.get_logger().info(
            'Target outline anchored on the fitted book plane in '
            f'{self.base_frame}: corners(mm)='
            + ', '.join(
                '(' + ', '.join(f'{value * 1000.0:.1f}' for value in corner) + ')'
                for corner in self.target_outline_base
            )
        )
        return mask, outline_px

    def project_target_outline_mask(
        self, bundle: CameraBundle, *, context: str
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Re-project the base-frame target outline into a newer frame."""
        if self.target_outline_base is None:
            self.get_logger().error('No anchored target outline is available')
            return None
        try:
            rotation, translation = self._lookup_rigid_transform(
                bundle.color_msg.header.frame_id,
                self.base_frame,
                bundle.color_msg.header.stamp,
            )
            corners_camera = rotation.apply(self.target_outline_base) + translation
            outline_px = project_points_to_pixels(corners_camera, bundle.color_info.k)
            mask = projected_quad_mask(bundle.color_bgr.shape, outline_px)
        except Exception as exc:
            self.get_logger().error(f'{context} outline re-projection failed: {exc}')
            return None
        if not self._check_target_mask(mask, context=context):
            return None
        return mask, outline_px

    # ------------------------------------------------------------------
    # Selected-book geometry only
    # ------------------------------------------------------------------
    def build_3d_map(self, bundle: CameraBundle) -> np.ndarray:
        height, width = bundle.color_bgr.shape[:2]
        intrinsic = np.asarray(bundle.color_info.k, dtype=float).reshape(3, 3)
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        depth = np.asarray(bundle.depth)
        z = (
            depth.astype(np.float32) * self.depth_unit_scale_m
            if np.issubdtype(depth.dtype, np.integer)
            else depth.astype(np.float32)
        )
        x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
        y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
        return np.dstack((x, y, z))

    def _fit_plane_ransac(self, points: np.ndarray, *, minimum_points: Optional[int] = None):
        minimum_points = self.minimum_geometry_points if minimum_points is None else int(minimum_points)
        points = np.asarray(points, dtype=float)
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.02)
        points = points[valid]
        if len(points) < minimum_points:
            raise ValueError(f'too few valid depth points: {len(points)} < {minimum_points}')

        # Remove gross depth contamination but keep a slanted book plane.
        median_z = float(np.median(points[:, 2]))
        mad_z = float(np.median(np.abs(points[:, 2] - median_z)))
        depth_band = max(0.025, 6.0 * 1.4826 * mad_z)
        cleaned = points[np.abs(points[:, 2] - median_z) <= depth_band]
        if len(cleaned) < minimum_points:
            cleaned = points

        rng = np.random.default_rng(7)
        best_mask = None
        best_count = 0
        best_residual = float('inf')
        for _ in range(self.ransac_iterations):
            sample = cleaned[rng.choice(len(cleaned), size=3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = float(np.linalg.norm(normal))
            if norm < 1e-9:
                continue
            normal /= norm
            distances = np.abs((cleaned - sample[0]) @ normal)
            mask = distances <= self.ransac_distance_m
            count = int(np.count_nonzero(mask))
            if count < 3:
                continue
            residual = float(np.mean(distances[mask]))
            if count > best_count or (count == best_count and residual < best_residual):
                best_mask = mask
                best_count = count
                best_residual = residual

        if best_mask is None or best_count < minimum_points:
            raise ValueError(f'RANSAC did not find a supported book plane: {best_count} < {minimum_points}')

        inliers = cleaned[best_mask]
        center = np.mean(inliers, axis=0)
        centered = inliers - center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        if normal[2] < 0.0:
            normal = -normal
        residuals = np.abs(centered @ normal)
        inlier_ratio = float(len(inliers) / max(len(cleaned), 1))
        residual = float(np.mean(residuals))

        if inlier_ratio < self.minimum_plane_inlier_ratio:
            raise ValueError(
                f'plane inlier ratio {inlier_ratio:.2f} is below '
                f'{self.minimum_plane_inlier_ratio:.2f}'
            )
        if residual > self.maximum_plane_residual_m:
            raise ValueError(
                f'plane residual {residual * 1000.0:.2f} mm is above '
                f'{self.maximum_plane_residual_m * 1000.0:.2f} mm'
            )
        return inliers, center, normal, vh, inlier_ratio, residual

    @staticmethod
    def _project_pixel(point: np.ndarray, camera_info: CameraInfo) -> np.ndarray:
        fx, fy = float(camera_info.k[0]), float(camera_info.k[4])
        cx, cy = float(camera_info.k[2]), float(camera_info.k[5])
        if point[2] <= 1e-6:
            raise ValueError('point projects behind camera')
        return np.array([
            fx * point[0] / point[2] + cx,
            fy * point[1] / point[2] + cy,
        ])

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        nonzero = cv2.findNonZero(np.where(mask, 255, 0).astype(np.uint8))
        if nonzero is None:
            raise ValueError('instance mask is empty')
        x, y, width, height = cv2.boundingRect(nonzero)
        return x, y, x + width, y + height

    def _spine_from_plane(self, inliers, center, normal, vh, camera_info) -> np.ndarray:
        axes = [np.asarray(vh[0], dtype=float), np.asarray(vh[1], dtype=float)]
        spans = []
        for axis in axes:
            axis = axis - float(np.dot(axis, normal)) * normal
            axis /= max(float(np.linalg.norm(axis)), 1.0e-9)
            projection = (inliers - center) @ axis
            spans.append(float(np.percentile(projection, 98) - np.percentile(projection, 2)))
        spine = axes[int(np.argmax(spans))]
        spine = spine - float(np.dot(spine, normal)) * normal
        spine /= max(float(np.linalg.norm(spine)), 1.0e-9)
        center_uv = self._project_pixel(center, camera_info)
        spine_uv = self._project_pixel(center + 0.03 * spine, camera_info)
        if spine_uv[1] > center_uv[1]:
            spine = -spine
        return spine

    def estimate_book_geometry(
        self, bundle: CameraBundle, mask: np.ndarray, *,
        use_aruco_orientation: bool = False,
    ) -> BookGeometry:
        map_3d = self.build_3d_map(bundle)
        target_fit = self._fit_plane_ransac(map_3d[mask > 0])
        inliers, center, target_normal, vh, inlier_ratio, residual = target_fit
        target_spine = self._spine_from_plane(inliers, center, target_normal, vh, bundle.color_info)
        aruco_estimate = None
        row_normal = np.asarray(target_normal, dtype=float)
        if use_aruco_orientation:
            aruco_estimate = estimate_aruco_insertion_axis(
                bundle.color_bgr,
                bundle.color_info.k,
                bundle.color_info.d,
                dictionary_name=self.aruco_dictionary,
                marker_id=self.aruco_target_id,
                marker_length_m=self.aruco_marker_length_m,
                maximum_reprojection_error_px=self.aruco_max_reprojection_error_px,
                minimum_side_px=self.aruco_min_side_px,
            )
            row_normal, constrained_spine, correction_deg, orientation_source = (
                select_aruco_ground_truth_axes(
                    aruco_estimate.insertion_axis_camera,
                    target_spine,
                    book_spine_trust_max_error_deg=(
                        self.aruco_book_spine_trust_max_error_deg
                    ),
                )
            )
        row_normal /= max(float(np.linalg.norm(row_normal)), 1.0e-9)
        # Preserve the directed marker -Z sign. Never flip it into a camera
        # hemisphere: doing so could reverse insertion and extraction.
        row_spine = np.asarray(
            constrained_spine if use_aruco_orientation else target_spine, dtype=float
        )
        if not use_aruco_orientation:
            row_spine = row_spine - float(np.dot(row_spine, row_normal)) * row_normal
            row_spine /= max(float(np.linalg.norm(row_spine)), 1.0e-9)
        center_uv = self._project_pixel(center, bundle.color_info)
        if self._project_pixel(center + 0.03 * row_spine, bundle.color_info)[1] > center_uv[1]:
            row_spine = -row_spine
        if use_aruco_orientation:
            self.get_logger().info(
                'ArUco insertion orientation accepted: '
                f'dictionary={self.aruco_dictionary}, id={self.aruco_target_id}, '
                f'size={self.aruco_marker_length_m*1000.0:.1f} mm, axis=marker -Z, '
                f'reprojection={aruco_estimate.reprojection_error_px:.2f} px, '
                f'spine_perpendicular_error={correction_deg:.2f} deg, '
                f'trust_threshold={self.aruco_book_spine_trust_max_error_deg:.2f} deg, '
                f'orientation_source={orientation_source}; '
                'book seam remains the insertion position reference'
            )
        target_right = np.cross(target_normal, target_spine)
        target_right /= max(float(np.linalg.norm(target_right)), 1.0e-9)
        if self._project_pixel(center + 0.02 * target_right, bundle.color_info)[0] < center_uv[0]:
            target_right = -target_right
        gap_right = np.cross(row_normal, row_spine)
        gap_right /= max(float(np.linalg.norm(gap_right)), 1.0e-9)
        if self._project_pixel(center + 0.02 * gap_right, bundle.color_info)[0] < center_uv[0]:
            gap_right = -gap_right
        centered = inliers - center
        right_projection = centered @ target_right
        spine_projection = centered @ target_spine
        left_edge = float(np.percentile(right_projection, 2))
        right_edge = float(np.percentile(right_projection, 98))
        width_m = right_edge - left_edge
        if not self.minimum_book_width_m <= width_m <= self.maximum_book_width_m:
            raise ValueError(
                f'book width {width_m*1000.0:.1f} mm is outside '
                f'[{self.minimum_book_width_m*1000.0:.1f}, {self.maximum_book_width_m*1000.0:.1f}] mm'
            )
        seam_center = center + right_edge * target_right + float(np.median(spine_projection)) * target_spine
        target_point = (
            seam_center
            + self.right_gap_tip_offset_m * gap_right
            + self.target_spine_up_offset_m * target_spine
        )
        tool_y = row_spine
        tool_z = row_normal
        tool_x = np.cross(tool_y, tool_z)
        tool_x /= max(float(np.linalg.norm(tool_x)), 1.0e-9)
        tool_y = np.cross(tool_z, tool_x)
        tool_y /= max(float(np.linalg.norm(tool_y)), 1.0e-9)
        rotation_camera = np.column_stack((tool_x, tool_y, tool_z))
        if np.linalg.det(rotation_camera) < 0.99:
            raise ValueError('failed to construct a right-handed grasp orientation')
        quaternion = Rotation.from_matrix(rotation_camera).as_quat()
        target = PoseStamped(); target.header = bundle.color_msg.header
        target.pose.position.x, target.pose.position.y, target.pose.position.z = map(float, target_point)
        target.pose.orientation.x, target.pose.orientation.y, target.pose.orientation.z, target.pose.orientation.w = map(float, quaternion)
        x1, y1, x2, y2 = self._mask_bbox(mask > 0)
        return BookGeometry(
            target_pose_camera=target, center_camera=center,
            normal_camera=row_normal, target_normal_camera=target_normal,
            spine_camera=row_spine, right_camera=gap_right, width_m=width_m,
            plane_inlier_ratio=inlier_ratio, plane_residual_m=residual,
            mask_bbox=(x1,y1,x2,y2),
            aruco_corners_px=(
                None if aruco_estimate is None else aruco_estimate.corners_px.copy()
            ),
            aruco_axis_camera=(
                None if aruco_estimate is None
                else aruco_estimate.insertion_axis_camera.copy()
            ),
            aruco_reprojection_error_px=(
                None if aruco_estimate is None
                else float(aruco_estimate.reprojection_error_px)
            ),
        )

    def save_detection_debug(
        self,
        bundle: CameraBundle,
        mask: np.ndarray,
        geometry: BookGeometry,
        filename: str,
    ) -> None:
        image = bundle.color_bgr.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        x1, y1, x2, y2 = geometry.mask_bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if geometry.outline_px is not None:
            outline = np.rint(geometry.outline_px).astype(np.int32)
            cv2.polylines(image, [outline], True, (255, 255, 0), 1, cv2.LINE_AA)
        if geometry.aruco_corners_px is not None:
            corners = np.rint(geometry.aruco_corners_px).astype(np.int32)
            cv2.polylines(image, [corners], True, (0, 255, 255), 2)
            center_marker = tuple(np.rint(np.mean(corners, axis=0)).astype(int))
            cv2.drawMarker(
                image, center_marker, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 18, 2
            )
        point = np.array([
            geometry.target_pose_camera.pose.position.x,
            geometry.target_pose_camera.pose.position.y,
            geometry.target_pose_camera.pose.position.z,
        ])
        pixel = self._project_pixel(point, bundle.color_info)
        cv2.drawMarker(
            image,
            tuple(np.rint(pixel).astype(int)),
            (255, 0, 255),
            cv2.MARKER_CROSS,
            20,
            2,
        )
        path = os.path.join(self.debug_output_dir, filename)
        if cv2.imwrite(path, image):
            self.get_logger().info(f'Detection debug saved to {path}')
        else:
            self.get_logger().warning(f'Could not save detection debug to {path}')
        overlay = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        overlay.header = bundle.color_msg.header
        self.debug_overlay_pub.publish(overlay)

    def detect_book(
        self, *, initial: bool, not_before_stamp_ns: Optional[int] = None,
        timeout_s: Optional[float] = None, debug_filename: str,
        use_aruco_orientation: bool = False,
    ) -> Optional[tuple[CameraBundle, np.ndarray, BookGeometry]]:
        # The operator outlines only the first initial frame. ArUco retry
        # frames and the stand-off frame re-project that base-frame outline.
        if initial:
            self.target_outline_base = None
        retry_count = self.aruco_retry_count if use_aruco_orientation else 0
        total_attempts = retry_count + 1
        next_stamp_ns = not_before_stamp_ns
        for attempt in range(1, total_attempts + 1):
            frame_timeout_s = (
                timeout_s if attempt == 1 else self.aruco_retry_frame_timeout_s
            )
            bundle = self.wait_for_camera_bundle(
                not_before_stamp_ns=next_stamp_ns,
                timeout_s=frame_timeout_s,
            )
            if bundle is None:
                self.get_logger().error('No synchronized RGB-D/TF observation')
                return None
            if self.target_outline_base is None:
                target = self.select_initial_target_mask(bundle)
            else:
                target = self.project_target_outline_mask(
                    bundle,
                    context='Initial retry' if initial else 'Stand-off',
                )
            if target is None:
                return None
            mask, outline_px = target
            try:
                geometry = self.estimate_book_geometry(
                    bundle, mask,
                    use_aruco_orientation=use_aruco_orientation,
                )
            except ArucoObservationError as exc:
                if attempt >= total_attempts:
                    self.get_logger().error(
                        f'ArUco observation failed after {total_attempts} '
                        f'attempt(s): {exc}'
                    )
                    return None
                next_stamp_ns = self._stamp_to_ns(bundle.color_msg.header.stamp) + 1
                self.get_logger().warning(
                    f'ArUco observation rejected on attempt {attempt}/'
                    f'{total_attempts}: {exc}. Retrying with a newer '
                    'synchronized RGB-D frame and the re-projected target outline.'
                )
                continue
            except ValueError as exc:
                self.get_logger().error(f'Book geometry rejected: {exc}')
                return None
            geometry.outline_px = outline_px
            self.save_detection_debug(bundle, mask, geometry, debug_filename)
            self.get_logger().info(
                f'Book geometry accepted on attempt {attempt}/{total_attempts}: '
                f'width={geometry.width_m*1000.0:.1f} mm, '
                f'target_inlier={geometry.plane_inlier_ratio:.2f}, '
                f'target_residual={geometry.plane_residual_m*1000.0:.2f} mm'
            )
            return bundle, mask, geometry
        return None

    # ------------------------------------------------------------------
    # TF and poses
    # ------------------------------------------------------------------
    def transform_pose_to_base(self, pose: PoseStamped) -> Optional[PoseStamped]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                pose.header.frame_id,
                self._query_time(pose.header.stamp),
            )
            rotation_base_camera = Rotation.from_quat([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ])
            translation = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ])
            point_camera = np.array([
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            ])
            pose_rotation_camera = Rotation.from_quat([
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ])
            point_base = rotation_base_camera.apply(point_camera) + translation
            rotation_base = rotation_base_camera * pose_rotation_camera
            quaternion = rotation_base.as_quat()

            result = PoseStamped()
            result.header.frame_id = self.base_frame
            result.header.stamp = self.get_clock().now().to_msg()
            result.pose.position.x = float(point_base[0])
            result.pose.position.y = float(point_base[1])
            result.pose.position.z = float(point_base[2])
            result.pose.orientation.x = float(quaternion[0])
            result.pose.orientation.y = float(quaternion[1])
            result.pose.orientation.z = float(quaternion[2])
            result.pose.orientation.w = float(quaternion[3])
            return result
        except Exception as exc:
            self.get_logger().error(f'Pose transform failed: {exc}')
            return None

    def get_current_pose(self) -> Optional[PoseStamped]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.eef_link, rclpy.time.Time()
            )
            pose = PoseStamped()
            pose.header.frame_id = self.base_frame
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            return pose
        except Exception as exc:
            self.get_logger().error(f'Current TCP pose unavailable: {exc}')
            return None

    def offset_pose_along_tool_z(self, pose: PoseStamped, distance_m: float) -> PoseStamped:
        result = copy.deepcopy(pose)
        rotation = Rotation.from_quat([
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ])
        position = np.array([
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        ])
        position += float(distance_m) * rotation.as_matrix()[:, 2]
        result.header.stamp = self.get_clock().now().to_msg()
        result.pose.position.x = float(position[0])
        result.pose.position.y = float(position[1])
        result.pose.position.z = float(position[2])
        return result

    @staticmethod
    def pose_position(pose: PoseStamped) -> np.ndarray:
        return np.array([
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        ], dtype=float)

    # ------------------------------------------------------------------
    # Minimal MoveIt planning and execution
    # ------------------------------------------------------------------
    def wait_for_fresh_joint_state(self, timeout_s: float = 2.0) -> bool:
        required = [f'joint_{index}' for index in range(1, 7)]
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            with self.joint_lock:
                positions = dict(self.joint_positions)
                received = self.joint_state_received_monotonic
            age = None if received is None else time.monotonic() - received
            if (
                all(name in positions for name in required)
                and age is not None
                and age <= self.joint_state_max_age_s
            ):
                return True
        self.get_logger().error('Current six-axis joint state is missing or stale')
        return False

    def build_robot_state(self) -> RobotState:
        state = RobotState()
        with self.joint_lock:
            positions = dict(self.joint_positions)
        state.joint_state.header.stamp = self.get_clock().now().to_msg()
        state.joint_state.name = list(positions.keys())
        state.joint_state.position = [positions[name] for name in state.joint_state.name]
        state.is_diff = False
        return state

    def record_current_six_joints(self) -> Optional[dict[str, float]]:
        if not self.wait_for_fresh_joint_state():
            return None
        required = [f'joint_{index}' for index in range(1, 7)]
        with self.joint_lock:
            current = dict(self.joint_positions)
        if any(name not in current for name in required):
            self.get_logger().error('Cannot record an incomplete six-joint state')
            return None
        recorded = {name: float(current[name]) for name in required}
        self.get_logger().info('Recorded exact initial arm joints: ' + ', '.join(
            f'{name}={value:.4f}' for name, value in recorded.items()
        ))
        return recorded

    def wait_for_joint_target(self, joint_targets: dict[str, float], *, timeout_s: float = 25.0) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        last_error = None
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            with self.joint_lock:
                current = dict(self.joint_positions)
            if any(name not in current for name in joint_targets):
                continue
            last_error = max(abs(float(current[name]) - float(target)) for name, target in joint_targets.items())
            if last_error <= self.return_joint_tolerance_rad:
                self.get_logger().info(f'Recorded joint target reached: max_error={last_error:.4f} rad')
                return True
        self.get_logger().error(
            f'Recorded joint target did not settle: max_error={float("nan") if last_error is None else last_error:.4f} rad'
        )
        return False

    @staticmethod
    def _build_exact_joint_constraints(joint_targets: dict[str, float]) -> Constraints:
        constraints = Constraints(); constraints.name = 'exact_recorded_six_joint_return'
        for joint_name in (f'joint_{index}' for index in range(1, 7)):
            joint = JointConstraint(); joint.joint_name = joint_name
            joint.position = float(joint_targets[joint_name])
            joint.tolerance_above = 1.0e-4; joint.tolerance_below = 1.0e-4; joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        return constraints

    def plan_recorded_joint_return(self, joint_targets: dict[str, float]):
        if not self.wait_for_fresh_joint_state():
            return None
        required = [f'joint_{index}' for index in range(1, 7)]
        if any(name not in joint_targets for name in required):
            self.get_logger().error('Recorded return target is incomplete'); return None
        start_state = self.build_robot_state()
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.pipeline_id = self.pilz_pipeline_id
        goal.request.planner_id = self.ptp_planner_id
        goal.request.num_planning_attempts = 1
        goal.request.allowed_planning_time = self.planning_time_s
        goal.request.max_velocity_scaling_factor = self.return_velocity_scaling
        goal.request.max_acceleration_scaling_factor = self.return_acceleration_scaling
        goal.request.start_state = start_state
        goal.request.goal_constraints = [self._build_exact_joint_constraints(joint_targets)]
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        future = self.move_group_client.send_goal_async(goal); self._wait_future(future)
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('Recorded-joint return planning was rejected'); return None
        result_future = handle.get_result_async(); self._wait_future(result_future)
        wrapped = result_future.result()
        if wrapped is None or wrapped.result.error_code.val != 1:
            code = None if wrapped is None else wrapped.result.error_code.val
            self.get_logger().error(f'Recorded-joint return planning failed, error={code}'); return None
        trajectory = wrapped.result.planned_trajectory
        names = list(trajectory.joint_trajectory.joint_names); points = trajectory.joint_trajectory.points
        if not names or not points:
            self.get_logger().error('Recorded-joint return trajectory is empty'); return None
        final_positions = dict(zip(names, points[-1].positions))
        if any(name not in final_positions for name in required):
            self.get_logger().error('Recorded-joint return trajectory omits arm joints'); return None
        endpoint_error = max(abs(float(final_positions[name]) - float(joint_targets[name])) for name in required)
        if endpoint_error > max(self.return_joint_tolerance_rad, 0.005):
            self.get_logger().error(f'Recorded-joint return endpoint error={endpoint_error:.4f} rad'); return None
        self.publish_display_trajectory(start_state, trajectory, 'exact recorded-six-joint return PTP')
        return trajectory

    def _build_pose_constraints(
        self,
        target: PoseStamped,
        *,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
    ) -> Constraints:
        constraints = Constraints()
        constraints.name = 'simple_book_grasp_pose_goal'

        position = PositionConstraint()
        position.header.frame_id = self.base_frame
        position.header.stamp = self.get_clock().now().to_msg()
        position.link_name = self.eef_link
        position.weight = 1.0
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [float(position_tolerance_m)]
        sphere_pose = Pose()
        sphere_pose.position = target.pose.position
        sphere_pose.orientation.w = 1.0
        position.constraint_region.primitives = [sphere]
        position.constraint_region.primitive_poses = [sphere_pose]
        constraints.position_constraints = [position]

        orientation = OrientationConstraint()
        orientation.header = position.header
        orientation.link_name = self.eef_link
        orientation.orientation = target.pose.orientation
        orientation.absolute_x_axis_tolerance = float(orientation_tolerance_rad)
        orientation.absolute_y_axis_tolerance = float(orientation_tolerance_rad)
        orientation.absolute_z_axis_tolerance = float(orientation_tolerance_rad)
        orientation.parameterization = OrientationConstraint.ROTATION_VECTOR
        orientation.weight = 1.0
        constraints.orientation_constraints = [orientation]
        return constraints

    @staticmethod
    def _trajectory_joint_delta(trajectory, joint_name: str, reference: float):
        names = list(trajectory.joint_trajectory.joint_names)
        if joint_name not in names:
            return None
        index = names.index(joint_name)
        values = [
            float(point.positions[index])
            for point in trajectory.joint_trajectory.points
            if len(point.positions) > index
        ]
        if not values:
            return None
        return max(abs(value - reference) for value in values)

    def _validate_planned_wrist(self, trajectory, planner_id: str) -> bool:
        with self.joint_lock:
            current = dict(self.joint_positions)
        joint_5_delta = self._trajectory_joint_delta(
            trajectory, 'joint_5', current.get('joint_5', 0.0)
        )
        joint_6_delta = self._trajectory_joint_delta(
            trajectory, 'joint_6', current.get('joint_6', 0.0)
        )
        if joint_6_delta is None:
            self.get_logger().error('Planned trajectory has no joint_6 samples')
            return False
        if str(planner_id).upper() == 'LIN':
            if joint_6_delta > self.max_lin_joint_6_delta_rad:
                self.get_logger().error(
                    'Rejected LIN wrist jump: '
                    f'{math.degrees(joint_6_delta):.1f} deg > '
                    f'{math.degrees(self.max_lin_joint_6_delta_rad):.1f} deg'
                )
                return False
            return True
        if joint_5_delta is None:
            self.get_logger().error('Planned trajectory has no joint_5 samples')
            return False
        if (
            joint_5_delta > self.max_ptp_joint_5_delta_rad
            or joint_6_delta > self.max_ptp_joint_6_delta_rad
        ):
            self.get_logger().error(
                'Rejected PTP wrist branch: '
                f'J5={math.degrees(joint_5_delta):.1f} deg, '
                f'J6={math.degrees(joint_6_delta):.1f} deg'
            )
            return False
        return True

    def publish_display_trajectory(self, start_state, trajectory, label: str) -> None:
        display = DisplayTrajectory()
        display.model_id = self.robot_model_id
        display.trajectory_start = start_state
        display.trajectory = [trajectory]
        self.display_pub.publish(display)
        self.get_logger().info(f'Published {label} trajectory to RViz')

    @staticmethod
    def _joint_map(state: RobotState) -> dict[str, float]:
        return {
            str(name): float(position)
            for name, position in zip(
                state.joint_state.name, state.joint_state.position
            )
        }

    def _ptp_seed_offsets(self, seed_index: int) -> tuple[float, ...]:
        """Return one deterministic measured-relative IK seed.

        Seed zero is the measured state. The remaining entries perturb
        shoulder, elbow, and wrist combinations in both directions. A retry
        is offered to Pilz only when IK produces a globally new six-joint
        output, so the same rejected branch is never planned twice.
        """
        step = self.ptp_ik_seed_step_rad
        zero = 0.0
        patterns = (
            (zero, zero, zero, zero, zero, step),
            (zero, zero, zero, zero, step, zero),
            (zero, zero, zero, step, zero, zero),
            (zero, zero, step, zero, zero, zero),
            (zero, step, zero, zero, zero, zero),
            (step, zero, zero, zero, zero, zero),
            (step, zero, -step, zero, zero, zero),
            (zero, zero, zero, step, -step, step),
            (step, -step, step, zero, zero, zero),
            (step, step, zero, zero, zero, zero),
            (step, -step, zero, zero, zero, zero),
            (zero, step, step, zero, zero, zero),
            (zero, step, -step, zero, zero, zero),
            (zero, zero, step, step, zero, zero),
            (zero, zero, step, -step, zero, zero),
            (zero, zero, zero, step, step, zero),
            (zero, zero, zero, step, -step, zero),
            (zero, zero, zero, zero, step, step),
            (zero, zero, zero, zero, step, -step),
        )
        if seed_index == 0:
            return (zero,) * 6
        catalog = []
        for scale in (1.0, 0.5, 1.5, 2.0):
            for pattern in patterns:
                scaled = tuple(scale * value for value in pattern)
                catalog.append(scaled)
                catalog.append(tuple(-value for value in scaled))
        return catalog[seed_index - 1]

    def _query_collision_aware_ik(
        self,
        target: PoseStamped,
        start_state: RobotState,
        offsets: tuple[float, ...],
    ) -> Optional[dict[str, float]]:
        if not self.ik_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('compute_ik service is unavailable')
            return None
        required = tuple(f'joint_{index}' for index in range(1, 7))
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.group_name
        request.ik_request.ik_link_name = self.eef_link
        request.ik_request.pose_stamped = target
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout.sec = int(self.ptp_ik_timeout_s)
        request.ik_request.timeout.nanosec = int(
            (self.ptp_ik_timeout_s - int(self.ptp_ik_timeout_s)) * 1.0e9
        )
        request.ik_request.robot_state = copy.deepcopy(start_state)
        names = list(request.ik_request.robot_state.joint_state.name)
        positions = list(request.ik_request.robot_state.joint_state.position)
        if any(name not in names for name in required):
            self.get_logger().error('Measured IK seed is missing one or more arm joints')
            return None
        for name, offset in zip(required, offsets):
            index = names.index(name)
            positions[index] = float(positions[index]) + float(offset)
        request.ik_request.robot_state.joint_state.position = positions
        future = self.ik_client.call_async(request)
        if not self._wait_future(future, timeout_s=self.ptp_ik_timeout_s + 1.0):
            self.get_logger().warning('Collision-aware IK request timed out')
            return None
        response = future.result()
        if response is None or response.error_code.val != 1:
            code = None if response is None else response.error_code.val
            self.get_logger().info(f'Collision-aware IK seed rejected, error={code}')
            return None
        solution = dict(zip(
            response.solution.joint_state.name,
            response.solution.joint_state.position,
        ))
        if any(name not in solution for name in required):
            self.get_logger().warning('Collision-aware IK returned an incomplete arm solution')
            return None
        return {name: float(solution[name]) for name in required}

    def _same_joint_solution(
        self, first: tuple[float, ...], second: tuple[float, ...]
    ) -> bool:
        return all(
            abs(left - right) <= self.ptp_unique_solution_tolerance_rad
            for left, right in zip(first, second)
        )

    @staticmethod
    def _trajectory_total_joint_travel(trajectory, reference_positions=None):
        """Return complete-path J1--J6 travel, not only endpoint distance."""
        required = tuple(f'joint_{index}' for index in range(1, 7))
        names = list(trajectory.joint_trajectory.joint_names)
        points = list(trajectory.joint_trajectory.points)
        if any(name not in names for name in required) or not points:
            return None
        indices = tuple(names.index(name) for name in required)
        per_joint = [0.0] * 6
        previous = None
        if reference_positions is not None:
            if any(name not in reference_positions for name in required):
                return None
            previous = tuple(float(reference_positions[name]) for name in required)
        for point in points:
            if any(index >= len(point.positions) for index in indices):
                return None
            current = tuple(float(point.positions[index]) for index in indices)
            if previous is not None:
                for index, (before, after) in enumerate(zip(previous, current)):
                    per_joint[index] += abs(after - before)
            previous = current
        return float(sum(per_joint)), tuple(per_joint)

    def _plan_explicit_ptp_candidate(
        self,
        target: PoseStamped,
        start_state: RobotState,
        joint_target: dict[str, float],
        attempt: int,
    ):
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.pipeline_id = self.pilz_pipeline_id
        goal.request.planner_id = self.ptp_planner_id
        goal.request.num_planning_attempts = 1
        goal.request.allowed_planning_time = self.planning_time_s
        goal.request.max_velocity_scaling_factor = self.ptp_velocity_scaling
        goal.request.max_acceleration_scaling_factor = self.ptp_acceleration_scaling
        goal.request.start_state = start_state
        goal.request.goal_constraints = [self._build_exact_joint_constraints(joint_target)]
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        future = self.move_group_client.send_goal_async(goal)
        self._wait_future(future)
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().warning(f'PTP candidate {attempt} planning request rejected')
            return None
        result_future = handle.get_result_async()
        self._wait_future(result_future)
        wrapped = result_future.result()
        if wrapped is None:
            self.get_logger().warning(f'PTP candidate {attempt} returned no result')
            return None
        trajectory = wrapped.result.planned_trajectory
        if wrapped.result.error_code.val != 1:
            if trajectory.joint_trajectory.points:
                self.publish_display_trajectory(
                    start_state, trajectory, f'PTP candidate {attempt} DEBUG ONLY'
                )
            self.get_logger().warning(
                f'PTP candidate {attempt} planning failed, '
                f'error={wrapped.result.error_code.val}'
            )
            return None
        if not trajectory.joint_trajectory.points:
            self.get_logger().warning(f'PTP candidate {attempt} returned an empty trajectory')
            return None
        names = list(trajectory.joint_trajectory.joint_names)
        endpoint = dict(zip(names, trajectory.joint_trajectory.points[-1].positions))
        endpoint_error = max(
            abs(float(endpoint.get(name, float('inf'))) - target_value)
            for name, target_value in joint_target.items()
        )
        if endpoint_error > 0.005:
            self.publish_display_trajectory(
                start_state, trajectory, f'PTP candidate {attempt} REJECTED'
            )
            self.get_logger().warning(
                f'PTP candidate {attempt} endpoint mismatch={endpoint_error:.4f} rad'
            )
            return None
        if not self._validate_planned_wrist(trajectory, self.ptp_planner_id):
            self.publish_display_trajectory(
                start_state, trajectory, f'PTP candidate {attempt} REJECTED'
            )
            return None
        # This path has passed every hard gate, but it is not executable until
        # all eligible candidates have been compared. Publish it as a
        # non-selected candidate; the winner is published again below.
        self.publish_display_trajectory(
            start_state,
            trajectory,
            f'PTP candidate {attempt} ELIGIBLE - SELECTION PENDING',
        )
        return start_state, trajectory, target

    def _plan_ptp_with_retries(self, target: PoseStamped):
        if not self.wait_for_fresh_joint_state():
            return None
        start_state = self.build_robot_state()
        measured = self._joint_map(start_state)
        required = tuple(f'joint_{index}' for index in range(1, 7))
        if any(name not in measured for name in required):
            self.get_logger().error('PTP retry search requires all six measured arm joints')
            return None
        unique_solutions: list[tuple[float, ...]] = []
        eligible_plans = []
        planned_candidates = 0
        maximum_candidates = self.planning_retry_count + 1
        for seed_index in range(self.ptp_ik_max_seed_attempts):
            if planned_candidates >= maximum_candidates:
                break
            offsets = self._ptp_seed_offsets(seed_index)
            solution = self._query_collision_aware_ik(target, start_state, offsets)
            if solution is None:
                continue
            signature = tuple(solution[name] for name in required)
            if any(self._same_joint_solution(signature, known) for known in unique_solutions):
                self.get_logger().info(
                    f'PTP seed {seed_index + 1} converged to a globally duplicate IK output; skipped'
                )
                continue
            unique_solutions.append(signature)
            deltas = tuple(solution[name] - measured[name] for name in required)
            self.get_logger().info(
                f'PTP unique candidate {len(unique_solutions)} from seed '
                f'{seed_index + 1}/{self.ptp_ik_max_seed_attempts}: '
                + ', '.join(
                    f'J{index}={math.degrees(delta):+.1f} deg'
                    for index, delta in enumerate(deltas, start=1)
                )
            )
            if (
                abs(deltas[4]) > self.max_ptp_joint_5_delta_rad
                or abs(deltas[5]) > self.max_ptp_joint_6_delta_rad
            ):
                self.get_logger().warning(
                    'Rejected IK target before Pilz: '
                    f'J5={math.degrees(abs(deltas[4])):.1f} deg, '
                    f'J6={math.degrees(abs(deltas[5])):.1f} deg'
                )
                continue
            planned_candidates += 1
            self.get_logger().info(
                f'Planning distinct PTP candidate {planned_candidates}/'
                f'{maximum_candidates} (initial + {self.planning_retry_count} retries)'
            )
            plan = self._plan_explicit_ptp_candidate(
                target, start_state, solution, planned_candidates
            )
            if plan is not None:
                travel = self._trajectory_total_joint_travel(plan[1], measured)
                if travel is None:
                    self.publish_display_trajectory(
                        start_state,
                        plan[1],
                        f'PTP candidate {planned_candidates} REJECTED',
                    )
                    self.get_logger().warning(
                        f'PTP candidate {planned_candidates} cannot be scored '
                        'because its complete J1--J6 samples are missing'
                    )
                    continue
                total_travel, per_joint_travel = travel
                eligible_plans.append((total_travel, planned_candidates, plan))
                self.get_logger().info(
                    f'PTP candidate {planned_candidates} eligible path score: '
                    f'total_joint_travel={math.degrees(total_travel):.1f} deg; '
                    + ', '.join(
                        f'J{index}={math.degrees(value):.1f} deg'
                        for index, value in enumerate(per_joint_travel, start=1)
                    )
                )
        if eligible_plans:
            total_travel, selected_attempt, selected_plan = min(
                eligible_plans, key=lambda item: (item[0], item[1])
            )
            self.publish_display_trajectory(
                selected_plan[0],
                selected_plan[1],
                f'PTP candidate {selected_attempt} SELECTED SHORTEST',
            )
            self.get_logger().info(
                f'Selected shortest of {len(eligible_plans)} eligible PTP '
                f'candidate(s): candidate={selected_attempt}, '
                f'total_joint_travel={math.degrees(total_travel):.1f} deg'
            )
            return selected_plan
        self.get_logger().error(
            'PTP planning exhausted distinct collision-aware candidates: '
            f'planned={planned_candidates}/{maximum_candidates}, '
            f'unique_ik={len(unique_solutions)}, seeds={self.ptp_ik_max_seed_attempts}'
        )
        return None

    def _plan_pose_goal_once(self, target: PoseStamped, planner_id: str, attempt: int):
        if not self.wait_for_fresh_joint_state():
            return None
        if target.header.frame_id != self.base_frame:
            self.get_logger().error('Motion target must be expressed in base frame')
            return None

        is_lin = str(planner_id).upper() == 'LIN'
        position_tolerance = (
            self.lin_position_tolerance_m if is_lin else self.ptp_position_tolerance_m
        )
        orientation_tolerance = (
            self.lin_orientation_tolerance_rad
            if is_lin else self.ptp_orientation_tolerance_rad
        )
        velocity = self.lin_velocity_scaling if is_lin else self.ptp_velocity_scaling
        acceleration = (
            self.lin_acceleration_scaling if is_lin else self.ptp_acceleration_scaling
        )
        start_state = self.build_robot_state()
        constraints = self._build_pose_constraints(
            target,
            position_tolerance_m=position_tolerance,
            orientation_tolerance_rad=orientation_tolerance,
        )

        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.pipeline_id = self.pilz_pipeline_id
        goal.request.planner_id = str(planner_id)
        goal.request.num_planning_attempts = 1
        goal.request.allowed_planning_time = self.planning_time_s
        goal.request.max_velocity_scaling_factor = velocity
        goal.request.max_acceleration_scaling_factor = acceleration
        goal.request.start_state = start_state
        goal.request.goal_constraints = [constraints]
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False

        future = self.move_group_client.send_goal_async(goal)
        self._wait_future(future)
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error(f'{planner_id} planning request was rejected')
            return None
        result_future = handle.get_result_async()
        self._wait_future(result_future)
        wrapped = result_future.result()
        if wrapped is None or wrapped.result.error_code.val != 1:
            code = None if wrapped is None else wrapped.result.error_code.val
            if (
                wrapped is not None
                and wrapped.result.planned_trajectory.joint_trajectory.points
            ):
                self.publish_display_trajectory(
                    start_state,
                    wrapped.result.planned_trajectory,
                    f'{planner_id} attempt {attempt} DEBUG ONLY',
                )
            self.get_logger().warning(
                f'{planner_id} planning attempt {attempt} failed, error={code}'
            )
            return None
        trajectory = wrapped.result.planned_trajectory
        if not trajectory.joint_trajectory.points:
            self.get_logger().error(f'{planner_id} returned an empty trajectory')
            return None
        if not self._validate_planned_wrist(trajectory, planner_id):
            self.publish_display_trajectory(
                start_state, trajectory, f'{planner_id} attempt {attempt} REJECTED'
            )
            return None
        self.publish_display_trajectory(
            start_state, trajectory, f'{planner_id} attempt {attempt} ACCEPTED'
        )
        return start_state, trajectory, target

    def plan_pose(self, target: PoseStamped, planner_id: str):
        if str(planner_id).upper() == 'PTP':
            return self._plan_ptp_with_retries(target)
        maximum_attempts = self.planning_retry_count + 1
        for attempt in range(1, maximum_attempts + 1):
            plan = self._plan_pose_goal_once(target, planner_id, attempt)
            if plan is not None:
                return plan
            if attempt < maximum_attempts:
                self.get_logger().warning(
                    f'{planner_id} planning retry {attempt}/'
                    f'{self.planning_retry_count}; target geometry and all safety guards remain unchanged'
                )
        self.get_logger().error(
            f'{planner_id} planning failed after the initial request and '
            f'{self.planning_retry_count} retries'
        )
        return None

    def _trajectory_start_matches_current(self, trajectory) -> bool:
        if not self.wait_for_fresh_joint_state():
            return False
        names = list(trajectory.joint_trajectory.joint_names)
        points = trajectory.joint_trajectory.points
        if not names or not points:
            return False
        start = list(points[0].positions)
        with self.joint_lock:
            current = dict(self.joint_positions)
        if any(name not in current for name in names):
            return False
        maximum_error = max(
            abs(current[name] - float(position))
            for name, position in zip(names, start)
        )
        if maximum_error > self.trajectory_start_tolerance_rad:
            self.get_logger().error(
                'Robot moved after planning; re-plan required. '
                f'Max joint error={maximum_error:.4f} rad'
            )
            return False
        return True

    def wait_for_pose(self, target: PoseStamped, timeout_s: float = 8.0) -> bool:
        target_position = self.pose_position(target)
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            current = self.get_current_pose()
            if current is None:
                continue
            error = float(np.linalg.norm(self.pose_position(current) - target_position))
            if error <= self.pose_verification_tolerance_m:
                return True
        self.get_logger().error('TCP did not settle at the planned target')
        return False

    def execute_trajectory(self, trajectory, target: PoseStamped) -> bool:
        if not self.motion_enabled:
            self.get_logger().warning('Motion disabled; trajectory was not executed')
            return False
        if not self._trajectory_start_matches_current(trajectory):
            return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        future = self.execute_client.send_goal_async(goal)
        self._wait_future(future)
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('Trajectory execution was rejected')
            return False
        result_future = handle.get_result_async()
        self._wait_future(result_future)
        wrapped = result_future.result()
        if wrapped is None or wrapped.result.error_code.val != 1:
            code = None if wrapped is None else wrapped.result.error_code.val
            self.get_logger().error(f'Trajectory execution failed, error={code}')
            return False
        with self.gripper_lock:
            hold_lost = self.gripper_hold_expected and self.gripper_hold_lost
        if hold_lost:
            self.get_logger().error('Execution ended after gripper hold loss')
            return False
        return self.wait_for_pose(target)

    def execute_joint_trajectory(self, trajectory, joint_targets: dict[str, float]) -> bool:
        if not self.motion_enabled:
            self.get_logger().warning('Motion disabled; trajectory was not executed'); return False
        if not self._trajectory_start_matches_current(trajectory):
            return False
        goal = ExecuteTrajectory.Goal(); goal.trajectory = trajectory
        future = self.execute_client.send_goal_async(goal); self._wait_future(future)
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('Recorded-joint return execution was rejected'); return False
        result_future = handle.get_result_async(); self._wait_future(result_future)
        wrapped = result_future.result()
        if wrapped is None or wrapped.result.error_code.val != 1:
            code = None if wrapped is None else wrapped.result.error_code.val
            self.get_logger().error(f'Recorded-joint return execution failed, error={code}'); return False
        with self.gripper_lock:
            hold_lost = self.gripper_hold_expected and self.gripper_hold_lost
        if hold_lost:
            self.get_logger().error('Return completed after gripper hold loss; automatic release is blocked'); return False
        return self.wait_for_joint_target(joint_targets)

    def return_to_recorded_joints(self, joint_targets: dict[str, float]) -> bool:
        trajectory = self.plan_recorded_joint_return(joint_targets)
        if trajectory is None:
            return False
        self.get_logger().info('Executing exact recorded-six-joint return PTP while maintaining hold')
        return self.execute_joint_trajectory(trajectory, joint_targets)

    def move_pose(self, target: PoseStamped, planner_id: str) -> bool:
        plan = self.plan_pose(target, planner_id)
        if plan is None:
            return False
        _, trajectory, expected = plan
        return self.execute_trajectory(trajectory, expected)

    # ------------------------------------------------------------------
    # Markers and gripper
    # ------------------------------------------------------------------
    def publish_pose_arrow(
        self,
        pose: PoseStamped,
        marker_id: int,
        color: tuple[float, float, float],
        length_m: float = 0.05,
    ) -> None:
        rotation = Rotation.from_quat([
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ])
        start = self.pose_position(pose)
        end = start + float(length_m) * rotation.as_matrix()[:, 2]
        marker = Marker()
        marker.header = pose.header
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'simple_book_grasp'
        marker.id = int(marker_id)
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.points = [
            Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
            Point(x=float(end[0]), y=float(end[1]), z=float(end[2])),
        ]
        marker.scale.x = 0.005
        marker.scale.y = 0.010
        marker.scale.z = 0.015
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.0)
        marker.lifetime.sec = 120
        self.marker_pub.publish(marker)

    def gripper_set_gap(self, gap_mm: float) -> bool:
        if not self.gripper_min_gap_mm <= gap_mm <= self.gripper_max_gap_mm:
            self.get_logger().error(f'Invalid gripper gap: {gap_mm:.1f} mm')
            return False
        if self.gripper_gap_pub.get_subscription_count() < 1:
            self.get_logger().error('No /gripper/gap subscriber')
            return False
        message = Float32()
        message.data = float(gap_mm)
        self.gripper_gap_pub.publish(message)
        time.sleep(self.gripper_settle_s)
        return True

    def gripper_grip(self, thickness_mm: float) -> bool:
        if not self.gripper_min_gap_mm <= thickness_mm <= self.gripper_max_gap_mm:
            self.get_logger().error(f'Invalid book thickness: {thickness_mm:.1f} mm')
            return False
        if self.gripper_cmd_pub.get_subscription_count() < 1:
            self.get_logger().error('No /gripper/command subscriber')
            return False
        message = String()
        message.data = f'H {thickness_mm:.1f}'
        self.gripper_cmd_pub.publish(message)
        time.sleep(self.gripper_settle_s)
        return True

    def gripper_high_voltage_on(self) -> bool:
        if not self.use_high_voltage_hold:
            return True
        if self.gripper_cmd_pub.get_subscription_count() < 1:
            self.get_logger().error('No /gripper/command subscriber')
            return False
        with self.gripper_lock:
            self.gripper_hold_expected = True
            self.gripper_hold_lost = False
        message = String()
        message.data = 'T'
        self.gripper_cmd_pub.publish(message)
        time.sleep(self.gripper_settle_s)
        return True

    def gripper_high_voltage_off(self) -> bool:
        if self.gripper_cmd_pub.get_subscription_count() < 1:
            self.get_logger().error('No /gripper/command subscriber; Y was not sent'); return False
        with self.gripper_lock:
            self.gripper_hold_expected = False
            self.gripper_hold_lost = False
        message = String(); message.data = 'Y'; self.gripper_cmd_pub.publish(message)
        self.get_logger().info('Sent gripper command Y')
        time.sleep(self.release_settle_s)
        return True

    def gripper_open(self) -> bool:
        if self.gripper_cmd_pub.get_subscription_count() < 1:
            self.get_logger().error('No /gripper/command subscriber; final O was not sent'); return False
        message = String(); message.data = 'O'; self.gripper_cmd_pub.publish(message)
        self.get_logger().info('Sent final gripper command O')
        time.sleep(self.release_settle_s)
        return True

    def _locked_spine_up_axis(self, locked_rotation: np.ndarray) -> Optional[np.ndarray]:
        spine_up = np.asarray(locked_rotation[:, 1], dtype=float).copy()
        spine_up /= max(float(np.linalg.norm(spine_up)), 1.0e-9)
        if spine_up[2] < 0.0:
            spine_up *= -1.0
        if spine_up[2] < self.minimum_spine_up_component:
            self.get_logger().error(
                'Extraction blocked: locked book-spine axis is not '
                f'upward enough ({spine_up[2]:.3f} < '
                f'{self.minimum_spine_up_component:.3f})'
            )
            return None
        return spine_up

    def _locked_offset_pose(
        self,
        current_pose: PoseStamped,
        locked_rotation: np.ndarray,
        *,
        outward_m: float = 0.0,
        spine_lift_m: float = 0.0,
        insertion_m: float = 0.0,
        spine_axis: Optional[np.ndarray] = None,
    ) -> PoseStamped:
        """Build a locked-quaternion pose from measured TCP and tool axes."""
        position = self.pose_position(current_pose).copy()
        position += (-float(outward_m) + float(insertion_m)) * locked_rotation[:, 2]
        lift_axis = locked_rotation[:, 1] if spine_axis is None else spine_axis
        lift_axis = np.asarray(lift_axis, dtype=float)
        lift_axis /= max(float(np.linalg.norm(lift_axis)), 1.0e-9)
        position += float(spine_lift_m) * lift_axis
        target = copy.deepcopy(current_pose)
        target.header.frame_id = self.base_frame
        target.header.stamp = self.get_clock().now().to_msg()
        target.pose.position.x, target.pose.position.y, target.pose.position.z = map(
            float, position
        )
        quaternion = Rotation.from_matrix(locked_rotation).as_quat()
        target.pose.orientation.x = float(quaternion[0])
        target.pose.orientation.y = float(quaternion[1])
        target.pose.orientation.z = float(quaternion[2])
        target.pose.orientation.w = float(quaternion[3])
        return target

    def move_locked_offset(
        self,
        locked_rotation: np.ndarray,
        *,
        outward_m: float = 0.0,
        spine_lift_m: float = 0.0,
        insertion_m: float = 0.0,
        spine_axis: Optional[np.ndarray] = None,
        planner_id: Optional[str] = None,
        label: str,
    ) -> bool:
        current_pose = self.get_current_pose()
        if current_pose is None:
            return False
        target = self._locked_offset_pose(
            current_pose,
            locked_rotation,
            outward_m=outward_m,
            spine_lift_m=spine_lift_m,
            insertion_m=insertion_m,
            spine_axis=spine_axis,
        )
        self.get_logger().info(
            f'{label}: outward={outward_m*1000.0:.1f} mm, '
            f'spine_lift={spine_lift_m*1000.0:.1f} mm, '
            f'insertion={insertion_m*1000.0:.1f} mm, '
            f'planner={planner_id or self.ptp_planner_id}'
        )
        return self.move_pose(target, planner_id or self.ptp_planner_id)

    # ------------------------------------------------------------------
    # One simple sequence
    # ------------------------------------------------------------------
    def run(self) -> bool:
        self.get_logger().info('Capturing the initial frame for the manual target outline')
        initial = self.detect_book(
            initial=True,
            timeout_s=None,
            debug_filename='book_grasp_initial.png',
            use_aruco_orientation=self.use_aruco_orientation,
        )
        if initial is None:
            return False
        _, _, initial_geometry = initial
        initial_target_base = self.transform_pose_to_base(
            initial_geometry.target_pose_camera
        )
        if initial_target_base is None:
            return False

        locked_quaternion = np.array([
            initial_target_base.pose.orientation.x,
            initial_target_base.pose.orientation.y,
            initial_target_base.pose.orientation.z,
            initial_target_base.pose.orientation.w,
        ])
        locked_rotation = Rotation.from_quat(locked_quaternion).as_matrix()
        insertion_axis_base = locked_rotation[:, 2]

        approach_pose = self.offset_pose_along_tool_z(
            initial_target_base, -self.approach_standoff_m
        )
        self.publish_pose_arrow(initial_target_base, 0, (1.0, 0.0, 0.0))
        self.publish_pose_arrow(approach_pose, 1, (0.0, 1.0, 0.0))

        approach_plan = self.plan_pose(approach_pose, self.ptp_planner_id)
        if approach_plan is None:
            return False
        if not self.motion_enabled:
            self.get_logger().info(
                'Dry run complete: initial detection and approach PTP are shown in RViz.'
            )
            return True
        if self.confirm_before_motion:
            input(
                'The simplified approach PTP is shown in RViz. '
                'Press Enter to execute the complete sequence, or Ctrl+C to abort...'
            )

        initial_joint_targets = self.record_current_six_joints()
        if initial_joint_targets is None:
            return False

        book_thickness_mm = initial_geometry.width_m * 1000.0
        open_gap_mm = min(
            book_thickness_mm + self.gripper_open_clearance_mm,
            self.gripper_max_gap_mm,
        )
        clearance_mm = open_gap_mm - book_thickness_mm
        if clearance_mm < self.gripper_min_clearance_mm:
            self.get_logger().error(
                f'Book is too thick: available clearance={clearance_mm:.1f} mm'
            )
            return False

        _, approach_trajectory, approach_target = approach_plan
        if not self.execute_trajectory(approach_trajectory, approach_target):
            return False

        # One and only one fresh visual adjustment.
        self.get_logger().info(
            'Waiting '
            f'{self.fine_perception_settle_s:.1f} s for the wrist camera image '
            'to settle before collecting the fine-perception observation'
        )
        settle_deadline = time.monotonic() + self.fine_perception_settle_s
        while rclpy.ok() and time.monotonic() < settle_deadline:
            rclpy.spin_once(
                self,
                timeout_sec=max(
                    0.0, min(0.05, settle_deadline - time.monotonic())
                ),
            )
        observation_after_motion_ns = self.get_clock().now().nanoseconds
        fresh = self.detect_book(
            initial=False,
            not_before_stamp_ns=observation_after_motion_ns,
            timeout_s=8.0,
            debug_filename='book_grasp_fine.png',
        )
        if fresh is None:
            self.get_logger().error('Fine adjustment failed; insertion is blocked')
            return False
        _, _, fresh_geometry = fresh
        fresh_target_base = self.transform_pose_to_base(
            fresh_geometry.target_pose_camera
        )
        if fresh_target_base is None:
            return False

        # Keep the initial orientation. Apply only translation in the book-front
        # plane; depth is used later to calculate insertion distance.
        fresh_target_base.pose.orientation.x = float(locked_quaternion[0])
        fresh_target_base.pose.orientation.y = float(locked_quaternion[1])
        fresh_target_base.pose.orientation.z = float(locked_quaternion[2])
        fresh_target_base.pose.orientation.w = float(locked_quaternion[3])
        current_pose = self.get_current_pose()
        if current_pose is None:
            return False
        initial_target_position = self.pose_position(initial_target_base)
        fresh_target_position = self.pose_position(fresh_target_base)
        locked_side_axis_base = locked_rotation[:, 0]
        locked_spine_axis_base = locked_rotation[:, 1]
        correction = compute_locked_side_correction(
            initial_target_position,
            fresh_target_position,
            locked_side_axis_base,
            locked_spine_axis_base,
            insertion_axis_base,
        )
        planar_correction = correction['side_vector']
        planar_distance = abs(float(correction['side_signed_m']))
        self.get_logger().info(
            'One-shot locked-seam adjustment: '
            f'tool+x={correction["side_signed_m"] * 1000.0:.1f} mm, '
            f'raw={correction["raw_m"] * 1000.0:.1f} mm, '
            f'ignored spine={correction["vertical_signed_m"] * 1000.0:.1f} mm, '
            f'ignored axial={correction["axial_signed_m"] * 1000.0:.1f} mm, '
            f'ignored residual={correction["ignored_m"] * 1000.0:.1f} mm'
        )
        if planar_distance > self.maximum_fine_correction_m:
            self.get_logger().error(
                f'Fine correction {planar_distance * 1000.0:.1f} mm exceeds '
                f'{self.maximum_fine_correction_m * 1000.0:.1f} mm'
            )
            return False
        if planar_distance > self.fine_correction_deadband_m:
            correction_pose = copy.deepcopy(current_pose)
            corrected_position = self.pose_position(current_pose) + planar_correction
            correction_pose.pose.position.x = float(corrected_position[0])
            correction_pose.pose.position.y = float(corrected_position[1])
            correction_pose.pose.position.z = float(corrected_position[2])
            correction_pose.pose.orientation = fresh_target_base.pose.orientation
            # This is a free-space book-front correction, not an in-gap
            # Cartesian insertion. Keep the commissioned correction policy on
            # constrained Pilz PTP; only the initial insertion uses LIN.
            if not self.move_pose(correction_pose, self.ptp_planner_id):
                return False

        if not self.gripper_set_gap(open_gap_mm):
            return False

        current_pose = self.get_current_pose()
        if current_pose is None:
            return False
        corrected_seam_position = initial_target_position + planar_correction
        measured_standoff_m = float(np.dot(
            corrected_seam_position - self.pose_position(current_pose),
            insertion_axis_base,
        ))
        if not (
            self.minimum_measured_standoff_m
            <= measured_standoff_m
            <= self.maximum_measured_standoff_m
        ):
            self.get_logger().error(
                f'Measured stand-off {measured_standoff_m * 1000.0:.1f} mm is outside '
                f'[{self.minimum_measured_standoff_m * 1000.0:.1f}, '
                f'{self.maximum_measured_standoff_m * 1000.0:.1f}] mm'
            )
            return False

        insertion_distance_m = measured_standoff_m + self.insert_depth_m
        insertion_pose = self.offset_pose_along_tool_z(
            current_pose, insertion_distance_m
        )
        insertion_pose.pose.orientation = fresh_target_base.pose.orientation
        self.get_logger().info(
            f'Insertion LIN: {insertion_distance_m * 1000.0:.1f} mm '
            f'({measured_standoff_m * 1000.0:.1f} mm to plane + '
            f'{self.insert_depth_m * 1000.0:.1f} mm into gap)'
        )
        if not self.move_pose(insertion_pose, self.lin_planner_id):
            self.get_logger().error(
                'Insertion failed. Do not issue an automatic diagonal recovery.'
            )
            return False

        if not self.gripper_grip(book_thickness_mm):
            return False
        if not self.gripper_high_voltage_on():
            return False
        with self.gripper_lock:
            if self.gripper_hold_lost:
                return False

        spine_up_axis = self._locked_spine_up_axis(locked_rotation)
        if spine_up_axis is None:
            return False

        final_outward_m = (
            insertion_distance_m
            + self.extract_extra_m
        )
        if not self.move_locked_offset(
            locked_rotation,
            outward_m=final_outward_m,
            spine_lift_m=self.final_lift_m,
            spine_axis=spine_up_axis,
            planner_id=self.lin_planner_id,
            label='Final locked-quaternion extraction LIN',
        ):
            self.get_logger().error(
                'Final extraction failed. H/T remains active; automatic return is blocked.'
            )
            return False

        self.get_logger().info('H/T extraction complete. Returning to the exact recorded six joints.')
        if not self.return_to_recorded_joints(initial_joint_targets):
            self.get_logger().error(
                'Recorded-six-joint return failed. H/T remains active and final Y/O release is blocked.'
            )
            return False
        if self.confirm_before_release:
            input(
                'Robot reached the recorded start while H/T remains active. '
                'Confirm the book is supported and press Enter for final Y then O; '
                'Ctrl+C keeps the hold active...'
            )
        if not self.gripper_high_voltage_off():
            return False
        if not self.gripper_open():
            return False
        self.get_logger().info(
            'Book-grasp sequence complete: returned to the recorded six joints and sent final Y then O.'
        )
        return True


def main() -> None:
    rclpy.init()
    node: Optional[ManualBookGraspNode] = None
    try:
        node = ManualBookGraspNode()
        success = node.run()
        if not success:
            node.get_logger().error('Manual book grasp sequence failed')
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().warning('Cancelled by operator')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
