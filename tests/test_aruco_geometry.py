from pathlib import Path
import sys

import cv2
import numpy as np
import pytest


SOURCE_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "tm5_manual_book_grasp" / "scripts"
)
sys.path.insert(0, str(SOURCE_DIR))

from aruco_geometry import (
    ArucoObservationError,
    constrain_marker_axis_to_book_spine,
    compute_locked_side_correction,
    estimate_aruco_insertion_axis,
    select_aruco_ground_truth_axes,
)


def _front_facing_marker(marker_id=0):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # OpenCV >= 4.7 renamed drawMarker to generateImageMarker.
    generate = getattr(cv2.aruco, "generateImageMarker", None) or cv2.aruco.drawMarker
    marker = generate(dictionary, marker_id, 200)
    canvas = np.full((600, 600), 255, dtype=np.uint8)
    canvas[200:400, 200:400] = marker
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def test_id_zero_50mm_marker_uses_directed_negative_marker_z():
    estimate = estimate_aruco_insertion_axis(
        _front_facing_marker(0),
        [600.0, 0.0, 300.0, 0.0, 600.0, 300.0, 0.0, 0.0, 1.0],
        np.zeros(5),
        dictionary_name="DICT_4X4_50",
        marker_id=0,
        marker_length_m=0.050,
    )
    assert estimate.marker_id == 0
    assert estimate.tvec_m[2] == pytest.approx(0.15, abs=0.01)
    assert estimate.insertion_axis_camera == pytest.approx([0.0, 0.0, 1.0], abs=0.02)
    assert estimate.reprojection_error_px <= 2.0


def test_wrong_marker_id_is_a_retryable_observation_failure():
    with pytest.raises(ArucoObservationError, match="id 0"):
        estimate_aruco_insertion_axis(
            _front_facing_marker(1),
            [600.0, 0.0, 300.0, 0.0, 600.0, 300.0, 0.0, 0.0, 1.0],
            np.zeros(5),
            marker_id=0,
            marker_length_m=0.050,
        )


def test_marker_axis_is_projected_to_seam_constraint_without_sign_flip():
    tool_z, tool_y, correction_deg = constrain_marker_axis_to_book_spine(
        [0.20, 0.10, 0.97], [0.0, 1.0, 0.0]
    )
    assert np.dot(tool_z, tool_y) == pytest.approx(0.0, abs=1.0e-12)
    assert np.dot(tool_z, [0.20, 0.10, 0.97]) > 0.0
    assert tool_z[2] > 0.0
    assert correction_deg > 0.0


def test_marker_axis_parallel_to_spine_fails_closed():
    with pytest.raises(ValueError, match="parallel"):
        constrain_marker_axis_to_book_spine([0.0, 1.0, 0.0], [0.0, 1.0, 0.0])


def test_book_spine_angle_is_used_when_perpendicular_error_is_within_10deg():
    angle = np.radians(9.0)
    detected_spine = [0.0, np.cos(angle), np.sin(angle)]
    tool_z, tool_y, error_deg, source = select_aruco_ground_truth_axes(
        [0.0, 0.0, 1.0],
        detected_spine,
        book_spine_trust_max_error_deg=10.0,
    )
    assert source == "book_spine"
    assert error_deg == pytest.approx(9.0)
    assert tool_y == pytest.approx(detected_spine)
    assert np.dot(tool_z, tool_y) == pytest.approx(0.0, abs=1.0e-12)
    assert not np.allclose(tool_z, [0.0, 0.0, 1.0])


def test_book_spine_angle_is_used_at_exactly_10deg_boundary():
    angle = np.radians(10.0)
    _, _, error_deg, source = select_aruco_ground_truth_axes(
        [0.0, 0.0, 1.0],
        [0.0, np.cos(angle), np.sin(angle)],
        book_spine_trust_max_error_deg=10.0,
    )
    assert error_deg == pytest.approx(10.0)
    assert source == "book_spine"


def test_aruco_z_is_preserved_when_spine_error_exceeds_10deg():
    angle = np.radians(20.0)
    detected_spine = [0.0, np.cos(angle), np.sin(angle)]
    tool_z, tool_y, error_deg, source = select_aruco_ground_truth_axes(
        [0.0, 0.0, 1.0],
        detected_spine,
        book_spine_trust_max_error_deg=10.0,
    )
    assert source == "aruco_ground_truth"
    assert error_deg == pytest.approx(20.0)
    assert tool_z == pytest.approx([0.0, 0.0, 1.0])
    assert tool_y == pytest.approx([0.0, 1.0, 0.0])
    assert np.dot(tool_z, tool_y) == pytest.approx(0.0, abs=1.0e-12)


def test_fresh_height_and_depth_are_ignored_for_second_stage_correction():
    result = compute_locked_side_correction(
        [0.0, 0.0, 0.0],
        [0.002, 0.040, -0.003],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    )
    assert result['side_signed_m'] == pytest.approx(0.002)
    assert result['side_vector'] == pytest.approx([0.002, 0.0, 0.0])
    assert result['vertical_signed_m'] == pytest.approx(0.040)
    assert result['axial_signed_m'] == pytest.approx(-0.003)


def test_simple_coordinator_keeps_correction_out_of_lin_path():
    source = (
        SOURCE_DIR / "manual_book_grasp.py"
    ).read_text()
    correction_start = source.index("One-shot locked-seam adjustment")
    correction_end = source.index("if not self.gripper_set_gap", correction_start)
    correction_block = source[correction_start:correction_end]
    assert "self.ptp_planner_id" in correction_block
    assert "self.lin_planner_id" not in correction_block


def test_simple_coordinator_inserts_once_then_ht_and_lin_extracts():
    source = (SOURCE_DIR / "manual_book_grasp.py").read_text()
    sequence_start = source.index(
        "if not self.gripper_grip(book_thickness_mm):",
        source.index("def run("),
    )
    sequence_end = source.index("H/T extraction complete", sequence_start)
    sequence = source[sequence_start:sequence_end]
    positions = [
        sequence.index("gripper_grip(book_thickness_mm)"),
        sequence.index("gripper_high_voltage_on"),
        sequence.index("Final locked-quaternion extraction LIN"),
    ]
    assert positions == sorted(positions)
    assert sequence.count("gripper_grip(book_thickness_mm)") == 1
    assert "First-grip outward/lift verification PTP" not in sequence
    assert "gripper_command_gap(open_gap_mm)" not in sequence
    assert "def gripper_command_gap" not in source
    assert "Downward compensation PTP" not in sequence
    assert "Second-grip locked-axis LIN reinsert" not in sequence
    assert "regrip_reinsert_m" not in source
    assert "regrip_extract_m" not in source
    assert "regrip_lift_m" not in source
    final_extract_start = source.index(
        "Final locked-quaternion extraction LIN",
        sequence_start,
    )
    final_extract_start = source.rfind(
        "if not self.move_locked_offset(",
        sequence_start,
        final_extract_start,
    )
    final_extract_end = source.index("H/T extraction complete", final_extract_start)
    final_extract_block = source[final_extract_start:final_extract_end]
    assert "planner_id=self.lin_planner_id" in final_extract_block
    assert "planner_id=self.ptp_planner_id" not in final_extract_block
    hold_t = source.index("gripper_high_voltage_on()", sequence_start)
    final_y = source.index("gripper_high_voltage_off()", sequence_start)
    assert hold_t < final_y
    assert "retreat_pose" not in sequence
    assert "gripper_open" not in sequence


def test_simple_coordinator_retries_distinct_collision_aware_ptp_candidates():
    source = (SOURCE_DIR / "manual_book_grasp.py").read_text()
    assert "declare_parameter('planning_retry_count', 3)" in source
    assert "from moveit_msgs.srv import GetPositionIK" in source
    assert "request.ik_request.avoid_collisions = True" in source
    assert "_same_joint_solution(signature, known)" in source
    assert "goal.request.goal_constraints = [self._build_exact_joint_constraints(joint_target)]" in source
    assert "maximum_candidates = self.planning_retry_count + 1" in source
    assert "Rejected IK target before Pilz" in source
    assert "self._validate_planned_wrist(trajectory, self.ptp_planner_id)" in source
    planner = source[
        source.index("def _plan_ptp_with_retries"):
        source.index("def _plan_pose_goal_once")
    ]
    assert "eligible_plans.append" in planner
    assert "min(\n                eligible_plans" in planner
    assert "_trajectory_total_joint_travel(plan[1], measured)" in planner
    assert "SELECTED SHORTEST" in planner
    assert "if plan is not None:\n                return plan" not in planner


def test_simple_coordinator_waits_before_fresh_capture_and_targets_15mm_up():
    source = (SOURCE_DIR / "manual_book_grasp.py").read_text()
    assert "declare_parameter('fine_perception_settle_s', 1.5)" in source
    assert "declare_parameter('target_spine_up_offset_m', 0.015)" in source
    geometry = source[source.index("seam_center ="):source.index("tool_y = row_spine")]
    assert "+ self.target_spine_up_offset_m * target_spine" in geometry
    run = source[source.index("def run("):]
    settle = run.index("settle_deadline = time.monotonic()")
    timestamp = run.index("observation_after_motion_ns = self.get_clock().now().nanoseconds")
    fresh = run.index("fresh = self.detect_book(")
    assert settle < timestamp < fresh


def test_simple_coordinator_retries_only_aruco_observations_on_newer_frames():
    source = (SOURCE_DIR / "manual_book_grasp.py").read_text()
    assert "declare_parameter('aruco_retry_count', 3)" in source
    assert "except ArucoObservationError as exc:" in source
    assert "except ValueError as exc:" in source
    assert "next_stamp_ns = self._stamp_to_ns(bundle.color_msg.header.stamp) + 1" in source
    assert "Retrying with a newer" in source
    detect = source[source.index("def detect_book("):source.index("# TF and poses")]
    assert "total_attempts = retry_count + 1" in detect
    assert "retry_count = self.aruco_retry_count if use_aruco_orientation else 0" in detect


def test_marker_translation_is_documented_as_diagnostic_only():
    source = (SOURCE_DIR / "aruco_geometry.py").read_text()
    coordinator = (SOURCE_DIR / "manual_book_grasp.py").read_text()
    assert "marker supplies direction only" in source
    assert "book seam remains the insertion position reference" in coordinator
    assert "insertion_axis = -rotation_camera_marker[:, 2]" in source
