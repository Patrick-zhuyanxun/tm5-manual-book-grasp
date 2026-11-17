"""Validated ArUco pose estimation for the book insertion-axis reference."""

from dataclasses import dataclass

import cv2
import numpy as np


class ArucoObservationError(ValueError):
    """One timestamped ArUco observation failed a production gate."""


@dataclass(frozen=True)
class ArucoAxisEstimate:
    marker_id: int
    corners_px: np.ndarray
    center_px: np.ndarray
    rvec: np.ndarray
    tvec_m: np.ndarray
    insertion_axis_camera: np.ndarray
    reprojection_error_px: float


def constrain_marker_axis_to_book_spine(marker_axis, book_spine):
    """Project directed marker ``-Z`` perpendicular to the book-spine line."""
    axis = np.asarray(marker_axis, dtype=np.float64).reshape(3)
    spine = np.asarray(book_spine, dtype=np.float64).reshape(3)
    if not np.isfinite(axis).all() or not np.isfinite(spine).all():
        raise ArucoObservationError("ArUco axis and book spine must be finite")
    axis_norm = float(np.linalg.norm(axis))
    spine_norm = float(np.linalg.norm(spine))
    if axis_norm < 1.0e-9 or spine_norm < 1.0e-9:
        raise ArucoObservationError("ArUco axis and book spine must be nonzero")
    axis /= axis_norm
    spine /= spine_norm
    projected = axis - float(np.dot(axis, spine)) * spine
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm < 1.0e-6:
        raise ArucoObservationError("ArUco insertion axis is parallel to the book spine")
    projected /= projected_norm
    correction_deg = float(
        np.degrees(np.arccos(np.clip(np.dot(axis, projected), -1.0, 1.0)))
    )
    # Orthogonal projection preserves the directed component by construction.
    if float(np.dot(axis, projected)) <= 0.0:
        raise ArucoObservationError("ArUco insertion-axis projection reversed direction")
    return projected, spine, correction_deg


def select_aruco_ground_truth_axes(
    marker_axis, book_spine, *, book_spine_trust_max_error_deg=10.0
):
    """Select tool Z/Y while treating directed ArUco ``-Z`` as ground truth.

    When the detected spine is already perpendicular to marker ``-Z`` within
    the configured error, let the spine define tool Y and minimally project
    marker Z onto its perpendicular plane. Otherwise preserve marker ``-Z``
    exactly and project only the detected spine to obtain an orthogonal tool Y.
    Marker X/Y are intentionally not used because their physical alignment to
    the book spine is not part of the marker contract.
    """
    if not 0.0 <= float(book_spine_trust_max_error_deg) <= 45.0:
        raise ValueError("book_spine_trust_max_error_deg must be within [0, 45]")
    spine_tool_z, normalized_spine, error_deg = (
        constrain_marker_axis_to_book_spine(marker_axis, book_spine)
    )
    if error_deg <= float(book_spine_trust_max_error_deg) + 1.0e-9:
        return spine_tool_z, normalized_spine, error_deg, "book_spine"

    tool_z = np.asarray(marker_axis, dtype=np.float64).reshape(3)
    tool_z /= max(float(np.linalg.norm(tool_z)), 1.0e-12)
    tool_y = normalized_spine - float(np.dot(normalized_spine, tool_z)) * tool_z
    tool_y_norm = float(np.linalg.norm(tool_y))
    if tool_y_norm < 1.0e-6:
        raise ArucoObservationError(
            "book spine is parallel to ArUco ground-truth insertion axis"
        )
    tool_y /= tool_y_norm
    return tool_z, tool_y, error_deg, "aruco_ground_truth"


def compute_locked_side_correction(
    initial_target_position,
    fresh_target_position,
    locked_side_axis,
    locked_spine_axis,
    locked_insertion_axis,
):
    """Keep only fresh displacement along locked tool ``+X``.

    Fresh target height, axial depth, and any residual component are diagnostic
    only. This mirrors the production ``planar_seam_target`` contract and
    prevents re-estimated book-spine sign/height offsets from commanding a
    large second-stage correction.
    """
    initial = np.asarray(initial_target_position, dtype=np.float64).reshape(3)
    fresh = np.asarray(fresh_target_position, dtype=np.float64).reshape(3)
    side = np.asarray(locked_side_axis, dtype=np.float64).reshape(3)
    spine = np.asarray(locked_spine_axis, dtype=np.float64).reshape(3)
    insertion = np.asarray(locked_insertion_axis, dtype=np.float64).reshape(3)
    vectors = (initial, fresh, side, spine, insertion)
    if not all(np.isfinite(vector).all() for vector in vectors):
        raise ValueError("seam correction vectors must be finite")
    side /= max(float(np.linalg.norm(side)), 1.0e-12)
    spine /= max(float(np.linalg.norm(spine)), 1.0e-12)
    insertion /= max(float(np.linalg.norm(insertion)), 1.0e-12)
    raw_delta = fresh - initial
    side_signed = float(np.dot(raw_delta, side))
    vertical_signed = float(np.dot(raw_delta, spine))
    axial_signed = float(np.dot(raw_delta, insertion))
    side_vector = side_signed * side
    ignored_vector = raw_delta - side_vector
    return {
        "raw_delta": raw_delta,
        "side_signed_m": side_signed,
        "side_vector": side_vector,
        "vertical_signed_m": vertical_signed,
        "axial_signed_m": axial_signed,
        "ignored_vector": ignored_vector,
        "ignored_m": float(np.linalg.norm(ignored_vector)),
        "raw_m": float(np.linalg.norm(raw_delta)),
    }


def estimate_aruco_insertion_axis(
    color_bgr,
    camera_matrix,
    distortion_coefficients,
    *,
    dictionary_name="DICT_4X4_50",
    marker_id=0,
    marker_length_m=0.050,
    maximum_reprojection_error_px=2.0,
    minimum_side_px=20.0,
):
    """Return marker ``-Z`` in camera coordinates, or raise ``ValueError``.

    The marker supplies direction only. Its translation and image centre are
    diagnostic and must not be used as the book-seam insertion target.
    """
    image = np.asarray(color_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("ArUco input must be one BGR image")
    if not np.isfinite(marker_length_m) or marker_length_m <= 0.0:
        raise ValueError("aruco_marker_length_m must be positive")
    if not hasattr(cv2, "aruco"):
        raise ValueError("OpenCV was built without cv2.aruco")
    dictionary_id = getattr(cv2.aruco, str(dictionary_name), None)
    if dictionary_id is None:
        raise ValueError(f"unknown ArUco dictionary: {dictionary_name}")

    intrinsic = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(distortion_coefficients, dtype=np.float64).reshape(-1)
    if not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise ValueError("invalid CameraInfo intrinsic matrix")

    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(
            dictionary, cv2.aruco.DetectorParameters()
        )
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=parameters
        )
    if ids is None:
        raise ArucoObservationError(f"ArUco marker id {marker_id} was not detected")
    ids_flat = np.asarray(ids, dtype=int).reshape(-1)
    matches = np.flatnonzero(ids_flat == int(marker_id))
    if len(matches) != 1:
        raise ArucoObservationError(
            f"expected exactly one ArUco marker id {marker_id}; found {len(matches)}"
        )

    marker_corners = np.asarray(corners[int(matches[0])], dtype=np.float64).reshape(4, 2)
    side_lengths = np.linalg.norm(
        marker_corners - np.roll(marker_corners, -1, axis=0), axis=1
    )
    if float(np.min(side_lengths)) < float(minimum_side_px):
        raise ArucoObservationError(
            f"ArUco marker is too small: min side {float(np.min(side_lengths)):.1f} px"
        )

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        [marker_corners.astype(np.float32)],
        float(marker_length_m),
        intrinsic,
        distortion,
    )
    rvec = np.asarray(rvecs[0], dtype=np.float64).reshape(3)
    tvec = np.asarray(tvecs[0], dtype=np.float64).reshape(3)
    rotation_camera_marker, _ = cv2.Rodrigues(rvec)
    insertion_axis = -rotation_camera_marker[:, 2]
    insertion_axis /= max(float(np.linalg.norm(insertion_axis)), 1.0e-12)

    half = 0.5 * float(marker_length_m)
    object_corners = np.array(
        [[-half, half, 0.0], [half, half, 0.0],
         [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )
    projected, _ = cv2.projectPoints(
        object_corners, rvec, tvec, intrinsic, distortion
    )
    projected = np.asarray(projected, dtype=np.float64).reshape(4, 2)
    reprojection_error = float(
        np.sqrt(np.mean(np.sum((projected - marker_corners) ** 2, axis=1)))
    )
    if not np.isfinite(reprojection_error) or reprojection_error > float(maximum_reprojection_error_px):
        raise ArucoObservationError(
            f"ArUco reprojection error {reprojection_error:.2f} px exceeds "
            f"{float(maximum_reprojection_error_px):.2f} px"
        )
    if not np.isfinite(tvec).all() or tvec[2] <= 0.0:
        raise ArucoObservationError("ArUco pose is not in front of the camera")

    return ArucoAxisEstimate(
        marker_id=int(marker_id),
        corners_px=marker_corners,
        center_px=np.mean(marker_corners, axis=0),
        rvec=rvec,
        tvec_m=tvec,
        insertion_axis_camera=insertion_axis,
        reprojection_error_px=reprojection_error,
    )


__all__ = [
    "ArucoObservationError",
    "ArucoAxisEstimate",
    "constrain_marker_axis_to_book_spine",
    "select_aruco_ground_truth_axes",
    "compute_locked_side_correction",
    "estimate_aruco_insertion_axis",
]
