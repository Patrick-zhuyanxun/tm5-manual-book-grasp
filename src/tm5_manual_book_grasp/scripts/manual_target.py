"""Operator four-corner target selection and its metric re-projection.

The operator outlines the target book once on the initial synchronized RGB-D
frame. The outline is anchored on the fitted book-front plane and stored in the
robot base frame. Every later observation (ArUco retry frames and the stand-off
validation frame) re-projects that metric outline through capture-time TF and
the live CameraInfo instead of asking the operator again.

Because the re-projected outline is fixed in the base frame, its right edge
reproduces the initial seam. The stand-off observation is therefore a fresh
depth/plane/thickness validation gate whose lateral correction is expected to
be near zero; it does not add new seam information.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import cv2
import numpy as np


def normalize_quad_corners(corners, image_shape):
    """Return one convex quadrilateral sorted around its centroid, or ``None``."""
    height, width = image_shape[:2]
    points = np.asarray(corners, dtype=float)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None
    points = np.rint(points).astype(np.int32)
    if (
        np.any(points[:, 0] < 0)
        or np.any(points[:, 0] >= width)
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] >= height)
    ):
        return None
    if len({(int(x), int(y)) for x, y in points}) != 4:
        return None

    # The operator may click corners in any order. Sort around the centroid to
    # produce a non-self-intersecting contour before checking convexity.
    center = points.astype(float).mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    contour = points.reshape((-1, 1, 2))
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 25.0:
        return None
    return tuple((int(x), int(y)) for x, y in points)


def quad_mask(image_shape, quad) -> np.ndarray:
    """Rasterize an in-image operator quad to a 0/255 ``uint8`` mask."""
    normalized = normalize_quad_corners(quad, image_shape)
    if normalized is None:
        raise ValueError('target outline is not a valid convex four-corner quad')
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(normalized, dtype=np.int32)], 255)
    return mask


def projected_quad_mask(image_shape, pixels) -> np.ndarray:
    """Rasterize a projected quad; parts outside the image are clipped.

    Unlike :func:`quad_mask`, projected corners may legitimately leave the
    image at close stand-off range. The caller must still enforce a minimum
    visible pixel count.
    """
    points = np.asarray(pixels, dtype=float)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError('projected outline must contain four finite pixels')
    contour = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
    if not cv2.isContourConvex(contour):
        raise ValueError('projected outline is not convex')
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour.reshape(-1, 2)], 255)
    return mask


def pixel_rays_to_plane(pixels, camera_matrix, plane_point, plane_normal) -> np.ndarray:
    """Intersect pinhole pixel rays with a camera-frame plane.

    Returns an ``N x 3`` array of metric camera-frame points. Distortion is
    ignored, matching the coordinator's depth back-projection.
    """
    intrinsic = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    point = np.asarray(plane_point, dtype=float).reshape(3)
    normal = np.asarray(plane_normal, dtype=float).reshape(3)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1.0e-9:
        raise ValueError('plane normal must be nonzero')
    normal = normal / normal_norm
    uv = np.asarray(pixels, dtype=float).reshape(-1, 2)
    rays = np.column_stack((
        (uv[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0],
        (uv[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1],
        np.ones(len(uv)),
    ))
    denominators = rays @ normal
    if np.any(np.abs(denominators) < 1.0e-6):
        raise ValueError('an outline corner ray is parallel to the book plane')
    scales = float(np.dot(point, normal)) / denominators
    if np.any(scales <= 0.0):
        raise ValueError('an outline corner intersects the book plane behind the camera')
    return rays * scales[:, None]


def project_points_to_pixels(points_camera, camera_matrix, *, minimum_depth_m=0.02) -> np.ndarray:
    """Project camera-frame points with the pinhole model."""
    intrinsic = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    points = np.asarray(points_camera, dtype=float).reshape(-1, 3)
    if np.any(points[:, 2] <= float(minimum_depth_m)):
        raise ValueError('a projected outline corner is behind or too close to the camera')
    return np.column_stack((
        intrinsic[0, 0] * points[:, 0] / points[:, 2] + intrinsic[0, 2],
        intrinsic[1, 1] * points[:, 1] / points[:, 2] + intrinsic[1, 2],
    ))


def select_manual_book_quad(
    window_name: str,
    image: np.ndarray,
    *,
    timeout_s: Optional[float] = None,
    overlay_lines: Sequence[str] = (),
):
    """Interactively collect a convex four-corner book outline.

    Left click adds a corner; right click or Backspace undoes the most recent
    point. Enter/Space accepts; C/Esc cancels. Returns ``None`` on cancel or
    timeout. Click order is normalized at acceptance.
    """
    points = []
    message = 'Left-click 4 corners. Right-click undo. Enter accept; C/Esc cancel.'
    started = time.monotonic()

    def mouse_callback(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    try:
        while True:
            if timeout_s is not None and time.monotonic() - started >= timeout_s:
                return None
            canvas = image.copy()
            if len(points) >= 2:
                cv2.polylines(
                    canvas, [np.asarray(points, dtype=np.int32)], False,
                    (0, 255, 255), 2, cv2.LINE_AA,
                )
            if len(points) == 4:
                cv2.polylines(
                    canvas, [np.asarray(points, dtype=np.int32)], True,
                    (0, 255, 0), 2, cv2.LINE_AA,
                )
            for index, point in enumerate(points, start=1):
                cv2.circle(canvas, point, 5, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    canvas, str(index), (point[0] + 7, point[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA,
                )
            for row, text in enumerate((message, *overlay_lines)):
                cv2.putText(
                    canvas, text, (12, 30 + 26 * row), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2, cv2.LINE_AA,
                )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord('c'), ord('C')):
                return None
            if key in (8, 127, ord('r'), ord('R')) and points:
                points.pop()
                continue
            if key in (13, 32):
                quad = normalize_quad_corners(points, image.shape)
                if quad is not None:
                    return quad
                message = 'Need four distinct convex corners inside the image. Right-click/Backspace undo.'
    finally:
        cv2.setMouseCallback(window_name, lambda *_args: None)
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)
