import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from manual_target import (
    normalize_quad_corners,
    pixel_rays_to_plane,
    project_points_to_pixels,
    projected_quad_mask,
    quad_mask,
)


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "src" / "tm5_manual_book_grasp" / "scripts"
PACKAGE_DIR = SCRIPTS_DIR.parent
COORDINATOR = SCRIPTS_DIR / "manual_book_grasp.py"
K = np.array([[910.0, 0.0, 640.0], [0.0, 910.0, 360.0], [0.0, 0.0, 1.0]])
IMAGE_SHAPE = (720, 1280, 3)


def _book_quad_on_plane(center, normal, spine, width=0.03, height=0.20):
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    spine = np.asarray(spine, float)
    spine -= np.dot(spine, normal) * normal
    spine /= np.linalg.norm(spine)
    right = np.cross(normal, spine)
    half_w, half_h = 0.5 * width, 0.5 * height
    return np.array([
        center - half_w * right - half_h * spine,
        center + half_w * right - half_h * spine,
        center + half_w * right + half_h * spine,
        center - half_w * right + half_h * spine,
    ])


def test_click_order_is_normalized_and_degenerate_quads_are_rejected():
    clicked = [(700, 100), (600, 500), (700, 500), (600, 100)]
    quad = normalize_quad_corners(clicked, IMAGE_SHAPE)
    assert quad is not None and set(quad) == set(clicked)
    assert normalize_quad_corners([(1, 1), (2, 2), (3, 3), (4, 4)], IMAGE_SHAPE) is None
    assert normalize_quad_corners([(0, 0), (10, 0), (10, 10)], IMAGE_SHAPE) is None
    assert normalize_quad_corners([(-1, 0), (10, 0), (10, 10), (0, 10)], IMAGE_SHAPE) is None
    with pytest.raises(ValueError):
        quad_mask(IMAGE_SHAPE, [(1, 1), (2, 2), (3, 3), (4, 4)])


def test_outline_rays_land_on_plane_and_project_back_to_the_clicked_pixels():
    plane_point = np.array([0.01, -0.02, 0.45])
    plane_normal = np.array([0.1, -0.05, 1.0])
    pixels = np.array([[600.0, 150.0], [690.0, 152.0], [688.0, 560.0], [602.0, 558.0]])
    points = pixel_rays_to_plane(pixels, K, plane_point, plane_normal)
    unit = plane_normal / np.linalg.norm(plane_normal)
    assert np.allclose((points - plane_point) @ unit, 0.0, atol=1.0e-12)
    assert np.allclose(project_points_to_pixels(points, K), pixels, atol=1.0e-9)


def test_plane_behind_camera_fails_closed():
    with pytest.raises(ValueError):
        pixel_rays_to_plane([[640.0, 360.0]], K, [0.0, 0.0, -0.4], [0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        project_points_to_pixels([[0.0, 0.0, -0.1]], K)


def test_projected_mask_clips_corners_outside_the_image():
    pixels = np.array([[500.0, -200.0], [700.0, -200.0], [700.0, 900.0], [500.0, 900.0]])
    mask = projected_quad_mask(IMAGE_SHAPE, pixels)
    assert mask.shape == IMAGE_SHAPE[:2]
    assert mask[0, 600] == 255 and mask[-1, 600] == 255 and mask[360, 400] == 0


def _coordinator_class():
    pytest.importorskip("rclpy")
    pytest.importorskip("cv_bridge")
    import manual_book_grasp

    return manual_book_grasp.ManualBookGraspNode


class _Logger:
    def __init__(self):
        self.errors = []

    def error(self, text):
        self.errors.append(text)

    info = warning = error


def _fake_node(cls, *, outline_base, base_from_camera):
    logger = _Logger()
    fake = SimpleNamespace(
        base_frame="base",
        minimum_mask_pixels=500,
        outline_border_margin_px=2,
        target_outline_base=outline_base,
        get_logger=lambda: logger,
    )
    rotation, translation = base_from_camera

    def lookup(target_frame, source_frame, _stamp):
        # Only camera <- base is needed for re-projection.
        assert (target_frame, source_frame) == ("camera_color_optical_frame", "base")
        inverse = rotation.inv()
        return inverse, -inverse.apply(translation)

    fake._lookup_rigid_transform = lookup
    fake._check_target_mask = lambda mask, *, context: cls._check_target_mask(
        fake, mask, context=context
    )
    return fake, logger


def _bundle():
    header = SimpleNamespace(frame_id="camera_color_optical_frame", stamp=None)
    return SimpleNamespace(
        color_bgr=np.zeros(IMAGE_SHAPE, np.uint8),
        color_info=SimpleNamespace(k=K.reshape(-1).tolist()),
        color_msg=SimpleNamespace(header=header),
    )


def test_base_frame_outline_reprojects_through_a_moved_camera():
    cls = _coordinator_class()
    # Initial camera at 0.45 m; the outline is anchored in camera 1 and stored
    # in base. Camera 2 (stand-off) is 0.25 m closer and slightly yawed.
    base_from_cam1 = (Rotation.from_euler("xyz", [0.0, 0.0, 0.0]), np.zeros(3))
    corners_cam1 = _book_quad_on_plane([0.0, 0.0, 0.45], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0])
    outline_base = base_from_cam1[0].apply(corners_cam1) + base_from_cam1[1]
    base_from_cam2 = (Rotation.from_euler("y", 4.0, degrees=True), np.array([0.004, 0.0, 0.25]))
    fake, logger = _fake_node(cls, outline_base=outline_base, base_from_camera=base_from_cam2)

    result = cls.project_target_outline_mask(fake, _bundle(), context="Stand-off")
    assert result is not None, logger.errors
    mask, outline_px = result

    inverse = base_from_cam2[0].inv()
    expected_cam2 = inverse.apply(outline_base - base_from_cam2[1])
    assert np.allclose(outline_px, project_points_to_pixels(expected_cam2, K))
    # At 0.20 m the 30-mm book spans ~136 px and the 200-mm height leaves the
    # image vertically; only top/bottom clipping is acceptable.
    columns = np.flatnonzero(mask.any(axis=0))
    assert 120 < columns[-1] - columns[0] < 160
    assert mask[0].any() and mask[-1].any()


def test_outline_touching_left_or_right_border_is_rejected():
    cls = _coordinator_class()
    corners = _book_quad_on_plane([0.13, 0.0, 0.20], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0])
    fake, logger = _fake_node(
        cls,
        outline_base=corners,
        base_from_camera=(Rotation.identity(), np.zeros(3)),
    )
    assert cls.project_target_outline_mask(fake, _bundle(), context="Stand-off") is None
    assert any("left/right image border" in text for text in logger.errors)


def test_coordinator_has_no_language_or_recognition_interface():
    source = COORDINATOR.read_text()
    for forbidden in (
        "/book_grasp/perception",
        "request_perception",
        "Book name/keyword",
        "use_local_row_consensus",
        "instance_labels",
        "query_text",
    ):
        assert forbidden not in source, forbidden


def test_only_the_first_initial_frame_asks_the_operator():
    source = COORDINATOR.read_text()
    detect = source[source.index("    def detect_book("):source.index("# TF and poses")]
    assert "if initial:\n            self.target_outline_base = None" in detect
    assert "if self.target_outline_base is None:\n                target = self.select_initial_target_mask(bundle)" in detect
    assert "self.project_target_outline_mask(" in detect
    run = source[source.index("    def run(self)"):]
    assert run.count("self.detect_book(") == 2
    assert "initial=True" in run and "initial=False" in run
    fresh = run[run.index("fresh = self.detect_book("):run.index("if fresh is None")]
    assert "use_aruco_orientation" not in fresh


def test_yaml_matches_node_and_keeps_motion_disabled():
    config = yaml.safe_load((PACKAGE_DIR / "config" / "manual_grasp.yaml").read_text())
    assert list(config) == ["manual_book_grasp"]
    params = config["manual_book_grasp"]["ros__parameters"]
    source = COORDINATOR.read_text()
    assert "super().__init__('manual_book_grasp')" in source
    assert "declare_parameter('motion_enabled', False)" in source
    assert params["motion_enabled"] is False
    assert params["confirm_before_motion"] is True
    assert params["confirm_before_release"] is True
    assert params["tf_buffer_cache_s"] >= params["manual_selection_timeout_s"] + 30.0
    declared = set(re.findall(r"declare_parameter\(\s*'([a-z0-9_]+)'", source))
    assert set(params) <= declared, sorted(set(params) - declared)


def test_package_installs_every_script_it_launches_or_imports():
    cmake = (PACKAGE_DIR / "CMakeLists.txt").read_text()
    launch = (PACKAGE_DIR / "launch" / "tm5-700_with_gripper.launch.py").read_text()
    for executable in re.findall(r'executable="([a-z_]+\.py)"', launch):
        assert f"scripts/{executable}" in cmake, executable
    for module in ("aruco_geometry.py", "manual_target.py"):
        assert f"scripts/{module}" in cmake
    for xacro in (PACKAGE_DIR / "urdf").glob("*.xacro"):
        text = xacro.read_text()
        assert "grasp_book" not in text.replace("tm5_manual_book_grasp", "")
        for mesh in re.findall(r"package://tm5_manual_book_grasp/([^\"]+)", text):
            assert (PACKAGE_DIR / mesh).is_file(), mesh
