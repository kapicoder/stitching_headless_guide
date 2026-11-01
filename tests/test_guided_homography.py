"""Homography registration, projective warping and mesh integration tests."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from algorithms.guided_affine import (
    GUIDED_SETTING_KEYS,
    GuidedAffineStitcher,
    _points,
    build_guided_settings,
)
from algorithms.opencv_stitcher import GuidedProgressStitcher
from config import DEFAULT_SETTINGS, STITCHER_SETTING_KEYS
from stitching_headless_api import app
from test_guided_affine import _images, _write_pair


def test_homography_is_the_guided_default_and_invalid_models_are_rejected():
    assert DEFAULT_SETTINGS["guided_transform_model"] == "homography"
    assert build_guided_settings({"algorithm": "guided_affine"})[
        "guided_transform_model"
    ] == "homography"
    assert build_guided_settings({
        "algorithm": "guided_affine",
        "guided_transform_model": "AFFINE",
    })["guided_transform_model"] == "affine"
    with pytest.raises(ValueError, match="guided_transform_model"):
        build_guided_settings({
            "algorithm": "guided_affine",
            "guided_transform_model": "similarity",
        })


def test_homography_requires_four_minimum_inliers():
    payload = {
        "algorithm": "guided_affine",
        "settings": {
            "guided_transform_model": "homography",
            "guided_min_inliers": 3,
        },
    }
    with pytest.raises(ValueError, match="guided_min_inliers 必须至少为4"):
        build_guided_settings(payload)
    with pytest.raises(ValueError, match="guided_min_inliers 必须至少为4"):
        GuidedAffineStitcher(
            guided_transform_model="homography",
            guided_min_inliers=3,
        )
    assert build_guided_settings({
        "algorithm": "guided_affine",
        "settings": {
            "guided_transform_model": "affine",
            "guided_min_inliers": 3,
        },
    })["guided_min_inliers"] == 3


def test_homography_propagation_maps_source_coordinates_to_reference_canvas():
    engine = GuidedAffineStitcher(
        show_progress=False,
        guided_transform_model="homography",
    )
    initial_a = np.eye(2, 3, dtype=np.float64)
    initial_b = np.array([[1.0, 0.0, 180.0], [0.0, 1.0, 0.0]])
    items = [
        {
            "size": (300, 200),
            "initial_matrix": matrix,
            "matrix": matrix.copy(),
            "correction_status": "fallback_initial",
        }
        for matrix in (initial_a, initial_b)
    ]
    source_b_to_a = np.array([
        [1.0, 0.01, -10.0],
        [-0.005, 1.0, 4.0],
        [2e-5, -1e-5, 1.0],
    ])

    engine._global_homography_adjust(items, [(0, 1, source_b_to_a, 40)])

    assert items[1]["correction_status"] == "refined"
    points = np.array([[10.0, 20.0], [120.0, 90.0], [270.0, 180.0]])
    np.testing.assert_allclose(
        _points(items[1]["matrix"], points),
        _points(source_b_to_a, points),
        atol=1e-8,
    )


def test_homography_global_adjustment_uses_all_edges_to_close_loops():
    engine = GuidedAffineStitcher(
        show_progress=False,
        guided_transform_model="homography",
    )
    items = [
        {
            "size": (400, 300),
            "initial_matrix": np.eye(2, 3, dtype=np.float64),
            "matrix": np.eye(2, 3, dtype=np.float64),
            "correction_status": "fallback_initial",
        }
        for _ in range(3)
    ]
    grid = np.array([
        (x, y)
        for y in (20.0, 80.0, 140.0, 200.0)
        for x in (10.0, 70.0, 130.0, 190.0)
    ])

    def translation(x):
        return np.array([
            [1.0, 0.0, x],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

    pairs = []
    for image_a, image_b, shift, inliers in (
        (0, 1, 100.0, 100),
        (1, 2, 100.0, 90),
        (0, 2, 180.0, 80),
    ):
        points_b = grid.copy()
        points_a = points_b + [shift, 0.0]
        pairs.append((
            image_a, image_b, translation(shift), inliers, points_a, points_b,
        ))

    engine._global_homography_adjust(items, pairs)

    mean_residuals = []
    for image_a, image_b, _, _, points_a, points_b in pairs:
        residual = np.linalg.norm(
            _points(items[image_a]["matrix"], points_a)
            - _points(items[image_b]["matrix"], points_b),
            axis=1,
        )
        mean_residuals.append(float(residual.mean()))
    assert all(item["correction_status"] in {"reference", "refined"} for item in items)
    assert max(mean_residuals) < 10.0


def test_homography_registration_passes_inliers_into_global_loop_adjustment(
    tmp_path, monkeypatch,
):
    engine = GuidedAffineStitcher(
        show_progress=False,
        guided_transform_model="homography",
    )
    image_paths = []
    image_corners = [[0, 0], [400, 0], [400, 300], [0, 300]]
    for index in range(3):
        path = tmp_path / f"loop_{index}.png"
        assert cv2.imwrite(str(path), np.zeros((300, 400, 3), dtype=np.uint8))
        image_paths.append(str(path))

    inlier_points_b = np.array([
        (x, y)
        for y in (20.0, 80.0, 140.0, 200.0)
        for x in (10.0, 70.0, 130.0, 190.0)
    ])
    outlier_points_b = np.array([
        (250.0, 30.0), (280.0, 90.0), (310.0, 150.0), (340.0, 210.0),
    ])
    points_b = np.concatenate((inlier_points_b, outlier_points_b))
    shifts = {(0, 1): (100.0, 16), (1, 2): (100.0, 16), (0, 2): (180.0, 16)}

    def fake_match_pair(a, b, _polygon_a, _polygon_b, _norm):
        image_a = image_paths.index(a["path"])
        image_b = image_paths.index(b["path"])
        shift, inlier_count = shifts[(image_a, image_b)]
        points_a = points_b + [shift, 0.0]
        points_a[16:] += [45.0, -30.0]
        homography = np.array([
            [1.0, 0.0, shift],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        inliers = np.arange(len(points_b)) < inlier_count
        return (
            {
                "status": "matched",
                "matches": len(points_b),
                "inliers": inlier_count,
                "rmse_pixels": 0.0,
            },
            {
                "homography_b_to_a": homography,
                "source_points_a": points_a,
                "source_points_b": points_b.copy(),
                "inliers": inliers,
            },
        )

    monkeypatch.setattr(engine, "_match_pair", fake_match_pair)
    images = [
        {"path": path, "corners": image_corners}
        for path in image_paths
    ]

    items = engine.register(images)

    assert engine.registration_status == "full"
    assert all(pair["status"] == "refined" for pair in engine.pair_diagnostics)
    residuals = []
    for image_a, image_b, (shift, _) in (
        (0, 1, shifts[(0, 1)]),
        (1, 2, shifts[(1, 2)]),
        (0, 2, shifts[(0, 2)]),
    ):
        points_a = inlier_points_b + [shift, 0.0]
        residuals.extend(np.linalg.norm(
            _points(items[image_a]["matrix"], points_a)
            - _points(items[image_b]["matrix"], inlier_points_b),
            axis=1,
        ))
    assert max(residuals) < 15.0


def test_resolution_scaling_and_roi_translation_preserve_projective_geometry():
    item = {
        "size": (400, 300),
        "matrix": np.array([
            [1.0, 0.02, 80.0],
            [-0.01, 1.0, 30.0],
            [1e-4, -2e-5, 1.0],
        ]),
    }
    transform = GuidedProgressStitcher._guided_matrix(
        item, (200, 150), 0.5, origin=(12, 7)
    )
    points = np.array([[0.0, 0.0], [50.0, 40.0], [199.0, 149.0]])
    expected = _points(item["matrix"], points * 2) * 0.5 - [12, 7]
    np.testing.assert_allclose(_points(transform, points), expected, atol=1e-8)


def test_mesh_constraints_exclude_global_homography_outliers(monkeypatch):
    captured = {}

    def capture_constraints(tile_shapes, valid_masks, seam_masks, constraints, movable, **kwargs):
        captured["constraints"] = constraints
        return [None] * len(tile_shapes), {"status": "no_change"}

    monkeypatch.setattr("algorithms.opencv_stitcher.solve_seam_mesh", capture_constraints)
    stitcher = object.__new__(GuidedProgressStitcher)
    stitcher.images = SimpleNamespace(
        _get_scaler=lambda _resolution: SimpleNamespace(scale=1.0)
    )
    stitcher.geometry = SimpleNamespace(
        pair_diagnostics=[{"image_a": 0, "image_b": 1, "status": "refined"}],
        registration_pairs=[{
            "image_a": 0,
            "image_b": 1,
            "source_points_a": np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float64),
            "source_points_b": np.array([[2, 2], [20, 20], [4, 4]], dtype=np.float64),
            "global_inliers": np.array([True, False, True]),
        }],
    )
    stitcher.items = [
        {"matrix": np.eye(3), "correction_status": "reference"},
        {"matrix": np.eye(3), "correction_status": "refined"},
    ]
    stitcher.mesh_grid_spacing = 128
    stitcher.mesh_seam_band_ratio = 0.25
    stitcher.mesh_max_displacement_ratio = 0.02
    stitcher.mesh_min_matches = 12

    stitcher._fit_seam_mesh(
        [np.zeros((24, 24, 3), np.uint8)] * 2,
        [np.ones((24, 24), np.uint8)] * 2,
        [np.ones((24, 24), np.uint8)] * 2,
        [(0, 0), (0, 0)],
    )

    constraint = captured["constraints"][0]
    np.testing.assert_array_equal(constraint["points_a"], [[1, 1], [3, 3]])
    np.testing.assert_array_equal(constraint["points_b"], [[2, 2], [4, 4]])


def test_guided_api_runs_homography_with_mesh_and_reports_model(tmp_path):
    first, second = _write_pair(tmp_path)
    response = app.test_client().post("/stitch", json={
        "images": _images(first, second, 174),
        "output_path": str(tmp_path / "homography.png"),
        "show_progress": False,
        "settings": {
            "algorithm": "guided_affine",
            "georef_mode": "none",
            "final_megapix": -1,
            "guided_mesh_enabled": True,
        },
    })

    assert response.status_code == 200, response.get_json()
    result = response.get_json()
    assert result["transform_model"] == "homography"
    assert result["settings"]["guided_transform_model"] == "homography"
    assert result["registration_status"] == "full"
    assert result["debug_info"]["pair_diagnostics"][0]["status"] == "refined"
    assert result["debug_info"]["seam_mesh"]["enabled"] is True
    assert result["output_width"] == pytest.approx(420, abs=2)


def test_homography_mesh_pipeline_warps_projective_images(tmp_path):
    first, second = _write_pair(tmp_path)
    images = _images(first, second, 174)
    settings = {
        **DEFAULT_SETTINGS,
        "guided_transform_model": "homography",
        "guided_mesh_enabled": True,
        "final_megapix": -1,
        "georef_mode": "none",
    }
    stitcher = GuidedProgressStitcher(
        images,
        {key: settings[key] for key in GUIDED_SETTING_KEYS},
        **{
            key: settings[key]
            for key in STITCHER_SETTING_KEYS
            if key in settings
        },
        show_progress=False,
    )

    panorama = stitcher.stitch([str(first), str(second)])

    assert panorama.shape[1] == pytest.approx(420, abs=2)
    assert stitcher.geometry.transform_model == "homography"
    assert stitcher.mesh_diagnostics["enabled"] is True
    assert stitcher.items[1]["matrix"].shape == (3, 3)
