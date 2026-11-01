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
