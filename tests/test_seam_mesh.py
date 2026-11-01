from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from algorithms.guided_affine import GUIDED_DEFAULT_SETTINGS, GUIDED_SETTING_KEYS
from algorithms.opencv_stitcher import GuidedProgressStitcher
from algorithms.seam_mesh import seam_influence, solve_seam_mesh, warp_with_mesh
from config import DEFAULT_SETTINGS, STITCHER_SETTING_KEYS, build_settings
from test_guided_affine import _images, _write_pair


def _split_masks(height: int, width: int):
    valid = np.full((height, width), 255, dtype=np.uint8)
    left = np.zeros_like(valid)
    right = np.zeros_like(valid)
    left[:, : width // 2] = 255
    right[:, width // 2 :] = 255
    return valid, left, right


def test_seam_influence_is_confined_to_a_band():
    valid, left, _ = _split_masks(120, 100)
    influence = seam_influence(valid, left, 20)
    assert influence[60, 50] > 0.9
    assert influence[60, 15] == pytest.approx(0.0)
    assert influence[60, 85] == pytest.approx(0.0)


def test_solver_fits_smooth_bounded_displacement_near_seam():
    height, width = 120, 100
    valid, seam_a, seam_b = _split_masks(height, width)
    y = np.linspace(18, 102, 20)
    points_a = np.column_stack((np.full(len(y), 50), y))
    points_b = np.column_stack((np.full(len(y), 42), y))
    constraints = [{
        "image_a": 0,
        "image_b": 1,
        "points_a": points_a,
        "points_b": points_b,
        "residuals": np.tile([-8.0, 0.0], (len(y), 1)),
    }]

    fields, info = solve_seam_mesh(
        [(height, width)] * 2,
        [valid, valid],
        [seam_a, seam_b],
        constraints,
        [True, True],
        grid_spacing=30,
        seam_band_ratio=0.2,
        max_displacement_ratio=0.15,
        min_matches=12,
    )

    assert info["status"] == "applied"
    assert info["matches_used"] == 20
    assert fields[0] is None
    assert fields[1] is not None
    assert info["images"][0]["status"] == "fixed_reference"
    assert info["images"][1]["status"] == "refined"
    assert info["images"][1]["max_displacement_pixels"] <= 15
    assert info["images"][1]["minimum_jacobian"] >= 0.25
    assert np.max(np.abs(fields[1]["dx"])) > 1
    assert np.all(fields[1]["dx"][[0, -1], :] == 0)
    assert np.all(fields[1]["dx"][:, [0, -1]] == 0)
    assert np.all(fields[1]["dy"][[0, -1], :] == 0)
    assert np.all(fields[1]["dy"][:, [0, -1]] == 0)


def test_solver_falls_back_when_seam_has_no_ownership_boundary():
    height, width = 80, 100
    valid = np.full((height, width), 255, dtype=np.uint8)
    points = np.array([[40.0, 40.0]] * 12)
    constraints = [{
        "image_a": 0,
        "image_b": 1,
        "points_a": points,
        "points_b": points,
        "residuals": np.tile([4.0, 0.0], (len(points), 1)),
    }]
    fields, info = solve_seam_mesh(
        [(height, width)] * 2,
        [valid, valid],
        [valid, valid],
        constraints,
        [True, True],
    )
    assert info["status"] == "fallback_insufficient_seam_matches"
    assert fields == [None, None]


def test_warp_with_mesh_uses_inverse_sampling_and_preserves_shape():
    image = np.tile(np.arange(32, dtype=np.uint8), (24, 1))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    mask = np.full(image.shape[:2], 255, dtype=np.uint8)
    field = {
        "dx": np.full((3, 3), 2, dtype=np.float32),
        "dy": np.zeros((3, 3), dtype=np.float32),
        "strength": np.ones(image.shape[:2], dtype=np.float32),
        "grid_x": np.linspace(0, image.shape[1] - 1, 3, dtype=np.float32),
        "grid_y": np.linspace(0, image.shape[0] - 1, 3, dtype=np.float32),
    }
    warped, warped_mask = warp_with_mesh(image, mask, field)
    assert warped.shape == image.shape
    assert warped_mask.shape == mask.shape
    assert warped[12, 16, 0] == image[12, 14, 0]
    assert warped_mask[12, 16] == 255

    high_image = cv2.resize(image, (64, 48), interpolation=cv2.INTER_NEAREST)
    high_mask = cv2.resize(mask, (64, 48), interpolation=cv2.INTER_NEAREST)
    high_warped, _ = warp_with_mesh(high_image, high_mask, field)
    assert high_warped[24, 32, 0] == high_image[24, 28, 0]


def test_final_mesh_mapping_accounts_for_low_and_output_canvas_origins():
    image = np.tile(np.arange(64, dtype=np.float32), (48, 1))
    mask = np.full(image.shape, 255, dtype=np.uint8)
    field = {
        "dx": np.full((3, 3), 4, dtype=np.float32),
        "dy": np.zeros((3, 3), dtype=np.float32),
        "strength": np.tile(
            np.linspace(0, 1, 32, dtype=np.float32), (24, 1)
        ),
        "grid_x": np.linspace(0, 31, 3, dtype=np.float32),
        "grid_y": np.linspace(0, 23, 3, dtype=np.float32),
    }

    warped, _ = warp_with_mesh(
        image,
        mask,
        field,
        low_to_output_scale=(2, 2),
        low_origin=(10, 0),
        output_origin=(18, 0),
    )

    low_x = (29 + 0.5 + 18) / 2 - 10 - 0.5
    expected = 29 - 8 * low_x / 31
    assert warped[24, 29] == pytest.approx(expected, abs=0.02)


@pytest.mark.parametrize(
    "key,value",
    [
        ("guided_mesh_grid_spacing", 15),
        ("guided_mesh_seam_band_ratio", 0.6),
        ("guided_mesh_max_displacement_ratio", 0.21),
        ("guided_mesh_min_matches", 7),
    ],
)
def test_invalid_mesh_settings_are_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        build_settings({"settings": {"algorithm": "stitcher", key: value}})


def test_mesh_default_matches_config_and_boolean_values_are_normalized():
    defaults = build_settings({"settings": {"algorithm": "stitcher"}})
    assert defaults["guided_mesh_enabled"] is DEFAULT_SETTINGS["guided_mesh_enabled"]
    enabled = build_settings({
        "settings": {"algorithm": "stitcher", "guided_mesh_enabled": "yes"}
    })
    assert enabled["guided_mesh_enabled"] is True


def test_guided_pipeline_reports_mesh_state_when_enabled(tmp_path: Path):
    first, second = _write_pair(tmp_path)
    images = _images(first, second, 170)
    settings = {
        **DEFAULT_SETTINGS,
        **GUIDED_DEFAULT_SETTINGS,
        "guided_mesh_enabled": True,
        "final_megapix": -1,
        "georef_mode": "none",
    }
    stitcher = GuidedProgressStitcher(
        images,
        {key: settings[key] for key in GUIDED_SETTING_KEYS},
        **{key: settings[key] for key in STITCHER_SETTING_KEYS if key in settings},
        show_progress=False,
    )
    panorama = stitcher.stitch([str(first), str(second)])
    diagnostics = stitcher.debug_info()["seam_mesh"]

    assert panorama is not None
    assert abs(panorama.shape[1] - 420) <= 2
    assert diagnostics["enabled"] is True
    assert diagnostics["status"] in {
        "applied", "no_change", "fallback_no_movable_images",
        "fallback_no_seam_support", "fallback_insufficient_seam_matches",
        "fallback_solver_failure", "fallback_mesh_too_large",
    }
