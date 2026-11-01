"""人工四角引导拼接的坐标、匹配和 API 回归测试。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from algorithms.guided_affine import (
    GUIDED_DEFAULT_SETTINGS,
    GUIDED_SETTING_KEYS,
    GuidedAffineStitcher,
    validate_images,
)
from algorithms.opencv_stitcher import GuidedProgressStitcher
from config import DEFAULT_SETTINGS, STITCHER_SETTING_KEYS
from image_io import guided_output_georeference
from stitching_headless_api import _build_settings, app


def _write_pair(tmp_path: Path, *, textured: bool = True):
    if textured:
        base = np.random.default_rng(7).integers(0, 256, (180, 420, 3), dtype=np.uint8)
    else:
        base = np.full((180, 420, 3), 120, np.uint8)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    assert cv2.imwrite(str(first), base[:, :250])
    assert cv2.imwrite(str(second), base[:, 170:420])
    return first, second


def _images(first: Path, second: Path, second_x: int):
    return [
        {"path": str(first), "corners": [[0, 0], [250, 0], [250, 180], [0, 180]]},
        {"path": str(second), "corners": [
            [second_x, 0], [second_x + 250, 0],
            [second_x + 250, 180], [second_x, 180],
        ]},
    ]


def test_guided_match_corrects_small_initial_error(tmp_path):
    first, second = _write_pair(tmp_path)
    stitcher = GuidedAffineStitcher(show_progress=False, final_megapix=-1)
    panorama = stitcher.stitch(_images(first, second, 174))
    assert panorama.shape[0] == 180
    assert abs(panorama.shape[1] - 420) <= 2
    assert stitcher.pair_diagnostics[0]["status"] == "refined"
    assert stitcher.pair_diagnostics[0]["inliers"] >= 8
    assert stitcher.output_offset == [0.0, 0.0]


def test_weak_texture_keeps_manual_position_and_negative_offset(tmp_path):
    first, second = _write_pair(tmp_path, textured=False)
    stitcher = GuidedAffineStitcher(show_progress=False, final_megapix=-1)
    panorama = stitcher.stitch(_images(first, second, -5))
    assert panorama.shape == (180, 255, 3)
    assert stitcher.output_offset == [5.0, 0.0]
    assert stitcher.pair_diagnostics[0]["status"] == "fallback_initial"


def test_feature_mask_is_confined_to_overlap(tmp_path):
    first, second = _write_pair(tmp_path)
    items = validate_images(_images(first, second, 170))
    area, polygon = GuidedAffineStitcher._overlap(items[0], items[1])
    assert area == 80 * 180
    items[0]["medium_mask"] = None
    mask = GuidedAffineStitcher._feature_mask(items[0], polygon, (180, 250))
    assert not mask[:, :169].any()
    assert mask[:, 172:248].all()


def test_search_margin_is_ten_percent_of_first_short_side(tmp_path):
    first, second = _write_pair(tmp_path)
    items = validate_images(_images(first, second, 170))
    margin = .10 * min(items[0]["size"])
    overlap, search_area, polygon_a, polygon_b = GuidedAffineStitcher._search_polygons(
        items[0], items[1], margin,
    )
    assert margin == 18
    assert overlap == 80 * 180
    assert search_area == 98 * 180
    assert np.min(polygon_a[:, 0]) == 152
    assert np.max(polygon_b[:, 0]) == 268


def test_near_miss_pair_is_considered_with_margin(tmp_path):
    first, second = _write_pair(tmp_path)
    items = validate_images(_images(first, second, 260))
    overlap, search_area, polygon_a, polygon_b = GuidedAffineStitcher._search_polygons(
        items[0], items[1], 18,
    )
    assert overlap == 0
    assert search_area == 8 * 180
    assert polygon_a.size and polygon_b.size
    exact = GuidedAffineStitcher._search_polygons(items[0], items[1], 0)
    assert exact[1] == 0


def test_three_images_refine_through_overlap_chain(tmp_path):
    base = np.random.default_rng(9).integers(0, 256, (180, 600, 3), dtype=np.uint8)
    images = []
    for index, (source_x, initial_x) in enumerate(((0, 0), (170, 174), (340, 336))):
        path = tmp_path / f"chain_{index}.png"
        assert cv2.imwrite(str(path), base[:, source_x:source_x + 260])
        images.append({"path": str(path), "corners": [
            [initial_x, 0], [initial_x + 260, 0],
            [initial_x + 260, 180], [initial_x, 180],
        ]})
    stitcher = GuidedAffineStitcher(show_progress=False, final_megapix=-1)
    panorama = stitcher.stitch(images)
    assert abs(panorama.shape[1] - 600) <= 2
    assert {(pair["image_a"], pair["image_b"]) for pair in stitcher.pair_diagnostics} == {(0, 1), (1, 2)}
    assert all(pair["status"] == "refined" for pair in stitcher.pair_diagnostics)


def test_visual_correction_is_not_limited_by_old_pixel_cap(tmp_path):
    first, second = _write_pair(tmp_path)
    stitcher = GuidedAffineStitcher(show_progress=False, final_megapix=-1)
    panorama = stitcher.stitch(_images(first, second, 195))
    assert abs(panorama.shape[1] - 420) <= 2
    assert stitcher.pair_diagnostics[0]["status"] == "refined"


def test_invalid_geometry_and_no_overlap(tmp_path):
    first, second = _write_pair(tmp_path)
    images = _images(first, second, 170)
    images[1]["corners"][2][0] += 40
    with pytest.raises(ValueError, match="仿射"):
        validate_images(images)
    images = _images(first, second, 400)
    with pytest.raises(ValueError, match="没有重叠"):
        GuidedAffineStitcher(show_progress=False).stitch(images)


def test_api_returns_compact_result_and_debug_info(tmp_path):
    first, second = _write_pair(tmp_path, textured=False)
    output = tmp_path / "result.png"
    response = app.test_client().post("/stitch", json={
        "images": _images(first, second, -5),
        "output_path": str(output),
        "show_progress": False,
        "settings": {
            "algorithm": "guided_affine", "final_megapix": -1,
            "georef_mode": "none",
        },
    })
    assert response.status_code == 200, response.get_json()
    data = response.get_json()
    assert output.is_file()
    assert data["settings"]["georef_mode"] == "none"
    assert data["output_width"] == 255
    assert set(data) == {
        "status", "output_path", "progress_path", "image_count", "valid_paths",
        "output_width", "output_height", "algorithm", "cog", "settings", "debug_info",
    }
    stages = data["debug_info"]["stages"]
    assert len(stages) == 17
    assert stages[-1]["index"] == stages[-1]["total"] == 17
    assert any(stage["message"] == "估计曝光补偿" for stage in stages)


def test_guided_uses_config_default_georef_mode_when_omitted():
    settings = _build_settings({"algorithm": "guided_affine"})
    assert settings["georef_mode"] == DEFAULT_SETTINGS["georef_mode"] == "affine_fit"


def test_guided_geotransform_uses_reference_scale_and_offset(tmp_path):
    gdal = pytest.importorskip("osgeo.gdal")
    osr = pytest.importorskip("osgeo.osr")

    reference = tmp_path / "reference.tif"
    dataset = gdal.GetDriverByName("GTiff").Create(str(reference), 20, 10, 1)
    dataset.SetGeoTransform((100.0, 2.0, 0.25, 200.0, -0.1, -3.0))
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(3857)
    dataset.SetProjection(spatial_ref.ExportToWkt())
    dataset = None

    result = guided_output_georeference(reference, 0.5, [10.0, 5.0])
    assert result is not None
    transform, projection, metadata = result
    assert transform == pytest.approx((57.5, 4.0, 0.5, 232.0, -0.2, -6.0))
    assert "3857" in projection
    assert metadata["method"] == "GUIDED_FIRST_IMAGE"


def test_guided_api_reuses_progress_exposure_seam_and_blend(tmp_path, monkeypatch):
    first, second = _write_pair(tmp_path)
    events = []
    for method in ("estimate_exposure_errors", "find_seam_masks", "initialize_composition"):
        original = getattr(GuidedProgressStitcher, method)

        def tracked(self, *args, _original=original, _method=method, **kwargs):
            events.append(_method)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(GuidedProgressStitcher, method, tracked)

    settings = {**DEFAULT_SETTINGS, **GUIDED_DEFAULT_SETTINGS, "final_megapix": -1}
    stitcher = GuidedProgressStitcher(
        _images(first, second, 174),
        {key: settings[key] for key in GUIDED_SETTING_KEYS if key in settings},
        **{key: settings[key] for key in STITCHER_SETTING_KEYS if key in settings},
        show_progress=False,
    )
    panorama = stitcher.stitch([str(first), str(second)])
    assert abs(panorama.shape[1] - 420) <= 2
    assert events == [
        "estimate_exposure_errors", "find_seam_masks", "initialize_composition",
    ]
    assert stitcher.pair_diagnostics[0]["status"] == "refined"
    assert len(stitcher.debug_info()["stages"]) == 17
