"""Registration units, reciprocal matching and anchored optimization regressions."""

import cv2
import numpy as np
import pytest

from algorithms.guided_affine import GuidedAffineStitcher, build_guided_settings
from stitching_headless_api import app
from test_guided_affine import _images, _write_pair


def test_mutual_match_rejects_many_to_one_correspondences():
    desc_a = np.array([[0, 0], [0.1, 0], [10, 10]], np.float32)
    desc_b = np.array([[0, 0], [10, 10]], np.float32)
    matches = GuidedAffineStitcher._mutual_matches(desc_a, desc_b, cv2.NORM_L2)
    assert [(m.queryIdx, m.trainIdx) for m in matches] == [(0, 0), (2, 1)]


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_ransac_threshold_is_rejected(value):
    with pytest.raises(ValueError, match='guided_ransac_threshold_px'):
        build_guided_settings({'algorithm': 'guided_affine', 'guided_ransac_threshold_px': value})
    with pytest.raises(ValueError, match='guided_ransac_threshold_px'):
        GuidedAffineStitcher(guided_ransac_threshold_px=value)


def test_ransac_threshold_accepts_nested_and_top_level_settings():
    settings = build_guided_settings({'settings': {'algorithm': 'guided_affine',
                                     'guided_ransac_threshold_px': 2.5},
                                     'guided_ransac_threshold_px': 4})
    assert settings['guided_ransac_threshold_px'] == 4
    assert build_guided_settings({'algorithm': 'guided_affine'})['guided_ransac_threshold_px'] == 3


def test_pair_margin_uses_smaller_transformed_short_side():
    engine = GuidedAffineStitcher()
    a = {'size': (400, 200), 'matrix': np.array([[2, 0, 0], [0, 2, 0]], float)}
    b = {'size': (100, 80), 'matrix': np.array([[0, -0.5, 30], [0.5, 0, 20]], float)}
    assert engine._pair_margin(a, b) == pytest.approx(4)
    for item in (a, b):
        item['matrix'] *= 3
    assert engine._pair_margin(a, b) == pytest.approx(12)


@pytest.mark.parametrize('canvas_scale', [0.5, 1, 4])
def test_ransac_runs_in_target_feature_pixels(monkeypatch, canvas_scale):
    engine = GuidedAffineStitcher(show_progress=False, guided_transform_model="affine")
    points_b = np.array([(x, y) for y in (40, 100, 160) for x in (40, 100, 160, 220)], float)
    points_a = points_b + [5, 7]
    points_a[:9] += np.random.default_rng(4).normal(0, 0.5, (9, 2))
    points_a[-3:] += [[30, -20], [-25, 40], [35, 25]]
    descriptors = np.eye(len(points_b), dtype=np.float32)
    detections = iter((points_a, points_b))

    def detect(detector, image, mask):
        points = next(detections)
        return [cv2.KeyPoint(float(x), float(y), 1) for x, y in points], descriptors

    monkeypatch.setattr(engine, '_detect_region', detect)
    image = np.zeros((300, 300, 3), np.uint8)
    # Different source sizes and anisotropic canvas transforms must not change
    # the RANSAC coordinate system or the number of accepted correspondences.
    a = {'size': (600, 900), 'medium': image, 'medium_mask': None,
         'matrix': np.array([[canvas_scale, 0, 0], [0, canvas_scale, 0]], float)}
    b = {'size': (900, 600), 'medium': image, 'medium_mask': None,
         'matrix': np.array([[canvas_scale, 0.2 * canvas_scale, 20], [0, canvas_scale, 10]])}
    polygon = np.array([[-100, -100], [5000, -100], [5000, 5000], [-100, 5000]], np.float32)
    cv2.setRNGSeed(17)
    info, pair = engine._match_pair(a, b, polygon, polygon, cv2.NORM_L2)
    assert info['status'] == 'matched'
    assert info['inliers'] == 9
    assert info['rmse_pixels'] < 2
    assert info['error_coordinate_system'] == 'image_a_medium_pixels'
    expected_a = (points_a[:9] * [2, 3]) @ a['matrix'][:, :2].T + a['matrix'][:, 2]
    np.testing.assert_allclose(pair[0], expected_a, rtol=1e-6)


def test_already_aligned_images_are_successfully_registered(tmp_path):
    first, second = _write_pair(tmp_path)
    engine = GuidedAffineStitcher(show_progress=False, guided_transform_model="affine")
    items = engine.register(_images(first, second, 170))
    assert engine.registration_status == 'full'
    assert engine.image_diagnostics[0]['status'] == 'reference'
    assert engine.image_diagnostics[1]['status'] == 'refined'
    np.testing.assert_allclose(items[0]['matrix'], items[0]['initial_matrix'], atol=0)
    np.testing.assert_allclose(items[1]['matrix'], items[1]['initial_matrix'], atol=0.1)


def test_disconnected_successful_group_keeps_its_manual_positions(tmp_path):
    paths = []
    for group in range(2):
        folder = tmp_path / str(group)
        folder.mkdir()
        paths.extend(_write_pair(folder))
    images = _images(paths[0], paths[1], 174)
    for path, x in zip(paths[2:], (600, 774)):
        images.append({'path': str(path), 'corners': [[x, 0], [x+250, 0], [x+250, 180], [x, 180]]})
    engine = GuidedAffineStitcher(show_progress=False)
    items = engine.register(images)
    assert engine.registration_status == 'partial'
    assert engine.pair_diagnostics[0]['status'] == 'refined'
    assert engine.pair_diagnostics[1]['inliers'] >= 8
    assert engine.pair_diagnostics[1]['status'] == 'fallback_initial'
    for item in items[2:]:
        assert item['correction_status'] == 'fallback_initial'
        np.testing.assert_array_equal(item['matrix'], item['initial_matrix'])


def test_failed_bridge_does_not_refine_detached_images(tmp_path):
    first, second = _write_pair(tmp_path)
    blank = tmp_path / 'blank.png'
    assert cv2.imwrite(str(blank), np.full((180, 250, 3), 120, np.uint8))
    images = _images(first, blank, 174)
    images.append({'path': str(second), 'corners': [[348, 0], [598, 0], [598, 180], [348, 180]]})
    engine = GuidedAffineStitcher(show_progress=False)
    items = engine.register(images)
    assert engine.registration_status == 'manual_only'
    assert all(d['status'] == 'fallback_initial' for d in engine.image_diagnostics[1:])
    for item in items[1:]:
        np.testing.assert_array_equal(item['matrix'], item['initial_matrix'])


def test_invalid_global_transform_is_not_applied():
    engine = GuidedAffineStitcher(show_progress=False)
    initial = np.eye(2, 3)
    items = [{'size': (250, 180), 'initial_matrix': initial.copy(),
              'matrix': initial.copy(), 'correction_status': 'fallback_initial'} for _ in range(3)]
    pa = np.array([(x, y) for y in (20, 80, 140) for x in (20, 80, 140)], float)
    reflected = pa * [-1, 1] + [200, 0]
    engine._global_adjust(items, [(0, 1, pa, reflected), (1, 2, reflected, reflected + [10, 0])])
    assert all(item['correction_status'] == 'fallback_initial' for item in items[1:])
    for item in items:
        np.testing.assert_array_equal(item['matrix'], initial)


@pytest.mark.parametrize('textured, expected', [(True, 'full'), (False, 'manual_only')])
def test_api_reports_registration_independently_of_output_success(tmp_path, textured, expected):
    first, second = _write_pair(tmp_path, textured=textured)
    output = tmp_path / 'result.png'
    response = app.test_client().post('/stitch', json={
        'images': _images(first, second, 174), 'output_path': str(output), 'show_progress': False,
        'settings': {'algorithm': 'guided_affine', 'georef_mode': 'none',
                     'guided_ransac_threshold_px': 2.5},
    })
    assert response.status_code == 200, response.get_json()
    data = response.get_json()
    assert data['status'] == 'done' and output.is_file()
    assert data['registration_status'] == expected
    assert len(data['image_diagnostics']) == 2
    assert data['debug_info']['pair_diagnostics'][0]['ransac_threshold_px'] == 2.5
