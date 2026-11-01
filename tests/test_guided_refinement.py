"""Bound full-affine corrections relative to the user's initial placement."""

import math

import numpy as np
import pytest

from algorithms.guided_affine import GuidedAffineStitcher, build_guided_settings
from stitching_headless_api import app
from test_guided_affine import _images, _write_pair


LIMIT_KEYS = (
    'guided_max_translation_ratio', 'guided_max_rotation_degrees',
    'guided_max_scale_change', 'guided_max_shear',
)


def _rotation(degrees):
    angle = math.radians(degrees)
    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]])


def _adjust(engine, linear, shift, *, initial=None):
    if initial is None:
        initial = np.eye(2, 3)
    anchor = np.eye(2, 3)
    items = [{'size': (250, 180), 'initial_matrix': matrix.copy(), 'matrix': matrix.copy(),
              'correction_status': 'fallback_initial'} for matrix in (anchor, initial)]
    source = np.array([(x, y) for y in (20, 80, 140) for x in (20, 80, 140, 220)], float)
    pb = source @ initial[:, :2].T + initial[:, 2]
    center = np.array([125, 90]) @ initial[:, :2].T + initial[:, 2]
    pa = (pb - center) @ linear.T + center + shift
    engine._global_adjust(items, [(0, 1, pa, pb)])
    return items


@pytest.mark.parametrize('key', LIMIT_KEYS)
@pytest.mark.parametrize('value', [-0.01, float('nan'), float('inf')])
def test_invalid_refinement_limits_are_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        build_guided_settings({'algorithm': 'guided_affine', key: value})
    with pytest.raises(ValueError, match=key):
        GuidedAffineStitcher(**{key: value})


def test_rotation_limit_cannot_exceed_half_turn():
    with pytest.raises(ValueError, match='180'):
        build_guided_settings({'algorithm': 'guided_affine', 'guided_max_rotation_degrees': 181})
    with pytest.raises(ValueError, match='180'):
        GuidedAffineStitcher(guided_max_rotation_degrees=181)


def test_full_affine_micro_correction_preserves_large_initial_rotation_and_scale():
    engine = GuidedAffineStitcher(
        show_progress=False, guided_refinement_limits_enabled=True
    )
    initial = np.column_stack((1.5 * _rotation(70), [700, -200]))
    correction = _rotation(2) @ np.array([[1.01, 0.01], [0.01, 0.99]])
    items = _adjust(engine, correction, [2, -2], initial=initial)
    assert items[1]['correction_status'] == 'refined'
    metrics = items[1]['attempted_correction']
    assert metrics['rotation_degrees'] == pytest.approx(2, abs=0.01)
    assert 0 < metrics['shear'] < 0.03
    assert metrics['translation_ratio'] < 0.05
    assert not np.allclose(items[1]['matrix'], initial)
    np.testing.assert_array_equal(items[0]['matrix'], items[0]['initial_matrix'])


@pytest.mark.parametrize('linear, shift, overrides, reason', [
    (np.eye(2), [12, 0], {}, '中心平移'),
    (_rotation(4), [0, 0], {}, '旋转角度'),
    (np.diag([1.04, 1]), [0, 0], {}, '缩放'),
    (np.array([[1, 0.04], [0.04, 1]]), [0, 0], {'guided_max_scale_change': 0.1}, '剪切'),
])
def test_excessive_correction_keeps_exact_initial_matrix(linear, shift, overrides, reason):
    engine = GuidedAffineStitcher(
        show_progress=False, guided_refinement_limits_enabled=True, **overrides
    )
    items = _adjust(engine, linear, shift)
    assert items[1]['correction_status'] == 'fallback_initial'
    assert reason in items[1]['correction_reason']
    np.testing.assert_array_equal(items[1]['matrix'], items[1]['initial_matrix'])


def test_refinement_boundary_is_inclusive():
    engine = GuidedAffineStitcher(guided_refinement_limits_enabled=True)
    metrics, reason = engine._check_refinement(_rotation(3) @ np.diag([1.03, 0.97]), np.array([9, 0]), 180)
    assert reason == ''
    assert metrics['translation_ratio'] == 0.05
    assert metrics['rotation_degrees'] == pytest.approx(3)


@pytest.mark.parametrize('canvas_scale', [0.5, 1, 3])
def test_translation_limit_uses_initial_canvas_short_side(canvas_scale):
    engine = GuidedAffineStitcher(
        show_progress=False, guided_refinement_limits_enabled=True
    )
    initial = np.column_stack((canvas_scale * np.eye(2), [400, 50]))
    items = _adjust(engine, np.eye(2), np.array([8, 0]) * canvas_scale, initial=initial)
    assert items[1]['correction_status'] == 'refined'
    assert items[1]['attempted_correction']['translation_ratio'] == pytest.approx(8/180, abs=1e-4)


def test_excessive_bridge_removes_downstream_refinement():
    engine = GuidedAffineStitcher(
        show_progress=False, guided_refinement_limits_enabled=True
    )
    initial = np.eye(2, 3)
    items = [{'size': (250, 180), 'initial_matrix': initial.copy(), 'matrix': initial.copy(),
              'correction_status': 'fallback_initial'} for _ in range(3)]
    pa = np.array([(x, y) for y in (20, 80, 140) for x in (20, 80, 140, 220)], float)
    pb = pa + [20, 0]
    # The third image needs almost no correction, but its only bridge to the
    # anchor needs excessive movement. It must not remain marked as refined.
    engine._global_adjust(items, [(0, 1, pa, pb), (1, 2, pb, pa)])
    assert '微调上限' in items[1]['correction_reason']
    for item in items[1:]:
        assert item['correction_status'] == 'fallback_initial'
        np.testing.assert_array_equal(item['matrix'], initial)


def test_small_shear_and_nonuniform_scale_are_refined(tmp_path):
    first, second = _write_pair(tmp_path)
    images = _images(first, second, 170)
    corners = np.array([[0, 0], [250, 0], [250, 180], [0, 180]], float)
    initial = np.array([[1.01, 0.01, 172], [0, 0.99, 1]])
    images[1]['corners'] = (corners @ initial[:, :2].T + initial[:, 2]).tolist()
    engine = GuidedAffineStitcher(show_progress=False)
    items = engine.register(images)
    assert engine.registration_status == 'full'
    transformed = corners @ items[1]['matrix'][:, :2].T + items[1]['matrix'][:, 2]
    np.testing.assert_allclose(transformed, corners + [170, 0], atol=0.2)


def test_refinement_limits_are_disabled_by_default():
    engine = GuidedAffineStitcher(show_progress=False)
    items = _adjust(engine, np.eye(2), [20, 0])

    assert engine.refinement_limits_enabled is False
    assert items[1]['correction_status'] == 'refined'
    assert items[1]['attempted_correction']['translation_ratio'] > 0.05


@pytest.mark.parametrize('limit, expected', [(0.05, 'full'), (0.01, 'manual_only')])
def test_api_applies_requested_micro_limits_and_reports_attempt(tmp_path, limit, expected):
    first, second = _write_pair(tmp_path)
    response = app.test_client().post('/stitch', json={
        'images': _images(first, second, 174), 'output_path': str(tmp_path / 'out.png'),
        'show_progress': False,
        'settings': {
            'algorithm': 'guided_affine', 'georef_mode': 'none',
            'guided_refinement_limits_enabled': True,
            'guided_max_translation_ratio': limit,
        },
    })
    assert response.status_code == 200, response.get_json()
    data = response.get_json()
    assert data['registration_status'] == expected
    assert data['refinement_limits_enabled'] is True
    assert data['refinement_limits']['translation_ratio'] == limit
    diagnostic = data['image_diagnostics'][1]
    assert diagnostic['attempted_correction']['translation_ratio'] == pytest.approx(4/180, abs=1e-4)
    if expected == 'manual_only':
        assert '微调上限' in diagnostic['reason']
        assert diagnostic['initial_transform'] == diagnostic['final_transform']
