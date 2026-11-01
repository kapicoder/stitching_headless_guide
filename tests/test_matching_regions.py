"""Region geometry, feature allocation, matching and API contract regressions."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from algorithms.guided_affine import (
    GuidedAffineStitcher, build_guided_settings, validate_matching_regions,
)
from stitching_headless_api import app
from test_guided_affine import _write_pair, _images


@pytest.mark.parametrize('value', [False, {}, [None], [{'id':'a','rect':[0,0,True,2]}],
    [{'id':'a','rect':[0,0,float('nan'),2]}], [{'id':'a','rect':[0,0,0,2]}],
    [{'id':'a','rect':[0,0,1,2]}]*2, [{'id':str(i),'rect':[0,0,1,2]} for i in range(33)]])
def test_invalid_regions(value):
    with pytest.raises(ValueError):
        validate_matching_regions(value)


def test_negative_and_partial_rectangles():
    value = [{'id':'a','rect':[-100,-20,500,60]}]
    assert validate_matching_regions(value) == value
    assert validate_matching_regions(None) == []


@pytest.mark.parametrize('key', ['guided_feature_density','guided_region_min_features','guided_pair_max_features'])
def test_removed_budget_settings_are_rejected(key):
    with pytest.raises(ValueError, match="不支持的参数"):
        build_guided_settings({'settings': {'algorithm':'guided_affine', key:2000}})


def _register(tmp_path, regions, *, textured=True, detector='sift'):
    first, second = _write_pair(tmp_path, textured=textured)
    engine = GuidedAffineStitcher(show_progress=False, matching_regions=regions, detector=detector)
    engine.register(_images(first,second,174))
    return engine


@pytest.mark.parametrize('detector', ['sift','orb','akaze','brisk'])
def test_selected_region_and_duplicate_area(tmp_path, detector):
    engine = _register(tmp_path, [{'id':'a','rect':[160,0,100,180]},
                                  {'id':'b','rect':[160,0,100,180]}], detector=detector)
    pair = engine.pair_diagnostics[0]
    assert pair['region_source'] == 'selected'
    assert pair['regions'][0]['area_a'] > 0
    assert pair['regions'][1]['area_a'] == pair['regions'][1]['budget_a'] == 0
    for side in ('a','b'):
        assert pair[f'features_{side}'] <= len(pair['regions']) * engine.nfeatures
        assert pair['regions'][0][f'budget_{side}'] == engine.nfeatures
        assert all(r[f'features_{side}'] <= r[f'budget_{side}'] for r in pair['regions'])


def test_different_region_areas_use_same_feature_budget(tmp_path):
    engine = _register(tmp_path, [
        {'id': 'small', 'rect': [160, 0, 100, 40]},
        {'id': 'large', 'rect': [160, 40, 100, 140]},
    ])
    regions = engine.pair_diagnostics[0]['regions']
    for side in ('a', 'b'):
        assert 0 < regions[0][f'area_{side}'] < regions[1][f'area_{side}']
        assert all(region[f'budget_{side}'] == engine.nfeatures for region in regions)


def test_unrelated_region_leaves_automatic_matching(tmp_path):
    pair = _register(tmp_path,[{'id':'away','rect':[-1000,-1000,10,10]}]).pair_diagnostics[0]
    assert pair['region_source'] == 'automatic'
    assert pair['status'] == 'refined'


def test_weak_region_never_falls_back_to_outside(tmp_path):
    pair = _register(tmp_path,[{'id':'a','rect':[170,0,60,180]}],textured=False).pair_diagnostics[0]
    assert pair['region_source'] == 'selected'
    assert pair['status'] == 'fallback_initial'
    assert '扩大区域' in pair['reason']


def test_selected_features_stay_in_mask(tmp_path, monkeypatch):
    original = GuidedAffineStitcher._detect_region
    calls = []
    def checked(detector, image, mask):
        points, desc = original(detector,image,mask)
        assert all(mask[int(p.pt[1]),int(p.pt[0])] for p in points)
        calls.append(cv2.countNonZero(mask))
        return points,desc
    monkeypatch.setattr(GuidedAffineStitcher,'_detect_region',staticmethod(checked))
    _register(tmp_path,[{'id':'a','rect':[175,20,60,130]}])
    assert len(calls) == 2


def test_empty_valid_mask_stays_selected(tmp_path, monkeypatch):
    monkeypatch.setattr('algorithms.guided_affine.read_valid_mask',
                        lambda path,size: np.zeros((size[1],size[0]),np.uint8))
    pair = _register(tmp_path,[{'id':'a','rect':[160,0,100,180]}]).pair_diagnostics[0]
    assert pair['region_source'] == 'selected' and pair['status'] == 'fallback_initial'
    assert pair['regions'][0]['reason'] == '区域无有效像素'


def test_api_diagnostics_and_invalid_request(tmp_path):
    first,second = _write_pair(tmp_path)
    payload = {'images':_images(first,second,174),'output_path':str(tmp_path/'out.png'),
               'show_progress':False,'settings':{'algorithm':'guided_affine','georef_mode':'none'},
               'matching_regions':[{'id':'a','rect':[160,0,100,180]}]}
    client = app.test_client()
    result = client.post('/stitch',json=payload)
    assert result.status_code == 200, result.get_json()
    pair = result.get_json()['debug_info']['pair_diagnostics'][0]
    assert pair['region_source'] == 'selected' and pair['inliers'] >= 8
    payload['matching_regions'][0]['rect'][2] = False
    assert client.post('/stitch',json=payload).status_code == 400
    payload['matching_regions'] = [{'id':'a','rect':[0,0,1,1]}]
    payload['settings']['algorithm'] = 'stitcher'
    assert client.post('/stitch',json=payload).status_code == 400


def test_three_image_chain_mixes_selected_and_automatic(tmp_path):
    base = np.random.default_rng(9).integers(0,256,(180,600,3),dtype=np.uint8)
    images = []
    for i,(source_x,initial_x) in enumerate(((0,0),(170,174),(340,336))):
        path = tmp_path/f'chain{i}.png'
        cv2.imwrite(str(path),base[:,source_x:source_x+260])
        images.append({'path':str(path),'corners':[[initial_x,0],[initial_x+260,0],
                                                 [initial_x+260,180],[initial_x,180]]})
    engine = GuidedAffineStitcher(show_progress=False,
                                 matching_regions=[{'id':'a','rect':[180,0,60,180]}])
    engine.register(images)
    assert [p['region_source'] for p in engine.pair_diagnostics] == ['selected','automatic']
    assert all(p['status']=='refined' for p in engine.pair_diagnostics)
    # A tall shared rectangle can select both pairs in a chain.
    engine.matching_regions = [{'id':'all','rect':[0,0,600,180]}]
    engine.register(images)
    assert all(p['region_source']=='selected' for p in engine.pair_diagnostics)


def test_conflicting_displacements_follow_selected_content(tmp_path):
    base = np.random.default_rng(37).integers(0,256,(400,620,3),dtype=np.uint8)
    first = base[:,:400].copy()
    second = base[:,200:600].copy()
    second[:190] = base[:190,210:610]
    paths = [tmp_path/'conflict_a.png',tmp_path/'conflict_b.png']
    for path,image in zip(paths,(first,second)):
        cv2.imwrite(str(path),image)
    images = [{'path':str(path),'corners':[[x,0],[x+400,0],[x+400,400],[x,400]]}
              for path,x in zip(paths,(0,204))]
    results = []
    for y in (10,220):
        engine = GuidedAffineStitcher(show_progress=False,
                         matching_regions=[{'id':'target','rect':[220,y,160,150]}])
        items = engine.register(images)
        assert engine.pair_diagnostics[0]['status']=='refined'
        results.append(items[1]['matrix'][0,2])
    assert abs(results[0]-210) < 1
    assert abs(results[1]-200) < 1


def test_rotated_scaled_mask_and_valid_hole():
    angle = np.pi/7
    matrix = np.array([[1.3*np.cos(angle),-1.3*np.sin(angle),-30],
                       [1.3*np.sin(angle),1.3*np.cos(angle),40]])
    valid = np.full((100,150),255,np.uint8)
    valid[30:50,40:80]=0
    item = {'matrix':matrix,'size':(300,200),'medium_mask':valid}
    polygon = np.array([[-30,40],[80,40],[80,160],[-30,160]],np.float32)
    mask = GuidedAffineStitcher._feature_mask(item,polygon,(100,150))
    assert mask.any() and not mask[30:50,40:80].any()
    # Inverse mapping respects the actual working dimensions, including rotation.
    ys,xs = np.nonzero(mask)
    world = np.column_stack((xs*2,ys*2)) @ matrix[:,:2].T + matrix[:,2]
    assert world[:,0].min() >= -34 and world[:,0].max() <= 84
    assert world[:,1].min() >= 36 and world[:,1].max() <= 164


def test_proxy_rejects_invalid_regions_before_upstream():
    import guided_validation
    client = guided_validation.create_app([]).test_client()
    assert client.post('/api/stitch',json={'matching_regions':False}).status_code == 400
