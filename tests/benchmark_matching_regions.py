"""Run from project root: PYTHONPATH=. python tests/benchmark_matching_regions.py.

Uses existing 4000x24000 JPEG fixtures so the feature path is measurable without
GDAL. Tests registration only; production TIFF decoding/fusion is not measured.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from algorithms.guided_affine import GuidedAffineStitcher


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    images = []
    for i in range(3):
        x = i*3000
        images.append({'path':str(root/'test_data/test_data_4000x24000'/f'stitch_test_{i+1:02d}.jpg'),
                       'corners':[[x,0],[x+4000,0],[x+4000,24000],[x,24000]]})
    reports = []
    for name,regions in [('automatic',[]),('selected',[
        {'id':'top','rect':[3000,0,1000,10000]},
        {'id':'bottom','rect':[3000,12000,1000,12000]},
    ])]:
        engine = GuidedAffineStitcher(show_progress=False,matching_regions=regions)
        start = time.perf_counter()
        engine.register(images)
        seconds = time.perf_counter()-start
        for pair in engine.pair_diagnostics:
            for side in ('a','b'):
                if pair['region_source']=='selected':
                    assert pair[f'features_{side}'] <= engine.pair_max_features
        reports.append({'case':name,'seconds':round(seconds,3),
                        'original_size':[4000,24000], 'medium_megapix':engine.medium_megapix,
                        'medium_scale':engine._scale((4000,24000),engine.medium_megapix),
                        'pairs':engine.pair_diagnostics})
    print(json.dumps(reports,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
