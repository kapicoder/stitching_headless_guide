"""拼接算法统一导出模块。

职责：向兼容入口提供三种算法实现；输入输出由各算法类定义。
流程：调用方选择算法类后执行其 stitch 方法。
"""

from .apap import ApapStitcher
from .guided_affine import (
    GUIDED_DEFAULT_SETTINGS,
    GUIDED_SETTING_KEYS,
    GuidedAffineStitcher,
    build_guided_settings,
)
from .local_refine import LocalRefineStitcher
from .opencv_stitcher import GuidedProgressStitcher, ProgressStitcher
from .origin import OriginStitcher

__all__ = [
    "ProgressStitcher",
    "GuidedProgressStitcher",
    "OriginStitcher",
    "ApapStitcher",
    "GuidedAffineStitcher",
    "GUIDED_SETTING_KEYS",
    "GUIDED_DEFAULT_SETTINGS",
    "build_guided_settings",
    "LocalRefineStitcher",
]
