"""拼接服务的统一配置模块。

职责：集中保存算法默认参数、各算法允许接收的参数，以及 API 配置的构建和校验。
输入：请求中的 settings 对象或顶层配置字段；输出：类型规范且通过约束检查的配置字典。
流程：复制默认值 -> 合并嵌套及顶层参数 -> 兼容旧字段 -> 类型转换 -> 合法性校验。
"""

from __future__ import annotations

import math
from typing import Any


DEFAULT_SETTINGS: dict[str, Any] = {
    # GDAL 仅负责 TIFF 解码，图像位置始终由视觉拼接算法计算。
    "algorithm": "guided_affine",
    
    "detector": "sift",
    "nfeatures": 4000,
    # 人工引导配准：外扩距离使用每对图像在画布中的较小短边。
    "guided_min_overlap_pixels": 100,
    "guided_overlap_margin_ratio": 0.10,
    "guided_min_inliers": 8,
    # RANSAC 误差使用目标特征图像素，不使用画布单位。
    "guided_ransac_threshold_px": 3.0,
    # 完整仿射微调上限，均相对于用户的初始布局。
    # 平移使用图像中心位移 / 初始画布短边，旋转使用极分解角度。
    "guided_max_translation_ratio": 0.05,
    "guided_max_rotation_degrees": 3.0,
    # 缩放限制极分解的两个主伸缩率，剪切为对称伸缩矩阵的归一化非对角项。
    "guided_max_scale_change": 0.03,
    "guided_max_shear": 0.03,
    "match_conf": 0.65,
    "confidence_threshold": 0.3,
    
    "matcher_type": "affine",
    "range_width": -1,    ##设置为-1为全匹配
    "estimator": "affine",
    "adjuster": "affine",
    "warper_type": "affine",
    
    "wave_correct_kind": "no",
    "compensator": "channel_blocks",
    
    "finder": "dp_color",
    "blender_type": "multiband",
    "blend_strength": 1,
    
    
    # ------------------------------------------------------------------
    # 拼接流程性能开关（默认值保持当前优化流程）
    # ------------------------------------------------------------------
    # medium_megapix：每张图在特征检测/匹配阶段使用的百万像素数。
    #   调低会加快读取和特征计算，但弱纹理图像可能更难匹配。
    # low_megapix：曝光估计、接缝查找和自动裁剪阶段的百万像素数。
    #   调低可提速，但可能降低接缝定位精度。
    # final_megapix：每张图在最终投影和融合阶段的百万像素数。
    #   它不是最终全景图的总像素数；-1 表示尽量使用原始分辨率。
    "medium_megapix": 2,
    "low_megapix": 0.3,
    "final_megapix": 10,

    # optimized_tiff_io：TIFF/COG 优化读取总开关。
    #   True：GDAL 直接解码到 medium/final 目标尺寸，可利用 overview；medium
    #         阶段在同一次 gdal.OpenEx 中读取原始尺寸和目标尺寸像素。
    #   False：完整读取 TIFF 后再由 OpenCV 缩放，用于与旧流程进行 A/B 对比。
    "optimized_tiff_io": True,

    # parallel_image_io：并行执行多张图像的 GDAL 解码和尺寸转换。
    #   False 用于单核、慢速机械盘或排查并发读取问题。
    "parallel_image_io": True,

    # image_io_workers：图像读取线程数，仅在 parallel_image_io=True 时生效。
    #   推荐：普通电脑/机械盘=1~2，SSD/NVMe=2~4；线程越多不一定越快。
    #   程序会自动限制实际线程数不超过当前需要读取的图片数量。
    "image_io_workers": 2,

    # prefetch_final_images：Final 阶段预取图像。
    #   True：处理当前图像时并行读取后续图像；需要 parallel_image_io=True。
    #   False：Final 阶段逐张同步读取，便于排查预取造成的磁盘争用。
    "prefetch_final_images": True,

    # cache_all_valid_masks：在单次拼接请求内缓存“整图有效”的判断。
    #   True：medium 阶段确认全有效后，low/final 不再重复打开和读取 mask，
    #         直接创建相应尺寸的全白 mask。
    #   False：每个阶段都重新向 GDAL 查询 mask，用于行为对照。
    "cache_all_valid_masks": True,

    # shared_warp_maps：最终阶段让图像与 mask 共用同一组投影映射。
    #   True 可避免重复计算 xmap/ymap；如需和第三方原始流程对比可设 False。
    "shared_warp_maps": True,

    # georef_mode：输出 COG 的近似地理参考策略。
    # affine_fit 复用 affine 投影拟合；失败时自动退回 copy。
    "georef_mode": "affine_fit",

    # ------------------------------------------------------------------    
    "crop": True,
    "apap_grid_parts": 2,
    "apap_sigma": 0.15,
    "apap_min_matches": 12,
    "apap_blend_width": 80,
}

# 直接执行 ``python stitiching.py`` 时使用的输入输出选项。
# 算法参数只在上面的 DEFAULT_SETTINGS 中维护。
# DEFAULT_IMAGE_PATHS 设置为非空列表时优先使用，并严格保留列表顺序；
# 设置为 None 或空列表时，改为扫描 DEFAULT_INPUT_FOLDER。
# DEFAULT_IMAGE_PATHS: list[str] | None = [
#     "./data/横须贺_20180517_tile_001.tif",
#     "./data/横须贺_20180517_tile_002.tif",
#     "./data/横须贺_20180517_tile_003.tif",
#     "./data/横须贺_20180517_tile_004.tif",
#     "./data/横须贺_20180517_tile_005.tif",
#     "./data/横须贺_20180517_tile_006.tif",
# ]
DEFAULT_IMAGE_PATHS: list[str] | None = [
    "./test_data/aerial_4000x24000/aerial_01.tif",
    "./test_data/aerial_4000x24000/aerial_02.tif",
    "./test_data/aerial_4000x24000/aerial_03.tif",
    "./test_data/aerial_4000x24000/aerial_04.tif",
    "./test_data/aerial_4000x24000/aerial_05.tif"

]
DEFAULT_INPUT_FOLDER = "./test_data/aerial_4000x24000"
DEFAULT_OUTPUT_PATH = "./output/aerial_test_data.jpg"
DEFAULT_IMAGE_LIMIT: int | None = None   #不设置图像数量限制
#DEFAULT_IMAGE_LIMIT = 5
DEFAULT_SHOW_PROGRESS = True

STITCHER_SETTING_KEYS = {
    "detector", "nfeatures", "match_conf", "matcher_type", "range_width",
    "estimator", "adjuster",
    "warper_type", "wave_correct_kind", "compensator", "nr_feeds",
    "block_size", "finder", "blender_type", "blend_strength",
    "confidence_threshold", "medium_megapix", "low_megapix",
    "final_megapix", "crop",
}

APAP_SETTING_KEYS = {
    "detector", "nfeatures", "medium_megapix", "confidence_threshold",
    "crop", "apap_grid_parts", "apap_grid_size", "apap_sigma",
    "apap_min_matches", "apap_blend_width",
}

LOCAL_REFINE_SETTING_KEYS = {
    "detector", "nfeatures", "medium_megapix", "confidence_threshold", "crop",
}
SETTING_KEYS = set(DEFAULT_SETTINGS)
LEGACY_SETTING_KEYS = {"apap_grid_size"}
ACCEPTED_SETTING_KEYS = SETTING_KEYS | LEGACY_SETTING_KEYS


def filter_settings(settings: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    """只保留某种算法支持的参数。"""
    return {key: value for key, value in settings.items() if key in keys}


def stitcher_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """只筛选第三方 Stitcher 支持的参数；默认值统一由 DEFAULT_SETTINGS 提供。"""
    return filter_settings(settings, STITCHER_SETTING_KEYS)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def build_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """合并并校验 API 请求中的拼接设置。"""
    settings = dict(DEFAULT_SETTINGS)
    nested = payload.get("settings")
    if nested is not None:
        if not isinstance(nested, dict):
            raise ValueError("settings 必须是 JSON object")
        unknown = sorted(set(nested) - ACCEPTED_SETTING_KEYS)
        if unknown:
            raise ValueError(f"settings 包含不支持的参数: {unknown}")
        settings.update(nested)
        if "apap_grid_size" in nested and "apap_grid_parts" not in nested:
            settings["apap_grid_parts"] = nested["apap_grid_size"]

    for key in SETTING_KEYS:
        if key in payload and payload[key] not in (None, ""):
            settings[key] = payload[key]
    if "apap_grid_size" in payload and "apap_grid_parts" not in payload:
        settings["apap_grid_parts"] = payload["apap_grid_size"]
    settings.pop("apap_grid_size", None)

    for key in {
        "nfeatures", "range_width", "image_io_workers", "apap_grid_parts", "apap_min_matches",
        "apap_blend_width",
        "guided_min_overlap_pixels", "guided_min_inliers",
    }:
        settings[key] = int(settings[key])
    for key in {
        "match_conf", "confidence_threshold", "medium_megapix", "low_megapix",
        "final_megapix", "apap_sigma",
        "guided_overlap_margin_ratio", "guided_ransac_threshold_px",
        "guided_max_translation_ratio", "guided_max_rotation_degrees",
        "guided_max_scale_change", "guided_max_shear",
    }:
        settings[key] = float(settings[key])
    settings["crop"] = _as_bool(settings["crop"])
    settings["optimized_tiff_io"] = _as_bool(settings["optimized_tiff_io"])
    settings["parallel_image_io"] = _as_bool(settings["parallel_image_io"])
    settings["prefetch_final_images"] = _as_bool(settings["prefetch_final_images"])
    settings["cache_all_valid_masks"] = _as_bool(settings["cache_all_valid_masks"])
    settings["shared_warp_maps"] = _as_bool(settings["shared_warp_maps"])
    settings["georef_mode"] = str(settings["georef_mode"]).strip().lower()
    if settings["georef_mode"] not in {"affine_fit", "copy", "none"}:
        raise ValueError("georef_mode 只支持 affine_fit、copy 或 none")

    if settings["nfeatures"] <= 0:
        raise ValueError("nfeatures 必须大于0")
    if settings["guided_min_overlap_pixels"] < 1 or settings["guided_min_inliers"] < 3:
        raise ValueError("guided_min_overlap_pixels 必须大于0，guided_min_inliers 必须至少为3")
    if not 0 <= settings["guided_overlap_margin_ratio"] <= 1:
        raise ValueError("guided_overlap_margin_ratio 必须在 0 到 1 之间")
    if (not math.isfinite(settings["guided_ransac_threshold_px"])
            or settings["guided_ransac_threshold_px"] <= 0):
        raise ValueError("guided_ransac_threshold_px 必须是大于0的有限数值")
    for key in ("guided_max_translation_ratio", "guided_max_rotation_degrees",
                "guided_max_scale_change", "guided_max_shear"):
        if not math.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f"{key} 必须是非负的有限数值")
    if settings["guided_max_rotation_degrees"] > 180:
        raise ValueError("guided_max_rotation_degrees 不能超过180度")
    if not 0 < settings["match_conf"] <= 1:
        raise ValueError("match_conf 必须大于0且小于等于1")
    if settings["range_width"] != -1 and settings["range_width"] < 1:
        raise ValueError("range_width 必须为 -1 或大于等于1")
    if settings["image_io_workers"] < 1:
        raise ValueError("image_io_workers 必须大于等于1")
    settings["algorithm"] = str(settings["algorithm"]).strip().lower()
    if settings["algorithm"] not in {"stitcher", "origin", "apap", "local_refine"}:
        raise ValueError("algorithm 只支持 stitcher、origin、apap 或 local_refine")
    if settings["confidence_threshold"] < 0:
        raise ValueError("confidence_threshold 不能小于0")
    if settings["medium_megapix"] <= 0:
        raise ValueError("medium_megapix 必须大于0")
    if settings["low_megapix"] <= 0:
        raise ValueError("low_megapix 必须大于0")
    if settings["final_megapix"] == 0:
        raise ValueError("final_megapix 不能为0；使用 -1 表示原始分辨率")
    if settings["apap_grid_parts"] <= 0:
        raise ValueError("apap_grid_parts 必须大于0")
    if settings["apap_sigma"] <= 0:
        raise ValueError("apap_sigma 必须大于0")
    if settings["apap_min_matches"] < 4:
        raise ValueError("apap_min_matches 必须大于等于4")
    if settings["apap_blend_width"] < 0:
        raise ValueError("apap_blend_width 不能小于0")
    return settings
