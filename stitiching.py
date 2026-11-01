"""图像拼接兼容入口与默认配置运行程序。

职责：保留旧模块名和公开导入，按 algorithm 分派到拆分后的实现，并支持直接运行。
输入：函数调用的图片路径和算法配置，或 config.py 中的运行默认项；输出：BGR 全景图及结果文件。
流程：收集输入 -> 选择算法 -> 执行拼接 -> 写出图片 -> 更新进度。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms import (
    ApapStitcher,
    GUIDED_DEFAULT_SETTINGS,
    GUIDED_SETTING_KEYS,
    GuidedAffineStitcher,
    GuidedProgressStitcher,
    LocalRefineStitcher,
    OriginStitcher,
    ProgressStitcher,
)
from algorithms.guided_affine import validate_matching_regions
from config import (
    APAP_SETTING_KEYS,
    DEFAULT_IMAGE_LIMIT,
    DEFAULT_IMAGE_PATHS,
    DEFAULT_INPUT_FOLDER,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SETTINGS,
    DEFAULT_SHOW_PROGRESS,
    LOCAL_REFINE_SETTING_KEYS,
    STITCHER_SETTING_KEYS,
    filter_settings as _filter_settings,
    stitcher_settings as _stitcher_settings,
)
from image_io import (
    collect_images,
    guided_output_georeference,
    is_tiff_path,
    write_output_image,
)
from progress import write_progress


def stitch_images(
    image_paths: list[str],
    settings: dict | None = None,
    show_progress: bool = True,
    output_path: str | Path | None = None,
    georef_context: dict | None = None,
    guided_images: list[dict] | None = None,
    result_context: dict | None = None,
    matching_regions: list[dict] | None = None,
) -> np.ndarray | None:
    """按配置选择算法，保持旧入口的参数和返回值。"""
    requested_algorithm = str(
        (settings or {}).get("algorithm", DEFAULT_SETTINGS["algorithm"])
    ).strip().lower()
    merged_settings = dict(DEFAULT_SETTINGS)
    if requested_algorithm == "guided_affine":
        merged_settings.update(GUIDED_DEFAULT_SETTINGS)
    if settings is not None:
        merged_settings.update(settings)
    settings = merged_settings
    algorithm = str(settings["algorithm"]).strip().lower()
    matching_regions = validate_matching_regions(matching_regions)
    if matching_regions and algorithm != "guided_affine":
        raise ValueError("matching_regions 仅支持 guided_affine")
    if algorithm == "guided_affine":
        if guided_images is None:
            raise ValueError("guided_affine 必须提供每张图片的四角坐标")
        if [str(item["path"]) for item in guided_images] != list(image_paths):
            raise ValueError("guided_images 的路径和顺序必须与 image_paths 一致")
        stitcher = GuidedProgressStitcher(
            guided_images=guided_images,
            matching_regions=matching_regions,
            guided_settings=_filter_settings(settings, GUIDED_SETTING_KEYS),
            **_stitcher_settings(settings),
            show_progress=show_progress,
            output_path=output_path,
            optimized_tiff_io=settings["optimized_tiff_io"],
            parallel_image_io=settings["parallel_image_io"],
            image_io_workers=settings["image_io_workers"],
            prefetch_final_images=settings["prefetch_final_images"],
            cache_all_valid_masks=settings["cache_all_valid_masks"],
            shared_warp_maps=settings["shared_warp_maps"],
        )
        panorama = stitcher.stitch(image_paths)
        scale = float(stitcher.output_scale)
        offset = [float(value) for value in stitcher.output_offset]
        canvas_to_output = [
            [scale, 0.0, offset[0]],
            [0.0, scale, offset[1]],
            [0.0, 0.0, 1.0],
        ]
        output_to_canvas = [
            [1.0 / scale, 0.0, -offset[0] / scale],
            [0.0, 1.0 / scale, -offset[1] / scale],
            [0.0, 0.0, 1.0],
        ]
        if georef_context is not None:
            georef_context["canvas_to_output_transform"] = canvas_to_output
            georef_context["output_to_canvas_transform"] = output_to_canvas
        geographic = None
        if settings["georef_mode"] == "affine_fit":
            geographic = guided_output_georeference(image_paths[0], scale, offset)
            if geographic is None:
                raise ValueError(
                    "guided_affine 使用 affine_fit 时，第一张图必须包含 CRS 和 GeoTransform"
                )
            if georef_context is not None:
                georef_context["guided_georeference"] = geographic
        if result_context is not None:
            result_context["output_offset"] = offset
            result_context["output_scale"] = scale
            result_context["canvas_to_output_transform"] = canvas_to_output
            result_context["output_to_canvas_transform"] = output_to_canvas
            if geographic is not None:
                geotransform, projection, metadata = geographic
                result_context["geotransform"] = list(geotransform)
                result_context["projection"] = projection
                result_context["georeference_method"] = metadata["method"]
            result_context["pair_diagnostics"] = stitcher.pair_diagnostics
            result_context["output_source"] = "algorithm_blended"
            result_context["fusion_method"] = settings["blender_type"]
            result_context["debug_info"] = stitcher.debug_info()
        return panorama
    if algorithm in {"stitcher", "opencv", "default"}:
        stitcher = ProgressStitcher(
            **_stitcher_settings(settings),
            show_progress=show_progress,
            output_path=output_path,
            optimized_tiff_io=settings["optimized_tiff_io"],
            parallel_image_io=settings["parallel_image_io"],
            image_io_workers=settings["image_io_workers"],
            prefetch_final_images=settings["prefetch_final_images"],
            cache_all_valid_masks=settings["cache_all_valid_masks"],
            shared_warp_maps=settings["shared_warp_maps"],
        )
        panorama = stitcher.stitch(image_paths)
        if georef_context is not None:
            georef_context["control_points"] = stitcher.georef_control_points
        if result_context is not None:
            result_context["output_source"] = "algorithm_blended"
            result_context["fusion_method"] = settings["blender_type"]
            result_context["debug_info"] = stitcher.debug_info()
        return panorama
    if algorithm == "origin":
        stitcher = OriginStitcher(**_stitcher_settings(settings))
        return stitcher.stitch(image_paths)
    if algorithm == "apap":
        stitcher = ApapStitcher(
            **_filter_settings(settings, APAP_SETTING_KEYS),
            show_progress=show_progress,
        )
        return stitcher.stitch(image_paths)
    if algorithm == "local_refine":
        stitcher = LocalRefineStitcher(
            **_filter_settings(settings, LOCAL_REFINE_SETTING_KEYS),
            show_progress=show_progress,
        )
        return stitcher.stitch(image_paths)
    raise ValueError("algorithm 只支持 stitcher、origin、apap、local_refine 或 guided_affine")

def main() -> int:
    input_folder = Path(DEFAULT_INPUT_FOLDER).expanduser().resolve()
    output_path = Path(DEFAULT_OUTPUT_PATH).expanduser().resolve()

    if DEFAULT_IMAGE_PATHS:
        image_paths = [
            str(Path(path).expanduser().resolve()) for path in DEFAULT_IMAGE_PATHS
        ]
        if DEFAULT_IMAGE_LIMIT is not None:
            image_paths = image_paths[:DEFAULT_IMAGE_LIMIT]
        print("输入方式: config.py 中的有序图像列表")
    else:
        image_paths = collect_images(input_folder, DEFAULT_IMAGE_LIMIT)
        print(f"输入目录: {input_folder}")
    print(f"找到图片: {len(image_paths)} 张")

    if len(image_paths) < 2:
        print("至少需要 2 张图片。")
        write_progress(output_path, 100.0, "failed: 至少需要 2 张图片")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_progress(output_path, 5.0, "读取输入图片")

    settings = dict(DEFAULT_SETTINGS)

    print("拼接参数:")
    for key, value in settings.items():
        print(f"  {key}: {value}")

    try:
        georef_context: dict = {}
        panorama = stitch_images(
            image_paths, settings, show_progress=DEFAULT_SHOW_PROGRESS,
            output_path=output_path, georef_context=georef_context,
        )
    except Exception as exc:  # noqa: BLE001
        write_progress(output_path, 100.0, f"failed: {exc}")
        raise

    if panorama is None:
        print("拼接失败: Stitcher 未返回图像。")
        write_progress(output_path, 100.0, "failed: Stitcher 未返回图像")
        return 1

    try:
        write_progress(output_path, 96.0, "保存输出图像")
        saved_path = write_output_image(
            output_path, panorama, image_paths,
            georef_mode=settings["georef_mode"], georef_context=georef_context,
        )
    except RuntimeError as exc:
        print(str(exc))
        write_progress(output_path, 100.0, f"failed: {exc}")
        return 1

    write_progress(output_path, 100.0, "completed")
    print(f"输出尺寸: {panorama.shape[1]}x{panorama.shape[0]}")
    print(f"已保存: {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
