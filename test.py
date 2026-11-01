"""直接调用 stitching.Stitcher 的最简拼接测试。

职责：不展开 Stitcher 内部阶段，使用 config.py 中的默认输入和算法参数完成一次拼接。
输入：优先使用 DEFAULT_IMAGE_PATHS 的有序列表，否则扫描 DEFAULT_INPUT_FOLDER。
输出：保存到 DEFAULT_OUTPUT_PATH，并在控制台打印输入顺序、输出尺寸和总耗时。
流程：收集路径 -> 创建 Stitcher -> 一次调用 stitch -> 保存结果。
"""

from __future__ import annotations

import time
from pathlib import Path

from algorithms import OriginStitcher
from config import (
    DEFAULT_IMAGE_LIMIT,
    DEFAULT_IMAGE_PATHS,
    DEFAULT_INPUT_FOLDER,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SETTINGS,
    stitcher_settings,
)
from image_io import collect_images, write_output_image


def get_image_paths() -> list[str]:
    """按照配置返回有序输入路径。"""
    if DEFAULT_IMAGE_PATHS:
        image_paths = [
            str(Path(path).expanduser().resolve()) for path in DEFAULT_IMAGE_PATHS
        ]
        if DEFAULT_IMAGE_LIMIT is not None:
            image_paths = image_paths[:DEFAULT_IMAGE_LIMIT]
        return image_paths

    input_folder = Path(DEFAULT_INPUT_FOLDER).expanduser().resolve()
    return collect_images(input_folder, DEFAULT_IMAGE_LIMIT)


def main() -> int:
    image_paths = get_image_paths()
    if len(image_paths) < 2:
        print("至少需要 2 张图片。")
        return 1

    missing_paths = [path for path in image_paths if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(f"输入图片不存在: {missing_paths}")

    print(f"参与拼接图片数: {len(image_paths)}")
    for index, path in enumerate(image_paths, 1):
        print(f"  {index}: {path}")

    settings = stitcher_settings(DEFAULT_SETTINGS)
    print("Stitcher 参数:")
    for key, value in settings.items():
        print(f"  {key}: {value}")

    started_at = time.perf_counter()
    stitcher = OriginStitcher(**settings)
    panorama = stitcher.stitch(image_paths)
    stitch_elapsed = time.perf_counter() - started_at

    if panorama is None:
        raise RuntimeError("拼接失败: Stitcher 未返回图像")

    output_path = Path(DEFAULT_OUTPUT_PATH).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = write_output_image(output_path, panorama, image_paths)
    total_elapsed = time.perf_counter() - started_at

    print(f"输出尺寸: {panorama.shape[1]}x{panorama.shape[0]}")
    print(f"拼接耗时: {stitch_elapsed:.2f}s")
    print(f"总耗时（含保存）: {total_elapsed:.2f}s")
    print(f"已保存: {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
