"""输入图片发现和结果写出模块。

职责：收集支持的图片、通过 GDAL 解码 TIFF/COG 及有效区域，并写出拼接结果。
输入：输入目录/图片路径、BGR ndarray 和输出路径；输出：排序后的路径或已保存文件路径。
流程：扫描输入 -> 统一为 OpenCV 的 uint8 BGR/mask -> 视觉拼接 -> 写出图片。
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


def _open_gdal_raster(path: Path):
    """只允许主 TIFF 和外部 overview，明确忽略同名外部 .msk。"""
    from osgeo import gdal  # type: ignore

    gdal.UseExceptions()
    name = path.name
    return gdal.OpenEx(
        str(path),
        gdal.OF_RASTER | gdal.OF_READONLY,
        sibling_files=[name, f"{name}.ovr", f"{name}.OVR"],
    )


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """将 GDAL 波段统一为 OpenCV 拼接需要的 uint8。"""
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if np.issubdtype(array.dtype, np.integer):
        limits = np.iinfo(array.dtype)
        scaled = (array.astype(np.float32) - float(limits.min))
        scaled *= 255.0 / float(limits.max - limits.min)
        return np.clip(scaled, 0, 255).astype(np.uint8)

    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.uint8)
    minimum = float(np.min(values[finite]))
    maximum = float(np.max(values[finite]))
    if minimum >= 0.0 and maximum <= 1.0:
        values = values * 255.0
    elif not (minimum >= 0.0 and maximum <= 255.0):
        if maximum <= minimum:
            return np.zeros(values.shape, dtype=np.uint8)
        values = (values - minimum) * (255.0 / (maximum - minimum))
    values[~finite] = 0
    return np.clip(values, 0, 255).astype(np.uint8)


def _gdal_color_band_indices(dataset) -> tuple[int, int, int] | None:
    """返回 RGB 波段索引；无颜色解释时按常见的前三波段处理。"""
    from osgeo import gdal  # type: ignore

    interpretations = {
        dataset.GetRasterBand(index).GetColorInterpretation(): index
        for index in range(1, dataset.RasterCount + 1)
    }
    required = (gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand)
    if all(value in interpretations for value in required):
        return tuple(interpretations[value] for value in required)
    if dataset.RasterCount >= 3:
        return 1, 2, 3
    return None


def read_input_image(
    path: str,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """读取 BGR 图像；TIFF 可由 GDAL 直接解码到目标尺寸。"""
    input_path = Path(path)
    if not is_tiff_path(input_path):
        flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        image = cv2.imread(str(input_path), flags)
        if image is None:
            raise RuntimeError(f"无法读取图片: {path}")
        if target_size is not None and (image.shape[1], image.shape[0]) != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR_EXACT)
        return np.ascontiguousarray(image)

    from osgeo import gdal  # type: ignore

    dataset = _open_gdal_raster(input_path)
    if dataset is None or dataset.RasterCount < 1:
        raise RuntimeError(f"无法通过 GDAL 读取 TIFF/COG: {path}")
    try:
        read_kwargs = {}
        if target_size is not None:
            width, height = (int(target_size[0]), int(target_size[1]))
            if width <= 0 or height <= 0:
                raise ValueError(f"无效的目标尺寸: {target_size}")
            read_kwargs = {
                "buf_xsize": width,
                "buf_ysize": height,
                "resample_alg": gdal.GRIORA_Bilinear,
            }
        rgb_indices = _gdal_color_band_indices(dataset)
        if rgb_indices is None:
            gray = _to_uint8(dataset.GetRasterBand(1).ReadAsArray(**read_kwargs))
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            red, green, blue = (
                _to_uint8(dataset.GetRasterBand(index).ReadAsArray(**read_kwargs))
                for index in rgb_indices
            )
            image = np.dstack((blue, green, red))
    finally:
        dataset = None

    if image.size == 0:
        raise RuntimeError(f"TIFF/COG 像素数据为空: {path}")
    return np.ascontiguousarray(image)


def read_input_image_with_size(
    path: str,
    target_size_factory: Callable[[tuple[int, int]], tuple[int, int]],
) -> tuple[np.ndarray, tuple[int, int]]:
    """单次打开输入，取得原始尺寸并按调用方计算的尺寸解码。"""
    input_path = Path(path)
    if not is_tiff_path(input_path):
        image = read_input_image(path)
        source_size = image.shape[1], image.shape[0]
        target_size = target_size_factory(source_size)
        if source_size != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR_EXACT)
        return np.ascontiguousarray(image), source_size

    from osgeo import gdal  # type: ignore

    dataset = _open_gdal_raster(input_path)
    if dataset is None or dataset.RasterCount < 1:
        raise RuntimeError(f"无法通过 GDAL 读取 TIFF/COG: {path}")
    try:
        source_size = int(dataset.RasterXSize), int(dataset.RasterYSize)
        width, height = target_size_factory(source_size)
        width, height = int(width), int(height)
        if width <= 0 or height <= 0:
            raise ValueError(f"无效的目标尺寸: {(width, height)}")
        read_kwargs = {
            "buf_xsize": width,
            "buf_ysize": height,
            "resample_alg": gdal.GRIORA_Bilinear,
        }
        rgb_indices = _gdal_color_band_indices(dataset)
        if rgb_indices is None:
            gray = _to_uint8(dataset.GetRasterBand(1).ReadAsArray(**read_kwargs))
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            red, green, blue = (
                _to_uint8(dataset.GetRasterBand(index).ReadAsArray(**read_kwargs))
                for index in rgb_indices
            )
            image = np.dstack((blue, green, red))
    finally:
        dataset = None

    if image.size == 0:
        raise RuntimeError(f"TIFF/COG 像素数据为空: {path}")
    return np.ascontiguousarray(image), source_size


def get_input_image_size(path: str) -> tuple[int, int]:
    """不解码像素地读取 TIFF/COG 的原始宽高。"""
    input_path = Path(path)
    if not is_tiff_path(input_path):
        image = read_input_image(path)
        return image.shape[1], image.shape[0]

    from osgeo import gdal  # type: ignore

    dataset = _open_gdal_raster(input_path)
    if dataset is None or dataset.RasterCount < 1:
        raise RuntimeError(f"无法通过 GDAL 读取 TIFF/COG 元数据: {path}")
    try:
        return int(dataset.RasterXSize), int(dataset.RasterYSize)
    finally:
        dataset = None


def read_valid_mask(path: str, size: tuple[int, int]) -> np.ndarray | None:
    """以目标尺寸读取 TIFF/COG 的 NoData/alpha mask；全有效时返回 None。"""
    input_path = Path(path)
    if not is_tiff_path(input_path):
        return None

    from osgeo import gdal  # type: ignore

    width, height = (int(size[0]), int(size[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"无效的 mask 尺寸: {size}")

    dataset = _open_gdal_raster(input_path)
    if dataset is None or dataset.RasterCount < 1:
        raise RuntimeError(f"无法通过 GDAL 读取 TIFF/COG mask: {path}")
    try:
        band_indices = _gdal_color_band_indices(dataset) or (1,)
        masks = []
        for index in sorted(set(band_indices)):
            band = dataset.GetRasterBand(index)
            if band.GetMaskFlags() & gdal.GMF_ALL_VALID:
                continue
            mask = band.GetMaskBand().ReadAsArray(
                buf_xsize=width,
                buf_ysize=height,
                resample_alg=gdal.GRIORA_NearestNeighbour,
            )
            masks.append(np.asarray(mask, dtype=np.uint8))

        if not masks:
            return None
        valid = masks[0]
        for mask in masks[1:]:
            valid = cv2.bitwise_and(valid, mask)
        valid = np.where(valid > 0, 255, 0).astype(np.uint8)
        return None if np.all(valid == 255) else valid
    finally:
        dataset = None


def collect_images(input_folder: Path, limit: int | None = None) -> list[str]:
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(str(input_folder / ext)))
        image_paths.extend(glob.glob(str(input_folder / ext.upper())))

    image_paths = sorted(set(image_paths))
    if limit is not None:
        image_paths = image_paths[:limit]
    return image_paths


def is_tiff_path(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}


def _select_approximate_georeference(
    image_paths: list[str],
) -> tuple[str, tuple[float, ...], str] | None:
    """选择地理位置最左上的 TIFF；无法比较时自然退回首个有效 TIFF。"""
    candidates: list[tuple[float, float, int, str, tuple[float, ...], str]] = []
    for index, raw_path in enumerate(image_paths):
        path = Path(raw_path)
        if not is_tiff_path(path):
            continue
        dataset = _open_gdal_raster(path)
        if dataset is None:
            continue
        try:
            projection = dataset.GetProjectionRef() or dataset.GetProjection() or ""
            geotransform = dataset.GetGeoTransform(can_return_null=True)
            if not projection or geotransform is None:
                continue
            transform = tuple(float(value) for value in geotransform)
            # 常见 north-up 数据中：X 越小越靠左，Y 越大越靠上。
            candidates.append(
                (transform[0], -transform[3], index, str(path), transform, projection)
            )
        finally:
            dataset = None

    if not candidates:
        return None
    # 不同 CRS 的原点数值不可直接比较，此时按用户输入顺序退回首个有效 TIFF。
    if len({candidate[5] for candidate in candidates}) != 1:
        _, _, _, source_path, transform, projection = min(
            candidates, key=lambda candidate: candidate[2]
        )
        return source_path, transform, projection
    _, _, _, source_path, transform, projection = min(candidates)
    return source_path, transform, projection


def _fit_affine_georeference(
    control_points: list[dict],
) -> tuple[tuple[float, ...], str, dict[str, float | int | str]] | None:
    """由全景像素/输入地理坐标对应点稳健拟合六参数 GeoTransform。"""
    references: dict[str, tuple[tuple[float, ...], str]] = {}
    for record in control_points:
        path = str(record["path"])
        if path in references:
            continue
        dataset = _open_gdal_raster(Path(path))
        if dataset is None:
            continue
        try:
            projection = dataset.GetProjectionRef() or dataset.GetProjection() or ""
            transform = dataset.GetGeoTransform(can_return_null=True)
            if projection and transform is not None:
                references[path] = (
                    tuple(float(value) for value in transform), projection
                )
        finally:
            dataset = None

    if not references or len({value[1] for value in references.values()}) != 1:
        return None

    panorama_points = []
    geo_points = []
    for record in control_points:
        reference = references.get(str(record["path"]))
        if reference is None:
            continue
        transform, _ = reference
        source_x = float(record["source_x"])
        source_y = float(record["source_y"])
        geo_x = transform[0] + source_x * transform[1] + source_y * transform[2]
        geo_y = transform[3] + source_x * transform[4] + source_y * transform[5]
        panorama_points.append(
            [1.0, float(record["panorama_x"]), float(record["panorama_y"])]
        )
        geo_points.append([geo_x, geo_y])

    design = np.asarray(panorama_points, dtype=np.float64)
    targets = np.asarray(geo_points, dtype=np.float64)
    if len(design) < 6 or np.linalg.matrix_rank(design) < 3:
        return None

    inliers = np.ones(len(design), dtype=bool)
    coefficients = None
    # 少量固定点使用三轮 MAD 剔除，耗时稳定且不依赖地理单位阈值。
    for _ in range(3):
        if np.count_nonzero(inliers) < 6:
            return None
        coefficients, _, rank, _ = np.linalg.lstsq(
            design[inliers], targets[inliers], rcond=None
        )
        if rank < 3:
            return None
        residuals = np.linalg.norm(design @ coefficients - targets, axis=1)
        active = residuals[inliers]
        median = float(np.median(active))
        mad = float(np.median(np.abs(active - median)))
        threshold = median + max(3.0 * 1.4826 * mad, 1e-12)
        updated = inliers & (residuals <= threshold)
        if np.array_equal(updated, inliers):
            break
        inliers = updated

    if coefficients is None or np.count_nonzero(inliers) < 6:
        return None
    coefficients, _, rank, _ = np.linalg.lstsq(
        design[inliers], targets[inliers], rcond=None
    )
    if rank < 3:
        return None
    residuals = np.linalg.norm(
        design[inliers] @ coefficients - targets[inliers], axis=1
    )
    rmse_units = float(np.sqrt(np.mean(residuals**2)))
    linear = np.array(
        [[coefficients[1, 0], coefficients[2, 0]],
         [coefficients[1, 1], coefficients[2, 1]]],
        dtype=np.float64,
    )
    area_scale = float(np.sqrt(abs(np.linalg.det(linear))))
    if not np.isfinite(area_scale) or area_scale <= 1e-15:
        return None
    rmse_pixels = rmse_units / area_scale
    # 一个全局仿射模型若连已有控制点都无法在 25 像素内解释，就不应覆盖
    # copy 回退结果；局部非线性误差仍会通过 APPROXIMATE 元数据明确标识。
    if not np.isfinite(rmse_pixels) or rmse_pixels > 25.0:
        return None
    geotransform = (
        float(coefficients[0, 0]),
        float(coefficients[1, 0]),
        float(coefficients[2, 0]),
        float(coefficients[0, 1]),
        float(coefficients[1, 1]),
        float(coefficients[2, 1]),
    )
    projection = next(iter(references.values()))[1]
    metadata: dict[str, float | int | str] = {
        "method": "AFFINE_FIT",
        "point_count": int(np.count_nonzero(inliers)),
        "candidate_count": int(len(design)),
        "rmse_pixels": rmse_pixels,
        "rmse_units": rmse_units,
    }
    return geotransform, projection, metadata


def guided_output_georeference(
    reference_path: str | Path,
    output_scale: float,
    output_offset: list[float] | tuple[float, float],
) -> tuple[tuple[float, ...], str, dict[str, str]] | None:
    """由第一张图的地理参考推导 guided 输出的 GeoTransform。"""
    dataset = _open_gdal_raster(Path(reference_path))
    if dataset is None:
        return None
    try:
        projection = dataset.GetProjectionRef() or dataset.GetProjection() or ""
        source = dataset.GetGeoTransform(can_return_null=True)
    finally:
        dataset = None
    if not projection or source is None:
        return None
    scale = float(output_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("guided_affine output_scale 必须是有限正数")
    offset_x, offset_y = map(float, output_offset)
    if not np.isfinite([offset_x, offset_y]).all():
        raise ValueError("guided_affine output_offset 必须是有限数值")
    source = tuple(float(value) for value in source)
    # output = canvas * scale + offset；第一张图像素坐标就是 canvas 坐标。
    canvas_origin_x = -offset_x / scale
    canvas_origin_y = -offset_y / scale
    geotransform = (
        source[0] + canvas_origin_x * source[1] + canvas_origin_y * source[2],
        source[1] / scale,
        source[2] / scale,
        source[3] + canvas_origin_x * source[4] + canvas_origin_y * source[5],
        source[4] / scale,
        source[5] / scale,
    )
    return geotransform, projection, {
        "method": "GUIDED_FIRST_IMAGE",
        "source": str(reference_path),
    }


def write_output_image(
    output_path: Path,
    image_bgr: np.ndarray,
    image_paths: list[str],
    georef_mode: str = "copy",
    georef_context: dict | None = None,
) -> str:
    if not is_tiff_path(output_path):
        if not cv2.imwrite(str(output_path), image_bgr):
            raise RuntimeError(f"写出失败: {output_path}")
        return str(output_path)

    from osgeo import gdal  # type: ignore
    gdal.UseExceptions()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    georef_mode = str(georef_mode).strip().lower()
    if georef_mode not in {"affine_fit", "copy", "none"}:
        raise ValueError("georef_mode 只支持 affine_fit、copy 或 none")
    fitted_georeference = None
    guided_georeference = (
        georef_context.get("guided_georeference")
        if georef_context is not None else None
    )
    if georef_mode == "affine_fit" and georef_context is not None and guided_georeference is None:
        fitted_georeference = _fit_affine_georeference(
            list(georef_context.get("control_points", []))
        )
    approximate_georeference = (
        _select_approximate_georeference(image_paths)
        if georef_mode != "none" and fitted_georeference is None and guided_georeference is None
        else None
    )

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp.tif")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        if output_path.exists():
            output_path.unlink()

        tmp_ds = gdal.GetDriverByName("GTiff").Create(
            str(tmp_path),
            width,
            height,
            3,
            gdal.GDT_Byte,
            options=[
                "TILED=YES",
                "BLOCKXSIZE=512",
                "BLOCKYSIZE=512",
                "COMPRESS=DEFLATE",
                "PREDICTOR=2",
                "BIGTIFF=IF_SAFER",
                "INTERLEAVE=PIXEL",
                "PHOTOMETRIC=RGB",
            ],
        )
        tmp_ds.SetMetadataItem("TIFFTAG_SOFTWARE", "stitching_headless")
        tmp_ds.SetMetadataItem("COG_LAYOUT", "true")
        if georef_context is not None:
            canvas_to_output = georef_context.get("canvas_to_output_transform")
            output_to_canvas = georef_context.get("output_to_canvas_transform")
            if canvas_to_output is not None:
                tmp_ds.SetMetadataItem(
                    "CANVAS_TO_OUTPUT_TRANSFORM",
                    json.dumps(canvas_to_output, separators=(",", ":")),
                )
            if output_to_canvas is not None:
                tmp_ds.SetMetadataItem(
                    "OUTPUT_TO_CANVAS_TRANSFORM",
                    json.dumps(output_to_canvas, separators=(",", ":")),
                )
            if canvas_to_output is not None or output_to_canvas is not None:
                tmp_ds.SetMetadataItem("STITCHING_COORDINATE_SYSTEM", "FIRST_IMAGE_PIXEL_CANVAS")
        if guided_georeference is not None:
            geotransform, projection, guided_metadata = guided_georeference
            tmp_ds.SetGeoTransform(geotransform)
            tmp_ds.SetProjection(projection)
            tmp_ds.SetMetadataItem("GEOREFERENCE_QUALITY", "REFERENCE_ANCHORED")
            tmp_ds.SetMetadataItem("GEOREFERENCE_METHOD", str(guided_metadata["method"]))
            tmp_ds.SetMetadataItem("GEOREFERENCE_SOURCE", str(guided_metadata["source"]))
            print(
                "[COG] 已根据 guided 第一张图写入 CRS/GeoTransform: "
                f"{guided_metadata['source']}",
                flush=True,
            )
        elif fitted_georeference is not None:
            geotransform, projection, fit_metadata = fitted_georeference
            tmp_ds.SetGeoTransform(geotransform)
            tmp_ds.SetProjection(projection)
            tmp_ds.SetMetadataItem("GEOREFERENCE_QUALITY", "APPROXIMATE")
            tmp_ds.SetMetadataItem("GEOREFERENCE_METHOD", str(fit_metadata["method"]))
            tmp_ds.SetMetadataItem("GEOREFERENCE_POINT_COUNT", str(fit_metadata["point_count"]))
            tmp_ds.SetMetadataItem("GEOREFERENCE_CANDIDATE_COUNT", str(fit_metadata["candidate_count"]))
            tmp_ds.SetMetadataItem("GEOREFERENCE_RMSE_PIXELS", f"{fit_metadata['rmse_pixels']:.6g}")
            tmp_ds.SetMetadataItem("GEOREFERENCE_RMSE_UNITS", f"{fit_metadata['rmse_units']:.6g}")
            print(
                "[COG] 已使用 affine 控制点拟合 CRS/GeoTransform: "
                f"points={fit_metadata['point_count']}/{fit_metadata['candidate_count']}, "
                f"rmse={fit_metadata['rmse_pixels']:.3f}px",
                flush=True,
            )
        elif approximate_georeference is not None:
            source_path, geotransform, projection = approximate_georeference
            tmp_ds.SetGeoTransform(geotransform)
            tmp_ds.SetProjection(projection)
            tmp_ds.SetMetadataItem("GEOREFERENCE_QUALITY", "APPROXIMATE")
            tmp_ds.SetMetadataItem("GEOREFERENCE_SOURCE", source_path)
            tmp_ds.SetMetadataItem("GEOREFERENCE_METHOD", "COPY")
            print(
                "[COG] affine 拟合不可用，已复制输入图 CRS/GeoTransform（仅粗略定位）: "
                f"{source_path}", flush=True,
            )
        else:
            print("[COG] 未写入 CRS/GeoTransform", flush=True)
        for band_index in range(3):
            tmp_ds.GetRasterBand(band_index + 1).WriteArray(image_rgb[:, :, band_index])
        tmp_ds.FlushCache()
        tmp_ds = None

        cog_ds = gdal.Translate(
            str(output_path),
            str(tmp_path),
            format="COG",
            creationOptions=[
                "COMPRESS=DEFLATE",
                "PREDICTOR=2",
                "BLOCKSIZE=512",
                "BIGTIFF=IF_SAFER",
                "OVERVIEWS=AUTO",
                "RESAMPLING=AVERAGE",
            ],
        )
        cog_ds.FlushCache()
        cog_ds = None
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return str(output_path)
