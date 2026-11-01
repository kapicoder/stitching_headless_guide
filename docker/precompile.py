"""Build-time warm-up for third-party native/JIT dependencies."""

import cv2
import numpy as np
from osgeo import gdal, gdal_array


def main() -> None:
    # 构建阶段直接验证 NumPy 扩展，避免运行时才出现 `_gdal_array` 缺失。
    if gdal_array.NumericTypeCodeToGDALTypeCode(np.dtype("uint8")) != gdal.GDT_Byte:
        raise RuntimeError("GDAL NumPy array bridge is unavailable")
    import largestinteriorrectangle
    from stitching import Stitcher

    mask = np.ones((64, 64), dtype=np.bool_)
    mask[:4, :] = False
    mask[:, :4] = False
    contour = np.array([[4, 4], [63, 4], [63, 63], [4, 63]], dtype=np.int32)
    largestinteriorrectangle.lir(mask)
    largestinteriorrectangle.lir(mask, contour)

    sample = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(sample, (16, 16), (112, 112), 255, 2)
    cv2.line(sample, (16, 112), (112, 16), 180, 2)
    for detector in (cv2.SIFT_create(nfeatures=64), cv2.ORB_create(nfeatures=64)):
        detector.detectAndCompute(sample, None)

    Stitcher(crop=True)
    if gdal.GetDriverByName("GTiff") is None or gdal.GetDriverByName("COG") is None:
        raise RuntimeError("GDAL image does not provide GTiff and COG drivers")


if __name__ == "__main__":
    main()
