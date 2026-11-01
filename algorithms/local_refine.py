"""全局对齐后局部微调的拼接算法。

职责：在全局单应性对齐基础上，按高度分带估计残差并对重叠区域做局部 remap。
输入：按顺序排列的图片路径；输出：局部校正并融合后的 BGR ndarray。
流程：自动匹配 -> 全局变换 -> 分带残差估计 -> 局部重映射 -> 羽化融合 -> 裁剪。
性能敏感点：全画布 warpPerspective、meshgrid/remap、distanceTransform 与浮点融合。
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .apap import ApapStitcher

class LocalRefineStitcher(ApapStitcher):
    BAND_COUNT = 5
    MIN_BAND_MATCHES = 4
    REFINE_FEATHER_RATIO = 0.12
    MAX_RESIDUAL_RATIO = 0.03

    def __init__(
        self,
        *,
        detector: str = "sift",
        nfeatures: int = 8000,
        medium_megapix: float = 1.0,
        confidence_threshold: float = 0.3,
        crop: bool = True,
        show_progress: bool = True,
    ) -> None:
        super().__init__(
            detector=detector,
            nfeatures=nfeatures,
            medium_megapix=medium_megapix,
            confidence_threshold=confidence_threshold,
            crop=crop,
            apap_grid_parts=1,
            apap_sigma=0.5,
            apap_min_matches=12,
            apap_blend_width=260,
            show_progress=show_progress,
        )

    def _progress(self, message: str) -> None:
        if not self.show_progress:
            return
        self._stage_index += 1
        elapsed = time.perf_counter() - self._started_at
        print(
            f"[LOCAL_REFINE {self._stage_index:02d}] {message} ({elapsed:.1f}s)",
            flush=True,
        )

    def stitch(self, image_paths: list[str]) -> np.ndarray | None:
        self._stage_index = 0
        self._started_at = time.perf_counter()

        self._progress("读取图片")
        images = [self._read_image(path) for path in image_paths]
        panorama = images[0]
        pano_mask = np.full(panorama.shape[:2], 255, dtype=np.uint8)

        for idx, image in enumerate(images[1:], 2):
            self._progress(f"全局对齐并局部校正第 {idx}/{len(images)} 张")
            panorama, pano_mask = self._append_image(panorama, pano_mask, image)

        if self.crop:
            self._progress("裁剪有效区域")
            panorama = self._crop_to_mask(panorama, pano_mask)
        return panorama

    def _append_image(
        self,
        panorama: np.ndarray,
        pano_mask: np.ndarray,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        src_pts, dst_pts = self._match_points(image, panorama)
        homography, inlier_mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            max(2.0, 8.0 * (1.0 - min(self.confidence_threshold, 0.95))),
        )
        if homography is None or inlier_mask is None:
            raise RuntimeError("local_refine 全局单应性估计失败")

        inliers = inlier_mask.ravel().astype(bool)
        min_inliers = max(8, self.apap_min_matches)
        if int(inliers.sum()) < min_inliers:
            raise RuntimeError(
                f"local_refine 有效匹配点不足: {int(inliers.sum())} < {min_inliers}"
            )
        src_pts = src_pts[inliers]
        dst_pts = dst_pts[inliers]
        base, base_mask, warped, warped_mask, shifted_homography, shift = (
            self._global_warp(panorama, pano_mask, image, homography)
        )
        shifted_dst_pts = dst_pts + np.float32(shift)

        try:
            refined, refined_mask = self._refine_warped_image(
                warped,
                warped_mask,
                base_mask,
                src_pts,
                shifted_dst_pts,
                shifted_homography,
                image.shape[0],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"local_refine 局部校正失败，回退全局融合: {exc}", flush=True)
            refined, refined_mask = warped, warped_mask

        blended = self._blend(base, base_mask, refined, refined_mask)
        merged_mask = cv2.bitwise_or(base_mask, refined_mask)
        return blended, merged_mask

    def _global_warp(
        self,
        panorama: np.ndarray,
        pano_mask: np.ndarray,
        image: np.ndarray,
        homography: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
        height, width = image.shape[:2]
        pano_height, pano_width = panorama.shape[:2]
        source_corners = np.float32(
            [[0, 0], [width, 0], [width, height], [0, height]]
        ).reshape(-1, 1, 2)
        pano_corners = np.float32(
            [[0, 0], [pano_width, 0], [pano_width, pano_height], [0, pano_height]]
        ).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(source_corners, homography)
        all_corners = np.vstack([pano_corners, warped_corners]).reshape(-1, 2)

        min_xy = np.floor(all_corners.min(axis=0)).astype(int)
        max_xy = np.ceil(all_corners.max(axis=0)).astype(int)
        shift_x = max(0, -int(min_xy[0]))
        shift_y = max(0, -int(min_xy[1]))
        out_width = int(max_xy[0] + shift_x)
        out_height = int(max_xy[1] + shift_y)
        if out_width <= 0 or out_height <= 0:
            raise RuntimeError("local_refine 输出画布尺寸异常")

        translation = np.array(
            [[1.0, 0.0, shift_x], [0.0, 1.0, shift_y], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        shifted_homography = translation @ homography
        base = cv2.warpPerspective(
            panorama,
            translation,
            (out_width, out_height),
            flags=cv2.INTER_LINEAR,
        )
        base_mask = cv2.warpPerspective(
            pano_mask,
            translation,
            (out_width, out_height),
            flags=cv2.INTER_NEAREST,
        )
        image_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        warped = cv2.warpPerspective(
            image,
            shifted_homography,
            (out_width, out_height),
            flags=cv2.INTER_LINEAR,
        )
        warped_mask = cv2.warpPerspective(
            image_mask,
            shifted_homography,
            (out_width, out_height),
            flags=cv2.INTER_NEAREST,
        )
        return base, base_mask, warped, warped_mask, shifted_homography, (shift_x, shift_y)

    def _refine_warped_image(
        self,
        warped: np.ndarray,
        warped_mask: np.ndarray,
        base_mask: np.ndarray,
        src_pts: np.ndarray,
        shifted_dst_pts: np.ndarray,
        shifted_homography: np.ndarray,
        source_height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        predicted = cv2.perspectiveTransform(
            src_pts.reshape(-1, 1, 2), shifted_homography
        ).reshape(-1, 2)
        residuals = shifted_dst_pts - predicted
        centers, band_offsets = self._estimate_band_offsets(
            predicted[:, 1],
            residuals,
            warped.shape[0],
            source_height,
        )

        grid_y = np.arange(warped.shape[0], dtype=np.float32)
        offset_x = np.interp(grid_y, centers, band_offsets[:, 0]).astype(np.float32)
        offset_y = np.interp(grid_y, centers, band_offsets[:, 1]).astype(np.float32)

        strength = self._refine_strength(base_mask, warped_mask)
        if float(strength.max()) <= 0:
            raise RuntimeError("没有可局部校正的重叠区域")

        x_coords, y_coords = np.meshgrid(
            np.arange(warped.shape[1], dtype=np.float32),
            grid_y,
        )
        map_x = x_coords - offset_x[:, None] * strength
        map_y = y_coords - offset_y[:, None] * strength
        refined = cv2.remap(
            warped,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        refined_mask = cv2.remap(
            warped_mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return refined, refined_mask

    def _estimate_band_offsets(
        self,
        y_values: np.ndarray,
        residuals: np.ndarray,
        canvas_height: int,
        source_height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        edges = np.linspace(0.0, float(canvas_height), self.BAND_COUNT + 1)
        centers = (edges[:-1] + edges[1:]) * 0.5
        offsets = np.full((self.BAND_COUNT, 2), np.nan, dtype=np.float32)

        max_residual = max(20.0, float(source_height) * self.MAX_RESIDUAL_RATIO)
        for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            in_band = (y_values >= lo) & (y_values < hi)
            if int(in_band.sum()) < self.MIN_BAND_MATCHES:
                continue
            median = np.median(residuals[in_band], axis=0)
            offsets[idx] = np.clip(median, -max_residual, max_residual)

        valid = np.isfinite(offsets[:, 0])
        if not valid.any():
            raise RuntimeError("重叠区分带匹配点不足")
        if valid.sum() == 1:
            offsets[:] = offsets[valid][0]
        else:
            offsets[:, 0] = np.interp(centers, centers[valid], offsets[valid, 0])
            offsets[:, 1] = np.interp(centers, centers[valid], offsets[valid, 1])

        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        for _ in range(2):
            padded = np.pad(offsets, ((1, 1), (0, 0)), mode="edge")
            offsets = (
                padded[:-2] * kernel[0]
                + padded[1:-1] * kernel[1]
                + padded[2:] * kernel[2]
            )
        return centers, offsets

    def _refine_strength(self, base_mask: np.ndarray, warped_mask: np.ndarray) -> np.ndarray:
        overlap = ((base_mask > 0) & (warped_mask > 0)).astype(np.uint8)
        if not overlap.any():
            return np.zeros(base_mask.shape, dtype=np.float32)

        base_distance = cv2.distanceTransform(
            (base_mask > 0).astype(np.uint8), cv2.DIST_L2, 3
        ).astype(np.float32)
        feather = max(160.0, base_mask.shape[1] * self.REFINE_FEATHER_RATIO)
        strength = np.clip(base_distance / feather, 0.0, 1.0)
        strength *= (warped_mask > 0).astype(np.float32)
        return strength

