"""APAP 局部单应性拼接算法。

职责：完成特征检测匹配、局部加权单应性网格变换、羽化融合与有效区域裁剪。
输入：按顺序排列的图片路径；输出：拼接后的 BGR ndarray。
流程：读取图片 -> 逐图特征匹配 -> 全局单应性 -> APAP 网格变换 -> 融合 -> 裁剪。
性能敏感点：SIFT/BFMatcher、逐网格 SVD、透视变换和全画布距离变换。
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from image_io import read_input_image

class ApapStitcher:
    def __init__(
        self,
        *,
        detector: str = "sift",
        nfeatures: int = 8000,
        medium_megapix: float = 1.0,
        confidence_threshold: float = 0.3,
        crop: bool = True,
        apap_grid_parts: int = 2,
        apap_grid_size: int | None = None,
        apap_sigma: float = 0.15,
        apap_min_matches: int = 12,
        apap_blend_width: int = 80,
        show_progress: bool = True,
    ) -> None:
        self.detector = detector
        self.nfeatures = nfeatures
        self.medium_megapix = medium_megapix
        self.confidence_threshold = confidence_threshold
        self.crop = crop
        self.apap_grid_parts = apap_grid_size if apap_grid_size is not None else apap_grid_parts
        self.apap_sigma = apap_sigma
        self.apap_min_matches = apap_min_matches
        self.apap_blend_width = apap_blend_width
        self.show_progress = show_progress
        self._stage_index = 0
        self._started_at = time.perf_counter()

    def _progress(self, message: str) -> None:
        if not self.show_progress:
            return
        self._stage_index += 1
        elapsed = time.perf_counter() - self._started_at
        print(f"[APAP {self._stage_index:02d}] {message} ({elapsed:.1f}s)", flush=True)

    def stitch(self, image_paths: list[str]) -> np.ndarray | None:
        self._stage_index = 0
        self._started_at = time.perf_counter()

        self._progress("读取图片")
        images = [self._read_image(path) for path in image_paths]
        panorama = images[0]
        pano_mask = np.full(panorama.shape[:2], 255, dtype=np.uint8)

        for idx, image in enumerate(images[1:], 2):
            self._progress(f"APAP 匹配并拼接第 {idx}/{len(images)} 张")
            panorama, pano_mask = self._append_image(panorama, pano_mask, image)

        if self.crop:
            self._progress("裁剪有效区域")
            panorama = self._crop_to_mask(panorama, pano_mask)
        return panorama

    def _read_image(self, path: str) -> np.ndarray:
        return read_input_image(path)

    def _append_image(
        self,
        panorama: np.ndarray,
        pano_mask: np.ndarray,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        src_pts, dst_pts = self._match_points(image, panorama)
        min_inliers = max(4, self.apap_min_matches)
        homography, inlier_mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            max(2.0, 8.0 * (1.0 - min(self.confidence_threshold, 0.95))),
        )
        if homography is None or inlier_mask is None:
            raise RuntimeError("APAP 全局单应性估计失败")

        inliers = inlier_mask.ravel().astype(bool)
        if int(inliers.sum()) < min_inliers:
            raise RuntimeError(
                f"APAP 有效匹配点不足: {int(inliers.sum())} < {min_inliers}"
            )
        src_pts = src_pts[inliers]
        dst_pts = dst_pts[inliers]

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
            raise RuntimeError("APAP 输出画布尺寸异常")

        translation = np.array(
            [[1.0, 0.0, shift_x], [0.0, 1.0, shift_y], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        shifted_homography = translation @ homography
        shifted_dst_pts = dst_pts + np.float32([shift_x, shift_y])

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

        warped, warped_mask = self._mesh_warp(
            image,
            (out_width, out_height),
            src_pts,
            shifted_dst_pts,
            shifted_homography,
        )
        blended = self._blend(base, base_mask, warped, warped_mask)
        merged_mask = cv2.bitwise_or(base_mask, warped_mask)
        return blended, merged_mask

    def _match_points(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        source_scale, source_small = self._resize_for_features(source)
        target_scale, target_small = self._resize_for_features(target)
        detector, norm_type = self._create_detector()

        keypoints_src, descriptors_src = detector.detectAndCompute(source_small, None)
        keypoints_dst, descriptors_dst = detector.detectAndCompute(target_small, None)
        if descriptors_src is None or descriptors_dst is None:
            raise RuntimeError("APAP 特征点不足，无法匹配")

        matcher = cv2.BFMatcher(norm_type)
        raw_matches = matcher.knnMatch(descriptors_src, descriptors_dst, k=2)
        good_matches = []
        ratio = min(0.92, max(0.55, 0.75 + (0.3 - self.confidence_threshold) * 0.2))
        for pair in raw_matches:
            if len(pair) != 2:
                continue
            first, second = pair
            if first.distance < ratio * second.distance:
                good_matches.append(first)

        if len(good_matches) < self.apap_min_matches:
            raise RuntimeError(
                f"APAP 匹配点不足: {len(good_matches)} < {self.apap_min_matches}"
            )

        src_pts = np.float32(
            [keypoints_src[match.queryIdx].pt for match in good_matches]
        )
        dst_pts = np.float32(
            [keypoints_dst[match.trainIdx].pt for match in good_matches]
        )
        src_pts /= source_scale
        dst_pts /= target_scale
        return src_pts, dst_pts

    def _resize_for_features(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        if self.medium_megapix <= 0:
            return 1.0, image
        pixels = image.shape[0] * image.shape[1]
        target_pixels = self.medium_megapix * 1_000_000
        if pixels <= target_pixels:
            return 1.0, image
        scale = float((target_pixels / pixels) ** 0.5)
        resized = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        return scale, resized

    def _create_detector(self):
        detector = self.detector.lower()
        if detector == "sift":
            return cv2.SIFT_create(nfeatures=self.nfeatures), cv2.NORM_L2
        if detector == "orb":
            return cv2.ORB_create(nfeatures=self.nfeatures), cv2.NORM_HAMMING
        if detector == "akaze":
            return cv2.AKAZE_create(), cv2.NORM_HAMMING
        if detector == "brisk":
            return cv2.BRISK_create(), cv2.NORM_HAMMING
        raise ValueError("APAP detector 只支持 sift、orb、akaze、brisk")

    def _mesh_warp(
        self,
        image: np.ndarray,
        output_size: tuple[int, int],
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        global_homography: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        out_width, out_height = output_size
        height, width = image.shape[:2]
        x_parts, y_parts = self._grid_shape(width, height, self.apap_grid_parts)
        xs = self._grid_axis(width, x_parts)
        ys = self._grid_axis(height, y_parts)
        vertex_map = {}

        for y in ys:
            for x in xs:
                source_vertex = np.float32([[[x, y]]])
                center = cv2.perspectiveTransform(source_vertex, global_homography)[0, 0]
                local_h = self._weighted_homography(
                    src_pts,
                    dst_pts,
                    center,
                    height,
                    global_homography,
                )
                warped = cv2.perspectiveTransform(source_vertex, local_h)[0, 0]
                if not np.all(np.isfinite(warped)):
                    warped = center
                vertex_map[(x, y)] = warped.astype(np.float32)

        canvas = np.zeros((out_height, out_width, 3), dtype=image.dtype)
        mask = np.zeros((out_height, out_width), dtype=np.uint8)

        for y0, y1 in zip(ys[:-1], ys[1:]):
            for x0, x1 in zip(xs[:-1], xs[1:]):
                patch = image[y0:y1, x0:x1]
                if patch.size == 0:
                    continue
                src_quad = np.float32(
                    [[0, 0], [x1 - x0, 0], [x1 - x0, y1 - y0], [0, y1 - y0]]
                )
                dst_quad = np.float32(
                    [
                        vertex_map[(x0, y0)],
                        vertex_map[(x1, y0)],
                        vertex_map[(x1, y1)],
                        vertex_map[(x0, y1)],
                    ]
                )
                self._warp_patch(patch, src_quad, dst_quad, canvas, mask)
        return canvas, mask

    def _grid_shape(self, width: int, height: int, parts: int) -> tuple[int, int]:
        parts = max(1, int(parts))
        if parts == 1:
            return 1, 1
        aspect = max(width, 1) / max(height, 1)
        x_parts = max(1, int(round((parts * aspect) ** 0.5)))
        y_parts = max(1, int(np.ceil(parts / x_parts)))
        while x_parts * y_parts < parts:
            if width / x_parts >= height / y_parts:
                x_parts += 1
            else:
                y_parts += 1
        return x_parts, y_parts

    def _grid_axis(self, max_value: int, parts: int) -> list[int]:
        parts = max(1, int(parts))
        return [int(round(max_value * idx / parts)) for idx in range(parts + 1)]

    def _weighted_homography(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        center: np.ndarray,
        image_height: int,
        fallback: np.ndarray,
    ) -> np.ndarray:
        distances = np.linalg.norm(dst_pts - center.reshape(1, 2), axis=1)
        sigma = self._sigma_pixels(image_height)
        weights = np.exp(-(distances**2) / (2.0 * sigma**2))
        weights = np.maximum(weights, 1e-4)
        strongest = np.argsort(weights)[-max(4, self.apap_min_matches) :]
        src = src_pts[strongest]
        dst = dst_pts[strongest]
        w = weights[strongest]

        rows = []
        for (x, y), (u, v), weight in zip(src, dst, w):
            scale = float(weight) ** 0.5
            rows.append(scale * np.array([-x, -y, -1, 0, 0, 0, u * x, u * y, u]))
            rows.append(scale * np.array([0, 0, 0, -x, -y, -1, v * x, v * y, v]))
        design = np.asarray(rows, dtype=np.float64)
        try:
            _, _, vh = np.linalg.svd(design)
            local_h = vh[-1].reshape(3, 3)
            if abs(local_h[2, 2]) < 1e-8:
                return fallback
            local_h = local_h / local_h[2, 2]
            if not np.all(np.isfinite(local_h)):
                return fallback
            return local_h
        except np.linalg.LinAlgError:
            return fallback

    def _sigma_pixels(self, image_height: int) -> float:
        if 0 < self.apap_sigma <= 1:
            return max(1.0, float(self.apap_sigma) * float(image_height))
        return max(1.0, float(self.apap_sigma))

    def _warp_patch(
        self,
        patch: np.ndarray,
        src_quad: np.ndarray,
        dst_quad: np.ndarray,
        canvas: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        height, width = canvas.shape[:2]
        min_xy = np.floor(dst_quad.min(axis=0)).astype(int)
        max_xy = np.ceil(dst_quad.max(axis=0)).astype(int)
        x0 = max(0, min_xy[0] - 2)
        y0 = max(0, min_xy[1] - 2)
        x1 = min(width, max_xy[0] + 2)
        y1 = min(height, max_xy[1] + 2)
        if x1 <= x0 or y1 <= y0:
            return

        transform = cv2.getPerspectiveTransform(src_quad, dst_quad)
        transform[0, 2] -= x0
        transform[1, 2] -= y0
        patch_mask = np.full(patch.shape[:2], 255, dtype=np.uint8)
        warped_patch = cv2.warpPerspective(
            patch,
            transform,
            (x1 - x0, y1 - y0),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_mask = cv2.warpPerspective(
            patch_mask,
            transform,
            (x1 - x0, y1 - y0),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        region = warped_mask > 0
        canvas[y0:y1, x0:x1][region] = warped_patch[region]
        mask[y0:y1, x0:x1][region] = 255

    def _blend(
        self,
        base: np.ndarray,
        base_mask: np.ndarray,
        warped: np.ndarray,
        warped_mask: np.ndarray,
    ) -> np.ndarray:
        base_valid = base_mask > 0
        warped_valid = warped_mask > 0
        overlap = base_valid & warped_valid
        output = base.copy()
        output[warped_valid & ~base_valid] = warped[warped_valid & ~base_valid]
        if not overlap.any():
            return output

        base_dist = cv2.distanceTransform(base_mask, cv2.DIST_L2, 3).astype(np.float32)
        warped_dist = cv2.distanceTransform(warped_mask, cv2.DIST_L2, 3).astype(np.float32)
        if self.apap_blend_width > 0:
            base_dist = np.minimum(base_dist, float(self.apap_blend_width))
            warped_dist = np.minimum(warped_dist, float(self.apap_blend_width))
        denom = base_dist + warped_dist + 1e-6
        alpha = (warped_dist / denom)[..., None]
        mixed = base.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha
        output[overlap] = mixed[overlap].astype(np.uint8)
        return output

    def _crop_to_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        coords = cv2.findNonZero(mask)
        if coords is None:
            return image
        x, y, w, h = cv2.boundingRect(coords)
        return image[y : y + h, x : x + w]


