"""由用户四角提供初始仿射位置，仅在预计重叠区进行视觉配准。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import DEFAULT_SETTINGS
from image_io import get_input_image_size, read_input_image_with_size, read_valid_mask
from progress import write_progress


GUIDED_DEFAULT_SETTINGS: dict[str, Any] = {
    key: DEFAULT_SETTINGS[key] for key in (
        "guided_min_overlap_pixels", "guided_overlap_margin_ratio",
        "guided_min_inliers", "guided_ransac_threshold_px",
        "guided_max_translation_ratio", "guided_max_rotation_degrees",
        "guided_max_scale_change", "guided_max_shear",
    )
}

# GuidedAffineStitcher 能直接接收的设置；由拼接分派层据此过滤全局配置。
GUIDED_SETTING_KEYS = {
    "detector", "nfeatures", "medium_megapix", "low_megapix",
    "final_megapix", "finder", "compensator", "blender_type",
    "blend_strength", "crop", "guided_min_overlap_pixels",
    "guided_overlap_margin_ratio", "guided_min_inliers",
    "guided_ransac_threshold_px",
    "guided_max_translation_ratio", "guided_max_rotation_degrees",
    "guided_max_scale_change", "guided_max_shear",
}


def build_guided_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """复用原始 config.py，并在算法模块内叠加 guided_affine 配置。"""
    try:
        import config as base
    except ImportError:  # pragma: no cover - package import fallback
        from .. import config as base

    defaults = {**base.DEFAULT_SETTINGS, **GUIDED_DEFAULT_SETTINGS}
    setting_keys = set(defaults)
    accepted_keys = setting_keys | base.LEGACY_SETTING_KEYS
    settings = dict(defaults)
    nested = payload.get("settings")
    if nested is not None:
        if not isinstance(nested, dict):
            raise ValueError("settings 必须是 JSON object")
        unknown = sorted(set(nested) - accepted_keys)
        if unknown:
            raise ValueError(f"settings 包含不支持的参数: {unknown}")
        settings.update(nested)
        if "apap_grid_size" in nested and "apap_grid_parts" not in nested:
            settings["apap_grid_parts"] = nested["apap_grid_size"]
    for key in setting_keys:
        if key in payload and payload[key] not in (None, ""):
            settings[key] = payload[key]
    if "apap_grid_size" in payload and "apap_grid_parts" not in payload:
        settings["apap_grid_parts"] = payload["apap_grid_size"]
    settings.pop("apap_grid_size", None)

    algorithm = str(settings["algorithm"]).strip().lower()
    base_values = {key: settings[key] for key in base.SETTING_KEYS}
    base_values["algorithm"] = "stitcher"
    result = base.build_settings({"settings": base_values})
    if algorithm != "guided_affine":
        raise ValueError("人工四角配置必须设置 algorithm=guided_affine")
    result["algorithm"] = algorithm
    return result


def validate_matching_regions(value: Any) -> list[dict[str, Any]]:
    """Validate rectangles in the initial shared canvas coordinate system."""
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 32:
        raise ValueError("matching_regions 必须是最多32个框的数组")
    result, ids = [], set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("matching_regions 元素必须是对象")
        identifier, rect = item.get("id"), item.get("rect")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in ids:
            raise ValueError("matching_regions 的 id 必须是唯一非空字符串")
        if (not isinstance(rect, (list, tuple)) or len(rect) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in rect)
                or rect[2] <= 0 or rect[3] <= 0):
            raise ValueError("matching_regions.rect 必须是有限的 [x,y,width,height]，宽高大于0")
        if not math.isfinite(rect[0] + rect[2]) or not math.isfinite(rect[1] + rect[3]):
            raise ValueError("matching_regions 的矩形边界必须是有限数值")
        ids.add(identifier)
        result.append({"id": identifier, "rect": list(rect)})
    return result


def _corners(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float64)


def _points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def validate_images(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """校验请求并生成原图像素到共享画布的初始仿射变换。"""
    if not isinstance(images, list) or len(images) < 2:
        raise ValueError("images 至少需要两张图片")
    validated = []
    for index, item in enumerate(images):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"].strip():
            raise ValueError(f"images[{index}].path 必须是非空路径")
        path = str(Path(item["path"].strip()).expanduser())
        if not Path(path).is_file():
            raise FileNotFoundError(f"输入图片不存在: {path}")
        try:
            target = np.asarray(item["corners"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"images[{index}].corners 必须是四个 [x,y] 数值点") from exc
        if target.shape != (4, 2) or not np.isfinite(target).all():
            raise ValueError(f"images[{index}].corners 必须是四个有限的 [x,y] 数值点")
        size = get_input_image_size(path)
        if min(size) < 2:
            raise ValueError(f"images[{index}] 图像尺寸过小")
        source = _corners(size)
        if index == 0:
            if not np.allclose(target, source, rtol=0, atol=1e-3):
                raise ValueError("第一张图四角必须是 (0,0)、(W,0)、(W,H)、(0,H)")
            matrix = np.eye(2, 3, dtype=np.float64)
        else:
            contour = target.astype(np.float32)
            if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 4:
                raise ValueError(f"images[{index}] 四角顺序错误、交叉或面积过小")
            design = np.column_stack((source, np.ones(4)))
            coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
            matrix = coefficients.T
            scale = math.sqrt(abs(np.linalg.det(matrix[:, :2])))
            error = np.max(np.linalg.norm(_points(matrix, source) - target, axis=1))
            if rank != 3 or np.linalg.det(matrix[:, :2]) <= 0 or not 0.2 <= scale <= 5:
                raise ValueError(f"images[{index}] 四角无法构成有效仿射变换")
            if error > max(2.0, 0.005 * min(size)):
                raise ValueError(f"images[{index}] 四角偏离仿射模型 {error:.1f}px，请检查点顺序或透视形变")
        validated.append({"path": path, "size": size, "matrix": matrix, "corners": target})
    return validated


class GuidedAffineStitcher:
    def __init__(
        self, *, detector="sift", nfeatures=4000, medium_megapix=2,
        low_megapix=0.3, final_megapix=10, finder="dp_color",
        compensator="channel_blocks", blender_type="multiband",
        blend_strength=1, crop=True,
        guided_min_overlap_pixels=GUIDED_DEFAULT_SETTINGS["guided_min_overlap_pixels"],
        guided_overlap_margin_ratio=GUIDED_DEFAULT_SETTINGS["guided_overlap_margin_ratio"],
        guided_min_inliers=GUIDED_DEFAULT_SETTINGS["guided_min_inliers"],
        guided_ransac_threshold_px=GUIDED_DEFAULT_SETTINGS["guided_ransac_threshold_px"],
        guided_max_translation_ratio=GUIDED_DEFAULT_SETTINGS["guided_max_translation_ratio"],
        guided_max_rotation_degrees=GUIDED_DEFAULT_SETTINGS["guided_max_rotation_degrees"],
        guided_max_scale_change=GUIDED_DEFAULT_SETTINGS["guided_max_scale_change"],
        guided_max_shear=GUIDED_DEFAULT_SETTINGS["guided_max_shear"],
        show_progress=True, output_path=None,
        matching_regions=None,
    ) -> None:
        self.matching_regions = validate_matching_regions(matching_regions)
        self.detector = detector
        self.nfeatures = int(nfeatures)
        self.medium_megapix = float(medium_megapix)
        self.low_megapix = float(low_megapix)
        self.final_megapix = float(final_megapix)
        self.finder = finder
        self.compensator = compensator
        self.blender_type = blender_type
        self.blend_strength = float(blend_strength)
        self.crop = bool(crop)
        self.min_overlap = int(guided_min_overlap_pixels)
        self.overlap_margin_ratio = float(guided_overlap_margin_ratio)
        self.min_inliers = int(guided_min_inliers)
        self.ransac_threshold_px = float(guided_ransac_threshold_px)
        if not math.isfinite(self.ransac_threshold_px) or self.ransac_threshold_px <= 0:
            raise ValueError("guided_ransac_threshold_px 必须是大于0的有限数值")
        self.refinement_limits = {
            "translation_ratio": float(guided_max_translation_ratio),
            "rotation_degrees": float(guided_max_rotation_degrees),
            "scale_change": float(guided_max_scale_change),
            "shear": float(guided_max_shear),
        }
        for name, value in self.refinement_limits.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"guided_max_{name} 必须是非负的有限数值")
        if self.refinement_limits["rotation_degrees"] > 180:
            raise ValueError("guided_max_rotation_degrees 不能超过180度")
        self.show_progress = show_progress
        self.output_path = output_path
        self.output_offset = [0.0, 0.0]
        self.output_scale = 1.0
        self.pair_diagnostics: list[dict[str, Any]] = []
        self.registration_status = "manual_only"
        self.image_diagnostics: list[dict[str, Any]] = []

    def _log(self, message: str, percentage: float) -> None:
        write_progress(self.output_path, percentage, message)
        if self.show_progress:
            print(f"[GUIDED] {message}", flush=True)

    @staticmethod
    def _scale(size: tuple[int, int], megapix: float) -> float:
        if megapix < 0:
            return 1.0
        return min(1.0, math.sqrt(megapix * 1_000_000 / (size[0] * size[1])))

    def _detector(self, budget=None):
        budget = self.nfeatures if budget is None else budget
        name = self.detector.lower()
        if name == "sift":
            return cv2.SIFT_create(nfeatures=budget), cv2.NORM_L2
        if name == "orb":
            return cv2.ORB_create(nfeatures=budget), cv2.NORM_HAMMING
        if name == "akaze":
            return cv2.AKAZE_create(), cv2.NORM_HAMMING
        if name == "brisk":
            return cv2.BRISK_create(), cv2.NORM_HAMMING
        raise ValueError("guided_affine detector 只支持 sift、orb、akaze、brisk")

    @staticmethod
    def _overlap(a: dict, b: dict) -> tuple[float, np.ndarray]:
        quad_a = _points(a["matrix"], _corners(a["size"])).astype(np.float32)
        quad_b = _points(b["matrix"], _corners(b["size"])).astype(np.float32)
        area, polygon = cv2.intersectConvexConvex(quad_a, quad_b)
        return float(area), polygon.reshape(-1, 2) if polygon is not None else np.empty((0, 2))

    def _pair_margin(self, a: dict, b: dict) -> float:
        """Use transformed edge lengths, so the margin is in canvas units."""
        short_sides = []
        for item in (a, b):
            quad = _points(item["matrix"], _corners(item["size"]))
            short_sides.append(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min())
        return float(self.overlap_margin_ratio * min(short_sides))

    @staticmethod
    def _mutual_matches(desc_a: np.ndarray, desc_b: np.ndarray, norm: int) -> list[cv2.DMatch]:
        """Require the ratio test in both directions and a reciprocal match."""
        matcher = cv2.BFMatcher(norm)
        forward = [pair[0] for pair in matcher.knnMatch(desc_a, desc_b, k=2)
                   if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        reverse = {(pair[0].trainIdx, pair[0].queryIdx)
                   for pair in matcher.knnMatch(desc_b, desc_a, k=2)
                   if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance}
        return [match for match in forward if (match.queryIdx, match.trainIdx) in reverse]

    @staticmethod
    def _expanded_quad(quad: np.ndarray, margin: float) -> np.ndarray:
        """把凸四边形的每条边沿共享画布坐标向外平移 margin。"""
        if margin <= 0:
            return quad.astype(np.float32)
        edges = np.roll(quad, -1, axis=0) - quad
        lengths = np.linalg.norm(edges, axis=1)
        orientation = np.sign(np.sum(
            quad[:, 0] * np.roll(quad[:, 1], -1)
            - quad[:, 1] * np.roll(quad[:, 0], -1)
        ))
        normals = orientation * np.column_stack((edges[:, 1], -edges[:, 0])) / lengths[:, None]
        offsets = np.sum(normals * quad, axis=1) + margin
        expanded = []
        for index in range(4):
            before = (index - 1) % 4
            vertex = np.linalg.solve(
                np.stack((normals[before], normals[index])),
                np.array([offsets[before], offsets[index]]),
            )
            expanded.append(vertex)
        return np.asarray(expanded, np.float32)

    @classmethod
    def _search_polygons(cls, a: dict, b: dict, margin: float):
        quad_a = _points(a["matrix"], _corners(a["size"])).astype(np.float32)
        quad_b = _points(b["matrix"], _corners(b["size"])).astype(np.float32)
        overlap_area, _ = cv2.intersectConvexConvex(quad_a, quad_b)
        area_a, polygon_a = cv2.intersectConvexConvex(quad_a, cls._expanded_quad(quad_b, margin))
        area_b, polygon_b = cv2.intersectConvexConvex(quad_b, cls._expanded_quad(quad_a, margin))
        return (
            float(overlap_area), min(float(area_a), float(area_b)),
            polygon_a.reshape(-1, 2) if polygon_a is not None else np.empty((0, 2)),
            polygon_b.reshape(-1, 2) if polygon_b is not None else np.empty((0, 2)),
        )

    @staticmethod
    def _feature_mask(item: dict, polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        inverse = cv2.invertAffineTransform(item["matrix"])
        source_polygon = _points(inverse, polygon.astype(np.float64))
        height, width = shape
        source_polygon[:, 0] *= width / item["size"][0]
        source_polygon[:, 1] *= height / item["size"][1]
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, np.rint(source_polygon).astype(np.int32), 255)
        valid = item["medium_mask"]
        if valid is not None:
            mask = cv2.bitwise_and(mask, valid)
        return mask

    @staticmethod
    def _detect_region(detector, image: np.ndarray, mask: np.ndarray):
        rect = cv2.boundingRect(mask)
        x, y, width, height = rect
        if width == 0 or height == 0:
            return [], None
        points, descriptors = detector.detectAndCompute(
            image[y:y + height, x:x + width], mask[y:y + height, x:x + width]
        )
        for point in points:
            point.pt = (point.pt[0] + x, point.pt[1] + y)
        return points, descriptors

    def _match_pair(
        self, a: dict, b: dict, polygon_a: np.ndarray,
        polygon_b: np.ndarray, norm: int,
    ) -> tuple[dict, tuple | None]:
        regions = []
        for region in self.matching_regions:
            x, y, w, h = region["rect"]
            quad = np.array([[x,y], [x+w,y], [x+w,y+h], [x,y+h]], np.float32)
            aa, ca = cv2.intersectConvexConvex(polygon_a.astype(np.float32), quad)
            ab, cb = cv2.intersectConvexConvex(polygon_b.astype(np.float32), quad)
            if aa > 0 and ab > 0:
                regions.append((region["id"], ca.reshape(-1, 2), cb.reshape(-1, 2)))
        selected = bool(regions)
        if not selected:
            regions = [(None, polygon_a, polygon_b)]
        masks_a, masks_b = [], []
        used_a = np.zeros(a["medium"].shape[:2], np.uint8)
        used_b = np.zeros(b["medium"].shape[:2], np.uint8)
        for _, ca, cb in regions:
            for item, polygon, used, masks in ((a, ca, used_a, masks_a), (b, cb, used_b, masks_b)):
                mask = self._feature_mask(item, polygon, used.shape)
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(used))
                used[:] = cv2.bitwise_or(used, mask)
                masks.append(mask)
        areas_a = [cv2.countNonZero(mask) for mask in masks_a]
        areas_b = [cv2.countNonZero(mask) for mask in masks_b]
        budgets_a = [self.nfeatures if area else 0 for area in areas_a]
        budgets_b = [self.nfeatures if area else 0 for area in areas_b]
        info = {"features_a": 0, "features_b": 0, "matches": 0,
                "inliers": 0, "rmse_pixels": None, "status": "fallback_initial",
                "error_coordinate_system": "image_a_medium_pixels",
                "ransac_threshold_px": self.ransac_threshold_px,
                "region_source": "selected" if selected else "automatic", "regions": []}
        all_a, all_b, owners = [], [], []
        for index, (identifier, _, _) in enumerate(regions):
            keys, descriptors = [], []
            for item, mask, budget in ((a, masks_a[index], budgets_a[index]),
                                       (b, masks_b[index], budgets_b[index])):
                detector, _ = self._detector(budget)
                points, desc = self._detect_region(detector, item["medium"], mask) if budget else ([], None)
                if selected and points:
                    keep = [k for k, point in enumerate(points)
                            if 0 <= point.pt[0] < mask.shape[1]
                            and 0 <= point.pt[1] < mask.shape[0]
                            and mask[int(point.pt[1]), int(point.pt[0])]]
                    keep.sort(key=lambda k: points[k].response, reverse=True)
                    keep = keep[:budget]
                    points, desc = [points[k] for k in keep], desc[keep] if keep else None
                keys.append(points)
                descriptors.append(desc)
            key_a, key_b = keys
            desc_a, desc_b = descriptors
            detail = {"id": identifier, "area_a": areas_a[index], "area_b": areas_b[index],
                      "budget_a": budgets_a[index], "budget_b": budgets_b[index],
                      "features_a": len(key_a), "features_b": len(key_b), "matches": 0, "inliers": 0}
            info["regions"].append(detail)
            info["features_a"] += len(key_a)
            info["features_b"] += len(key_b)
            if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
                detail["reason"] = "区域无有效像素" if not areas_a[index] or not areas_b[index] else "区域特征不足"
                continue
            good = self._mutual_matches(desc_a, desc_b, norm)
            detail["matches"] = len(good)
            if not good:
                detail["reason"] = "描述子匹配不足"
            owners.extend([index] * len(good))
            all_a.extend(key_a[m.queryIdx].pt for m in good)
            all_b.extend(key_b[m.trainIdx].pt for m in good)
        if not all_a:
            info["reason"] = "区域特征或匹配不足；请扩大区域、增加框或调整 medium_megapix"
            return info, None
        pa, pb = np.asarray(all_a, np.float64), np.asarray(all_b, np.float64)
        unique = np.arange(len(pa))
        if selected:
            _, unique = np.unique(np.round(np.column_stack((pa, pb)), 4), axis=0, return_index=True)
        pa, pb = pa[unique], pb[unique]
        owners = np.asarray(owners)[unique]
        info["matches"] = len(pa)
        if len(pa) < self.min_inliers:
            info["reason"] = "描述子匹配点不足"
            return info, None
        relative, inlier_mask = cv2.estimateAffine2D(
            pb, pa, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold_px,
            maxIters=2000, confidence=0.995,
        )
        if relative is None or inlier_mask is None or not self._valid_affine(relative):
            info["reason"] = "RANSAC 估计失败"
            return info, None
        active = inlier_mask.ravel().astype(bool)
        info["inliers"] = int(active.sum())
        for index, detail in enumerate(info["regions"]):
            detail["inliers"] = int(np.count_nonzero(active & (owners == index)))
            if not detail["inliers"] and not detail.get("reason"):
                detail["reason"] = "无 RANSAC 内点"
        if info["inliers"] < self.min_inliers:
            info["reason"] = "RANSAC 内点不足"
            return info, None
        corrected = _points(relative, pb[active])
        info["rmse_pixels"] = float(np.sqrt(np.mean(np.sum((corrected - pa[active]) ** 2, axis=1))))
        pa *= [a["size"][0] / a["medium"].shape[1], a["size"][1] / a["medium"].shape[0]]
        pb *= [b["size"][0] / b["medium"].shape[1], b["size"][1] / b["medium"].shape[0]]
        world_a = _points(a["matrix"], pa)
        world_b = _points(b["matrix"], pb)
        info["status"] = "matched"
        return info, (world_a[active], world_b[active])

    @staticmethod
    def _valid_affine(matrix: np.ndarray) -> bool:
        if not np.isfinite(matrix).all():
            return False
        singular = np.linalg.svd(matrix[:, :2], compute_uv=False)
        return bool(np.linalg.det(matrix[:, :2]) > 0
                    and singular[-1] > 1e-8 * singular[0])

    @staticmethod
    def _reference_component(pairs: list[tuple], excluded: set[int] | tuple[int, ...] = ()) -> set[int]:
        adjacency: dict[int, set[int]] = {}
        for i, j, *_ in pairs:
            if i not in excluded and j not in excluded:
                adjacency.setdefault(i, set()).add(j)
                adjacency.setdefault(j, set()).add(i)
        connected, pending = {0}, [0]
        while pending:
            for index in adjacency.get(pending.pop(), set()) - connected:
                connected.add(index)
                pending.append(index)
        return connected

    def _check_refinement(
        self, linear: np.ndarray, center_shift: np.ndarray, short_side: float,
    ) -> tuple[dict[str, float], str]:
        """Measure the correction about the image center, not the canvas origin.

        Polar decomposition separates rotation from symmetric stretch. Principal
        stretches measure scale; normalized off-diagonal stretch measures shear.
        """
        u, singular, vt = np.linalg.svd(linear)
        rotation = u @ vt
        stretch = vt.T @ np.diag(singular) @ vt
        metrics = {
            "translation_ratio": float(np.linalg.norm(center_shift) / short_side),
            "rotation_degrees": abs(math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))),
            "scale_change": float(np.max(np.abs(singular - 1.0))),
            "shear": float(abs(stretch[0, 1]) / math.sqrt(stretch[0, 0] * stretch[1, 1])),
        }
        exceeded = [name for name, value in metrics.items()
                    if value > self.refinement_limits[name] + 1e-9]
        labels = {"translation_ratio": "中心平移", "rotation_degrees": "旋转角度",
                  "scale_change": "缩放", "shear": "剪切"}
        reason = ""
        if exceeded:
            reason = "仿射修正超过微调上限（" + "、".join(labels[name] for name in exceeded) + "），保留人工位置"
        return metrics, reason

    def _global_adjust(self, items: list[dict], pairs: list[tuple]) -> None:
        # Solve dimensionless residuals, with a weak positional prior at all four
        # corners. Its strength is independent of raw image and canvas sizes.
        quads = [_points(item["initial_matrix"], _corners(item["size"])) for item in items]
        centers = [quad.mean(axis=0) for quad in quads]
        spans = [float(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).max())
                 for quad in quads]
        short_sides = [float(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1).min())
                       for quad in quads]
        unit = spans[0]
        excluded: set[int] = set()
        items[0]["correction_status"] = "reference"
        for item in items[1:]:
            item["correction_reason"] = "没有通过可靠匹配连接到第一张参考图"

        def basis(index, point):
            x, y = (point - centers[index]) / spans[index]
            return np.array([[1, 0, x, y, 0, 0], [0, 1, 0, 0, x, y]], np.float64)

        # If a transform is invalid, remove it and solve the remaining anchored
        # component again; a failed bridge must not leave detached images refined.
        while True:
            connected = self._reference_component(pairs, excluded)
            variables = {index: slot for slot, index in enumerate(sorted(connected - {0}))}
            if not variables:
                return
            count = 6 * len(variables)
            rows, rhs = [], []
            for i, j, pa, pb in pairs:
                if i not in connected or j not in connected:
                    continue
                sample = np.linspace(0, len(pa) - 1, min(len(pa), 150), dtype=int)
                for point_a, point_b in zip(pa[sample], pb[sample]):
                    block = np.zeros((2, count), np.float64)
                    for index, point, sign in ((i, point_a, 1), (j, point_b, -1)):
                        if index in variables:
                            slot = variables[index]
                            block[:, 6 * slot:6 * (slot + 1)] = (
                                sign * spans[index] / unit * basis(index, point)
                            )
                    rows.extend(block)
                    rhs.extend((point_b - point_a) / unit)
            for index, slot in variables.items():
                for corner in quads[index]:
                    block = np.zeros((2, count), np.float64)
                    block[:, 6 * slot:6 * (slot + 1)] = 0.01 * basis(index, corner)
                    rows.extend(block)
                    rhs.extend((0.0, 0.0))
            try:
                solution, _, _, _ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
            except np.linalg.LinAlgError:
                for index in variables:
                    items[index]["correction_reason"] = "全局位置优化求解失败，保留人工位置"
                return
            candidates, invalid = {}, set()
            for index, slot in variables.items():
                tx, ty, ax, bx, ay, by = solution[6 * slot:6 * (slot + 1)]
                linear = np.array([[1 + ax, bx], [ay, 1 + by]])
                translation = spans[index] * np.array([tx, ty]) + centers[index] - linear @ centers[index]
                initial = items[index]["initial_matrix"]
                matrix = np.column_stack((linear @ initial[:, :2],
                                          linear @ initial[:, 2] + translation))
                if not self._valid_affine(matrix):
                    invalid.add(index)
                    items[index]["correction_reason"] = "全局优化产生无效仿射变换，保留人工位置"
                else:
                    metrics, reason = self._check_refinement(
                        linear, spans[index] * np.array([tx, ty]), short_sides[index],
                    )
                    items[index]["attempted_correction"] = metrics
                    if reason:
                        invalid.add(index)
                        items[index]["correction_reason"] = reason
                    else:
                        candidates[index] = matrix
            if invalid:
                excluded.update(invalid)
                continue
            for index, matrix in candidates.items():
                items[index]["matrix"] = matrix
                items[index]["correction_status"] = "refined"
                items[index].pop("correction_reason", None)
            return

    def registration_info(self) -> dict[str, Any]:
        """Return registration quality independently of file-generation success."""
        return {
            "registration_status": self.registration_status,
            "image_diagnostics": self.image_diagnostics,
            "refinement_limits": self.refinement_limits,
        }

    @staticmethod
    def _bounds(items: list[dict], scale: float):
        all_points = np.concatenate([_points(item["matrix"], _corners(item["size"])) for item in items])
        minimum = np.floor(all_points.min(axis=0) * scale).astype(int)
        maximum = np.ceil(all_points.max(axis=0) * scale).astype(int)
        size = (int(maximum[0] - minimum[0]), int(maximum[1] - minimum[1]))
        if min(size) <= 0 or size[0] * size[1] > 500_000_000:
            raise ValueError(f"输出画布尺寸不合理: {size}")
        return minimum, size

    @staticmethod
    def _warp(item: dict, scale: float, origin: np.ndarray, image: np.ndarray, mask: np.ndarray):
        source_scale = np.array([image.shape[1] / item["size"][0], image.shape[0] / item["size"][1]])
        matrix = item["matrix"].copy()
        matrix[:, :2] = scale * matrix[:, :2] / source_scale[np.newaxis, :]
        matrix[:, 2] = scale * matrix[:, 2] - origin
        quad = _points(matrix, _corners((image.shape[1], image.shape[0])))
        low = np.floor(quad.min(axis=0)).astype(int)
        high = np.ceil(quad.max(axis=0)).astype(int)
        size = tuple((high - low).astype(int))
        if min(size) <= 0:
            raise RuntimeError("投影后的图像尺寸无效")
        matrix[:, 2] -= low
        warped = cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR)
        valid = cv2.warpAffine(mask, matrix, size, flags=cv2.INTER_NEAREST)
        return (int(low[0]), int(low[1])), warped, valid

    def _read_scaled(self, item: dict, scale: float):
        def target(size):
            return max(1, round(size[0] * scale)), max(1, round(size[1] * scale))
        image, _ = read_input_image_with_size(item["path"], target)
        mask = read_valid_mask(item["path"], (image.shape[1], image.shape[0]))
        if mask is None:
            mask = np.full(image.shape[:2], 255, np.uint8)
        return image, mask

    def _exposure(self):
        types = {"no": cv2.detail.ExposureCompensator_NO,
                 "gain": cv2.detail.ExposureCompensator_GAIN,
                 "gain_blocks": cv2.detail.ExposureCompensator_GAIN_BLOCKS,
                 "channel": cv2.detail.ExposureCompensator_CHANNELS,
                 "channel_blocks": cv2.detail.ExposureCompensator_CHANNELS_BLOCKS}
        if self.compensator not in types:
            raise ValueError(f"不支持的 guided_affine compensator: {self.compensator}")
        return cv2.detail.ExposureCompensator_createDefault(types[self.compensator])

    def _seam(self):
        if self.finder == "no":
            return cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_NO)
        if self.finder == "voronoi":
            return cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM)
        if self.finder in {"dp_color", "dp_colorgrad"}:
            return cv2.detail_DpSeamFinder("COLOR" if self.finder == "dp_color" else "COLOR_GRAD")
        if self.finder in {"gc_color", "gc_colorgrad"}:
            return cv2.detail_GraphCutSeamFinder("COST_COLOR" if self.finder == "gc_color" else "COST_COLOR_GRAD")
        raise ValueError(f"不支持的 guided_affine finder: {self.finder}")

    def _blender(self, size):
        if self.blender_type == "multiband":
            blender = cv2.detail_MultiBandBlender()
            blender.setNumBands(max(1, min(8, round(math.log2(max(size)) - 1) if self.blend_strength > 0 else 1)))
        elif self.blender_type == "feather":
            blender = cv2.detail_FeatherBlender()
            blender.setSharpness(1.0 / max(1.0, self.blend_strength * 100.0))
        elif self.blender_type == "no":
            blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
        else:
            raise ValueError(f"不支持的 guided_affine blender_type: {self.blender_type}")
        blender.prepare((0, 0, size[0], size[1]))
        return blender

    def register(self, images: list[dict[str, Any]], medium_images=None) -> list[dict]:
        """只执行人工仿射约束和重叠区匹配，供原拼接流程复用。"""
        items = validate_images(images)
        self.pair_diagnostics = []
        self._log("读取特征分辨率图像", 15.0)
        for index, item in enumerate(items):
            item["initial_matrix"] = item["matrix"].copy()
            scale = self._scale(item["size"], self.medium_megapix)
            item["medium"] = medium_images[index] if medium_images is not None else self._read_scaled(item, scale)[0]
            item["medium_mask"] = read_valid_mask(item["path"],
                                                  (item["medium"].shape[1], item["medium"].shape[0]))
            item["correction_status"] = "fallback_initial"
        _, norm = self._detector()
        accepted = []
        self._log("检测重叠区特征并匹配", 30.0)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                margin = self._pair_margin(items[i], items[j])
                area, search_area, polygon_a, polygon_b = self._search_polygons(
                    items[i], items[j], margin,
                )
                if search_area < self.min_overlap:
                    continue
                info, match = self._match_pair(items[i], items[j], polygon_a, polygon_b, norm)
                self.pair_diagnostics.append({"image_a": i, "image_b": j,
                                              "overlap_pixels": round(area, 2),
                                              "search_area_pixels": round(search_area, 2),
                                              "search_margin_pixels": round(margin, 2), **info})
                if match is not None:
                    accepted.append((i, j, *match))
        for item in items:
            item.pop("medium")
            item.pop("medium_mask")
        self._global_adjust(items, accepted)
        self._log("完成全局位置修正", 50.0)
        registered = {index for index, item in enumerate(items)
                      if item["correction_status"] in {"reference", "refined"}}
        for record in self.pair_diagnostics:
            if record["status"] == "matched":
                if record["image_a"] in registered and record["image_b"] in registered:
                    record["status"] = "refined"
                else:
                    record["status"] = "fallback_initial"
                    record["reason"] = "未连接到参考图或全局修正无效，保留人工位置"
        if len(registered) == len(items):
            self.registration_status = "full"
        elif len(registered) > 1:
            self.registration_status = "partial"
        else:
            self.registration_status = "manual_only"
        self.image_diagnostics = [
            {"image_index": index, "path": item["path"],
             "status": item["correction_status"],
             "reason": item.get("correction_reason", ""),
             "attempted_correction": item.get("attempted_correction"),
             "initial_transform": item["initial_matrix"].tolist(),
             "final_transform": item["matrix"].tolist()}
            for index, item in enumerate(items)
        ]
        return items

    def stitch(self, images: list[dict[str, Any]]) -> np.ndarray:
        items = self.register(images)

        max_area = max(item["size"][0] * item["size"][1] for item in items)
        low_scale = min(1.0, math.sqrt(self.low_megapix * 1_000_000 / max_area))
        final_scale = 1.0 if self.final_megapix < 0 else min(1.0, math.sqrt(self.final_megapix * 1_000_000 / max_area))
        low_origin, _ = self._bounds(items, low_scale)
        final_origin, final_size = self._bounds(items, final_scale)
        self._log("低分辨率曝光补偿与接缝查找", 60.0)
        low_data = []
        for item in items:
            image, mask = self._read_scaled(item, low_scale)
            low_data.append(self._warp(item, low_scale, low_origin, image, mask))
        corners = [data[0] for data in low_data]
        warped_images = [data[1] for data in low_data]
        seam_masks = [data[2] for data in low_data]
        exposure = self._exposure()
        exposure.feed(corners, warped_images, seam_masks)
        for index, (corner, image, mask) in enumerate(low_data):
            exposure.apply(index, corner, image, mask)
        self._seam().find([image.astype(np.float32) for image in warped_images], corners, seam_masks)

        self._log("最终分辨率投影与融合", 80.0)
        blender = self._blender(final_size)
        for index, item in enumerate(items):
            image, mask = self._read_scaled(item, final_scale)
            corner, warped, valid = self._warp(item, final_scale, final_origin, image, mask)
            exposure.apply(index, corner, warped, valid)
            ratio = final_scale / low_scale
            low_corner = np.asarray(corners[index], np.float64)
            final_corner = np.asarray(corner, np.float64)
            translation = (low_corner + low_origin) * ratio - final_origin - final_corner
            seam_transform = np.array([[ratio, 0, translation[0]], [0, ratio, translation[1]]])
            seam = cv2.warpAffine(seam_masks[index], seam_transform,
                                  (warped.shape[1], warped.shape[0]), flags=cv2.INTER_NEAREST)
            combined = cv2.bitwise_and(valid, seam)
            blender.feed(warped.astype(np.int16), combined, corner)
        result, result_mask = blender.blend(None, None)
        panorama = np.clip(result, 0, 255).astype(np.uint8)
        self.output_scale = final_scale
        self.output_offset = [-float(final_origin[0]), -float(final_origin[1])]
        if self.crop:
            points = cv2.findNonZero(result_mask)
            if points is not None:
                x, y, width, height = cv2.boundingRect(points)
                panorama = panorama[y:y + height, x:x + width]
                self.output_offset[0] -= x
                self.output_offset[1] -= y
        return panorama
