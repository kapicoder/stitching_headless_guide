"""Regularized seam-local mesh fitting and image remapping."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def seam_influence(
    valid_mask: np.ndarray,
    seam_mask: np.ndarray,
    band_width: float,
) -> np.ndarray:
    """Return a soft band around the ownership boundary selected by the seam finder."""
    valid = np.asarray(valid_mask) > 0
    owned = (np.asarray(seam_mask) > 0) & valid
    unowned = valid & ~owned
    if not owned.any() or not unowned.any():
        return np.zeros(valid.shape, dtype=np.float32)

    kernel = np.ones((3, 3), dtype=np.uint8)
    owned_grown = cv2.dilate(owned.astype(np.uint8), kernel)
    unowned_grown = cv2.dilate(unowned.astype(np.uint8), kernel)
    boundary = (owned_grown > 0) & (unowned_grown > 0) & valid
    if not boundary.any():
        return np.zeros(valid.shape, dtype=np.float32)

    distance_input = np.where(boundary, 0, 255).astype(np.uint8)
    distance = cv2.distanceTransform(distance_input, cv2.DIST_L2, 3)
    influence = np.clip(1.0 - distance / max(float(band_width), 1.0), 0.0, 1.0)
    influence *= valid.astype(np.float32)
    return influence.astype(np.float32)


def _grid_axis(length: int, spacing: int) -> np.ndarray:
    count = min(9, max(2, int(math.ceil(max(length - 1, 1) / spacing)) + 1))
    return np.linspace(0.0, float(max(length - 1, 0)), count, dtype=np.float64)


def _sample(image: np.ndarray, point: np.ndarray) -> float:
    x, y = map(float, point)
    height, width = image.shape
    if x < 0 or y < 0 or x > width - 1 or y > height - 1:
        return 0.0
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    tx, ty = x - x0, y - y0
    top = float(image[y0, x0]) * (1 - tx) + float(image[y0, x1]) * tx
    bottom = float(image[y1, x0]) * (1 - tx) + float(image[y1, x1]) * tx
    return top * (1 - ty) + bottom * ty


def _basis(
    point: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[tuple[int, float], ...]:
    x, y = map(float, point)
    ix = min(max(int(np.searchsorted(xs, x, side="right")) - 1, 0), len(xs) - 2)
    iy = min(max(int(np.searchsorted(ys, y, side="right")) - 1, 0), len(ys) - 2)
    tx = float(np.clip((x - xs[ix]) / max(xs[ix + 1] - xs[ix], 1e-9), 0, 1))
    ty = float(np.clip((y - ys[iy]) / max(ys[iy + 1] - ys[iy], 1e-9), 0, 1))
    width = len(xs)
    return (
        (iy * width + ix, (1 - tx) * (1 - ty)),
        (iy * width + ix + 1, tx * (1 - ty)),
        ((iy + 1) * width + ix, (1 - tx) * ty),
        ((iy + 1) * width + ix + 1, tx * ty),
    )


def _jacobian_minimum(dx: np.ndarray, dy: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> float:
    dx_dy, dx_dx = np.gradient(dx, ys, xs)
    dy_dy, dy_dx = np.gradient(dy, ys, xs)
    determinant = (1.0 + dx_dx) * (1.0 + dy_dy) - dx_dy * dy_dx
    return float(np.min(determinant))


def solve_seam_mesh(
    tile_shapes: list[tuple[int, int]],
    valid_masks: list[np.ndarray],
    seam_masks: list[np.ndarray],
    pair_constraints: list[dict[str, Any]],
    movable: list[bool],
    *,
    grid_spacing: int = 128,
    seam_band_ratio: float = 0.15,
    max_displacement_ratio: float = 0.02,
    min_matches: int = 12,
) -> tuple[list[dict[str, np.ndarray] | None], dict[str, Any]]:
    """Fit per-image displacement meshes from inliers near the current seam."""
    count = len(tile_shapes)
    fields: list[dict[str, np.ndarray] | None] = [None] * count
    info: dict[str, Any] = {
        "enabled": True,
        "status": "fallback_no_seam_support",
        "matches_used": 0,
        "images": [],
        "coordinate_system": "low_resolution_projected_image_pixels",
        "georeference_note": (
            "局部网格变形改变了非参考图像的局部位置，仿射 GeoTransform 仅为近似描述"
        ),
    }
    if not (len(valid_masks) == len(seam_masks) == len(movable) == count):
        raise ValueError("网格输入的图像、mask 和可变形标记数量必须一致")

    strengths: list[np.ndarray] = []
    axes: list[tuple[np.ndarray, np.ndarray]] = []
    node_strengths: list[np.ndarray] = []
    for index, ((height, width), valid, seam) in enumerate(
        zip(tile_shapes, valid_masks, seam_masks)
    ):
        band_width = max(8.0, min(height, width) * seam_band_ratio)
        strength = seam_influence(valid, seam, band_width)
        xs, ys = _grid_axis(width, grid_spacing), _grid_axis(height, grid_spacing)
        samples = np.array(
            [[_sample(strength, np.array([x, y])) for x in xs] for y in ys],
            dtype=np.float64,
        )
        # Keep the outer image boundary fixed so local remapping cannot clip the tile.
        samples[[0, -1], :] = 0.0
        samples[:, [0, -1]] = 0.0
        strengths.append(strength)
        axes.append((xs, ys))
        node_strengths.append(samples)
        info["images"].append({
            "image_index": index,
            "seam_band_pixels": round(float(band_width), 2),
            "seam_support_pixels": int(np.count_nonzero(strength > 0.1)),
            "status": (
                "fixed_reference" if index == 0
                else "fixed_manual_position" if not movable[index]
                else "pending"
            ),
            "max_displacement_pixels": 0.0,
        })

    variable_offsets: dict[int, int] = {}
    variable_sizes: dict[int, int] = {}
    variable_count = 0
    for index, (is_movable, (xs, ys)) in enumerate(zip(movable, axes)):
        if not is_movable or index == 0:
            continue
        node_count = len(xs) * len(ys)
        variable_offsets[index] = variable_count
        variable_sizes[index] = node_count
        variable_count += node_count
    if variable_count == 0:
        info["status"] = "fallback_no_movable_images"
        return fields, info
    if variable_count > 800:
        info["status"] = "fallback_mesh_too_large"
        for image in info["images"]:
            image["status"] = "fallback_mesh_too_large"
        return fields, info

    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    used_matches = 0
    for pair in pair_constraints:
        image_a, image_b = int(pair["image_a"]), int(pair["image_b"])
        points_a = np.asarray(pair["points_a"], dtype=np.float64)
        points_b = np.asarray(pair["points_b"], dtype=np.float64)
        residuals = np.asarray(pair["residuals"], dtype=np.float64)
        if not (len(points_a) == len(points_b) == len(residuals)):
            continue
        sample = np.arange(len(points_a))
        if len(sample) > 200:
            sample = np.linspace(0, len(sample) - 1, 200, dtype=int)
        xs_a, ys_a = axes[image_a]
        xs_b, ys_b = axes[image_b]
        for match_index in sample:
            point_a, point_b = points_a[match_index], points_b[match_index]
            support_a = _sample(strengths[image_a], point_a)
            support_b = _sample(strengths[image_b], point_b)
            if max(support_a, support_b) < 0.2:
                continue
            residual = residuals[match_index]
            if not np.isfinite(residual).all():
                continue
            row = np.zeros(variable_count, dtype=np.float64)
            confidence = math.sqrt(max(support_a, support_b))
            huber_delta = 8.0
            residual_length = float(np.linalg.norm(residual))
            if residual_length > huber_delta:
                confidence *= math.sqrt(huber_delta / residual_length)
            for image_index, point, sign, support, grid_x, grid_y in (
                (image_a, point_a, 1.0, support_a, xs_a, ys_a),
                (image_b, point_b, -1.0, support_b, xs_b, ys_b),
            ):
                if image_index not in variable_offsets:
                    continue
                offset = variable_offsets[image_index]
                for node, coefficient in _basis(point, grid_x, grid_y):
                    row[offset + node] += sign * coefficient * support * confidence
            if np.linalg.norm(row) < 1e-6:
                continue
            rows.append(row)
            targets.append(np.asarray(residual, dtype=np.float64) * confidence)
            used_matches += 1
            if used_matches >= 3000:
                break
        if used_matches >= 3000:
            break

    info["matches_used"] = used_matches
    if used_matches < min_matches:
        info["status"] = "fallback_insufficient_seam_matches"
        for image in info["images"]:
            image["status"] = "fallback_insufficient_seam_matches"
        return fields, info

    # The anchor term holds mesh nodes away from the seam at their affine positions.
    for index, offset in variable_offsets.items():
        xs, ys = axes[index]
        values = node_strengths[index]
        nx = len(xs)
        ny = len(ys)
        for iy in range(ny):
            for ix in range(nx):
                node = iy * nx + ix
                is_edge = ix in {0, nx - 1} or iy in {0, ny - 1}
                weight = 10.0 if is_edge else 0.08 + 3.0 * (1.0 - values[iy, ix])
                row = np.zeros(variable_count, dtype=np.float64)
                row[offset + node] = weight
                rows.append(row)
                targets.append(np.zeros(2, dtype=np.float64))

                neighbors = []
                if ix > 0:
                    neighbors.append(node - 1)
                if ix + 1 < nx:
                    neighbors.append(node + 1)
                if iy > 0:
                    neighbors.append(node - nx)
                if iy + 1 < ny:
                    neighbors.append(node + nx)
                if len(neighbors) >= 3 and not is_edge:
                    smooth_row = np.zeros(variable_count, dtype=np.float64)
                    smooth_row[offset + node] = 1.0
                    for neighbor in neighbors:
                        smooth_row[offset + neighbor] -= 1.0 / len(neighbors)
                    rows.append(smooth_row * 1.5)
                    targets.append(np.zeros(2, dtype=np.float64))

    try:
        matrix = np.vstack(rows)
        target = np.vstack(targets)
        solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    except (np.linalg.LinAlgError, MemoryError):
        info["status"] = "fallback_solver_failure"
        for image in info["images"]:
            image["status"] = "fallback_solver_failure"
        return fields, info

    applied = False
    for index, offset in variable_offsets.items():
        xs, ys = axes[index]
        shape = (len(ys), len(xs))
        dx = solution[offset:offset + variable_sizes[index], 0].reshape(shape)
        dy = solution[offset:offset + variable_sizes[index], 1].reshape(shape)
        dx[[0, -1], :] = 0.0
        dx[:, [0, -1]] = 0.0
        dy[[0, -1], :] = 0.0
        dy[:, [0, -1]] = 0.0
        max_allowed = max(1.0, min(tile_shapes[index]) * max_displacement_ratio)
        node_magnitude = np.hypot(dx, dy)
        maximum = float(node_magnitude.max())
        if maximum > max_allowed:
            dx *= max_allowed / maximum
            dy *= max_allowed / maximum

        minimum_jacobian = _jacobian_minimum(dx, dy, xs, ys)
        if minimum_jacobian < 0.25:
            low, high = 0.0, 1.0
            for _ in range(18):
                mid = (low + high) * 0.5
                if _jacobian_minimum(dx * mid, dy * mid, xs, ys) >= 0.25:
                    low = mid
                else:
                    high = mid
            dx *= low
            dy *= low
            minimum_jacobian = _jacobian_minimum(dx, dy, xs, ys)

        maximum = float(np.hypot(dx, dy).max())
        image_info = info["images"][index]
        image_info["max_displacement_pixels"] = round(maximum, 3)
        image_info["minimum_jacobian"] = round(minimum_jacobian, 4)
        if maximum < 0.25:
            image_info["status"] = "no_change"
            continue
        fields[index] = {
            "dx": dx.astype(np.float32),
            "dy": dy.astype(np.float32),
            "strength": strengths[index].astype(np.float32),
            "grid_x": xs.astype(np.float32),
            "grid_y": ys.astype(np.float32),
        }
        image_info["status"] = "refined"
        applied = True

    info["status"] = "applied" if applied else "no_change"
    return fields, info


def warp_with_mesh(
    image: np.ndarray,
    mask: np.ndarray,
    field: dict[str, np.ndarray] | None,
    *,
    inplace: bool = False,
    low_to_output_scale: tuple[float, float] | None = None,
    low_origin: tuple[float, float] = (0.0, 0.0),
    output_origin: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a forward displacement mesh using inverse sampling."""
    if field is None:
        return image, mask
    height, width = image.shape[:2]
    low_height, low_width = field["strength"].shape
    if low_to_output_scale is None:
        scale_x, scale_y = width / low_width, height / low_height
    else:
        scale_x, scale_y = map(float, low_to_output_scale)
    low_origin_x, low_origin_y = map(float, low_origin)
    output_origin_x, output_origin_y = map(float, output_origin)
    if (
        not np.isfinite((scale_x, scale_y, low_origin_x, low_origin_y,
                         output_origin_x, output_origin_y)).all()
        or scale_x <= 0
        or scale_y <= 0
    ):
        raise ValueError("网格重映射的比例和原点必须是有限有效数值")

    support_y, support_x = np.nonzero(field["strength"] > 1e-3)
    if not len(support_x):
        return image, mask
    support_output_x = (
        (support_x + 0.5 + low_origin_x) * scale_x - output_origin_x - 0.5
    )
    support_output_y = (
        (support_y + 0.5 + low_origin_y) * scale_y - output_origin_y - 0.5
    )
    pad_x, pad_y = max(2, math.ceil(2 * scale_x)), max(2, math.ceil(2 * scale_y))
    x0 = max(0, int(math.floor(support_output_x.min())) - pad_x)
    y0 = max(0, int(math.floor(support_output_y.min())) - pad_y)
    x1 = min(width, int(math.ceil(support_output_x.max() + 1)) + pad_x)
    y1 = min(height, int(math.ceil(support_output_y.max() + 1)) + pad_y)
    if x1 <= x0 or y1 <= y0:
        return image, mask
    roi_width, roi_height = x1 - x0, y1 - y0

    # Build sampling maps only for the seam corridor, not for a potentially huge tile.
    low_x = (
        (np.arange(x0, x1, dtype=np.float32) + 0.5 + output_origin_x)
        / scale_x - low_origin_x - 0.5
    )
    low_y = (
        (np.arange(y0, y1, dtype=np.float32) + 0.5 + output_origin_y)
        / scale_y - low_origin_y - 0.5
    )
    low_x = np.clip(low_x, 0, low_width - 1)
    low_y = np.clip(low_y, 0, low_height - 1)
    map_low_x = np.broadcast_to(low_x[None, :], (roi_height, roi_width)).copy()
    map_low_y = np.broadcast_to(low_y[:, None], (roi_height, roi_width)).copy()
    strength = cv2.remap(
        field["strength"], map_low_x, map_low_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    grid_x = np.asarray(field["grid_x"], dtype=np.float32)
    grid_y = np.asarray(field["grid_y"], dtype=np.float32)
    node_x = np.interp(low_x, grid_x, np.arange(len(grid_x))).astype(np.float32)
    node_y = np.interp(low_y, grid_y, np.arange(len(grid_y))).astype(np.float32)
    map_node_x = np.broadcast_to(node_x[None, :], (roi_height, roi_width)).copy()
    map_node_y = np.broadcast_to(node_y[:, None], (roi_height, roi_width)).copy()
    dx = cv2.remap(
        field["dx"], map_node_x, map_node_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ) * scale_x * strength
    dy = cv2.remap(
        field["dy"], map_node_x, map_node_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ) * scale_y * strength
    yy, xx = np.indices((roi_height, roi_width), dtype=np.float32)
    map_x = xx + x0 - dx
    map_y = yy + y0 - dy
    remapped_image = cv2.remap(
        image, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    remapped_mask = cv2.remap(
        mask, map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped = image if inplace else image.copy()
    warped_mask = mask if inplace else mask.copy()
    warped[y0:y1, x0:x1] = remapped_image
    warped_mask[y0:y1, x0:x1] = remapped_mask
    return warped, warped_mask
