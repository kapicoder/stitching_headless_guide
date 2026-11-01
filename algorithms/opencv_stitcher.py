"""OpenCV Stitcher 流程封装。

职责：封装第三方 stitching.Stitcher，并维持现有阶段进度、接缝保护和融合行为。
输入：图片路径及 Stitcher 参数；输出：拼接后的 BGR ndarray。
流程：多级缩放与特征匹配 -> 相机估计/校正 -> 投影和接缝 -> 曝光补偿 -> 融合。
性能敏感点：特征检测匹配、最终分辨率投影、接缝查找及全景融合。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from stitching import Stitcher
from stitching.images import Images
from stitching.seam_finder import SeamFinder

from algorithms.guided_affine import GuidedAffineStitcher, _matrix3, _points
from algorithms.seam_mesh import solve_seam_mesh, warp_with_mesh
from image_io import (
    is_tiff_path,
    read_input_image,
    read_input_image_with_size,
    read_valid_mask,
)
from progress import write_progress


class ProgressStitcher(Stitcher):
    # 两个 worker 可以隐藏大图解码延迟，又不会像“每张图一个线程”那样让
    # 多张 4000x24000 原图同时驻留内存。GDAL/OpenCV 的读取和 resize 在 C 层执行。
    MASK_ERODE_KERNEL_SIZE = 3
    MASK_ERODE_ITERATIONS = 1
    MIN_SEAM_FEATHER_SIGMA = 1.0

    def __init__(
        self,
        *args,
        show_progress=True,
        output_path=None,
        optimized_tiff_io=True,
        parallel_image_io=True,
        image_io_workers=2,
        prefetch_final_images=True,
        cache_all_valid_masks=True,
        shared_warp_maps=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.show_progress = show_progress
        self.output_path = output_path
        self.optimized_tiff_io = bool(optimized_tiff_io)
        self.parallel_image_io = bool(parallel_image_io)
        self.image_io_workers = max(1, int(image_io_workers))
        self.prefetch_final_images = bool(prefetch_final_images)
        self.cache_all_valid_masks = bool(cache_all_valid_masks)
        self.shared_warp_maps = bool(shared_warp_maps)
        self._custom_panorama = None
        self._stage_index = 0
        self._stage_total = 22
        self.stage_events: list[dict] = []
        self._started_at = time.perf_counter()
        self._filename_inputs = False
        self._timing_lock = Lock()
        self._timings: dict[str, list[float | int]] = {}
        self.georef_control_points: list[dict[str, float | str]] = []

    def _progress(self, percentage: float, message: str) -> None:
        self._stage_index += 1
        elapsed = time.perf_counter() - self._started_at
        self.stage_events.append({
            "index": self._stage_index,
            "total": self._stage_total,
            "percentage": float(percentage),
            "message": message,
            "elapsed_seconds": round(elapsed, 3),
        })
        write_progress(self.output_path, percentage, message)
        if not self.show_progress:
            return
        print(
            f"[{self._stage_index:02d}/{self._stage_total}] "
            f"{message} ({elapsed:.1f}s)",
            flush=True,
        )

    def stitch(self, images, feature_masks=None):
        self._stage_index = 0
        self._stage_total = (
            19 if self._uses_seam_mesh()
            else 17 if self._uses_guided_geometry()
            else 22
        )
        self.stage_events = []
        self._started_at = time.perf_counter()
        self._custom_panorama = None
        self.georef_control_points = []
        with self._timing_lock:
            self._timings.clear()
        self._filename_inputs = all(isinstance(image, str) for image in images)
        self._all_valid_mask_paths: set[str] = set()
        self._progress(11.0, "读取图片并初始化缩放参数")
        self.images = Images.of(
            images, self.medium_megapix, self.low_megapix, self.final_megapix
        )
        self._progress(12.0, "缩放到特征检测分辨率")
        imgs = self.resize_medium_resolution()
        if self._uses_guided_geometry():
            self._progress(15.0, "读取人工四角并检测预计重叠区域")
            cameras = self._prepare_guided_geometry(imgs)
            self._progress(45.0, "完成重叠区匹配与全局位置修正")
        else:
            self._progress(15.0, "检测特征点")
            if feature_masks is None and self._filename_inputs:
                medium_masks = self._read_medium_feature_masks(imgs)
                if medium_masks is None:
                    features = self.find_features(imgs, [])
                else:
                    features = self.detector.detect_with_masks(imgs, medium_masks)
            else:
                features = self.find_features(
                    imgs, [] if feature_masks is None else feature_masks
                )
            self._progress(20.0, "匹配图像特征")
            matches = self.match_features(features)
            self._progress(25.0, "筛选属于同一拼接图的图片")
            imgs, features, matches = self.subset(imgs, features, matches)
            print(f"参与拼接图片数: {len(self.images.names)}", flush=True)
            self._progress(30.0, "估计相机参数")
            cameras = self.estimate_camera_parameters(features, matches)
            self._progress(35.0, "优化相机参数")
            try:
                cameras = self.refine_camera_parameters(features, matches, cameras)
            except Exception:  # noqa: BLE001 - 保留 OpenCV/StitchingError 原异常
                self._print_camera_adjustment_diagnostics(
                    features,
                    matches,
                    cameras,
                )
                raise
            self._progress(40.0, "执行波形校正")
            cameras = self.perform_wave_correction(cameras)
            self._progress(45.0, "估计投影缩放")
            self.estimate_scale(cameras)
        self._progress(50.0, "缩放到接缝/曝光估计分辨率")
        imgs = self.resize_low_resolution(imgs)
        self._progress(55.0, "低分辨率投影变换")
        imgs, masks, corners, sizes = self.warp_low_resolution(imgs, cameras)
        self._progress(60.0, "准备并执行低分辨率裁剪")
        self.prepare_cropper(imgs, masks, corners, sizes)
        imgs, masks, corners, sizes = self.crop_low_resolution(
            imgs, masks, corners, sizes
        )
        self._progress(65.0, "估计曝光补偿")
        self.estimate_exposure_errors(corners, imgs, masks)
        self._progress(70.0, "查找接缝掩膜")
        if self._uses_seam_mesh():
            valid_masks = [self._as_array(mask).copy() for mask in masks]
            seam_masks = self.find_seam_masks(
                imgs, corners, [mask.copy() for mask in valid_masks]
            )
            initial_seam_masks = [self._as_array(mask).copy() for mask in seam_masks]
            self._progress(72.0, "估计接缝附近的网格位移")
            try:
                self._fit_seam_mesh(imgs, valid_masks, seam_masks, corners)
                if any(field is not None for field in self.seam_mesh_fields):
                    refined_images, refined_masks = [], []
                    for index, field in enumerate(self.seam_mesh_fields):
                        image, mask = warp_with_mesh(
                            imgs[index], valid_masks[index], field
                        )
                        refined_images.append(image)
                        refined_masks.append(mask)
                    refined_seams = self.find_seam_masks(
                        refined_images, corners,
                        [mask.copy() for mask in refined_masks],
                    )
                    imgs = refined_images
                    valid_masks = refined_masks
                    seam_masks = [self._as_array(mask) for mask in refined_seams]
                    self.mesh_diagnostics["seam_refined"] = True
                else:
                    seam_masks = initial_seam_masks
            except (cv2.error, ValueError, RuntimeError, np.linalg.LinAlgError, MemoryError) as exc:
                self.seam_mesh_fields = [None] * len(imgs)
                self.mesh_diagnostics.update({
                    "status": "fallback_local_error",
                    "reason": str(exc)[:240],
                })
                for image_info in self.mesh_diagnostics.get("images", []):
                    image_info["status"] = "fallback_local_error"
                seam_masks = initial_seam_masks
            self._progress(74.0, "完成网格校正与接缝更新")
        else:
            seam_masks = self.find_seam_masks(imgs, corners, masks)
        self._progress(75.0, "缩放到最终输出分辨率")
        imgs = self.resize_final_resolution()
        self._progress(80.0, "最终分辨率投影变换")
        imgs, masks, corners, sizes = self.warp_final_resolution(imgs, cameras)
        uncropped_corners = list(corners)
        self._progress(83.0, "执行最终裁剪")
        imgs, masks, corners, sizes = self.crop_final_resolution(
            imgs, masks, corners, sizes
        )
        self._capture_georef_control_points(cameras, uncropped_corners, corners, sizes)

        # ！   可以直接注释掉下面这行
        # 腐蚀接缝掩膜以保护最终融合区域，避免接缝处出现明显的融合痕迹。
        # masks = self._protect_final_masks(masks)
        self._progress(86.0, "应用曝光补偿并调整接缝掩膜")
        # 保留父类的流式行为：曝光补偿、接缝 mask 缩放和融合按图片逐张执行，
        # 避免在进入融合前同时持有全部最终分辨率图像。
        imgs = self.compensate_exposure_errors(corners, imgs)
        self.set_masks(iter(masks))
        seam_masks = self.resize_seam_masks(seam_masks)
        self._progress(90.0, "初始化融合器")

        started_at = time.perf_counter()
        self.initialize_composition(corners, sizes)
        self._record_timing("final_initialize_blender", started_at)
        self._progress(90.0, "融合图像")
        started_at = time.perf_counter()
        self._blend_images_with_progress(
            imgs,
            seam_masks,
            corners,
        )
        self._record_timing("final_pipeline_and_feed", started_at)
        self._progress(95.0, "合成最终全景图")
        started_at = time.perf_counter()
        panorama = self.create_final_panorama()
        self._record_timing("final_blender_blend", started_at)
        self._progress(95.0, "最终全景图完成")
        self._print_timing_summary()
        return panorama

    def debug_info(self) -> dict:
        """返回本次请求自己的阶段与耗时，不读取其他请求的容器日志。"""
        with self._timing_lock:
            timings = {
                name: {"total_seconds": round(float(total), 3), "count": int(count)}
                for name, (total, count) in self._timings.items()
            }
        return {
            "stages": list(self.stage_events),
            "timings": timings,
            "image_count": len(self.images.names),
        }

    def _uses_guided_geometry(self):
        return False

    def _uses_seam_mesh(self):
        return False

    def _prepare_guided_geometry(self, imgs):
        raise NotImplementedError

    def _capture_georef_control_points(
        self, cameras, uncropped_corners, final_corners, final_sizes
    ) -> None:
        """复用最终 affine warper，将固定输入采样点映射到最终全景像素。"""
        if not self._filename_inputs or self.warper.warper_type != "affine":
            return
        source_sizes = self.images.get_scaled_img_sizes(Images.Resolution.FINAL)
        original_sizes = self.images.sizes
        aspect = self.images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)
        warper = cv2.PyRotationWarper("affine", self.warper.scale * aspect)
        if self.cropper.do_crop:
            crop_aspect = self.images.get_ratio(
                Images.Resolution.LOW, Images.Resolution.FINAL
            )
            crop_rectangles = [
                rectangle.times(crop_aspect)
                for rectangle in self.cropper.intersection_rectangles
            ]
        else:
            crop_rectangles = [None] * len(source_sizes)

        if self.cropper.do_crop and crop_rectangles:
            # crop_rois 会把裁剪后的 corners 重新归零；恢复其在 warper
            # 全局坐标系中的原点，才能得到 panorama ndarray 的像素坐标。
            output_x = (
                uncropped_corners[0][0]
                + crop_rectangles[0].x
                - final_corners[0][0]
            )
            output_y = (
                uncropped_corners[0][1]
                + crop_rectangles[0].y
                - final_corners[0][1]
            )
        else:
            output_x, output_y, _, _ = cv2.detail.resultRoi(
                corners=final_corners, sizes=final_sizes
            )

        records = []
        for index, (name, source_size, original_size, camera) in enumerate(
            zip(self.images.names, source_sizes, original_sizes, cameras)
        ):
            source_width, source_height = map(float, source_size)
            original_width, original_height = map(float, original_size)
            raw_corner_x, raw_corner_y = uncropped_corners[index]
            crop_rectangle = crop_rectangles[index]
            for y_ratio in (0.05, 0.5, 0.95):
                for x_ratio in (0.05, 0.5, 0.95):
                    source_x = x_ratio * max(source_width - 1.0, 0.0)
                    source_y = y_ratio * max(source_height - 1.0, 0.0)
                    warped_x, warped_y = warper.warpPoint(
                        (source_x, source_y), self.warper.get_K(camera, aspect), camera.R
                    )
                    if crop_rectangle is not None:
                        local_x = warped_x - raw_corner_x
                        local_y = warped_y - raw_corner_y
                        if not (
                            crop_rectangle.x <= local_x < crop_rectangle.x2
                            and crop_rectangle.y <= local_y < crop_rectangle.y2
                        ):
                            continue
                    records.append({
                        "path": str(name),
                        "source_x": source_x * original_width / source_width,
                        "source_y": source_y * original_height / source_height,
                        "panorama_x": float(warped_x - output_x),
                        "panorama_y": float(warped_y - output_y),
                    })
        self.georef_control_points = records
        if self.show_progress:
            print(f"[地理参考] 已生成 {len(records)} 个 affine 候选控制点", flush=True)

    def _print_camera_adjustment_diagnostics(
        self,
        features,
        matches,
        cameras,
    ) -> None:
        """仅在相机优化失败时输出匹配图和初始相机数值。"""
        threshold = float(self.settings["confidence_threshold"])
        names = list(self.images.names)
        print("相机参数优化失败诊断:", flush=True)
        print(
            f"  images={len(names)}, confidence_threshold={threshold}, "
            f"match_conf={self.settings['match_conf']}, "
            f"range_width={self.settings['range_width']}",
            flush=True,
        )

        for index, (name, feature) in enumerate(zip(names, features), 1):
            keypoint_count = len(feature.getKeypoints())
            print(
                f"  image[{index}] keypoints={keypoint_count} path={name}",
                flush=True,
            )

        adjacency = [set() for _ in names]
        selected_edges = 0
        print(
            "  pair matches: raw / inliers / confidence / selected / H_finite / H_cond",
            flush=True,
        )
        for match in matches:
            source = int(match.src_img_idx)
            target = int(match.dst_img_idx)
            if source < 0 or target < 0 or source >= target:
                continue
            raw_count = len(match.getMatches())
            inliers = int(match.num_inliers)
            confidence = float(match.confidence)
            selected = confidence > threshold
            if selected:
                adjacency[source].add(target)
                adjacency[target].add(source)
                selected_edges += 1

            homography = match.H
            if hasattr(homography, "get"):
                homography = homography.get()
            homography = np.asarray(homography) if homography is not None else None
            finite = bool(
                homography is not None
                and homography.shape == (3, 3)
                and np.all(np.isfinite(homography))
            )
            condition = float("inf")
            if finite:
                try:
                    condition = float(np.linalg.cond(homography))
                except np.linalg.LinAlgError:
                    condition = float("inf")
            print(
                f"    {source + 1}->{target + 1}: "
                f"{raw_count} / {inliers} / {confidence:.4f} / "
                f"{selected} / {finite} / {condition:.3e}",
                flush=True,
            )

        visited = set()
        if names:
            stack = [0]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                stack.extend(adjacency[current] - visited)
        print(
            f"  selected_edges={selected_edges}, "
            f"connected_images={len(visited)}/{len(names)}",
            flush=True,
        )

        for index, camera in enumerate(cameras, 1):
            rotation = np.asarray(camera.R)
            finite = bool(
                np.isfinite(camera.focal)
                and np.isfinite(camera.aspect)
                and np.isfinite(camera.ppx)
                and np.isfinite(camera.ppy)
                and np.all(np.isfinite(rotation))
            )
            determinant = float("nan")
            if rotation.shape == (3, 3) and np.all(np.isfinite(rotation)):
                determinant = float(np.linalg.det(rotation))
            print(
                f"  camera[{index}] focal={camera.focal:.6g}, "
                f"aspect={camera.aspect:.6g}, "
                f"pp=({camera.ppx:.6g},{camera.ppy:.6g}), "
                f"R_det={determinant:.6g}, finite={finite}",
                flush=True,
            )

    def _record_timing(self, name: str, started_at: float) -> None:
        elapsed = time.perf_counter() - started_at
        with self._timing_lock:
            total, count = self._timings.get(name, [0.0, 0])
            self._timings[name] = [float(total) + elapsed, int(count) + 1]

    def _print_timing_summary(self) -> None:
        if not self.show_progress:
            return
        labels = {
            "final_read_resize": "读取并缩放（并行累计）",
            "final_build_maps": "生成共享投影映射",
            "final_warp_image": "投影图像",
            "final_warp_mask": "投影 mask",
            "final_crop_image": "裁剪图像",
            "final_crop_mask": "裁剪 mask",
            "final_exposure": "曝光补偿",
            "final_resize_seam": "缩放接缝 mask",
            "final_initialize_blender": "初始化融合器",
            "final_blender_feed": "写入融合器",
            "final_pipeline_and_feed": "最终流水线墙钟时间",
            "final_blender_blend": "融合器最终输出",
        }
        print("最终阶段耗时明细:", flush=True)
        print(
            "  性能开关: "
            f"optimized_tiff_io={self.optimized_tiff_io}, "
            f"parallel_image_io={self.parallel_image_io}, "
            f"image_io_workers={self.image_io_workers}, "
            f"prefetch_final_images={self.prefetch_final_images}, "
            f"cache_all_valid_masks={self.cache_all_valid_masks}, "
            f"shared_warp_maps={self.shared_warp_maps}",
            flush=True,
        )
        for name, label in labels.items():
            if name not in self._timings:
                continue
            total, count = self._timings[name]
            print(
                f"  {label}: {float(total):.3f}s ({int(count)} 次)",
                flush=True,
            )

    def resize_medium_resolution(self):
        """并发解码文件输入，缩放算法及结果与父类保持一致。"""
        if not self._filename_inputs or getattr(self.images, "_sizes_set", False):
            return super().resize_medium_resolution()

        names = self.images.names
        resolution = Images.Resolution.MEDIUM

        if self.optimized_tiff_io and is_tiff_path(Path(names[0])):
            # 同一次打开中读取尺寸并解码；取得尺寸后初始化三个缩放器。
            def initialize_and_get_target(size):
                self.images._set_scales(size)
                return self.images._get_scaler(resolution).get_scaled_img_size(size)

            first_resized, first_size = read_input_image_with_size(
                names[0], initialize_and_get_target
            )
        else:
            first_img = read_input_image(names[0])
            first_size = Images.get_image_size(first_img)
            self.images._set_scales(first_size)
            first_resized = Images.resize_img_by_scaler(
                self.images._get_scaler(resolution), first_size, first_img
            )

        if self.parallel_image_io:
            worker_count = min(self.image_io_workers, max(1, len(names) - 1))
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="stitch-read",
            ) as executor:
                remaining = list(
                    executor.map(
                        self._read_and_resize_unknown_size,
                        names[1:],
                        [resolution] * (len(names) - 1),
                    )
                )
        else:
            remaining = [
                self._read_and_resize_unknown_size(name, resolution)
                for name in names[1:]
            ]

        self.images._sizes = [first_size, *(size for _, size in remaining)]
        self.images._sizes_set = True
        return [first_resized, *(img for img, _ in remaining)]

    def resize_final_resolution(self):
        """预取下一张最终分辨率图，让解码与当前图的投影/融合重叠。"""
        if not self._filename_inputs:
            return self._resize_final_and_release_sources()
        if not self.parallel_image_io or not self.prefetch_final_images:
            return self._resize_final_file_images()
        return self._prefetch_final_file_images()

    def warp_low_resolution(self, imgs, cameras):
        """低分辨率投影时同步使用 COG NoData/alpha 有效区域。"""
        source_sizes = self.images.get_scaled_img_sizes(Images.Resolution.LOW)
        aspect = self.images.get_ratio(
            Images.Resolution.MEDIUM, Images.Resolution.LOW
        )
        corners, warped_sizes = self.warper.warp_rois(
            source_sizes, cameras, aspect
        )
        warped_imgs = list(self.warper.warp_images(imgs, cameras, aspect))
        warped_masks = [
            self._warp_source_mask(index, size, camera, aspect)
            for index, (size, camera) in enumerate(zip(source_sizes, cameras))
        ]
        return warped_imgs, warped_masks, corners, warped_sizes

    def _read_medium_feature_masks(self, imgs):
        """在特征分辨率下读取 COG NoData/alpha mask。"""
        masks = []
        for name, img in zip(self.images.names, imgs):
            mask = read_valid_mask(name, (img.shape[1], img.shape[0]))
            if mask is None and self.cache_all_valid_masks:
                self._all_valid_mask_paths.add(name)
            masks.append(mask)
        if not any(mask is not None for mask in masks):
            return None
        return [
            mask
            if mask is not None
            else np.full(img.shape[:2], 255, dtype=np.uint8)
            for img, mask in zip(imgs, masks)
        ]

    def _source_valid_mask(self, index, size):
        if self._filename_inputs:
            name = self.images.names[index]
            if self.cache_all_valid_masks and name in self._all_valid_mask_paths:
                return np.full((size[1], size[0]), 255, dtype=np.uint8)
            mask = read_valid_mask(name, tuple(size))
            if mask is not None:
                return mask
            if self.cache_all_valid_masks:
                self._all_valid_mask_paths.add(name)
        return np.full((size[1], size[0]), 255, dtype=np.uint8)

    def _warp_source_mask(self, index, size, camera, aspect):
        mask = self._source_valid_mask(index, size)
        warper = cv2.PyRotationWarper(
            self.warper.warper_type,
            self.warper.scale * aspect,
        )
        _, warped = warper.warp(
            mask,
            self.warper.get_K(camera, aspect),
            camera.R,
            cv2.INTER_NEAREST,
            cv2.BORDER_CONSTANT,
        )
        return warped

    def warp_final_resolution(self, imgs, cameras):
        """图像与 mask 共用每台相机的最终分辨率投影映射。"""
        if not self.shared_warp_maps:
            return self._warp_final_with_separate_maps(imgs, cameras)

        source_sizes = self.images.get_scaled_img_sizes(Images.Resolution.FINAL)
        aspect = self.images.get_ratio(
            Images.Resolution.MEDIUM, Images.Resolution.FINAL
        )
        corners, warped_sizes = self.warper.warp_rois(
            source_sizes, cameras, aspect
        )
        rotation_warper = cv2.PyRotationWarper(
            self.warper.warper_type,
            self.warper.scale * aspect,
        )
        map_cache: dict[int, list] = {}

        def acquire_maps(index, camera):
            cached = map_cache.get(index)
            if cached is None:
                started_at = time.perf_counter()
                roi, xmap, ymap = rotation_warper.buildMaps(
                    source_sizes[index],
                    self.warper.get_K(camera, aspect),
                    camera.R,
                )
                self._record_timing("final_build_maps", started_at)
                actual_size = (xmap.shape[1], xmap.shape[0])
                if tuple(roi[:2]) != tuple(corners[index]):
                    raise RuntimeError("最终投影映射与预估 corner 不一致")
                if actual_size != tuple(warped_sizes[index]):
                    raise RuntimeError("最终投影映射与预估 size 不一致")
                cached = [xmap, ymap, 0]
                map_cache[index] = cached
            return cached[0], cached[1]

        def release_maps(index):
            cached = map_cache[index]
            cached[2] += 1
            if cached[2] == 2:
                del map_cache[index]

        def warp_images():
            for index, (img, camera) in enumerate(zip(imgs, cameras)):
                xmap, ymap = acquire_maps(index, camera)
                started_at = time.perf_counter()
                warped = cv2.remap(
                    img,
                    xmap,
                    ymap,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
                self._record_timing("final_warp_image", started_at)
                release_maps(index)
                yield warped

        def warp_masks():
            for index, (size, camera) in enumerate(zip(source_sizes, cameras)):
                xmap, ymap = acquire_maps(index, camera)
                started_at = time.perf_counter()
                mask = self._source_valid_mask(index, size)
                warped = cv2.remap(
                    mask,
                    xmap,
                    ymap,
                    cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                self._record_timing("final_warp_mask", started_at)
                release_maps(index)
                yield warped

        return warp_images(), warp_masks(), corners, warped_sizes

    def _warp_final_with_separate_maps(self, imgs, cameras):
        """使用第三方 Stitcher 原有的图像/mask 独立投影方式并记录耗时。"""
        source_sizes = self.images.get_scaled_img_sizes(Images.Resolution.FINAL)
        aspect = self.images.get_ratio(
            Images.Resolution.MEDIUM, Images.Resolution.FINAL
        )
        corners, warped_sizes = self.warper.warp_rois(
            source_sizes, cameras, aspect
        )

        def warp_images():
            for img, camera in zip(imgs, cameras):
                started_at = time.perf_counter()
                warped = self.warper.warp_image(img, camera, aspect)
                self._record_timing("final_warp_image", started_at)
                yield warped

        def warp_masks():
            for index, (size, camera) in enumerate(zip(source_sizes, cameras)):
                started_at = time.perf_counter()
                warped = self._warp_source_mask(index, size, camera, aspect)
                self._record_timing("final_warp_mask", started_at)
                yield warped

        return warp_images(), warp_masks(), corners, warped_sizes

    def crop_final_resolution(self, imgs, masks, corners, sizes):
        aspect = self.images.get_ratio(
            Images.Resolution.LOW, Images.Resolution.FINAL
        )
        imgs = self._crop_final_items(imgs, aspect, "final_crop_image")
        masks = self._crop_final_items(masks, aspect, "final_crop_mask")
        corners, sizes = self.cropper.crop_rois(corners, sizes, aspect)
        return imgs, masks, corners, sizes

    def _crop_final_items(self, items, aspect, timing_name):
        for index, item in enumerate(items):
            started_at = time.perf_counter()
            cropped = self.cropper.crop_img(item, index, aspect)
            self._record_timing(timing_name, started_at)
            yield cropped

    def compensate_exposure_errors(self, corners, imgs):
        for index, (corner, img) in enumerate(zip(corners, imgs)):
            mask = self.get_mask(index)
            started_at = time.perf_counter()
            compensated = self.compensator.apply(index, corner, img, mask)
            self._record_timing("final_exposure", started_at)
            yield compensated

    def resize_seam_masks(self, seam_masks):
        for index, seam_mask in enumerate(seam_masks):
            mask = self.get_mask(index)
            started_at = time.perf_counter()
            resized = SeamFinder.resize(seam_mask, mask)
            self._record_timing("final_resize_seam", started_at)
            yield resized

    def _read_and_resize_unknown_size(self, name, resolution):
        if not self.optimized_tiff_io or not is_tiff_path(Path(name)):
            img = read_input_image(name)
            size = Images.get_image_size(img)
            resized = Images.resize_img_by_scaler(
                self.images._get_scaler(resolution), size, img
            )
            return resized, size
        scaler = self.images._get_scaler(resolution)
        return read_input_image_with_size(name, scaler.get_scaled_img_size)

    def _read_direct_or_resize(self, name, size, resolution):
        scaler = self.images._get_scaler(resolution)
        if self.optimized_tiff_io and is_tiff_path(Path(name)):
            target_size = scaler.get_scaled_img_size(size)
            return read_input_image(name, target_size=target_size)
        img = read_input_image(name)
        return Images.resize_img_by_scaler(scaler, size, img)

    def _read_and_resize_known_size(self, item):
        name, size = item
        started_at = time.perf_counter()
        resized = self._read_direct_or_resize(
            name, size, Images.Resolution.FINAL
        )
        self._record_timing("final_read_resize", started_at)
        return resized

    def _prefetch_final_file_images(self):
        items = list(zip(self.images.names, self.images.sizes))
        worker_count = min(self.image_io_workers, len(items))
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="stitch-prefetch",
        ) as executor:
            yield from executor.map(self._read_and_resize_known_size, items)

    def _resize_final_file_images(self):
        for item in zip(self.images.names, self.images.sizes):
            yield self._read_and_resize_known_size(item)

    def _resize_final_and_release_sources(self):
        scaler = self.images._get_scaler(Images.Resolution.FINAL)
        for size, img in zip(self.images.sizes, self.images):
            resized = Images.resize_img_by_scaler(scaler, size, img)
            # 对 ndarray 输入也不要让生成器在投影期间继续引用原图。
            del img
            yield resized

    def _blend_images_with_progress(
        self,
        imgs,
        seam_masks,
        corners,
    ) -> None:
        total = len(self.images.names)
        for idx, (img, mask, corner) in enumerate(zip(imgs, seam_masks, corners), 1):
            blend_progress = 90.0 + 5.0 * idx / max(total, 1)
            write_progress(self.output_path, blend_progress, f"融合图像 {idx}/{total}")
            if self.show_progress:
                print(f"  融合 {idx}/{total}: {self.images.names[idx - 1]}", flush=True)
            if self.timelapser.do_timelapse:
                self.timelapser.process_and_save_frame(
                    self.images.names[idx - 1], img, corner
                )
            else:
                started_at = time.perf_counter()
                self.blender.feed(img, mask, corner)
                self._record_timing("final_blender_feed", started_at)

    def create_final_panorama(self):
        if self._custom_panorama is not None:
            return self._custom_panorama
        return super().create_final_panorama()

    def _use_custom_seam_feather(self) -> bool:
        return (
            not self.timelapser.do_timelapse
            and getattr(self.blender, "blender_type", None) == "feather"
        )
    #！
    # 修改成下面的之后，将不再使用自定义的接缝羽化融合，而是使用默认的 feather blender。
    # def _use_custom_seam_feather(self) -> bool:
    #     return False
    
    def _create_seam_feather_panorama(
        self,
        imgs,
        seam_masks,
        corners,
        sizes,
    ) -> np.ndarray:
        dst_x, dst_y, dst_w, dst_h = cv2.detail.resultRoi(
            corners=corners,
            sizes=sizes,
        )
        accumulator = np.zeros((dst_h, dst_w, 3), dtype=np.float32)
        weight_sum = np.zeros((dst_h, dst_w), dtype=np.float32)
        union = np.zeros((dst_h, dst_w), dtype=bool)
        sigma = self._seam_feather_sigma(dst_w, dst_h)
        total = len(self.images.names)

        for idx, (img, mask, corner) in enumerate(zip(imgs, seam_masks, corners), 1):
            blend_progress = 90.0 + 5.0 * idx / max(total, 1)
            write_progress(self.output_path, blend_progress, f"融合图像 {idx}/{total}")
            if self.show_progress:
                print(f"  融合 {idx}/{total}: {self.images.names[idx - 1]}", flush=True)

            img = self._as_array(img)
            mask = self._as_array(mask)
            mask = np.asarray(mask, dtype=np.uint8)
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            x0 = int(corner[0] - dst_x)
            y0 = int(corner[1] - dst_y)
            x1 = x0 + img.shape[1]
            y1 = y0 + img.shape[0]
            valid = mask > 0
            if not np.any(valid):
                continue

            weight = cv2.GaussianBlur(
                valid.astype(np.float32),
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
            )
            weight[weight < 1e-6] = 0.0
            roi_union = union[y0:y1, x0:x1]
            roi_union |= valid
            accumulator[y0:y1, x0:x1] += img.astype(np.float32) * weight[..., None]
            weight_sum[y0:y1, x0:x1] += weight

        panorama = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
        covered = union & (weight_sum > 1e-6)
        blended = accumulator[covered] / weight_sum[covered, None]
        panorama[covered] = np.clip(blended, 0, 255).astype(np.uint8)
        return panorama

    def _seam_feather_sigma(self, width: int, height: int) -> float:
        blend_width = (width * height) ** 0.5 * float(self.blender.blend_strength) / 100
        return max(self.MIN_SEAM_FEATHER_SIGMA, blend_width / 3.0)

    @staticmethod
    def _as_array(value):
        if hasattr(value, "get") and not isinstance(value, np.ndarray):
            return value.get()
        return value

    def _protect_final_masks(self, masks) -> list[np.ndarray]:
        kernel = np.ones(
            (self.MASK_ERODE_KERNEL_SIZE, self.MASK_ERODE_KERNEL_SIZE),
            dtype=np.uint8,
        )
        protected = []
        for mask in masks:
            if hasattr(mask, "get") and not isinstance(mask, np.ndarray):
                mask = mask.get()
            mask = np.asarray(mask, dtype=np.uint8)
            if mask.size == 0:
                protected.append(mask)
                continue
            protected.append(
                cv2.erode(mask, kernel, iterations=self.MASK_ERODE_ITERATIONS)
            )
        return protected


class GuidedProgressStitcher(ProgressStitcher):
    """人工四角负责配准；沿用 ProgressStitcher 的曝光、接缝和融合阶段。"""

    def __init__(self, guided_images, guided_settings, *args, matching_regions=None, **kwargs):
        # 原 cropper 的最大内接矩形会截掉旋转图像的非重叠边缘。
        # guided 模式保留整个有效范围，最终由 blender 的 ROI 收紧外框。
        kwargs["crop"] = False
        super().__init__(*args, **kwargs)
        self.guided_images = guided_images
        mesh_settings = dict(guided_settings)
        self.mesh_enabled = bool(mesh_settings.pop("guided_mesh_enabled", False))
        self.mesh_grid_spacing = int(mesh_settings.pop("guided_mesh_grid_spacing", 128))
        self.mesh_seam_band_ratio = float(mesh_settings.pop("guided_mesh_seam_band_ratio", 0.15))
        self.mesh_max_displacement_ratio = float(
            mesh_settings.pop("guided_mesh_max_displacement_ratio", 0.02)
        )
        self.mesh_min_matches = int(mesh_settings.pop("guided_mesh_min_matches", 12))
        self.geometry = GuidedAffineStitcher(
            **mesh_settings, show_progress=False, output_path=None,
            matching_regions=matching_regions,
        )
        self.items = []
        self.output_offset = [0.0, 0.0]
        self.output_scale = 1.0
        self.pair_diagnostics = []
        self.seam_mesh_fields: list[dict[str, np.ndarray] | None] = []
        self.mesh_low_corners: list[tuple[float, float]] = []
        self.mesh_low_scale: float | None = None
        self.mesh_diagnostics: dict = {
            "enabled": self.mesh_enabled,
            "status": "pending" if self.mesh_enabled else "disabled",
            "matches_used": 0,
            "images": [],
        }

    def _uses_seam_mesh(self):
        return self.mesh_enabled

    def _fit_seam_mesh(self, imgs, valid_masks, seam_masks, corners) -> None:
        low_scale = self.images._get_scaler(Images.Resolution.LOW).scale
        self.mesh_low_scale = float(low_scale)
        self.mesh_low_corners = [tuple(map(float, corner)) for corner in corners]
        accepted_pairs = {
            (record["image_a"], record["image_b"])
            for record in self.geometry.pair_diagnostics
            if record.get("status") == "refined"
        }
        constraints = []
        for pair in self.geometry.registration_pairs:
            image_a, image_b = pair["image_a"], pair["image_b"]
            if (image_a, image_b) not in accepted_pairs:
                continue
            item_a, item_b = self.items[image_a], self.items[image_b]
            source_points_a = np.asarray(pair["source_points_a"], dtype=np.float64)
            source_points_b = np.asarray(pair["source_points_b"], dtype=np.float64)
            global_inliers = pair.get("global_inliers")
            if global_inliers is not None:
                inliers = np.asarray(global_inliers, dtype=bool).reshape(-1)
                if len(inliers) != len(source_points_a) or len(inliers) != len(source_points_b):
                    continue
                source_points_a = source_points_a[inliers]
                source_points_b = source_points_b[inliers]
            if not len(source_points_a):
                continue
            world_a = _points(item_a["matrix"], source_points_a)
            world_b = _points(item_b["matrix"], source_points_b)
            constraints.append({
                "image_a": image_a,
                "image_b": image_b,
                "points_a": world_a * low_scale - np.asarray(corners[image_a]),
                "points_b": world_b * low_scale - np.asarray(corners[image_b]),
                "residuals": (world_b - world_a) * low_scale,
            })
        movable = [
            item["correction_status"] in {"reference", "refined"}
            for item in self.items
        ]
        self.seam_mesh_fields, self.mesh_diagnostics = solve_seam_mesh(
            [img.shape[:2] for img in imgs],
            valid_masks,
            [self._as_array(mask) for mask in seam_masks],
            constraints,
            movable,
            grid_spacing=self.mesh_grid_spacing,
            seam_band_ratio=self.mesh_seam_band_ratio,
            max_displacement_ratio=self.mesh_max_displacement_ratio,
            min_matches=self.mesh_min_matches,
        )

    def debug_info(self) -> dict:
        info = super().debug_info()
        info["seam_mesh"] = self.mesh_diagnostics
        return info

    def _uses_guided_geometry(self):
        return True

    def _prepare_guided_geometry(self, imgs):
        self.items = self.geometry.register(self.guided_images, imgs)
        self.pair_diagnostics = self.geometry.pair_diagnostics
        return self.items

    def _guided_rois(self, resolution):
        scale = self.images._get_scaler(resolution).scale
        sizes = self.images.get_scaled_img_sizes(resolution)
        corners, warped_sizes = [], []
        for item, size in zip(self.items, sizes):
            matrix = self._guided_matrix(item, size, scale)
            quad = np.array(
                [[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]],
                dtype=np.float64,
            )
            transformed = np.round(_points(matrix, quad), 8)
            if not np.isfinite(transformed).all():
                raise RuntimeError("单应投影产生非有限图像边界")
            low = np.floor(transformed.min(axis=0)).astype(int)
            high = np.ceil(transformed.max(axis=0)).astype(int)
            warped_size = high - low
            if min(warped_size) <= 0 or int(warped_size[0]) * int(warped_size[1]) > 500_000_000:
                raise RuntimeError("单张投影图像尺寸不合理")
            corners.append(tuple(map(int, low)))
            warped_sizes.append(tuple(map(int, warped_size)))
        minimum = np.min(np.asarray(corners), axis=0)
        maximum = np.max(np.asarray(corners) + np.asarray(warped_sizes), axis=0)
        canvas_size = maximum - minimum
        if int(canvas_size[0]) * int(canvas_size[1]) > 500_000_000:
            raise RuntimeError("单应投影后的全景画布尺寸不合理")
        return corners, warped_sizes

    @staticmethod
    def _guided_matrix(item, image_size, output_scale, origin=(0, 0)):
        """Map current-resolution source pixels into the requested canvas."""
        source_scale = np.asarray(item["size"], dtype=np.float64) / np.asarray(
            image_size, dtype=np.float64
        )
        matrix = np.asarray(item["matrix"], dtype=np.float64)
        if matrix.shape == (2, 3):
            result = matrix.copy()
            result[:, :2] *= output_scale * source_scale
            result[:, 2] = output_scale * result[:, 2] - np.asarray(origin)
            return result
        scaled_source = np.diag([source_scale[0], source_scale[1], 1.0])
        scaled_output = np.diag([output_scale, output_scale, 1.0])
        translated_output = np.array(
            [[1.0, 0.0, -origin[0]], [0.0, 1.0, -origin[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return translated_output @ scaled_output @ _matrix3(matrix) @ scaled_source

    def _guided_warp(self, index, img, mask, resolution, corner, size):
        item = self.items[index]
        scale = self.images._get_scaler(resolution).scale
        matrix = self._guided_matrix(
            item, (img.shape[1], img.shape[0]), scale, corner
        )
        if matrix.shape == (3, 3):
            warped_img = cv2.warpPerspective(
                img, matrix, size, flags=cv2.INTER_LINEAR
            )
            warped_mask = cv2.warpPerspective(
                mask, matrix, size, flags=cv2.INTER_NEAREST
            )
        else:
            warped_img = cv2.warpAffine(img, matrix, size, flags=cv2.INTER_LINEAR)
            warped_mask = cv2.warpAffine(mask, matrix, size, flags=cv2.INTER_NEAREST)
        if (
            resolution == Images.Resolution.FINAL
            and self.mesh_enabled
            and self.seam_mesh_fields[index] is not None
        ):
            if self.mesh_low_scale is None or index >= len(self.mesh_low_corners):
                raise RuntimeError("局部网格缺少低分辨率坐标信息")
            mesh_scale = scale / self.mesh_low_scale
            warped_img, warped_mask = warp_with_mesh(
                warped_img,
                warped_mask,
                self.seam_mesh_fields[index],
                inplace=True,
                low_to_output_scale=(mesh_scale, mesh_scale),
                low_origin=self.mesh_low_corners[index],
                output_origin=corner,
            )
        return warped_img, warped_mask

    def warp_low_resolution(self, imgs, cameras):
        corners, sizes = self._guided_rois(Images.Resolution.LOW)
        warped_imgs, warped_masks = [], []
        for index, (img, corner, size) in enumerate(zip(imgs, corners, sizes)):
            valid = self._source_valid_mask(index, (img.shape[1], img.shape[0]))
            warped_img, warped_mask = self._guided_warp(
                index, img, valid, Images.Resolution.LOW, corner, size,
            )
            warped_imgs.append(warped_img)
            warped_masks.append(warped_mask)
        return warped_imgs, warped_masks, corners, sizes

    def warp_final_resolution(self, imgs, cameras):
        corners, sizes = self._guided_rois(Images.Resolution.FINAL)
        cache = {}

        def warped_images():
            for index, img in enumerate(imgs):
                valid = self._source_valid_mask(index, (img.shape[1], img.shape[0]))
                pair = self._guided_warp(
                    index, img, valid, Images.Resolution.FINAL,
                    corners[index], sizes[index],
                )
                cache[index] = pair[1]
                yield pair[0]

        def warped_masks():
            for index in range(len(corners)):
                yield cache.pop(index)

        return warped_images(), warped_masks(), corners, sizes

    def _capture_georef_control_points(
        self, cameras, uncropped_corners, final_corners, final_sizes,
    ):
        x, y, _, _ = cv2.detail.resultRoi(
            corners=final_corners, sizes=final_sizes,
        )
        self.output_offset = [-float(x), -float(y)]
        self.output_scale = float(
            self.images._get_scaler(Images.Resolution.FINAL).scale
        )
