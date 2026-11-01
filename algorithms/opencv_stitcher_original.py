"""OpenCV Stitcher 优化前源码备份。

此文件保留并行解码和最终分辨率预取改造之前的完整实现，仅供对照测试；
项目默认入口仍然是 algorithms/opencv_stitcher.py。
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from stitching import Stitcher
from stitching.images import Images

from progress import write_progress


class ProgressStitcher(Stitcher):
    MASK_ERODE_KERNEL_SIZE = 3
    MASK_ERODE_ITERATIONS = 1
    MIN_SEAM_FEATHER_SIGMA = 1.0

    def __init__(self, *args, show_progress=True, output_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_progress = show_progress
        self.output_path = output_path
        self._custom_panorama = None
        self._stage_index = 0
        self._stage_total = 21
        self._started_at = time.perf_counter()

    def _progress(self, percentage: float, message: str) -> None:
        self._stage_index += 1
        write_progress(self.output_path, percentage, message)
        if not self.show_progress:
            return
        elapsed = time.perf_counter() - self._started_at
        print(
            f"[{self._stage_index:02d}/{self._stage_total}] "
            f"{message} ({elapsed:.1f}s)",
            flush=True,
        )

    def stitch(self, images, feature_masks=[]):
        self._stage_index = 0
        self._started_at = time.perf_counter()
        self._custom_panorama = None
        self._progress(11.0, "读取图片并初始化缩放参数")
        self.images = Images.of(
            images, self.medium_megapix, self.low_megapix, self.final_megapix
        )
        self._progress(12.0, "缩放到特征检测分辨率")
        imgs = self.resize_medium_resolution()
        self._progress(15.0, "检测特征点")
        features = self.find_features(imgs, feature_masks)
        self._progress(20.0, "匹配图像特征")
        matches = self.match_features(features)
        self._progress(25.0, "筛选属于同一拼接图的图片")
        imgs, features, matches = self.subset(imgs, features, matches)
        print(f"参与拼接图片数: {len(self.images.names)}", flush=True)
        self._progress(30.0, "估计相机参数")
        cameras = self.estimate_camera_parameters(features, matches)
        self._progress(35.0, "优化相机参数")
        cameras = self.refine_camera_parameters(features, matches, cameras)
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
        seam_masks = self.find_seam_masks(imgs, corners, masks)
        self._progress(75.0, "缩放到最终输出分辨率")
        imgs = self.resize_final_resolution()
        self._progress(80.0, "最终分辨率投影变换")
        imgs, masks, corners, sizes = self.warp_final_resolution(imgs, cameras)
        self._progress(83.0, "执行最终裁剪")
        imgs, masks, corners, sizes = self.crop_final_resolution(
            imgs, masks, corners, sizes
        )

        # ！   可以直接注释掉下面这行
        # 腐蚀接缝掩膜以保护最终融合区域，避免接缝处出现明显的融合痕迹。
        # masks = self._protect_final_masks(masks)
        self._progress(86.0, "应用曝光补偿并调整接缝掩膜")
        self.set_masks(iter(masks))
        # 保留父类的流式行为：曝光补偿、接缝 mask 缩放和融合按图片逐张执行，
        # 避免在进入融合前同时持有全部最终分辨率图像。
        imgs = self.compensate_exposure_errors(corners, imgs)
        self.set_masks(iter(masks))
        seam_masks = self.resize_seam_masks(seam_masks)
        self._progress(90.0, "初始化融合器")

        self.initialize_composition(corners, sizes)
        self._progress(90.0, "融合图像")
        self._blend_images_with_progress(imgs, seam_masks, corners)
        self._progress(95.0, "最终全景图完成")
        return self.create_final_panorama()

    def _blend_images_with_progress(self, imgs, seam_masks, corners) -> None:
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
                self.blender.feed(img, mask, corner)

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
