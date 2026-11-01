"""第三方 stitching.Stitcher 原始流程封装。

职责：提供名为 ``origin`` 的算法入口，直接执行第三方 Stitcher.stitch，不展开或改写内部阶段。
输入：Stitcher 支持的配置参数和有序图片路径；输出：第三方库生成的 BGR 全景图。
流程：初始化原生 Stitcher -> 一次调用 stitch -> 返回结果。
"""

from stitching import Stitcher

from image_io import read_input_image


class OriginStitcher(Stitcher):
    """保持第三方 Stitcher 原始行为的算法类。"""

    def stitch(self, images, feature_masks=None):
        decoded = [
            read_input_image(image) if isinstance(image, str) else image
            for image in images
        ]
        return super().stitch(decoded, [] if feature_masks is None else feature_masks)
