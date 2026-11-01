"""拼接进度与阶段计时工具。

职责：写入 API/CLI 共用的 progress.txt，并提供不改变现有返回结构的轻量计时器。
输入：输出路径、百分比、状态文本或阶段名称；输出：进度文件及内存中的耗时记录。
流程：定位输出目录 -> 原子语义覆盖进度文本；计时器按阶段记录 perf_counter 差值。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


def write_progress(output_path, progress_percentage, status) -> None:
    """将进度百分比写入与输出图像同级的 progress.txt。"""
    if output_path is None:
        return
    output_dir = os.path.dirname(str(output_path))
    progress_filepath = os.path.join(output_dir, "progress.txt")
    try:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(progress_filepath, "w", encoding="utf-8") as file:
            file.write(f"{progress_percentage:.2f},{status}")
    except Exception as exc:  # noqa: BLE001 - 保持旧进度写入的容错行为
        print(f"写入进度文件失败: {exc}")


@dataclass
class StageTimer:
    """记录阶段耗时；当前仅供内部诊断，不改变 API 响应。"""

    started_at: float = field(default_factory=time.perf_counter)
    last_at: float = field(init=False)
    durations: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.last_at = self.started_at

    def mark(self, stage: str) -> float:
        now = time.perf_counter()
        duration = now - self.last_at
        self.durations[stage] = duration
        self.last_at = now
        return duration

    @property
    def total(self) -> float:
        return time.perf_counter() - self.started_at
