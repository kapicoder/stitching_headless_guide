"""无界面图像拼接 HTTP API。

职责：解析和校验请求、调用统一拼接入口、保存结果并提供进度查询。
输入：JSON 请求或 progress 查询参数；输出：兼容原格式的 JSON 响应。
流程：解析选项 -> 收集图片 -> 执行拼接 -> 写出结果 -> 返回元数据。
"""

from __future__ import annotations

import itertools
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from flask import Flask, jsonify, request
import largestinteriorrectangle as lir
import numpy as np

try:
    from config import DEFAULT_SETTINGS, build_settings
except ImportError:  # pragma: no cover - package import fallback
    from .config import DEFAULT_SETTINGS, build_settings

try:
    from algorithms.guided_affine import build_guided_settings, validate_matching_regions
except ImportError:  # pragma: no cover - package import fallback
    from .algorithms.guided_affine import build_guided_settings, validate_matching_regions

try:
    from stitiching import (
        collect_images,
        is_tiff_path,
        stitch_images,
        write_output_image,
    )
except ImportError:  # pragma: no cover - package import fallback
    from .stitiching import (
        collect_images,
        is_tiff_path,
        stitch_images,
        write_output_image,
    )


def _warmup_numba_lir() -> None:
    """在 worker 接收首个请求前加载 Numba/LIR 的常用签名。"""
    started_at = time.perf_counter()
    mask = np.ones((256, 256), dtype=np.bool_)
    mask[:8, :] = False
    mask[:, :8] = False
    contour = np.array(
        [[8, 8], [255, 8], [255, 255], [8, 255]],
        dtype=np.int32,
    )

    lir.lir(mask)
    lir.lir(mask, contour)
    print(
        "[启动预热] Numba/LIR 加载完成 "
        f"用时={time.perf_counter() - started_at:.3f}秒",
        flush=True,
    )


_warmup_numba_lir()


app = Flask(__name__)
_stitch_request_ids = itertools.count(1)
_stitch_task_lock = Lock()


def _payload_value(payload: Dict[str, Any], key: str, fallback: Any) -> Any:
    if payload.get(key) not in (None, ""):
        return payload[key]
    return fallback


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_optional_path(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(Path(str(value)).expanduser())


def _as_image_paths(value: Any) -> list[str] | None:
    if value in (None, ""):
        return None
    if not isinstance(value, list):
        raise ValueError("image_paths 必须是路径数组")
    paths = [str(item).strip() for item in value if str(item).strip()]
    if not paths:
        raise ValueError("image_paths 不能为空")
    return paths


def _as_guided_images(value: Any) -> list[dict] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("images 必须是至少包含两张图片的数组")
    images = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"].strip():
            raise ValueError(f"images[{index}].path 必须是非空路径")
        images.append({"path": str(Path(item["path"].strip()).expanduser()),
                       "corners": item.get("corners")})
    return images


def _progress_path(output_path: str) -> str:
    output_dir = os.path.dirname(output_path) or "."
    return os.path.join(output_dir, "progress.txt")


def _write_progress(output_path: str | None, percentage: float, status: str) -> None:
    if not output_path:
        return
    progress_path = _progress_path(output_path)
    progress_dir = os.path.dirname(progress_path)
    if progress_dir:
        os.makedirs(progress_dir, exist_ok=True)
    with open(progress_path, "w", encoding="utf-8") as f:
        f.write(f"{percentage:.1f},{status}")


def _read_progress_file(output_path: str) -> Tuple[str, str]:
    progress_path = _progress_path(output_path)
    if not os.path.exists(progress_path):
        raise FileNotFoundError(progress_path)
    with open(progress_path, "r", encoding="utf-8") as f:
        return f.read().strip(), progress_path


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    if isinstance(value, tuple):
        return [_serialize(v) for v in value]
    return value


def _collect_from_payload(options: Dict[str, Any]) -> list[str]:
    if options["image_paths"] is not None:
        image_paths = [str(Path(path).expanduser()) for path in options["image_paths"]]
        if options["limit"] is not None:
            image_paths = image_paths[: options["limit"]]
        return image_paths

    input_folder = Path(options["input_folder"]).expanduser()
    return collect_images(input_folder, options["limit"])


def _build_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    nested = payload.get("settings")
    nested_algorithm = nested.get("algorithm") if isinstance(nested, dict) else None
    requested = payload.get("algorithm")
    if requested in (None, ""):
        requested = nested_algorithm
    if str(requested or DEFAULT_SETTINGS["algorithm"]).strip().lower() == "guided_affine":
        return build_guided_settings(payload)
    return build_settings(payload)


def _build_options(payload: Dict[str, Any]) -> Dict[str, Any]:
    options = {
        "images": _as_guided_images(payload.get("images")),
        "image_paths": _as_image_paths(_payload_value(payload, "image_paths", None)),
        "input_folder": _as_optional_path(_payload_value(payload, "input_folder", None)),
        "output_path": str(_payload_value(payload, "output_path", "")).strip(),
        "overwrite": _as_bool(_payload_value(payload, "overwrite", True)),
        "limit": _payload_value(payload, "limit", None),
        "show_progress": _as_bool(_payload_value(payload, "show_progress", True)),
        "settings": _build_settings(payload),
    }

    options["matching_regions"] = validate_matching_regions(payload.get("matching_regions"))
    if options["matching_regions"] and options["settings"]["algorithm"] != "guided_affine":
        raise ValueError("matching_regions 仅支持 guided_affine")
    if not options["output_path"]:
        raise ValueError("output_path 必须是非空字符串")
    if options["images"] is not None:
        if options["image_paths"] is not None or options["input_folder"] is not None:
            raise ValueError("images 不能与 image_paths 或 input_folder 同时使用")
        if options["settings"]["algorithm"] != "guided_affine":
            raise ValueError("images 输入必须设置 algorithm=guided_affine")
        if "limit" in payload:
            raise ValueError("images 输入不支持 limit，请直接提供需要的图片")
        if options["settings"]["georef_mode"] not in {"none", "affine_fit"}:
            raise ValueError("guided_affine 的 georef_mode 只支持 none 或 affine_fit")
        if (options["settings"]["georef_mode"] == "affine_fit"
                and Path(options["output_path"]).suffix.lower() not in {".tif", ".tiff"}):
            raise ValueError("guided_affine 使用 affine_fit 时输出路径必须是 .tif 或 .tiff")
    elif options["settings"]["algorithm"] == "guided_affine":
        raise ValueError("guided_affine 必须提供 images 及每张图片的四角坐标")
    elif options["image_paths"] is None and options["input_folder"] is None:
        raise ValueError("必须提供 images、image_paths 或 input_folder")
    if options["limit"] is not None:
        options["limit"] = int(options["limit"])
        if options["limit"] <= 0:
            raise ValueError("limit 必须大于0")
    return options


def _run_headless_stitch(options: Dict[str, Any]) -> Dict[str, Any]:
    output_path = options["output_path"]
    output = Path(output_path).expanduser()
    if output.exists() and not options["overwrite"]:
        raise FileExistsError(f"输出文件已存在: {output_path}")

    _write_progress(output_path, 5.0, "collecting_images")
    image_paths = ([item["path"] for item in options["images"]]
                   if options["images"] is not None else _collect_from_payload(options))
    missing = [path for path in image_paths if not Path(path).expanduser().exists()]
    if missing:
        raise FileNotFoundError(f"输入图片不存在: {missing}")
    if len(image_paths) < 2:
        raise ValueError("至少需要 2 张图片")

    output.parent.mkdir(parents=True, exist_ok=True)
    algorithm = options["settings"]["algorithm"]
    _write_progress(output_path, 10.0, f"{algorithm}_stitching")

    georef_context: dict = {}
    result_context: dict = {}
    panorama = stitch_images(
        image_paths,
        options["settings"],
        show_progress=options["show_progress"],
        output_path=output_path,
        georef_context=georef_context,
        guided_images=options["images"],
        matching_regions=options["matching_regions"],
        result_context=result_context,
    )
    if panorama is None:
        raise RuntimeError("拼接失败: Stitcher 未返回图像")

    if "output_source" in result_context:
        print(
            f"[API] 输出来源: {result_context['output_source']} "
            f"(融合方式: {result_context['fusion_method']}; "
            "非重叠区域来自单张输入图，但仍经过投影/曝光处理)",
            flush=True,
        )

    _write_progress(output_path, 95.0, "saving")
    saved_path = write_output_image(
        output, panorama, image_paths,
        georef_mode=options["settings"]["georef_mode"],
        georef_context=georef_context,
    )

    _write_progress(output_path, 100.0, "completed")
    response = {
        "status": "done",
        "output_path": saved_path,
        "progress_path": _progress_path(saved_path),
        "image_count": len(image_paths),
        "valid_paths": image_paths,
        "output_width": int(panorama.shape[1]),
        "output_height": int(panorama.shape[0]),
        "algorithm": algorithm,
        "cog": is_tiff_path(Path(saved_path)),
        "settings": options["settings"],
    }
    if "debug_info" in result_context:
        response["debug_info"] = result_context["debug_info"]
        if "pair_diagnostics" in result_context:
            response["debug_info"]["pair_diagnostics"] = result_context["pair_diagnostics"]
    return response


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok", "module": "stitching_headless"}), 200


def _handle_headless_stitching_image() -> Any:
    # 记录请求开始时间和请求 ID
    request_started_at = time.perf_counter()
    request_id = next(_stitch_request_ids)
    remote_addr = request.remote_addr
    print(
        f"[API] 收到拼接请求 #{request_id} from={remote_addr}",
        flush=True,
    )
    if not request.is_json:
        return jsonify({"error": "请求体必须为 JSON"}), 400

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "请求体必须是 JSON object"}), 400

    output_path = payload.get("output_path") if isinstance(payload.get("output_path"), str) else None
    if _stitch_task_lock.locked():
        print(f"[API] 拼接请求 #{request_id} 等待执行", flush=True)

    # OpenCV 拼接本身会使用多个内部线程并占用较多内存。请求在此
    # 串行排队，避免多个拼接任务同时竞争 CPU/内存并相互覆盖进度。
    with _stitch_task_lock:
        print(f"[API] 开始拼接请求 #{request_id}", flush=True)
        try:
            options = _build_options(payload)
            response = _run_headless_stitch(options)
        except (FileNotFoundError, KeyError, TypeError, ValueError, FileExistsError) as exc:
            app.logger.exception(
                "[API] 拼接请求 #%s 失败 from=%s output_path=%r",
                request_id,
                remote_addr,
                output_path,
            )
            try:
                _write_progress(output_path, 0.0, "failed")
            except Exception:  # noqa: BLE001
                app.logger.exception(
                    "[API] 拼接请求 #%s 写入失败进度时再次异常",
                    request_id,
                )
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            app.logger.exception(
                "[API] 拼接请求 #%s 失败 from=%s output_path=%r",
                request_id,
                remote_addr,
                output_path,
            )
            try:
                _write_progress(output_path, 0.0, "failed")
            except Exception:  # noqa: BLE001
                app.logger.exception(
                    "[API] 拼接请求 #%s 写入失败进度时再次异常",
                    request_id,
                )
            return jsonify({"error": str(exc)}), 500

        elapsed_seconds = time.perf_counter() - request_started_at
        print(
            f"[API] 完成拼接请求 #{request_id} "
            f"output_path={response['output_path']!r} "
            f"总用时={elapsed_seconds:.3f}秒",
            flush=True,
        )

    return jsonify(_serialize(response)), 200


@app.route("/headless_stitching_image", methods=["POST"])
def headless_stitching_image() -> Any:
    return _handle_headless_stitching_image()


@app.route("/visible_stitching_image", methods=["POST"])
def visible_stitching_image() -> Any:
    return _handle_headless_stitching_image()


@app.route("/stitch", methods=["POST"])
def run_stitch() -> Any:
    return _handle_headless_stitching_image()


@app.route("/progress", methods=["GET"])
def get_progress() -> Any:
    output_path = request.args.get("output_path", "").strip()
    if not output_path:
        return jsonify({"error": "缺少 output_path 查询参数"}), 400

    progress_path = _progress_path(output_path)
    try:
        raw, _ = _read_progress_file(output_path)
    except FileNotFoundError:
        return jsonify({"exists": False, "progress_path": progress_path}), 200

    percentage = None
    status = None
    if raw:
        parts = raw.split(",", 1)
        try:
            percentage = float(parts[0])
        except ValueError:
            percentage = None
        if len(parts) > 1:
            status = parts[1]

    return jsonify(
        {
            "exists": True,
            "progress_path": progress_path,
            "percentage": percentage,
            "status": status,
            "raw": raw,
        }
    ), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
