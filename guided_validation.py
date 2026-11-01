"""在浏览器中摆放图像，并调用容器中的 guided_affine API 进行验证。

示例：
    python guided_validation.py test_data/aerial_4000x24000/aerial_0{1,2}.tif
    # 打开 http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from flask import Flask, Response, jsonify, render_template, request, send_file
from PIL import Image

from algorithms.guided_affine import validate_matching_regions


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "output"
UPLOAD_ROOT = OUTPUT_ROOT / "guided_uploads"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
OUTPUT_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _image_size(path: Path) -> tuple[int, int]:
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as exc:
            raise RuntimeError("读取 TIFF 预览需要安装 rasterio") from exc
        with rasterio.open(path) as source:
            return source.width, source.height
    with Image.open(path) as source:
        return source.size


def _preview(path: Path, max_side: int = 1600) -> bytes:
    if path.suffix.lower() in {".tif", ".tiff"}:
        import numpy as np
        import rasterio
        from rasterio.enums import Resampling

        with rasterio.open(path) as source:
            ratio = min(1.0, max_side / max(source.width, source.height))
            width = max(1, round(source.width * ratio))
            height = max(1, round(source.height * ratio))
            indexes = list(range(1, min(3, source.count) + 1))
            bands = source.read(indexes, out_shape=(len(indexes), height, width),
                                resampling=Resampling.bilinear)
            mask = source.dataset_mask(out_shape=(height, width),
                                       resampling=Resampling.nearest)
        if len(indexes) < 3:
            bands = np.repeat(bands[:1], 3, axis=0)
        if bands.dtype != np.uint8:
            bands = bands.astype(np.float32)
            valid = bands[:, mask > 0]
            if valid.size:
                low, high = np.percentile(valid, [2, 98])
                bands = np.clip((bands - low) * 255.0 / max(1.0, high - low), 0, 255)
            bands = bands.astype(np.uint8)
        rgba = np.dstack((*bands[:3], mask)).astype(np.uint8)
        image = Image.fromarray(rgba, "RGBA")
    else:
        with Image.open(path) as source:
            if path.suffix.lower() in {".jpg", ".jpeg"}:
                source.draft("RGB", (max_side, max_side))
            source.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            image = source.convert("RGBA")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _resolve_images(paths: list[str], input_folder: str | None) -> list[dict]:
    if input_folder:
        if paths:
            raise ValueError("图片路径与 --input-folder 只能使用一种输入方式")
        folder = Path(input_folder).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"输入目录不存在: {folder}")
        candidates = sorted(path for path in folder.iterdir()
                            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    else:
        candidates = [Path(raw).expanduser().resolve() for raw in paths]
    images = []
    for path in candidates:
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            raise ValueError(f"不是可读取的图片: {path}")
        try:
            relative = path.relative_to(PROJECT_ROOT)
        except ValueError as exc:
            raise ValueError(f"图片必须位于项目目录内，才能通过 Docker 挂载访问: {path}") from exc
        width, height = _image_size(path)
        images.append({"name": path.name, "host_path": str(path),
                       "container_path": f"/workspace/{relative.as_posix()}",
                       "width": width, "height": height})
    return images


def create_app(
    image_paths: list[str], *, input_folder: str | None = None,
    api_url: str = "http://127.0.0.1:8002", preview_max_side: int = 1600,
) -> Flask:
    images = _resolve_images(image_paths, input_folder)
    images_lock = Lock()
    app = Flask(__name__, template_folder=str(PROJECT_ROOT / "validation_templates"))
    app.config.update(VALIDATION_IMAGES=images, VALIDATION_API_URL=api_url.rstrip("/"),
                      VALIDATION_PREVIEW_MAX_SIDE=preview_max_side,
                      MAX_CONTENT_LENGTH=1024 * 1024 * 1024)

    @app.after_request
    def disable_html_cache(response):
        """避免浏览器继续运行与最新 API 响应不兼容的旧版内联脚本。"""
        if response.mimetype == "text/html":
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    def public_images():
        return [{"index": index, "name": item["name"],
                 "width": item["width"], "height": item["height"],
                 "preview_url": f"/preview/{index}?v={Path(item['host_path']).stem}"}
                for index, item in enumerate(images)]

    @app.get("/")
    def index():
        return render_template("guided_validation.html", images=public_images(),
                               api_url=app.config["VALIDATION_API_URL"])

    @app.post("/api/images")
    def upload_images():
        files = request.files.getlist("images")
        if len(files) < 2 or any(not file.filename for file in files):
            return jsonify({"error": "请一次选择至少两张图片"}), 400
        saved = []
        new_images = []
        try:
            UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
            for file in files:
                name = file.filename.replace("\\", "/").split("/")[-1]
                suffix = Path(name).suffix.lower()
                if suffix not in IMAGE_SUFFIXES:
                    raise ValueError(f"不支持的图片格式: {name}")
                path = UPLOAD_ROOT / f"{uuid4().hex}{suffix}"
                file.save(path)
                saved.append(path)
                width, height = _image_size(path)
                if width < 2 or height < 2:
                    raise ValueError(f"图像尺寸过小: {name}")
                relative = path.relative_to(PROJECT_ROOT)
                new_images.append({"name": name, "host_path": str(path),
                                   "container_path": f"/workspace/{relative.as_posix()}",
                                   "width": width, "height": height})
        except Exception as exc:  # noqa: BLE001 - 上传失败时需清理本请求已保存的文件
            for path in saved:
                path.unlink(missing_ok=True)
            return jsonify({"error": f"读取图片失败: {exc}"}), 400
        with images_lock:
            images[:] = new_images
            return jsonify({"images": public_images()})

    @app.get("/preview/<int:index>")
    def preview(index: int):
        if index < 0 or index >= len(images):
            return jsonify({"error": "图片索引不存在"}), 404
        try:
            data = _preview(Path(images[index]["host_path"]), preview_max_side)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"预览生成失败: {exc}"}), 500
        return Response(data, mimetype="image/png")

    @app.get("/result/<path:name>")
    def result(name: str):
        if Path(name).name != name or name in {"", ".", ".."}:
            return jsonify({"error": "无效的结果文件名"}), 400
        path = OUTPUT_ROOT / name
        if not path.is_file():
            return jsonify({"error": "结果文件不存在"}), 404
        try:
            return Response(_preview(path, 1800), mimetype="image/png")
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"结果预览失败: {exc}"}), 500

    @app.get("/download/<path:name>")
    def download(name: str):
        if Path(name).name != name or name in {"", ".", ".."}:
            return jsonify({"error": "无效的结果文件名"}), 400
        path = OUTPUT_ROOT / name
        if not path.is_file():
            return jsonify({"error": "结果文件不存在"}), 404
        return send_file(path, as_attachment=True, download_name=name)

    @app.get("/api/progress")
    def progress():
        """查询共用进度文件；版本号用于忽略本次请求开始前的旧进度。"""
        path = OUTPUT_ROOT / "progress.txt"
        try:
            version = path.stat().st_mtime_ns
        except FileNotFoundError:
            version = 0
        if request.args.get("baseline") == "1":
            return jsonify({"version": str(version)})
        since_raw = request.args.get("since", "0")
        if not since_raw.isdecimal():
            return jsonify({"error": "无效的进度版本号"}), 400
        if version <= int(since_raw):
            return jsonify({"exists": False, "version": str(version)})
        try:
            raw = path.read_text(encoding="utf-8").strip()
            percentage_text, status = raw.split(",", 1)
            percentage = float(percentage_text)
            if not math.isfinite(percentage):
                raise ValueError("无效的进度数值")
        except (FileNotFoundError, ValueError):
            return jsonify({"exists": False, "version": str(version)})
        return jsonify({"exists": True, "version": str(version),
                        "percentage": max(0.0, min(100.0, percentage)),
                        "status": status})

    @app.post("/api/stitch")
    def stitch():
        with images_lock:
            current_images = list(images)
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "请求必须是 JSON object"}), 400
        try:
            matching_regions = validate_matching_regions(body.get("matching_regions"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        positions = body.get("images")
        if not isinstance(positions, list) or len(positions) < 2 or len(positions) != len(current_images):
            return jsonify({"error": "提交的图片数量与加载的图片不一致"}), 400
        guided_images = []
        for index, position in enumerate(positions):
            if not isinstance(position, dict) or position.get("index") != index:
                return jsonify({"error": f"第 {index + 1} 张图片的顺序无效"}), 400
            corners = position.get("corners")
            if not isinstance(corners, list) or len(corners) != 4:
                return jsonify({"error": f"第 {index + 1} 张图片缺少四角坐标"}), 400
            if any(not isinstance(point, list) or len(point) != 2 or
                   any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                       not math.isfinite(value) for value in point) for point in corners):
                return jsonify({"error": f"第 {index + 1} 张图片坐标无效"}), 400
            guided_images.append({"path": current_images[index]["container_path"], "corners": corners})
        first = guided_images[0]["corners"]
        width, height = current_images[0]["width"], current_images[0]["height"]
        expected = [[0, 0], [width, 0], [width, height], [0, height]]
        if any(abs(first[k][axis] - expected[k][axis]) > 1e-3
               for k in range(4) for axis in range(2)):
            return jsonify({"error": "第一张图片必须固定在原点且不能旋转或缩放"}), 400

        output_name = str(body.get("output_name", "guided_result.jpg")).strip()
        if (not output_name or Path(output_name).name != output_name or
                output_name in {".", ".."} or Path(output_name).suffix.lower() not in OUTPUT_SUFFIXES):
            return jsonify({"error": "结果文件名只能是 output 目录中的 JPG、PNG 或 TIFF 文件名"}), 400
        settings = {"algorithm": "guided_affine", "georef_mode": "none"}
        for key in ("final_megapix", "medium_megapix", "low_megapix", "blender_type",
                    "guided_ransac_threshold_px", "guided_max_translation_ratio",
                    "guided_max_rotation_degrees", "guided_max_scale_change", "guided_max_shear"):
            if key in body:
                settings[key] = body[key]
        payload = {"images": guided_images, "matching_regions": matching_regions, "output_path": f"/workspace/output/{output_name}",
                   "overwrite": True, "show_progress": True, "settings": settings}
        upstream = Request(f"{api_url.rstrip('/')}/headless_stitching_image",
                           data=json.dumps(payload).encode("utf-8"),
                           headers={"Content-Type": "application/json"}, method="POST")
        try:
            # 本地 Docker API 不应被系统 HTTP_PROXY/HTTPS_PROXY 转发。
            with build_opener(ProxyHandler({})).open(upstream, timeout=3600) as response:
                result_data = json.load(response)
        except HTTPError as exc:
            try:
                message = json.load(exc).get("error", exc.reason)
            except (ValueError, AttributeError):
                message = exc.reason
            return jsonify({"error": f"拼接 API 返回 {exc.code}: {message}"}), 502
        except (URLError, TimeoutError, OSError) as exc:
            return jsonify({"error": f"无法连接拼接 API ({api_url}): {exc}"}), 502
        result_data["preview_url"] = f"/result/{output_name}"
        result_data["download_url"] = f"/download/{output_name}"
        result_data["submitted_payload"] = payload
        return jsonify(result_data)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="人工摆放图像并验证 guided_affine 拼接")
    parser.add_argument("images", nargs="*", help="可选：预加载图片；也可以在网页中读取，第一张固定为坐标参考")
    parser.add_argument("--input-folder", help="按文件名排序加载目录中的图片")
    parser.add_argument("--api-url", default="http://127.0.0.1:8002",
                        help="stitching-headless-api:latest 的宿主机地址")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--preview-max-side", type=int, default=1600)
    args = parser.parse_args()
    if args.preview_max_side < 100 or args.port < 1 or args.port > 65535:
        parser.error("预览尺寸或端口无效")
    try:
        app = create_app(args.images, input_folder=args.input_folder,
                         api_url=args.api_url, preview_max_side=args.preview_max_side)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"验证画布: http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
