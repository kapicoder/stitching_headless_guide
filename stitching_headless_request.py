"""拼接 HTTP API 的 Python/命令行客户端。

职责：构造兼容请求、调用拼接与进度接口，并提供命令行参数解析。
输入：Python 参数或 CLI 参数；输出：requests.Response 或解析后的 JSON 字典。
流程：收集参数 -> 构造 JSON -> 发送 HTTP 请求 -> 校验状态 -> 返回结果。
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import requests
from requests import Response

from config import DEFAULT_SETTINGS


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    settings = {
        "algorithm": args.algorithm,
        "detector": args.detector,
        "nfeatures": args.nfeatures,
        "match_conf": args.match_conf,
        "matcher_type": args.matcher_type,
        "range_width": args.range_width,
        "estimator": args.estimator,
        "adjuster": args.adjuster,
        "warper_type": args.warper_type,
        "wave_correct_kind": args.wave_correct_kind,
        "compensator": args.compensator,
        "confidence_threshold": args.confidence_threshold,
        "medium_megapix": args.medium_megapix,
        "low_megapix": args.low_megapix,
        "final_megapix": args.final_megapix,
        "crop": args.crop,
        "apap_grid_parts": args.apap_grid_parts,
        "apap_grid_size": args.apap_grid_size,
        "apap_sigma": args.apap_sigma,
        "apap_min_matches": args.apap_min_matches,
        "apap_blend_width": args.apap_blend_width,
    }
    payload: Dict[str, Any] = {
        "image_paths": args.image_paths or None,
        "input_folder": args.input_folder,
        "output_path": args.output_path,
        "overwrite": args.overwrite,
        "limit": args.limit,
        "show_progress": args.show_progress,
        "settings": {key: value for key, value in settings.items() if value is not None},
    }
    return {key: value for key, value in payload.items() if value is not None}


def call_stitch(
    base_url: str,
    payload: Dict[str, Any],
    verify_ssl: bool,
    timeout: int,
) -> Response:
    url = f"{base_url.rstrip('/')}/headless_stitching_image"
    session = requests.Session()
    session.trust_env = False
    resp = session.post(url, json=payload, verify=verify_ssl, timeout=timeout)
    resp.raise_for_status()
    return resp


def call_progress(
    base_url: str,
    output_path: str,
    verify_ssl: bool,
    timeout: int,
) -> Response:
    url = f"{base_url.rstrip('/')}/progress"
    session = requests.Session()
    session.trust_env = False
    resp = session.get(url, params={"output_path": output_path}, verify=verify_ssl, timeout=timeout)
    resp.raise_for_status()
    return resp


def request_stitch(
    *,
    base_url: str = "http://localhost:8000",
    image_paths: List[str] | None = None,
    input_folder: str | None = None,
    output_path: str,
    overwrite: bool = True,
    limit: int | None = None,
    show_progress: bool = True,
    algorithm: str | None = None,
    settings: Dict[str, Any] | None = None,
    verify_ssl: bool = True,
    timeout: int = 900,
) -> Dict[str, Any]:
    if algorithm is not None:
        settings = dict(settings or {})
        settings["algorithm"] = algorithm
    payload = {
        "image_paths": image_paths,
        "input_folder": input_folder,
        "output_path": output_path,
        "overwrite": overwrite,
        "limit": limit,
        "show_progress": show_progress,
        "settings": settings,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    resp = call_stitch(base_url=base_url, payload=payload, verify_ssl=verify_ssl, timeout=timeout)
    return resp.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Request /headless_stitching_image API.")
    parser.add_argument("image_paths", nargs="*", help="按拼接顺序传入的图像路径")
    parser.add_argument("--base-url", dest="base_url", type=str, default="http://localhost:8000")
    parser.add_argument("--input-folder", dest="input_folder", type=str, default=None)
    parser.add_argument(
        "-o",
        "--output-path",
        dest="output_path",
        type=str,
        required=True,
        help="输出路径；后缀为 .tif/.tiff 时服务会写出 COG，但不附加地理参考。",
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--algorithm",
        choices=["stitcher", "origin", "apap", "local_refine"],
        default=None,
    )
    parser.add_argument("--detector", type=str, default=None)
    parser.add_argument("--nfeatures", type=int, default=None)
    parser.add_argument("--match-conf", dest="match_conf", type=float, default=None)
    parser.add_argument("--matcher-type", dest="matcher_type", type=str, default=None)
    parser.add_argument("--range-width", dest="range_width", type=int, default=None)
    parser.add_argument("--estimator", type=str, default=None)
    parser.add_argument("--adjuster", type=str, default=None)
    parser.add_argument("--warper-type", dest="warper_type", type=str, default=None)
    parser.add_argument("--wave-correct-kind", dest="wave_correct_kind", type=str, default=None)
    parser.add_argument("--compensator", type=str, default=None)
    parser.add_argument("--confidence-threshold", dest="confidence_threshold", type=float, default=None)
    parser.add_argument("--medium-megapix", dest="medium_megapix", type=float, default=None)
    parser.add_argument("--low-megapix", dest="low_megapix", type=float, default=None)
    parser.add_argument("--final-megapix", dest="final_megapix", type=float, default=None)
    parser.add_argument("--crop", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--apap-grid-parts", dest="apap_grid_parts", type=int, default=None)
    parser.add_argument(
        "--apap-grid-size",
        dest="apap_grid_size",
        type=int,
        default=None,
        help="兼容旧参数；现在按 apap_grid_parts 的切分份数处理。",
    )
    parser.add_argument(
        "--apap-sigma",
        dest="apap_sigma",
        type=float,
        default=None,
        help="APAP 局部权重半径；0~1 表示图像高度比例，如 0.15 表示 0.15*高度；大于 1 时按像素处理。",
    )
    parser.add_argument("--apap-min-matches", dest="apap_min_matches", type=int, default=None)
    parser.add_argument("--apap-blend-width", dest="apap_blend_width", type=int, default=None)
    parser.add_argument("--insecure", action="store_true", help="关闭 SSL 验证")
    parser.add_argument("--timeout", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    resp = call_stitch(
        base_url=args.base_url,
        payload=payload,
        verify_ssl=not args.insecure,
        timeout=args.timeout,
    )
    print(resp.json())


if __name__ == "__main__":
    main()
