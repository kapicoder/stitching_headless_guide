"""验证画布对容器 API 的路径和四角坐标代理行为。"""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path

from PIL import Image
import pytest

import guided_validation


@pytest.fixture(autouse=True)
def test_project(tmp_path, monkeypatch):
    """Use small generated TIFFs instead of depending on local aerial datasets."""
    templates = guided_validation.PROJECT_ROOT / "validation_templates"
    (tmp_path / "validation_templates").symlink_to(templates, target_is_directory=True)
    folder = tmp_path / "test_data" / "aerial_4000x24000"
    folder.mkdir(parents=True)
    for index in (1, 2):
        Image.new("RGB", (400, 240), "gray").save(folder / f"aerial_0{index}.tif")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(guided_validation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(guided_validation, "OUTPUT_ROOT", output)
    monkeypatch.setattr(guided_validation, "UPLOAD_ROOT", output / "guided_uploads")
    monkeypatch.chdir(tmp_path)


def _app():
    return guided_validation.create_app([
        "test_data/aerial_4000x24000/aerial_01.tif",
        "test_data/aerial_4000x24000/aerial_02.tif",
    ])


def _positions():
    return [
        {"index": 0, "corners": [[0, 0], [400, 0], [400, 240], [0, 240]]},
        {"index": 1, "corners": [[300, 0], [700, 0], [700, 240], [300, 240]]},
    ]


def test_page_and_tiff_preview():
    client = _app().test_client()
    page = client.get("/")
    assert page.status_code == 200
    assert "人工引导拼接" in page.get_data(as_text=True)
    preview = client.get("/preview/0")
    assert preview.status_code == 200
    assert preview.data.startswith(b"\x89PNG")


def test_progress_endpoint_ignores_previous_run(monkeypatch, tmp_path):
    monkeypatch.setattr(guided_validation, "OUTPUT_ROOT", tmp_path)
    progress = tmp_path / "progress.txt"
    progress.write_text("100.0,completed", encoding="utf-8")
    client = guided_validation.create_app([]).test_client()
    baseline = client.get("/api/progress?baseline=1").get_json()["version"]
    assert client.get(f"/api/progress?since={baseline}").get_json()["exists"] is False
    progress.write_text("65.00,估计曝光补偿", encoding="utf-8")
    next_version = int(baseline) + 1_000_000_000
    os.utime(progress, ns=(next_version, next_version))
    current = client.get(f"/api/progress?since={baseline}").get_json()
    assert current["exists"] is True
    assert current["percentage"] == 65.0
    assert current["status"] == "估计曝光补偿"
    assert client.get("/api/progress?since=bad").status_code == 400


def test_proxy_uses_container_paths_and_submits_four_corners(monkeypatch):
    captured = {}

    class FakeOpener:
        def open(self, upstream, timeout):
            captured["url"] = upstream.full_url
            captured["payload"] = json.loads(upstream.data)
            return io.BytesIO(json.dumps({"status": "done", "output_width": 7000,
                                          "output_height": 24000}).encode())

    monkeypatch.setattr(guided_validation, "build_opener", lambda handler: FakeOpener())
    response = _app().test_client().post("/api/stitch", json={
        "images": _positions(), "output_name": "manual_result.tif",
        "final_megapix": 2, "blender_type": "feather", "guided_ransac_threshold_px": 2.5,
        "matching_regions": [{"id":"a", "rect":[3000,0,1000,24000]}],
    })
    assert response.status_code == 200
    payload = captured["payload"]
    assert payload["matching_regions"] == [{"id":"a", "rect":[3000,0,1000,24000]}]
    assert captured["url"] == "http://127.0.0.1:8002/headless_stitching_image"
    assert payload["images"][0]["path"] == "/workspace/test_data/aerial_4000x24000/aerial_01.tif"
    assert payload["images"][1]["corners"] == _positions()[1]["corners"]
    assert payload["output_path"] == "/workspace/output/manual_result.tif"
    assert payload["settings"]["algorithm"] == "guided_affine"
    assert payload["settings"]["georef_mode"] == "none"
    assert payload["settings"]["guided_ransac_threshold_px"] == 2.5
    assert response.get_json()["preview_url"] == "/result/manual_result.tif"
    assert response.get_json()["download_url"] == "/download/manual_result.tif"


def test_proxy_rejects_moved_reference_and_output_path():
    client = _app().test_client()
    positions = _positions()
    positions[0]["corners"][0][0] = 1
    response = client.post("/api/stitch", json={"images": positions})
    assert response.status_code == 400
    assert "第一张" in response.get_json()["error"]
    response = client.post("/api/stitch", json={
        "images": _positions(), "output_name": "../elsewhere.jpg",
    })
    assert response.status_code == 400


def test_browser_can_upload_images_then_stitch(monkeypatch):
    def png(color):
        data = io.BytesIO()
        Image.new("RGB", (24, 16), color).save(data, format="PNG")
        data.seek(0)
        return data

    captured = {}

    class FakeOpener:
        def open(self, upstream, timeout):
            captured["payload"] = json.loads(upstream.data)
            return io.BytesIO(json.dumps({"status": "done", "output_width": 42,
                                          "output_height": 16}).encode())

    monkeypatch.setattr(guided_validation, "build_opener", lambda handler: FakeOpener())
    with tempfile.TemporaryDirectory(dir=guided_validation.OUTPUT_ROOT) as folder:
        monkeypatch.setattr(guided_validation, "UPLOAD_ROOT", Path(folder))
        client = guided_validation.create_app([]).test_client()
        assert client.get("/").status_code == 200
        response = client.post("/api/images", data={"images": [
            (png("red"), "first.png"), (png("blue"), "second.png"),
        ]}, content_type="multipart/form-data")
        assert response.status_code == 200, response.get_json()
        images = response.get_json()["images"]
        assert [item["name"] for item in images] == ["first.png", "second.png"]
        assert client.get(images[0]["preview_url"]).status_code == 200
        result = client.post("/api/stitch", json={"images": [
            {"index": 0, "corners": [[0, 0], [24, 0], [24, 16], [0, 16]]},
            {"index": 1, "corners": [[18, 0], [42, 0], [42, 16], [18, 16]]},
        ]})
        assert result.status_code == 200, result.get_json()
        assert captured["payload"]["images"][0]["path"].startswith("/workspace/output/")


def test_upload_rejects_bad_file_without_replacing_current_images(monkeypatch):
    with tempfile.TemporaryDirectory(dir=guided_validation.OUTPUT_ROOT) as folder:
        monkeypatch.setattr(guided_validation, "UPLOAD_ROOT", Path(folder))
        client = _app().test_client()
        response = client.post("/api/images", data={"images": [
            (io.BytesIO(b"not an image"), "broken.png"),
            (io.BytesIO(b"not an image"), "second.png"),
        ]}, content_type="multipart/form-data")
        assert response.status_code == 400
        assert not list(Path(folder).iterdir())
        assert client.get("/preview/0").status_code == 200
