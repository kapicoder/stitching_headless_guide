# 浏览器验证工具

`guided_validation.py` 是 `guided_affine` 人工四角拼接模式的浏览器验证工具。它允许用户在同一个画布中读取多张图像，通过拖动、旋转和缩放进行粗略重合，然后把各图四角坐标提交给拼接 API。

网页服务本身不执行拼接，而是完成以下工作：

1. 读取或上传 JPG、PNG、TIFF/COG 图像；
2. 生成适合浏览器显示的低分辨率预览；
3. 在 Canvas 中调整图像位置、旋转和缩放；
4. 将原图尺度的四角坐标提交给 `guided_affine` API；
5. 轮询并显示拼接进度；
6. 显示、缩放和下载拼接结果。

先在宿主机启动拼接 API，并确认 `8002` 端口健康检查通过：

```bash
cd /home/stitching/stitching_headless_guide
HOST_PORT=8002 docker compose up -d stitching-headless-api
curl --noproxy '*' --fail http://127.0.0.1:8002/health
```

返回 `"status":"ok"` 后，再执行下面的网页启动命令。网页容器与 API 是两个独立服务，
只启动网页或给网页容器额外添加 `-p 8000:8000` 都不会启动拼接 API。

在宿主机执行：

```bash
docker run --rm \
  --name guided-validation-web \
  -p 8766:8766 \
  --add-host host.docker.internal:host-gateway \
  --user "$(id -u):$(id -g)" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /home/stitching/stitching_headless_guide:/workspace:ro \
  -v /home/stitching/stitching_headless_guide/output:/workspace/output \
  -w /workspace \
  ultralytics/ultralytics:latest \
  python guided_validation.py \
    --api-url http://host.docker.internal:8002 \
    --host 0.0.0.0 \
    --port 8766
```

然后访问：

```text
http://服务器IP:8766
```

在本机浏览器中访问时可以使用：

```text
http://127.0.0.1:8766
```

`host.docker.internal` 让网页容器访问宿主机发布在 `8002` 端口的拼接 API；`-p 8766:8766` 将网页端口发布到宿主机。

项目目录使用只读挂载，只有 `output` 目录可写。上传图像、进度文件和拼接结果均保存在 `output` 中。`--user` 用于避免容器生成属于 root 的文件。

## 后台运行

需要网页在终端退出后继续运行时，添加 `-d`：

```bash
docker run -d \
  --name guided-validation-web \
  -p 8766:8766 \
  --add-host host.docker.internal:host-gateway \
  --user "$(id -u):$(id -g)" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /home/stitching/stitching_headless_guide:/workspace:ro \
  -v /home/stitching/stitching_headless_guide/output:/workspace/output \
  -w /workspace \
  ultralytics/ultralytics:latest \
  python guided_validation.py \
    --api-url http://host.docker.internal:8002 \
    --host 0.0.0.0 \
    --port 8766
```

查看日志：

```bash
docker logs -f guided-validation-web
```

停止网页：

```bash
docker stop guided-validation-web
```

## 预加载项目中的图像

除了在网页中上传图像，也可以在启动时提供项目内的图像路径：

```bash
docker run --rm \
  --name guided-validation-web \
  -p 8766:8766 \
  --add-host host.docker.internal:host-gateway \
  --user "$(id -u):$(id -g)" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /home/stitching/stitching_headless_guide:/workspace:ro \
  -v /home/stitching/stitching_headless_guide/output:/workspace/output \
  -w /workspace \
  ultralytics/ultralytics:latest \
  python guided_validation.py \
    data/横须贺_20180517_tile_001.tif \
    data/横须贺_20180517_tile_002.tif \
    --api-url http://host.docker.internal:8002 \
    --host 0.0.0.0 \
    --port 8766
```

也可以按文件名顺序加载整个目录：

```bash
python guided_validation.py \
  --input-folder data \
  --api-url http://host.docker.internal:8002 \
  --host 0.0.0.0 \
  --port 8766
```

使用目录方式时，应将这条命令替换到前面 `docker run` 命令的镜像名称之后。

## 网页操作流程

1. 点击读取图像，一次选择至少两张图片；也可以使用启动时预加载的图片。
2. 第一张图固定为参考图，其左上角是人工画布原点 `(0,0)`。
3. 选择其他图像并直接拖动图像调整位置。
4. 使用图像上的控制手柄进行旋转和缩放。
5. 将预计重叠内容粗略对齐；算法会在预计重叠区域及外扩区域内进行特征匹配。
6. 设置输出文件名、最终分辨率和融合方式。
7. 点击开始拼接，等待进度弹窗完成。
8. 在结果区域缩放查看图像，或下载输出文件。

推荐将输出文件名设置为：

```text
guided_result.tif
```

TIFF/COG 可以保存人工画布与输出像素之间的 Transform 元数据：

```text
CANVAS_TO_OUTPUT_TRANSFORM
OUTPUT_TO_CANVAS_TRANSFORM
STITCHING_COORDINATE_SYSTEM
```

JPG和PNG没有使用统一GIS字段保存该矩阵，因此矩阵不会嵌入这些格式。

## 输出目录

上传的文件保存在：

```text
/home/stitching/stitching_headless_guide/output/guided_uploads/
```

拼接结果保存在：

```text
/home/stitching/stitching_headless_guide/output/
```

拼接进度保存在：

```text
/home/stitching/stitching_headless_guide/output/progress.txt
```

## 检查 TIFF/COG 元数据

```bash
gdalinfo /home/stitching/stitching_headless_guide/output/guided_result.tif
```

只查看拼接 Transform：

```bash
gdalinfo /home/stitching/stitching_headless_guide/output/guided_result.tif \
  | grep -E "TRANSFORM|STITCHING_COORDINATE"
```

检查图像金字塔时，查看各波段的 `Overviews`。正常COG还应包含：

```text
LAYOUT=COG
COMPRESSION=DEFLATE
```

## 常见问题

### 网页能打开，但无法调用 API

先检查：

```bash
docker ps -a --filter name=stitching-headless-api-guide
curl --noproxy '*' --fail http://127.0.0.1:8002/health
```

如果 API 容器已退出，回到项目目录执行 `HOST_PORT=8002 docker compose up -d stitching-headless-api`，
再用 `docker compose logs --tail=50 stitching-headless-api` 检查启动日志。

并确认网页容器的 API 地址是 `http://host.docker.internal:8002`。宿主机健康检查通过后，
还可从网页容器内验证连接（容器名按实际运行的名称替换）：

```bash
docker exec guided-validation-web python -c 'from urllib.request import build_opener, ProxyHandler; print(build_opener(ProxyHandler({})).open("http://host.docker.internal:8002/health", timeout=5).read().decode())'
```

`Network is unreachable` 表示网络连接未建立，不能据此判断拼接算法失败。
先确认 API 正在运行，再分别验证宿主机和网页容器到 API 的连接。

### 端口8766已被占用

查找占用进程：

```bash
ss -ltnp | grep :8766
```

可以将启动参数改为其他端口，例如：

```text
--port 8767
```

### 修改代码后页面仍显示旧行为

重启网页容器：

```bash
docker restart guided-validation-web
```

页面已设置禁止HTML缓存，但已有页面仍建议使用 `Ctrl+F5` 强制刷新一次。

### 修改 API 代码或配置后没有生效

Gunicorn没有启用自动重载，需要重启拼接API：

```bash
docker restart stitching-headless-api-guide
```

### 无法写入上传文件或结果

确认宿主机当前用户对以下目录拥有写权限：

```text
/home/stitching/stitching_headless_guide/output
```

并确认 `output` 使用的是可写挂载，而不是只读挂载。

## 多区域框选配准

摆放好图片后，将侧栏模式切换为“选择拼接区域”，在画布空白处拖动添加框。
拖动框内部移动，拖四角缩放，使用按钮删除选中框或清空所有框。
空格拖动画布、滚轮缩放继续可用。返回“摆放图片”后，框保持共享画布坐标，
图片移动后会重新显示关联图像对；重置位置或重新读取图片会清空所有框。
最多添加 32 个框。

框标签显示关联图像对，无关联时显示“未覆盖候选图像对”。关联依据是框与
某个候选图像对两侧搜索区域都有正面积交集。标签使用默认搜索外扩比例 0.1
乘以每对图片在画布中的较小短边，并使用最小搜索面积 100 预估，实际匹配结果以 API 诊断为准。

有关联框的图像对只使用框内特征，无关联框的图像对继续自动匹配。
同一框内两侧互相匹配，不跨框匹配。重叠框按添加顺序分配像素，后添加的框
不再使用前面框占用的有效像素。框仅影响配准，仍然融合输出完整图像。
框选匹配失败时保留手工位置，不扩大到框外寻找特征。

### API 请求与配置

`/api/stitch` 代理和 `/stitch`、`/headless_stitching_image` 主接口都接受顶层
`matching_regions`。省略或传空列表维持原流程；非空列表仅支持 `guided_affine`。
以下为主接口完整示例，路径需替换为服务进程实际可访问的路径：

```json
{
  "images": [
    {
      "path": "/workspace/test_data/test_data_4000x24000/stitch_test_01.jpg",
      "corners": [[0,0],[4000,0],[4000,24000],[0,24000]]
    },
    {
      "path": "/workspace/test_data/test_data_4000x24000/stitch_test_02.jpg",
      "corners": [[3000,0],[7000,0],[7000,24000],[3000,24000]]
    }
  ],
  "matching_regions": [
    {"id": "region_1", "rect": [3000,0,1000,10000]},
    {"id": "region_2", "rect": [3000,12000,1000,12000]}
  ],
  "output_path": "/workspace/output/regions_result.jpg",
  "settings": {
    "algorithm": "guided_affine",
    "georef_mode": "none",
    "medium_megapix": 2,
    "nfeatures": 4000
  }
}
```

框使用初始共享画布的 `[x,y,width,height]`，允许负坐标和部分越界，服务端裁剪。
ID 必须唯一且非空，坐标必须为有限数值，宽高必须大于零，不接受布尔坐标。
非法输入返回 HTTP 400。

特征数量统一使用 `config.py` 中的 `nfeatures`，默认 4000，不再按区域面积动态分配。
每个框每侧的非空有效 mask 都使用这一预算；空 mask 预算为零。
多个框的总特征数可以超过 `nfeatures`，不再设置图像对的额外总预算上限。
SIFT/ORB 使用该预算检测，框选结果按响应值筛选到预算内；
AKAZE/BRISK 在框选结果中同样按响应值截取。无关联框的图像对继续使用 `nfeatures`。

预算不是实际特征点数量保证。保持现有缩图分辨率，不额外读取高分辨率区域。
弱纹理或缩图后太小的框可能仍匹配失败，需要扩大框、增加其他框或提高
`medium_megapix`。现有仿射模型也不能保证强视差下所有框都精确重合。

结果 `debug_info.pair_diagnostics` 包含 `region_source`、每框 ID、两侧面积、预算、
实际特征数、匹配数、内点数及失败原因。每框 `matches` 是去重前候选数，图像对
`matches` 是双向比值筛选后合并去重的数量。

默认 RANSAC 阈值为 3 个特征图像素，在 `config.py` 的
`guided_ransac_threshold_px` 中调整，也支持请求参数。RANSAC 使用完整仿射模型，
在特征图坐标中估计，不受人工画布缩放影响。仅当图片通过成功匹配链连接到第一张
参考图时才进行全局优化，其余图片保留人工位置；不会自动扩大搜索范围。

页面依据 `registration_status` 显示“完整配准”“部分配准”或“仅按人工布局合成”，
并通过 `image_diagnostics` 列出每张图片的状态和保留人工位置的原因。
请求成功只表示结果文件已生成，不代表自动配准成功。

直接调用同样支持：

```python
stitch_images(paths, settings, guided_images=images, matching_regions=regions)
GuidedAffineStitcher(matching_regions=regions).register(images)
```

### 历史验证记录（2026-10-26，移除动态预算之前）

```bash
PYTHONPATH=.:tests pytest -q tests
node tests/test_guided_canvas.js
PYTHONPATH=. python tests/benchmark_matching_regions.py
```

Python 测试 53 项通过，1 项因缺少 `osgeo` 跳过；Node 画布测试通过。
包含四种检测器、区域有效像素/空洞、旋转缩放、重复框去重、预算上下限、
框选/自动混合链、字段透传和框内外冲突位移测试。
冲突位移测试分别选择上下区域，得到约 210px 和 200px 的对应平移。

大图测试使用现有三张 `4000×24000` JPEG，仅测配准阶段，无额外读取适配。
`medium_megapix=2`，比例约 0.14434；本机单次结果如下，耗时不代表稳定性能结论：

| 模式 | 总耗时 | 图像对 | 特征数 A/B | 匹配数 | 内点数 | 结果 |
|---|---:|---|---:|---:|---:|---|
| 全自动 | 2.059s | 1↔2 | 597/676 | 216 | 141 | 配准成功 |
| 全自动 | 同上 | 2↔3 | 659/621 | 272 | 155 | 配准成功 |
| 两框选择 | 2.254s | 1↔2 | 409/438 | 189 | 125 | 配准成功 |
| 两框选择 | 同上 | 2↔3（自动） | 659/621 | 272 | 155 | 配准成功 |

两框选择仅覆盖 1↔2，总特征数在每侧 12000 的上限内，2↔3 保持自动流程。
当前环境缺少 GDAL Python 包，Docker 的 WSL 集成也不可用，因此没有验证生产
GDAL TIFF/COG 解码与完整大图融合链路；没有修改该链路或新增依赖。
