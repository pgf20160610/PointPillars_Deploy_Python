# PointPillars_deploy

面向 KITTI 3D 目标检测的 PointPillars 部署工程。仓库围绕 **zhulf0804/PointPillars `epoch_160.pth` checkpoint**，提供从 PyTorch 权重导出拆分 ONNX、ONNXRuntime/MNN 推理、结果诊断、BEV/相机图可视化到输出对比的一套脚本化流程。

当前工程的部署拆分方式：

- **PFN**：`pillar_features + pillar_mask -> pillar_embed`
- **Backbone/Neck/Head**：`bev_feature -> cls_preds, box_preds, dir_cls_preds`
- voxelization、scatter、anchor decode、NMS、可视化等逻辑保留在 Python/C++ 侧，便于部署端对齐和调试。

---

## 目录结构

```text
configs/                  # KITTI/运行时配置
data/                     # 示例 KITTI 输入：点云、图像、标定、标签
models/                   # checkpoint、ONNX、MNN 与 manifest
models/checkpoints/       # PyTorch checkpoint
outputs_demo/             # 示例推理与可视化输出
pointpillars/             # PointPillars 模型、ops、IO/后处理/可视化工具
ThirdParty/               # 第三方运行库，如 onnxruntime C/C++ 包
tools/                    # 模型准备、推理、验证、可视化、对比脚本
```

关键默认文件：

| 路径 | 说明 |
| --- | --- |
| `configs/pointpillars_kitti.yaml` | KITTI PointPillars 主配置，包含点云范围、voxel、anchors、NMS、可视化参数 |
| `configs/runtime.yaml` | 部署运行时配置示例 |
| `models/model_manifest.json` | 模型资产下载、导出、验证、简化、MNN 转换的统一 manifest |
| `data/sample.bin` | KITTI velodyne 点云示例 |
| `data/sample.png` | KITTI image_2 图像示例 |
| `data/sample_calib.txt` | KITTI calib 标定示例 |
| `data/sample_label.txt` | KITTI label_2 标签示例 |

---

## 环境依赖

建议使用 Python 3.9+。核心脚本按需依赖以下包：

```bash
python3 -m pip install numpy opencv-python pyyaml onnx onnxruntime onnxsim torch
```

说明：

- `torch`：真实 checkpoint 导出 ONNX、PyTorch 参考推理需要。
- `onnxruntime`：ONNX 推理和 `tools/verify_onnx.py` 需要。
- `onnxsim`：`tools/simplify_onnx.sh` 需要。
- `opencv-python`：图像/BEV 可视化需要。
- MNN 推理/转换需要额外安装 MNN Python 包或准备 `MNNConvert` 可执行文件。

仓库已包含 ONNXRuntime C/C++ 包：`ThirdParty/onnxruntime-linux-x64-1.16.3/`。

---

## 快速开始

### 1. 准备模型

默认 manifest 使用 checkpoint-first 流程：下载 checkpoint、导出拆分 ONNX、验证、简化、转换 MNN。

```bash
tools/prepare_models.sh
```

如果只需要生成/验证 ONNX，不转换 MNN：

```bash
tools/prepare_models.sh --skip-mnn
```

如果只想跑工具链冒烟测试，不依赖真实 checkpoint：

```bash
tools/prepare_models.sh --dummy-export --skip-mnn
```

完整流水线等价于：

```bash
python tools/download_models.py --manifest models/model_manifest.json
python tools/export_pointpillars_onnx.py --manifest models/model_manifest.json
python tools/verify_onnx.py --manifest models/model_manifest.json
tools/simplify_onnx.sh
tools/convert_mnn.sh
```

### 2. ONNX 推理并保存结果

```bash
python tools/infer_pointpillars_onnx.py \
  --pc-path data/sample.bin \
  --calib-path data/sample_calib.txt \
  --img-path data/sample.png \
  --gt-path data/sample_label.txt \
  --pfn models/pfn_sim.onnx \
  --backbone models/backbone_head_sim.onnx \
  --backend onnx \
  --save-dir outputs/onnx \
  --visualize
```

常见输出：

```text
outputs/onnx/detections_runtime.json
outputs/onnx/vis_bev.png
outputs/onnx/vis_image.png
```

### 3. PyTorch 参考推理

用于和部署端/ONNX 端进行行为对齐：

```bash
python tools/infer_pointpillars_pytorch.py \
  --ckpt models/checkpoints/epoch_160.pth \
  --pc-path data/sample.bin \
  --calib-path data/sample_calib.txt \
  --img-path data/sample.png \
  --gt-path data/sample_label.txt \
  --device cpu \
  --save-dir outputs/pytorch_ref \
  --save-bev \
  --save-image \
  --show-gt
```

---

## 模型 manifest

`models/model_manifest.json` 是模型准备流水线的中心配置。当前默认内容包含：

- `assets`：checkpoint 下载地址与输出路径；默认输出到 `models/checkpoints/epoch_160.pth`。
- `export`：导出参数，默认通过 `tools/export_pointpillars_split.py` 从 checkpoint 导出：
  - `models/pfn.onnx`
  - `models/backbone_head.onnx`
- `verify`：ONNXRuntime 验证参数。
- `simplify`：ONNX 简化输入形状与输出路径：
  - `models/pfn_sim.onnx`
  - `models/backbone_head_sim.onnx`
- `mnn`：MNN 输出路径：
  - `models/pfn.mnn`
  - `models/backbone_head.mnn`

默认导出命令模板：

```bash
python tools/export_pointpillars_split.py \
  --ckpt {checkpoint} \
  --config {config} \
  --pfn-out {pfn_output} \
  --backbone-out {backbone_output} \
  --opset {opset}
```

---

## tools 脚本说明

### 模型准备与转换

| 脚本 | 用途 | 示例 |
| --- | --- | --- |
| `tools/prepare_models.sh` | 一键执行下载、导出、验证、简化、MNN 转换 | `tools/prepare_models.sh --skip-mnn` |
| `tools/download_models.py` | 根据 manifest 下载模型资产 | `python tools/download_models.py --manifest models/model_manifest.json` |
| `tools/export_pointpillars_onnx.py` | 通用导出编排，可调用外部导出命令或 dummy 导出 | `python tools/export_pointpillars_onnx.py --manifest models/model_manifest.json` |
| `tools/export_pointpillars_split.py` | 自包含导出 zhulf0804 checkpoint 到拆分 ONNX | `python tools/export_pointpillars_split.py --ckpt models/checkpoints/epoch_160.pth --config configs/pointpillars_kitti.yaml --pfn-out models/pfn.onnx --backbone-out models/backbone_head.onnx` |
| `tools/verify_onnx.py` | 使用 ONNXRuntime 检查 PFN/Backbone ONNX 是否可运行 | `python tools/verify_onnx.py --manifest models/model_manifest.json` |
| `tools/simplify_onnx.sh` | 使用 onnxsim 固定输入形状并简化 ONNX | `tools/simplify_onnx.sh` |
| `tools/convert_mnn.sh` | 使用 MNNConvert 将 ONNX 转为 MNN | `tools/convert_mnn.sh --mnnconvert /path/to/MNNConvert` |
| `tools/sha256_file.py` | 计算模型文件 sha256 | `python tools/sha256_file.py models/pfn.onnx models/backbone_head.onnx` |

`prepare_models.sh` 常用选项：

```text
--skip-download     跳过下载
--skip-export       跳过 ONNX 导出
--skip-verify       跳过 ONNXRuntime 验证
--skip-simplify     跳过 ONNX 简化
--skip-mnn          跳过 MNN 转换
--dummy-export      使用内置 dummy 模型做工具链冒烟测试
--dry-run           仅打印命令
--mnnconvert PATH   指定 MNNConvert
--fp16              MNN 转换启用 fp16
--static            MNN 转换保存静态模型
```

### 推理、诊断与可视化

| 脚本 | 用途 | 示例 |
| --- | --- | --- |
| `tools/infer_pointpillars_onnx.py` | 使用拆分 ONNX/MNN 执行 PointPillars 推理 | 见“快速开始” |
| `tools/infer_pointpillars_pytorch.py` | PyTorch checkpoint 参考推理与可视化 | 见“快速开始” |
| `tools/visualize_detection.py` | 根据检测 JSON 重新绘制 BEV/相机图 | `python tools/visualize_detection.py --detections outputs/onnx/detections_runtime.json --bev-reference-style` |
| `tools/replot_bev_ref.py` | 按 zhulf0804/PointPillars 兼容 yaw 约定重绘 BEV | `python tools/replot_bev_ref.py --pc-path data/sample.bin --detections outputs/onnx/detections_runtime.json` |
| `tools/diagnose_pointpillars_output.py` | 诊断解码后的检测 JSON | `python tools/diagnose_pointpillars_output.py outputs/onnx/detections_runtime.json` |
| `tools/compare_outputs.py` | 比较两个检测 JSON 的 box 数值是否在容差内一致 | `python tools/compare_outputs.py --ref outputs_demo/onnx/detections_runtime.json --test outputs/onnx/detections_runtime.json --tolerance 1e-3` |
| `tools/generate_test_data.py` | 从 mini KITTI 数据集中抽取单帧到 `data/` | `python tools/generate_test_data.py --dataset-root /path/to/mini-kitti-3d --frame 000000 --force` |

ONNX/MNN 推理常用参数：

```text
--pc-path              KITTI velodyne .bin 点云，必填
--calib-path           KITTI calib 文件
--img-path             KITTI image_2 图像
--gt-path              KITTI label_2 标签
--pfn                  PFN ONNX/MNN 模型，必填
--backbone             Backbone/Head ONNX/MNN 模型，必填
--backend {onnx,mnn}   推理后端，默认 onnx
--runtime-device       运行设备，默认 cpu
--num-threads          线程数，默认 4
--score-thr            分数阈值，默认 0.1
--nms-thr              NMS 阈值，默认 0.01
--nms-pre              NMS 前保留数量，默认 100
--max-num              最大输出数量，默认 50
--save-dir             输出目录
--visualize            同时保存 BEV 和相机图可视化
```

---

## 配置说明

`configs/pointpillars_kitti.yaml` 使用扁平 YAML 风格，主要参数：

```yaml
point_cloud_range: [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
voxel_size: [0.16, 0.16, 4.0]
grid_size: [432, 496, 1]
max_points_per_pillar: 32
max_pillars: 12000
num_pillar_features: 10
pfn_out_channels: 64
class_names: [Pedestrian, Cyclist, Car]
score_threshold: 0.1
nms_threshold: 0.01
nms_pre: 100
max_detections: 100
```

部署/可视化相关默认项：

```yaml
pfn_model: models/pfn_sim.onnx
backbone_model: models/backbone_head_sim.onnx
visualize: false
image_path: data/sample.png
calib_path: data/sample_calib.txt
label_path: data/sample_label.txt
bev_output: outputs/pc_pred_cpp.png
image_output: outputs/img_3dbbox_cpp.png
bev_reference_style: true
```

---

## 输出 JSON 格式

推理输出为 detection list，每个元素通常包含：

```json
{
  "x": 10.30704402923584,
  "y": 0.007722944021224976,
  "z": -1.7239704132080078,
  "w": 1.6548254489898682,
  "l": 3.5103259086608887,
  "h": 1.5759927034378052,
  "yaw": -1.6265372037887573,
  "score": 0.9654676914215088,
  "cls_id": 2,
  "class_name": "Car"
}
```

其中 `x/y/z/w/l/h/yaw` 为 LiDAR 坐标系下 3D box 参数，`cls_id` 与类别映射：

```text
0 -> Pedestrian
1 -> Cyclist
2 -> Car
```

---

## 结果对齐建议

1. 先运行 PyTorch 参考推理，保存 `outputs/pytorch_ref/detections_pytorch_refactor.json`。
2. 再运行 ONNX 推理，保存 `outputs/onnx/detections_runtime.json`。
3. 用 `tools/compare_outputs.py` 或诊断脚本检查差异。

示例：

```bash
python tools/compare_outputs.py \
  --ref outputs_demo/onnx/detections_runtime.json \
  --test outputs/onnx/detections_runtime.json \
  --tolerance 1e-3
```

若相机图出现异常长射线或投影异常，可优先检查：

- `--calib-path` 是否匹配当前点云/图像；
- 是否关闭了 `--no-image-range-filter`；
- `--min-camera-depth` 与 `--max-proj-span-ratio` 是否过宽；
- box 坐标系和 yaw 约定是否与 zhulf0804/PointPillars 源实现一致。

---

## 常见问题

### 1. `onnxsim is not installed`

安装 onnxsim：

```bash
python3 -m pip install onnxsim
```

或在一键流程中跳过简化：

```bash
tools/prepare_models.sh --skip-simplify
```

### 2. `MNNConvert not found`

安装/编译 MNN 后指定可执行文件：

```bash
tools/convert_mnn.sh --mnnconvert /path/to/MNNConvert
```

或一键流程跳过 MNN：

```bash
tools/prepare_models.sh --skip-mnn
```

### 3. 只验证工具链，不想下载真实 checkpoint

```bash
tools/prepare_models.sh --dummy-export --skip-mnn
```

### 4. 真实导出失败

确认：

- `models/checkpoints/epoch_160.pth` 存在；
- `torch` 和 `onnx` 已安装；
- `models/model_manifest.json` 中 `export.external_export_command` 指向正确导出脚本；
- 当前默认 `tools/export_pointpillars_split.py` 针对 zhulf0804/PointPillars `epoch_160.pth` 的模块命名。

可以先 dry-run：

```bash
python tools/export_pointpillars_onnx.py --manifest models/model_manifest.json --dry-run
```

---

## 参考命令合集

```bash
# 下载 checkpoint
python tools/download_models.py --manifest models/model_manifest.json

# 从 checkpoint 导出拆分 ONNX
python tools/export_pointpillars_split.py \
  --ckpt models/checkpoints/epoch_160.pth \
  --config configs/pointpillars_kitti.yaml \
  --pfn-out models/pfn.onnx \
  --backbone-out models/backbone_head.onnx \
  --opset 17

# 验证 ONNX
python tools/verify_onnx.py --manifest models/model_manifest.json

# 简化 ONNX
tools/simplify_onnx.sh

# 转 MNN
tools/convert_mnn.sh --mnnconvert /path/to/MNNConvert

# ONNX 推理 + 可视化
python tools/infer_pointpillars_onnx.py \
  --pc-path data/sample.bin \
  --calib-path data/sample_calib.txt \
  --img-path data/sample.png \
  --gt-path data/sample_label.txt \
  --pfn models/pfn_sim.onnx \
  --backbone models/backbone_head_sim.onnx \
  --backend onnx \
  --save-dir outputs/onnx \
  --visualize

# 重新可视化检测 JSON
python tools/visualize_detection.py \
  --points data/sample.bin \
  --detections outputs/onnx/detections_runtime.json \
  --image data/sample.png \
  --calib data/sample_calib.txt \
  --label data/sample_label.txt \
  --bev-output outputs/onnx/vis_bev_replot.png \
  --image-output outputs/onnx/vis_image_replot.png \
  --bev-reference-style
```

---

## 备注

- 本 README 根据当前 `tools/` 脚本、`configs/` 配置、`models/model_manifest.json` 与 `outputs_demo/` 示例自动整理生成。
- 若后续修改脚本参数或 manifest，请同步更新 README 中的命令示例。

参考工程[zhulf0804/PointPillars](https://github.com/zhulf0804/PointPillars)
