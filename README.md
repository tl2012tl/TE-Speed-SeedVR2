# TE-Speed-SeedVR2 1.0

[English](#english) | [中文](#中文)

## English

SeedVR2 image and video super-resolution nodes for ComfyUI, accelerating SeedVR2 upscaling end to end with a TE frame-aware cache, a streaming VAE and automatic chunking.

TE-Speed-SeedVR2 builds on ComfyUI's built-in SeedVR2 implementation (models from [SeedVR2](https://github.com/ByteDance-Seed/SeedVR)), providing a single-node workflow, a TE frame-aware DiT cache, SageAttention, a streaming VAE and automatic spatial tiling for end-to-end acceleration of SeedVR2 upscaling inside ComfyUI.

### 1.0

- Supports SeedVR2 3B / 7B, compatible with Comfy-Org `int8_convrot` and the official FP8 EMA safetensors.
- Exclusive TE frame-aware DiT cache.
- SageAttention variable-length attention acceleration, applied to this node's DiT only.
- Streaming VAE: no VRAM accumulation.
- Supports both images and video (ComfyUI `IMAGE` frame batches).

### Requirements

- Windows x64
- ComfyUI
- ComfyUI running on a Python 3.13 portable environment
- NVIDIA GPU
- Optional: SageAttention; needed for `attention_backend=sage`, use `comfy_auto` when it is not installed

You can download the ComfyUI TE all-in-one package v260619 torch2.12 cuda132 python3.13
Link: https://pan.quark.cn/s/63ed157ef549

### Model directory

Models go in:

```text
ComfyUI/models/SEEDVR2/
```

Model download: https://pan.quark.cn/s/0af979e0f8ea

### Nodes

#### TE-Speed-SeedVR2 Loader

Model loading node: selects the native SeedVR2 DiT and VAE.

- `dit_name`: the DiT checkpoint under `ComfyUI/models/SEEDVR2/` (incompatible files are hidden automatically).
- `vae_name`: the SeedVR2 VAE in the same directory.

Outputs `model` and `vae`.

#### TE-Speed-SeedVR2

One-node restore and upscale node: resize, encode, conditioning, chunking, sampling, decode and colour correction all happen inside it.

- `model` / `vae`: connect the Loader outputs.
- `images`: ComfyUI `IMAGE` frame batch; the video layout is `(frames, height, width, channels)`.
- `max_long_edge`: target long edge, default 1920; scales by the input aspect ratio, no integer scale factor is exposed.
- `seed`: random seed.
- `steps`: sampling steps, default 1 (SeedVR2's native default is `euler` + `simple`).
- `processing_control_value`: TE frame-cache threshold, default 0.10; `0` disables the cache.
- `mode`: `quality` / `balanced` / `speed`, controls how aggressive the cache is, default `balanced`.
- `color_correction_method`: `lab` / `wavelet` / `adain` / `none`, default `lab`; from fastest to slowest this is generally `none` → `adain` → `wavelet` → `lab`.
- `attention_backend`: `sage` or `comfy_auto`, default `sage`.

Optional parameters:

- `vae_tiling`: `auto` / `enabled` / `disabled`, default `auto`; decides spatial tiling from the available VRAM.
- `vae_tile_size`: spatial tile edge in pixels, default 512; under `auto` it only acts as an upper bound. Keep 512 on 16 GiB cards.
- `vae_tile_overlap`: tile overlap in pixels, default 64.
- `video_batch_size`: frames per DiT temporal chunk, default 0 (automatic). You can set `5`, `9`, `13`, `21`, ... manually; it is lowered automatically when VRAM is short.
- `temporal_overlap`: latent frames shared by adjacent DiT chunks, default 1 (about 4 video frames), crossfaded before decoding; `0` gives maximum throughput.
- `vram_budget_gb`: DiT working VRAM budget in GiB, default 0 (80% of physical VRAM); lower it when other tasks occupy VRAM.
- `spatial_window_batch`: spatial windows processed per attention call, default 0 (automatic); lower values reduce peak VRAM.

---

## 中文

SeedVR2 图片与视频超分辨率 ComfyUI 节点，用 TE 帧感知缓存、流式 VAE 与自动分块为 SeedVR2 放大带来全链路加速。

TE-Speed-SeedVR2 以 ComfyUI 内置的 SeedVR2 实现为基础（模型来自 [SeedVR2](https://github.com/ByteDance-Seed/SeedVR)），提供单节点全流程、TE 帧感知 DiT 缓存、SageAttention 注意力、流式 VAE 与自动空间分块，为 ComfyUI 内的 SeedVR2 放大带来全链路加速。

### 1.0

- 支持 SeedVR2 3B / 7B，兼容 Comfy-Org `int8_convrot` 与官方 FP8 EMA safetensors。
- 独家 TE 帧感知 DiT 缓存。
- SageAttention 变长注意力加速，仅作用于本节点的 DiT。
- 流式 VAE ：不堆积显存。
- 支持图片与视频（ComfyUI `IMAGE` 帧批次）。

### 环境要求

- Windows x64
- ComfyUI
- ComfyUI 使用 Python 3.13 的便携环境
- NVIDIA GPU
- 可选：SageAttention；`attention_backend=sage` 时需要，未安装请改用 `comfy_auto`

可下载 ComfyUI TE整合包 v260619 torch2.12. cuda132 python3.13
链接：https://pan.quark.cn/s/63ed157ef549

### 模型目录

模型放在：

```text
ComfyUI/models/SEEDVR2/
```

模型下载：https://pan.quark.cn/s/0af979e0f8ea

### 节点

#### TE-Speed-SeedVR2 Loader

模型加载节点，选择原生 SeedVR2 DiT 与 VAE。

- `dit_name`：`ComfyUI/models/SEEDVR2/` 下的 DiT 检查点（自动隐藏不兼容的文件）。
- `vae_name`：同目录下的 SeedVR2 VAE。

输出 `model` 与 `vae`。

#### TE-Speed-SeedVR2

一键修复与放大节点，内部完成缩放、编码、条件、分块、采样、解码与色彩校正。

- `model` / `vae`：连接 Loader 输出。
- `images`：ComfyUI `IMAGE` 帧批次，视频布局 `(frames, height, width, channels)`。
- `max_long_edge`：长边目标尺寸，默认 1920；按输入宽高比缩放，不暴露整数倍率。
- `seed`：随机种子。
- `steps`：采样步数，默认 1（SeedVR2 原生默认 `euler` + `simple`）。
- `processing_control_value`：TE 帧缓存阈值，默认 0.10；`0` 关闭缓存。
- `mode`：`quality` / `balanced` / `speed`，控制缓存的激进程度，默认 `balanced`。
- `color_correction_method`：`lab` / `wavelet` / `adain` / `none`，默认 `lab`；速度由快到慢一般为 `none` → `adain` → `wavelet` → `lab`。
- `attention_backend`：`sage` 或 `comfy_auto`，默认 `sage`。

可选参数：

- `vae_tiling`：`auto` / `enabled` / `disabled`，默认 `auto`，按可用显存自动决定空间分块。
- `vae_tile_size`：空间分块边长（像素），默认 512；`auto` 下仅作上限。16 GiB 卡建议保持 512。
- `vae_tile_overlap`：分块重叠像素，默认 64。
- `video_batch_size`：DiT 时间分块帧数，默认 0（自动）。可手动填 `5`、`9`、`13`、`21` 等，显存不足会自动下调。
- `temporal_overlap`：相邻 DiT 块共享的 latent 帧数，默认 1（约 4 视频帧），解码前交叉淡化；`0` 为最大吞吐。
- `vram_budget_gb`：DiT 工作显存预算（GiB），默认 0（用物理显存的 80%）；有其他任务占显存时可调小。
- `spatial_window_batch`：每次注意力处理的空间窗口数，默认 0（自动）；调小可降低显存峰值。
