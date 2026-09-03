# Resilient

**简体中文** | [English](README.md)

Resilient 是一个以可复现性为首要目标、基于 [FastWAM](https://github.com/yuantianyuan01/FastWAM) 开展 LIBERO 评测和后续研究的代码库。由于 FastWAM 的 Hydra 配置和评测入口依赖仓库相对路径，上游代码直接整合在仓库根目录。

## 当前状态

- FastWAM 上游固定在提交 `7faa71108368fbb3b6885649f112af607427a2d4`。
- 基线目标为 `libero_uncond_2cam224.pt` 及其配套数据统计文件。
- 原始 checkpoint 使用 `EVALUATION.sigma_shift=5.0` 复现。
- FastWAM 核心默认行为未被修改；项目扩展放在 `src/resilient/` 和 `scripts/resilient/`。
- 已在下方硬件上验证独立 Python/CUDA 环境、全部固定资产、LIBERO EGL 无界面 reset 与单回合集成评测；完整评测尚待执行。

## 仓库结构

```text
Resilient/
├── configs/                       # 上游 Hydra 配置
├── experiments/libero/            # 上游 LIBERO 评测代码
├── src/fastwam/                   # 固定版本的 FastWAM 上游实现
├── src/resilient/                 # Resilient 扩展与适配器
├── scripts/resilient/             # 可复现的下载与验证工具
├── environment/                   # 环境规范、锁文件与说明
├── manifests/                     # 上游及外部资产固定信息
├── reproduce/fastwam_libero/      # 最小与完整复现入口
├── reports/baselines/             # 精简、可提交的复现记录
├── checkpoints/fastwam_release/   # 下载权重，不上传 Git
├── data/lerobot_v30/              # LIBERO LeRobot 3.0 数据，不上传 Git
├── third_party/LIBERO/            # 固定版本的仿真器源码，不上传 Git
├── runs/                          # 训练输出，不上传 Git
├── evaluate_results/              # 原始评测输出，不上传 Git
└── AILOG/WORKLOG.md                # 本地中文工作日志，不上传 Git
```

## 硬件基线

| 项目 | 要求/建议 | 本次验证机器 |
| --- | --- | --- |
| 系统 | Linux x86_64 | Linux 5.4，x86_64 |
| Python | CPython 3.10.20 | uv 管理的独立 CPython 3.10.20 |
| 环境工具 | uv 0.11.7 | uv 0.11.7 |
| GPU | 支持 BF16 的 NVIDIA GPU；最小评测使用 1 张 | 8 × RTX 6000 Ada，每张 48 GB |
| NVIDIA 驱动 | 兼容 CUDA 12.8 PyTorch wheel | 570.133.07 |
| 系统内存 | 待完整评测确认 | 503 GiB |
| 磁盘 | 环境、权重、Git LFS 对象、数据与输出合计至少 80 GB | 配置前可用约 695 GB |

固定版本下载 Wan 公共组件需要 Git LFS；本次验证版本为 3.6.1。

FastWAM 默认启动 8 个持久化 LIBERO worker。完整评测时可通过 `MULTIRUN.num_gpus` 或下方脚本的 `NUM_GPUS` 调整 GPU 数量。

## 环境安装

环境由以下同步维护的规范组成：

- `.python-version` 固定已验证的 Python 运行时。
- `requirements.txt` 固定全部 FastWAM 运行依赖，包括 CUDA 12.8 版 PyTorch。
- `requirements-libero.txt` 固定仅供 LIBERO 仿真使用的依赖，避免降级 FastWAM 当前依赖栈。
- `requirements-dev.txt` 增加开发和测试工具。
- `environment/pip-freeze-cu128.txt` 锁定本次验证的全部直接与传递依赖。
- `environment/environment.yml` 提供等价的可选 Conda 引导规范。

```bash
uv python install 3.10.20
uv venv --python 3.10.20 .venv
source .venv/bin/activate

uv pip install --index-strategy unsafe-best-match -r requirements.txt
uv pip install -r requirements-libero.txt
uv pip install --no-deps -e .

# 可选：安装开发工具
uv pip install --index-strategy unsafe-best-match -r requirements-dev.txt
```

锁文件完成验证后，如需精确重建，可用以下命令替换前两条依赖安装命令：

```bash
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r environment/pip-freeze-cu128.txt
```

由于 requirements 文件增加了 PyTorch CUDA wheel 索引，uv 需要 `unsafe-best-match` 参数；所有软件包版本仍为精确固定。独立运行时可以避免本机 Conda 动态库泄漏导致的 FFmpeg/torchcodec 冲突。

按已测试的精确提交安装官方 LIBERO 仿真器。禁止安装其历史完整 requirements，因为它会降级 FastWAM 当前依赖栈：

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
uv pip install --no-deps --editable third_party/LIBERO \
  --config-settings editable_mode=compat
python scripts/resilient/configure_libero.py
export LIBERO_CONFIG_PATH="$(pwd)/AILOG/libero"
```

LIBERO 的 namespace 源码布局需要兼容 editable 模式。配置工具会核对固定提交，并仅在被忽略的 `AILOG/libero/config.yaml` 中生成当前机器的仿真资产绝对路径；仓库不会提交机器路径。MuJoCo 固定为 `3.3.2`。

## 外部资产

### FastWAM 已发布 checkpoint

来源：<https://huggingface.co/yuanty/fastwam>

该模型卡目前没有声明权重许可证。可为本项目复现下载使用，但在发布者明确授权条款前不要再分发 checkpoint。

```bash
huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  --revision 8eaceeb24c3cc92ff2a9c9a9d266a4941b836705 \
  --local-dir ./checkpoints/fastwam_release
```

目标路径：

```text
checkpoints/fastwam_release/libero_uncond_2cam224.pt
checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
```

checkpoint 大小为 12,041,735,140 字节，SHA-256 为 `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`。统计文件大小为 40,939 字节，SHA-256 为 `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638`。两者均通过 `scripts/resilient/verify_assets.py` 校验。

### Wan 推理公共组件

评测还需要 Wan UMT5 文本编码器、tokenizer 和 Wan 2.2 VAE。使用 Git LFS 稀疏检出下载精确的 ModelScope Git 提交：

```bash
git lfs version
bash scripts/resilient/download_model_components.sh
```

脚本不会覆盖非空且不受其管理的目录，也会拒绝错误提交的已有检出。这些组件约占 12.8 GB。`manifests/assets.json` 记录了精确路径、大小、SHA-256、版本和 Apache-2.0 许可证。评测包装脚本保留 FastWAM 原本的 ModelScope 默认值，校验后直接使用本地文件。

### LIBERO LeRobot 3.0 数据集

训练和后续工作使用 <https://huggingface.co/datasets/yuanty/LIBERO-fastwam> 上按 CC-BY-4.0 发布的固定快照：

```bash
huggingface-cli download yuanty/LIBERO-fastwam \
  --repo-type dataset \
  --include "lerobot_v30/**" \
  --revision ee018b997c430bb12b5bf3c892d744798c5a2f91 \
  --local-dir ./data
```

目标目录：

```text
data/lerobot_v30/libero_10_no_noops_lerobot/
data/lerobot_v30/libero_goal_no_noops_lerobot/
data/lerobot_v30/libero_object_no_noops_lerobot/
data/lerobot_v30/libero_spatial_no_noops_lerobot/
```

已发布 checkpoint 的仿真评测不会读取训练数据，但后续研究会使用该数据。该快照包含 46 个文件，共 4,706,213,231 字节；按“相对路径＋内容”计算的确定性目录 SHA-256 为 `fb0532da60ac971d785f9135d0f015746a105d4a768fb3ec0ba15399dea40e7b`。计算时排除仓库的 `.gitkeep` 占位符。`manifests/assets.json` 保存这些信息，实际数据文件禁止上传 Git。

## 复现 LIBERO 基线

首先验证环境和资产：

```bash
python scripts/resilient/capture_environment.py --output AILOG/environment.json
python scripts/resilient/verify_assets.py
```

评测脚本根据仓库位置设置 `LIBERO_CONFIG_PATH`、`DIFFSYNTH_MODEL_BASE_PATH`、`DIFFSYNTH_DOWNLOAD_SOURCE=modelscope` 与 `MUJOCO_GL=egl`；这些值都可通过环境变量覆盖，受版本控制的代码中没有机器专属路径。

运行单任务、单回合的集成评测：

```bash
GPU_ID=0 bash reproduce/fastwam_libero/evaluate_minimal.sh
```

`GPU_ID` 默认为 `0`；如果没有显式设置 `CUDA_VISIBLE_DEVICES`，脚本也会用该值限制可见 GPU，确保选择的是对应物理 GPU，而不只是改变结果文件标签。

运行完整四套件评测（40 个任务 × 每任务 50 回合）：

```bash
NUM_GPUS=8 bash reproduce/fastwam_libero/evaluate_full.sh
```

原始视频、worker 日志和逐任务输出保留在 `evaluate_results/` 下并被 Git 忽略。验证完成后，只将精简指标和来源信息复制到 `reports/baselines/`。

### 已验证的最小结果

2026-09-04 使用发布 checkpoint，在 seed 42、10 个推理步、sigma shift 5.0、启用动作编译的设置下，`libero_spatial` 第 0 个任务单回合成功。rollout 阶段含首次 TorchInductor 编译共用时 114.52 秒。精简来源信息见 `reports/baselines/minimal-validation.json`。该结果仅为集成检查，不具有统计意义。

## 扩展开关与基线保护

以下规则为强制要求：

1. 新功能优先放在 `src/resilient/`，避免修改 `src/fastwam/`。
2. 如果必须修改 FastWAM 上游文件，新行为必须由明确的 Hydra/CLI 参数控制。
3. 每个 Resilient 开关必须默认关闭或保持上游原始值，保证文档中的基线命令行为不变。
4. 新开关必须在同一提交中加入下表、中英文 README、Hydra 配置和测试。
5. 无法避免的上游补丁记录到 `docs/upstream-patches.md`。

| 开关 | 默认值 | 作用范围 | 对基线的影响 |
| --- | --- | --- | --- |
| 暂无 | — | 尚未修改 FastWAM 源码 | 无 |

## 开发检查

轻量 CI 与完整 GPU 环境分离：

```bash
python -m pip install -r requirements-ci.txt
PYTHONPATH=src python -m pytest -m "not gpu and not libero"
ruff check src/resilient tests scripts/resilient
```

一次性调试脚本和输出使用后必须删除。可长期复用的 GPU/LIBERO 检查放入 `tests/integration/` 并添加明确标记。

## 可复现性约定

- 只使用仓库相对路径或用户传入路径，禁止提交机器专属的硬编码绝对路径。
- 记录 Git SHA、上游 SHA、命令、Hydra 覆盖参数、随机种子、硬件、环境、资产版本/校验值和结果。
- 依赖变化时同步更新环境文件和中英文 README。
- 数据集、checkpoint、仿真资产、缓存、视频、日志和原始实验结果不得上传 Git。
- 源码和代码注释使用英文；`README.md` 使用英文；`README.zh-CN.md` 使用简体中文；被忽略的 `AILOG/WORKLOG.md` 使用中文。

## 上游与许可证

FastWAM 来自 <https://github.com/yuantianyuan01/FastWAM>，保留 MIT License 和完整 Git 历史。来源信息见 `LICENSE` 与 `manifests/upstream.json`。
