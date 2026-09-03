# Resilient

**简体中文** | [English](README.md)

Resilient 是一个以可复现性为首要目标、基于 [FastWAM](https://github.com/yuantianyuan01/FastWAM) 开展 LIBERO 评测和后续研究的代码库。由于 FastWAM 的 Hydra 配置和评测入口依赖仓库相对路径，上游代码直接整合在仓库根目录。

## 当前状态

- FastWAM 上游固定在提交 `7faa71108368fbb3b6885649f112af607427a2d4`。
- 基线目标为 `libero_uncond_2cam224.pt` 及其配套数据统计文件。
- 原始 checkpoint 使用 `EVALUATION.sigma_shift=5.0` 复现。
- FastWAM 核心默认行为未被修改；项目扩展放在 `src/resilient/` 和 `scripts/resilient/`。
- 正在下方记录的硬件上验证环境与资产；完整评测完成后补充复现结果。

## 仓库结构

```text
Resilient/
├── configs/                       # 上游 Hydra 配置
├── experiments/libero/            # 上游 LIBERO 评测代码
├── src/fastwam/                   # 固定版本的 FastWAM 上游实现
├── src/resilient/                 # Resilient 扩展与适配器
├── scripts/resilient/             # 可复现的下载与验证工具
├── environment/                   # Conda 规范与环境说明
├── manifests/                     # 上游及外部资产固定信息
├── reproduce/fastwam_libero/      # 最小与完整复现入口
├── reports/baselines/             # 精简、可提交的复现记录
├── checkpoints/fastwam_release/   # 下载权重，不上传 Git
├── data/lerobot_v30/              # LIBERO LeRobot 3.0 数据，不上传 Git
├── runs/                          # 训练输出，不上传 Git
├── evaluate_results/              # 原始评测输出，不上传 Git
└── AILOG/WORKLOG.md                # 本地中文工作日志，不上传 Git
```

## 硬件基线

| 项目 | 要求/建议 | 本次验证机器 |
| --- | --- | --- |
| 系统 | Linux x86_64 | Linux 5.4，x86_64 |
| Python | CPython 3.10.8 | CPython 3.10.8 |
| GPU | 支持 BF16 的 NVIDIA GPU；最小评测使用 1 张 | 8 × RTX 6000 Ada，每张 48 GB |
| NVIDIA 驱动 | 兼容 CUDA 12.8 PyTorch wheel | 570.133.07 |
| 系统内存 | 待完整评测确认 | 503 GiB |
| 磁盘 | 环境与 checkpoint 至少 30 GB；数据和结果需更多空间 | 配置前可用约 695 GB |

FastWAM 默认启动 8 个持久化 LIBERO worker。完整评测时可通过 `MULTIRUN.num_gpus` 或下方脚本的 `NUM_GPUS` 调整 GPU 数量。

## 环境安装

环境由三个同步维护的文件组成：

- `environment/environment.yml` 固定 Python 和 pip。
- `requirements.txt` 固定全部 FastWAM 运行依赖，包括 CUDA 12.8 版 PyTorch。
- `requirements-dev.txt` 增加开发和测试工具。

```bash
conda env create -f environment/environment.yml
conda activate resilient-fastwam

python -m pip install -r requirements.txt
python -m pip install --no-deps -e .

# 可选：安装开发工具
python -m pip install -r requirements-dev.txt
```

官方 LIBERO 包将按 `manifests/upstream.json` 中记录的精确 commit 单独安装，MuJoCo 固定为 `3.3.2`。完成兼容性验证后，会在这里补充最终验证过的安装命令。

## 外部资产

### FastWAM 已发布 checkpoint

来源：<https://huggingface.co/yuanty/fastwam>

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

checkpoint 大小为 12,041,735,140 字节，SHA-256 为 `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`。统计文件下载后补充校验值，并通过 `scripts/resilient/verify_assets.py` 验证。

### LIBERO LeRobot 3.0 数据集

训练和后续工作使用 <https://huggingface.co/datasets/yuanty/LIBERO-fastwam> 的固定快照：

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

已发布 checkpoint 的仿真评测不会读取训练数据，但后续研究会使用该数据。`manifests/assets.json` 记录资产版本、大小和校验值；实际文件禁止上传 Git。

## 复现 LIBERO 基线

首先验证环境和资产：

```bash
python scripts/resilient/capture_environment.py --output AILOG/environment.json
python scripts/resilient/verify_assets.py
```

运行单任务、单回合的集成评测：

```bash
bash reproduce/fastwam_libero/evaluate_minimal.sh
```

运行完整四套件评测（40 个任务 × 每任务 50 回合）：

```bash
NUM_GPUS=8 bash reproduce/fastwam_libero/evaluate_full.sh
```

原始视频、worker 日志和逐任务输出保留在 `evaluate_results/` 下并被 Git 忽略。验证完成后，只将精简指标和来源信息复制到 `reports/baselines/`。

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
