# Resilient

**简体中文** | [English](README.md)

Resilient 是一个以可复现性为首要目标、基于 [FastWAM](https://github.com/yuantianyuan01/FastWAM) 开展 LIBERO 评测和后续研究的代码库。由于 FastWAM 的 Hydra 配置和评测入口依赖仓库相对路径，上游代码直接整合在仓库根目录。

## 当前状态

- FastWAM 上游固定在提交 `7faa71108368fbb3b6885649f112af607427a2d4`。
- 基线目标为 `libero_uncond_2cam224.pt` 及其配套数据统计文件。
- 原始 checkpoint 使用 `EVALUATION.sigma_shift=5.0` 复现。
- FastWAM 默认行为未改变；核心代码只增加一个已登记、默认关闭的去噪轨迹返回开关，项目具体
  实现位于 `src/resilient/`。
- 已在下方硬件上验证独立 Python/CUDA 环境、全部固定资产、LIBERO EGL 无界面 reset、单回合集成评测与完整 2,000 回合基准。
- 带 severity 的 Fault 层和 OPSD-Flow 已完成 CPU、配置与仿真器烟测，尚未验证正式多卡 OPSD
  训练。

## 仓库结构

```text
Resilient/
├── configs/                       # Hydra 配置，包含 Fault/adapter/OPSD 参数
├── experiments/libero/            # LIBERO 评测及默认关闭的 Fault hook
├── src/fastwam/                   # 固定 FastWAM 及已登记的默认关闭补丁
├── src/resilient/                 # Resilient 扩展与适配器
├── scripts/resilient/             # 可复现的评测、下载与验证工具
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
| GPU | 支持 BF16 的 NVIDIA GPU；建议每个 worker 至少 32 GB；1 张可运行，8 张复现本次并行评测 | 8 × RTX 6000 Ada，每张 48 GB；每个 worker 实测 24,728–25,593 MiB |
| NVIDIA 驱动 | 兼容 CUDA 12.8 PyTorch wheel | 570.133.07 |
| 系统内存 | 8 个 worker 建议至少 128 GiB；内存较小时减少 worker 数 | 503 GiB；运行中一次快照约使用 90 GiB |
| 磁盘 | 环境、权重、Git LFS 对象、数据与输出合计至少 80 GB | 配置前可用约 695 GB |

固定版本下载 Wan 公共组件需要 Git LFS；本次验证版本为 3.6.1。

FastWAM 默认启动 8 个持久化 LIBERO worker。完整评测时可通过 `MULTIRUN.num_gpus` 或下方脚本的 `NUM_GPUS` 调整 GPU 数量。本次 8-GPU 验证运行包含 worker 启动和模型加载约用时 46 分钟；减少 GPU 可降低内存需求，但会增加总耗时。

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

### 已验证的完整结果

2026-09-04 完成全部 40 个任务、2,000 个 episode，失败队列为空。独立校验确认：40 个结果文件互不重复、每任务均为 50 回合、成功/失败 episode 划分完整，且 2,000 个回放视频均非空。本次复现总成功率为 **97.15%**（1,943/2,000），比 [Fast-WAM arXiv v2 表 2](https://arxiv.org/html/2603.16666) 报告的 97.6% 低 0.45 个百分点。

| 套件 | 本次复现 | 成功数 | 论文结果 | 差异 |
| --- | ---: | ---: | ---: | ---: |
| LIBERO-Spatial | 97.0% | 485/500 | 98.2% | -1.2 个百分点 |
| LIBERO-Object | 99.8% | 499/500 | 100.0% | -0.2 个百分点 |
| LIBERO-Goal | 96.6% | 483/500 | 97.0% | -0.4 个百分点 |
| LIBERO-Long（`libero_10`） | 95.2% | 476/500 | 95.2% | 0.0 个百分点 |
| **平均** | **97.15%** | **1,943/2,000** | **97.6%** | **-0.45 个百分点** |

本次使用 seed 42、10 个推理步、sigma shift 5.0、CFG 1.0、动作编译、MuJoCo 3.3.2、EGL 渲染和 8 个持久 worker。并行运行墙钟时间约 46 分钟；各任务耗时之和为 19,253.27 秒。机器可读记录和完整来源信息见 `reports/baselines/fastwam-libero-full.json`。

### 已验证的最小结果

2026-09-04 使用发布 checkpoint，在 seed 42、10 个推理步、sigma shift 5.0、启用动作编译的设置下，`libero_spatial` 第 0 个任务单回合成功。rollout 阶段含首次 TorchInductor 编译共用时 114.52 秒。精简来源信息见 `reports/baselines/minimal-validation.json`。该结果仅为集成检查，不具有统计意义。

## 可复用 Fault 管线与评测

Fault 定义是 `configs/fault/` 下与模型解耦的 YAML。`FaultPipeline` 按声明顺序提供 reset、
observation、action、step 前后、临时挂起、断点状态和 detach 接口。后续图像污渍/遮挡、执行器
限制、关节运动退化和动力学故障可通过插件加入，不需要继续在 Fast-WAM 中增加故障分支。每个
Fault 都有稳定的 `family` 和独立的物理 `severity`（名称、数值、单位及可选等级），因此绕轴
10/20/30 度属于同一 Fault family 的三个 severity。

已提交的相机示例为 `configs/fault/visual/wrist_camera_local_z.yaml`。用 4 卡评测默认的手腕相机
绕局部 +Z 轴 30 度：

```bash
python scripts/resilient/evaluate_fault.py \
  --fault-config configs/fault/visual/wrist_camera_local_z.yaml \
  --gpus 0,1,2,3 \
  --checkpoint checkpoints/fastwam_release/libero_uncond_2cam224.pt
```

`--severity 10/20/30` 可在不改变 Fault family 的情况下覆盖单个 Fault 的物理程度。八卡传入
8 个 GPU ID。默认输出根目录为被忽略的 `evaluate_results/faults/`；规范子目录名包含 family、
目标、severity、单位与 checkpoint，例如
`visual-camera_pose-robot0_eye_in_hand-rotation_angle-p30p0degree__model-libero_uncond_2cam224/`。
其中包含逐任务结果/视频、`summary.json`、`fault_summary.md`、`fault_manifest.json` 和
`fault_comparison.png`。2×2 对比图从同一个仿真状态取得原始/故障第三人称与手腕图像，并在
故障图左上角标注完整参数。增加 `--create-only` 可在不加载 Fast-WAM、不运行 episode 的情况
下验证对比图渲染和多 worker 任务配置。

评测 OPSD 训练得到的 LoRA 时，发布版 Fast-WAM checkpoint 仍作为基座，同时传入最终 adapter
及生成它的解析后训练配置：

```bash
python scripts/resilient/evaluate_fault.py \
  --fault-config configs/fault/visual/wrist_camera_local_z.yaml \
  --gpus 0,1,2,3,4,5,6,7 \
  --checkpoint checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --dataset-stats checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  --opsd-adapter runs/opsd/wrist_camera_local_z_30deg/checkpoints/state/step_00000005/adapter.pt \
  --opsd-config runs/opsd/wrist_camera_local_z_30deg/resolved_config.yaml
```

`--opsd-adapter` 与 `--opsd-config` 必须同时提供。规范结果目录名会追加 adapter checkpoint
标识，manifest 同时记录基座模型、adapter 和配置；不传这两个参数时仍走原 Fast-WAM 基线加载
路径。

相机平移单位为米，沿原始相机局部轴；姿态按局部 X、Y、Z 顺序进行右手系旋转，单位为度。
MuJoCo 相机沿局部 `-Z` 观察，局部 `+X` 对应原始图像右方，局部 `+Y` 对应原始图像上方。
插件在 Teacher 请求特权图像时会临时恢复原始位姿，配对过程不会推进仿真时间。

## 面向 Fast-WAM 的 OPSD-Flow 适配

本实现参考 `manifests/upstream.json` 固定版本的 <https://github.com/siyan-zhao/OPSD>，采用
clean-room 方式适配。所检查版本没有仓库 LICENSE，因此没有复制其源码。核心约束保持不变：
Teacher **不会重新生成答案或动作轨迹**。Student 只运行一次 Fast-WAM 原始采样器；冻结的
Teacher 仅用特权条件评价这条 Student 轨迹上的每一个 latent。

主要模块如下：

| 路径 | 作用 |
| --- | --- |
| `src/resilient/faults/` | 可复用、带 severity 的机器人 Fault 插件与配对观测 |
| `src/resilient/opsd/teacher_inputs/` | 可替换的特权输入策略；当前仅把故障图像替换为干净图像 |
| `src/resilient/opsd/adapters.py` | 与 Fast-WAM 对齐的 LoRA 发现、冻结审计、启停和权重状态 |
| `src/resilient/opsd/model_adapter.py` | 使用 Fast-WAM 动作向量场评价 Student latent，不调用 scheduler step |
| `src/resilient/opsd/losses.py` | 逐坐标裁剪的 flow-matching 损失 |
| `src/resilient/opsd/trainer.py` | 限制显存的逐步打分、多卡梯度更新与断点恢复 |
| `configs/opsd/fastwam_libero.yaml` | 单 epoch 算法与运行参数 |
| `scripts/resilient/train_opsd.sh` | 严格限制为 4/8 卡的 Accelerate 启动器 |

Fast-WAM 原训练冻结 VAE/text encoder，训练两个 MoT expert 及 proprio encoder。OPSD 会冻结
整个发布 checkpoint，并向 `video_expert`、`action_expert`、`proprio_encoder` 内全部
`nn.Linear` 插入 LoRA；VAE 和 text encoder 不插入。这与原训练的高层模块范围一致，但只适配
线性权重，不更新 expert 中的 Conv3d、归一化、modulation 或 bias。训练前会输出逐层目标和
可训练参数数量的审计 JSON。

在 Student 去噪第 `k` 步，同一个 latent `x_k` 分别计算带 LoRA 的
`v_S(x_k,c_fault)` 和关闭 LoRA 后的 `stopgrad(v_T(x_k,c_clean))`。若有动作有效位 mask `m`，
该步损失是在有效 horizon/action 坐标上平均 `min((v_S-v_T)^2, 0.05)`；各步再按
`|delta_sigma_k| / sum_j |delta_sigma_j|` 加权并对 batch 求平均。因此 Teacher 只提供 Student
点上的目标向量场，不执行去噪、不调用 scheduler step，也不重新生成动作。

`opsd.rollout.num_inference_steps` 直接引用 `${eval_num_inference_steps}`，并将采集到的 timestep
和 delta 与 Fast-WAM scheduler 再校验。发布版 LIBERO 配置解析后严格为 **10 步**，不是 OPSD
独立选择的超参数；动作 horizon 同样直接引用 `data.train.num_frames - 1`，当前为 32。环境和
推理 seed 均引用项目 seed（默认 42），与 Fast-WAM 基线评测一致；进程内训练随机数使用
`seed + rank`。

模型构造也与发布 checkpoint 的评测路径一致：`load_text_encoder=true`、
`skip_dit_load_from_pretrain=true`、`action_dit_pretrained_path=null`。程序先构造网络并载入已
记录的共用 VAE/text 资产，再由 `ckpt` 提供两个训练后 expert；运行时会拒绝重复下载或加载
额外 video/action DiT 预训练权重的配置。

先运行不使用 CUDA 的配置烟测：

```bash
python scripts/resilient/train_opsd.py opsd.validate_only=true
```

GPU 空闲后，分别用 4 卡或 8 卡启动配置中的单个 epoch：

```bash
bash scripts/resilient/train_opsd.sh 4
bash scripts/resilient/train_opsd.sh 8 output_dir=runs/opsd/my_run
```

如需选择非默认物理卡，使用 `CUDA_VISIBLE_DEVICES`，例如
`CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/resilient/train_opsd.sh 4`。
启动器使用 OPSD 专用的 DeepSpeed ZeRO-2 配置，并显式设置
`train_micro_batch_size_per_gpu=1`，与运行时逐条轨迹更新一致；Fast-WAM 原训练使用的共享
DeepSpeed 配置保持不变。

所有参数均保留在 YAML：Fault/severity 位于 `configs/fault/`，LoRA 位于 `configs/adapter/`，
Teacher 输入位于 `configs/teacher_input/`，优化和 rollout 参数位于
`configs/opsd/fastwam_libero.yaml`。按需使用仓库相对路径覆盖 `ckpt`、
`opsd.dataset_stats_path` 和 `output_dir`。恢复训练时传入
`resume=runs/opsd/my_run/checkpoints/state/step_XXXXXXXX`，或在显式复用同一输出目录时使用
`resume=auto`。恢复点必须处于完整 task 边界，且解析后的配置哈希必须一致。训练状态、LoRA
checkpoint、指标和来源记录均写入被忽略的 `runs/`。

参考硬件仍为 Linux、8×NVIDIA RTX 6000 Ada 48 GB、CUDA 12.8 和 bf16；默认支持 4 卡与 8
卡。由于尚未正式运行或计时 OPSD 训练，目前不声明其最终峰值显存需求。本阶段已验证 CPU
单元测试、Hydra 配置组合、LoRA 插入/
审计，以及真实 LIBERO 相机干净/故障配对渲染，但尚未验证模型载入与多卡 optimizer step。

## 扩展开关与基线保护

以下规则为强制要求：

1. 新功能优先放在 `src/resilient/`，避免修改 `src/fastwam/`。
2. 如果必须修改 FastWAM 上游文件，新行为必须由明确的 Hydra/CLI 参数控制。
3. 每个 Resilient 开关必须默认关闭或保持上游原始值，保证文档中的基线命令行为不变。
4. 新开关必须在同一提交中加入下表、中英文 README、Hydra 配置和测试。
5. 无法避免的上游补丁记录到 `docs/upstream-patches.md`。

| 开关 | 默认值 | 作用范围 | 对基线的影响 |
| --- | --- | --- | --- |
| `EVALUATION.fault.pipeline.enabled` | `false` | 在 LIBERO evaluator 中启用有序机器人 Fault 管线 | 关闭时 evaluator 走完全相同的上游 reset/step 路径 |
| `FastWAM.infer_action` 的 `return_denoising_trace` | `false` | 为 OPSD 返回分离的 Student 去噪前 latent/timestep/delta | 为 false 时返回结构与动作采样不变 |
| `EVALUATION.opsd_adapter.enabled` | `false` | 在 Fast-WAM 基座 checkpoint 后注入并加载 OPSD LoRA | 关闭时不注入任何模块，基线评测不变 |

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
