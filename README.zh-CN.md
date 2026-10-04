# Resilient

**简体中文** | [English](README.md)

Resilient 是一个以可复现性为首要目标、基于 [FastWAM](https://github.com/yuantianyuan01/FastWAM) 开展 LIBERO、RoboTwin 评测和后续研究的代码库。由于 FastWAM 的 Hydra 配置和评测入口依赖仓库相对路径，上游代码直接整合在仓库根目录。

## 当前状态

- FastWAM 上游固定在提交 `7faa71108368fbb3b6885649f112af607427a2d4`。
- 基线目标为 `libero_uncond_2cam224.pt` 及其配套数据统计文件。
- 原始 checkpoint 使用 `EVALUATION.sigma_shift=5.0` 复现。
- FastWAM 默认行为未改变；核心代码只增加一个已登记、默认关闭的去噪轨迹返回开关，项目具体
  实现位于 `src/resilient/`。
- 已在下方硬件上验证独立 Python/CUDA 环境、全部固定资产、LIBERO EGL 无界面 reset、单回合集成评测与完整 2,000 回合基准。
- 带 severity 的 Fault 层和 OPSD-Flow 已完成 CPU、配置与仿真器烟测，尚未验证正式多卡 OPSD
  训练。
- 核心的仿真器级 Fault 目录现已提供五类视觉 Fault、五类具身 Fault、可复用 YAML 配置及
  时间对齐的图像/视频演示。
- 面向仿真器级第一关节 50% 运动保留 Fault 的 Outcome-Guided FPO 已完成实现及 CPU/配置测试；
  尚未进行真实多卡优化烟测。
- 官方 RoboTwin checkpoint 复现现已具备固定的50任务协议、独立兼容环境、可续跑的物理GPU
  调度、资产预检和论文对比报告工具。

## 仓库结构

```text
Resilient/
├── configs/                       # Hydra 配置，包含 Fault/OPSD/outcome-FPO 参数
├── experiments/libero/            # LIBERO 评测及默认关闭的 Fault hook
├── experiments/robotwin/          # 复用的 FastWAM RoboTwin 策略与评测器
├── src/fastwam/                   # 固定 FastWAM 及已登记的默认关闭补丁
├── src/resilient/                 # Resilient 扩展与适配器
├── scripts/resilient/             # 可复现的评测、下载与验证工具
├── environment/                   # 环境规范、锁文件与说明
├── manifests/                     # 上游及外部资产固定信息
├── reproduce/fastwam_libero/      # 最小与完整复现入口
├── reproduce/fastwam_robotwin/    # RoboTwin 论文协议和逐任务参考结果
├── reports/baselines/             # 精简、可提交的复现记录
├── checkpoints/fastwam_release/   # 下载权重，不上传 Git
├── data/lerobot_v30/              # LIBERO LeRobot 3.0 数据，不上传 Git
├── third_party/LIBERO/            # 固定版本的仿真器源码，不上传 Git
├── third_party/RoboTwin/           # 固定版本的 RoboTwin 评测代码
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
| 磁盘 | 两套仿真环境、权重、资产、数据与输出合计至少 120 GB | 配置前可用约 695 GB |

固定版本下载 Wan 公共组件需要 Git LFS；本次验证版本为 3.6.1。

FastWAM 默认启动 8 个持久化 LIBERO worker。完整评测时可通过 `MULTIRUN.num_gpus` 或下方脚本的 `NUM_GPUS` 调整 GPU 数量。本次 8-GPU 验证运行包含 worker 启动和模型加载约用时 46 分钟；减少 GPU 可降低内存需求，但会增加总耗时。

## 环境安装

环境由以下同步维护的规范组成：

- `.python-version` 固定已验证的 Python 运行时。
- `requirements.txt` 固定全部 FastWAM 运行依赖，包括 CUDA 12.8 版 PyTorch。
- `requirements-base.txt` 保存 LIBERO 与 RoboTwin 共用的 FastWAM 依赖。
- `requirements-libero.txt` 固定仅供 LIBERO 仿真使用的依赖，避免降级 FastWAM 当前依赖栈。
- `requirements-robotwin*.txt` 分别固定 NumPy 1.26 的 RoboTwin 运行依赖和 CuRobo 构建依赖。
- `requirements-dev.txt` 增加开发和测试工具。
- `environment/pip-freeze-cu128.txt` 锁定本次验证的全部直接与传递依赖。
- `environment/environment.yml` 提供等价的可选 Conda 引导规范。
- `environment/pip-freeze-robotwin-cu128.txt` 锁定已验证的 FastWAM/RoboTwin 组合环境。

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

## 复现 RoboTwin 基线

RoboTwin 使用独立环境：固定的 `mplib==0.2.1` 要求 NumPy 小于2，而 LIBERO 的 OpenCV
环境要求 NumPy 2。已验证组合保持 FastWAM 为 PyTorch 2.7.1/CUDA 12.8，并使用 NumPy
1.26.4、SAPIEN 3.0.0b1、MPLib 0.2.1、Open3D 0.18.0、Warp 1.12.0，以及
`manifests/upstream.json` 中固定的、仅限非商业用途的 CuRobo v0.7.7。该 CuRobo 版本仍调用
已在 Warp 1.13 删除的 `warp.torch` API，因此必须将 Warp 固定为1.12.0。编译 CuRobo 时必须
选择 CUDA 12.8 toolkit；主机的 `/usr/local/cuda` 可能指向其他版本。

```bash
bash scripts/resilient/create_robotwin_environment.sh .venv-robotwin
source .venv-robotwin/bin/activate
```

脚本依次安装 `requirements-robotwin-build.txt`、`requirements-robotwin.txt`，再关闭构建隔离
安装固定的 `requirements-robotwin-curobo.txt`。如需重建完整依赖，可使用
`environment/pip-freeze-robotwin-cu128.txt`；CuRobo 仍需针对目标 GPU 架构重新编译，并通过
`RESILIENT_CUDA_HOME` 指向 CUDA 12.8。`environment/environment-robotwin.yml` 是不包含
CuRobo 编译步骤的 Conda 引导文件，之后仍需运行上述 CuRobo 安装步骤。

RoboTwin 渲染还需要可用的 Vulkan runtime、`ffmpeg` 和兼容 CUDA 12.8 的 NVIDIA 驱动；
容器必须开放 graphics capability。已验证机器为8张48 GB RTX 6000 Ada。实测之前每张卡只
启动一个模型 worker。仿真器源码固定为 `third_party/RoboTwin` 下的 commit
`bf44be51cf5717a5595ce59447f2cf5263d2aa95`，禁止替换为持续变化的 RoboTwin main 分支。
目录内保留的上游 requirements 只用于来源审计，不能直接安装，否则会破坏 FastWAM 依赖。

下载并校验官方模型和仿真资产：

```bash
bash scripts/resilient/download_model_components.sh
python scripts/resilient/download_robotwin_assets.py --component all
python scripts/resilient/verify_robotwin.py --verify-hashes \
  --output AILOG/robotwin-preflight.json
```

第一条命令安装全局资产清单中固定的共用 Wan VAE、文本编码器和 tokenizer。RoboTwin 下载工具
固定 revision，将缓存放入被忽略的 `AILOG/`，解压前核对大小和 SHA-256，并拒绝替换已有非空
资产目录。RoboTwin 专用目标布局为：

```text
checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
third_party/RoboTwin/assets/background_texture/
third_party/RoboTwin/assets/embodiments/aloha-agilex/
third_party/RoboTwin/assets/objects/
```

官方 embodiment 压缩包中的 CuRobo planner 文件以 `*_tmp.yml` 模板形式提供。下载器会在
安装后按照当前 `third_party/RoboTwin` 路径生成机器本地的 `.yml` 文件；仓库迁移后必须重新
执行 simulator 下载命令。预检会拒绝缺失的 planner 配置以及由其他 checkout 路径生成的配置。

checkpoint 大小为12,041,813,092字节，SHA-256 为
`776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63`；统计文件大小为
88,715字节，SHA-256 为 `7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095`。
[checkpoint 仓库](https://huggingface.co/yuanty/fastwam)和
[RoboTwin 资产仓库](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)均未声明许可证，
因此不得再分发。三个仿真资产压缩包的 SHA-256 分别为：`background_texture.zip`
`54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070`、`embodiments.zip`
`6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6`、`objects.zip`
`6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27`。固定 revision、大小和
目标目录记录在 `manifests/assets.json` 与 `third_party/RoboTwin/ASSETS.md`。checkpoint 评测
不需要 RoboTwin 训练数据集。

按成本逐级运行，并只选择启动时空闲的 GPU：

```bash
# 单任务，clean/randomized各1回合。
python scripts/resilient/evaluate_robotwin_baseline.py \
  --mode smoke --gpu-ids 4 --task click_alarmclock

# 50任务，clean/randomized各1回合。
python scripts/resilient/evaluate_robotwin_baseline.py \
  --mode coverage --gpu-ids 4,5,6,7

# 论文协议：50任务 x（100 clean + 100 randomized）= 10,000回合。
python scripts/resilient/evaluate_robotwin_baseline.py \
  --mode paper --gpu-ids 4,5,6,7 --run-id fastwam_robotwin_paper
```

启动器默认拒绝带活跃计算进程的目标 GPU。它固定 unseen instruction、32步 action horizon、
24步重规划、10步 flow、sigma shift 5.0、CFG 1.0、CPU动作噪声，并关闭 Fault 和 adapter。
视频默认关闭，因为它不影响控制且10,000个视频会大量占用磁盘；调试时可传 `--save-videos`。
只有 protocol fingerprint 完全一致时才会续跑已完成 phase。评测器通过
`DIFFSYNTH_MODEL_BASE_PATH` 从仓库相对的 `checkpoints/` 解析共用 Wan VAE、文本编码器和
tokenizer；若外部环境已显式设置该变量，则以外部设置为准。每个 worker 在只暴露一张物理卡
后还会设置 `SAPIEN_RENDER_DEVICE=cuda:0`，避免 Vulkan 静默使用另一张卡；不设置该变量时
仍保持 RoboTwin 上游行为。

论文结果为 clean 91.88%、randomized 91.78%、展示平均值91.8%。完整附录表3位于
`reproduce/fastwam_robotwin/paper_reference.csv`。论文评测会自动输出总体和逐任务 CSV、JSON、
Markdown 对比及 Wilson 95%区间。原始结果位于被 Git 忽略的
`evaluate_results/robotwin/`；详细协议约束见 `reproduce/fastwam_robotwin/README.md`。

## 可复用 Fault 管线与评测

> **项目核心接口：**Fault 在创建仿真器时通过 LIBERO `FaultedEnvironment` 或 RoboTwin
> SAPIEN 适配器安装。Fast-WAM、FPO、OPSD 及后续任意策略均不包含具体 Fault 实现。因此相同
> family 的实现可以复用于评测、训练、shadow rollout 和多 Fault 联合实验。

Fault 定义是 `configs/fault/` 下与模型解耦的 YAML。创建环境时，
`get_libero_env(..., fault_pipeline=pipeline)` 会将启用的 pipeline 安装到透明的
`FaultedEnvironment`。之后 reset、状态加载、observation、action、simulator step、临时
挂起、断点状态及 detach 均由环境拥有；策略只调用标准 `reset`、`set_init_state`
和 `step`，不包含 Fault hook。后续图像污渍/遮挡、执行器限制、关节运动退化、动力学
故障及多 Fault 顺序组合均可以插件形式加入，不需要在 Fast-WAM 或其他策略中增加故障
分支。每个 Fault 都有稳定的 `family` 和独立的物理 `severity`（名称、数值、单位及可选
等级），因此绕轴 10/20/30 度属于同一 Fault family 的三个 severity。

### 十类 Fault 目录

可直接调用的配置位于 `configs/fault/catalog/`。其中 severity 故意设得较重，以便肉眼检查演示；
正式实验可修改 `severity` 及对应物理参数。

| 编号 | Family | 底层作用方式 | 强故障演示配置 |
| --- | --- | --- | --- |
| V1 相机旋转 | `visual.camera_pose` | 修改 MuJoCo 相机外参四元数 | `visual/camera_rotation.yaml`：双相机局部 +Z 45 度 |
| V2 相机平移 | `visual.camera_pose` | 修改 MuJoCo 相机外参位置 | `visual/camera_translation.yaml`：双相机局部 +X 0.12 m |
| V3 失焦模糊 | `visual.defocus_blur` | 在环境拥有的相机传感器中做 Gaussian 光学退化 | `visual/defocus_blur.yaml`：sigma 8 px |
| V4 局部遮挡 | `visual.local_occlusion` | 在环境拥有的相机传感器中加入柔边墨汁圆斑 | `visual/local_occlusion.yaml`：8 个固定圆斑，最大直径为图宽 0.44 |
| V5 光照变化 | `visual.illumination` | observation 离开环境前做仿射 RGB 响应 | `visual/illumination_change.yaml`：gain 0.2 并带颜色偏移 |
| E1 关节运动退化 | `structure.joint_motion` | 按比例保留每个 dynamics step 的实际位移与速度 | `structure/joint_motion_degradation.yaml`：关节 1 保留 0.2 |
| E2 关节位置偏置 | `structure.joint_position_bias` | 每次加载状态后引入一次固定且不累加的关节零点偏置 | `structure/joint_position_bias.yaml`：关节 1 +20 度 |
| E3 关节回差 | `structure.joint_backlash` | 每次运动反向后先消耗空行程 | `structure/joint_backlash.yaml`：关节 1 间隙 3 度 |
| E4 关节范围限制 | `structure.joint_range_limit` | 把实际关节位置裁剪到缩小后的绝对边界 | `structure/joint_range_limitation.yaml`：关节 1 限于 [-10,+10] 度 |
| E5 周期性关节冻结 | `structure.periodic_joint_freeze` | 关节越过按角度周期分布的坏齿位置时冻结 | `structure/periodic_joint_freeze.yaml`：每 10 度触发并保持 35 个控制步 |

同一第一关节的E3单档较强测评，可用
`configs/fault/evaluation/structure/joint_backlash_6deg.yaml`。该配置同时设置物理间隙
`gap_deg`与对应的severity元数据；不能只修改该Fault的`severity.value`。

V1/V2 修改 MuJoCo 渲染几何；V3--V5 模拟相机硬件/传感器输出，在任何模型预处理之前由
`FaultedEnvironment` 执行；E1--E5 在每个环境 dynamics step 后修改 MuJoCo 实际关节状态。
因此十类 Fault 均不位于模型、某个特定 controller 的 evaluator 或 recovery 算法中。
E2/E3/E5 的有状态变量会写入 `fault_runtime_state_dict()`，相同状态的 shadow 环境能够精确复现。

加载一个目录 Fault；需要联合 Fault 时，按期望顺序合并多个配置的 `faults` 项：

```python
from omegaconf import OmegaConf

from experiments.libero.libero_utils import get_libero_env
from resilient.faults import build_fault_pipeline

fault_config = OmegaConf.to_container(
    OmegaConf.load("configs/fault/catalog/structure/joint_backlash.yaml"),
    resolve=True,
)
pipeline = build_fault_pipeline(fault_config)
env, task_description = get_libero_env(task, 256, seed=42, fault_pipeline=pipeline)
observation = env.reset()
observation, reward, done, info = env.step(action)
```

在相应插件支持时，`targets` 可包含多个相机或标量 MuJoCo 关节名。每个插件都会校验单位和范围，
并输出含 `family`、目标、物理 severity、参数和注入层的可移植 metadata。完整公式与扩展接口见
`configs/fault/README.md`。

一次生成五张带标注的 2×2 视觉对比图和五段时间对齐的具身对比视频：

```bash
CUDA_VISIBLE_DEVICES=0 EGL_DEVICE_ID=0 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:third_party/LIBERO:. \
python scripts/resilient/demonstrate_fault_catalog.py \
  --catalog configs/fault/demo_catalog.yaml \
  --faults all \
  --output-root evaluate_results/fault_catalog
```

也可在 `--faults` 后只传需要生成的目录 ID。视觉产物名为 `nominal_vs_fault.png`，具身产物名为
`nominal_vs_fault.mp4`，每个画面左上角都有说明。具身演示从同一 simulator state 出发、回放同一
组 `JOINT_POSITION` 命令、帧数严格一致，并报告目标关节及末端执行器的偏差。所有具身演示
action 只有失效 joint1 对应维度非零，其余六个机械臂关节命令保持为零。运动阶段完整写在
`configs/fault/demo_catalog.yaml`，回差与周期卡钝演示包含反向。全量运行写入
`catalog_manifest.json`；子集运行写入
`catalog_manifest__<selected-ids>.json`，不会覆盖全量 manifest。默认输出目录和 JSON 摘要属于
生成物，继续由 Git 忽略。

### RoboTwin 十类 Fault 与双臂演示

RoboTwin 复用相同的 Fault family 注册实现和数学变换。单独放在
`configs/fault/robotwin/catalog/` 下的配置只负责适配物理目标名：三个相机
（`head_camera`、`left_camera`、`right_camera`）和两条六自由度 ALOHA 机械臂。V1/V2 修改
SAPIEN 相机 entity 位姿；V3--V5 在环境相机传感器返回层生效；E1--E5 在
`Robot.set_arm_joints()` 内、写入 SAPIEN articulation drive target 前变换目标。任何 Fault
都不作用于 Fast-WAM action tensor 或模型预处理层。

RoboTwin controller 提供的是绝对关节目标。E1 因此同时保存上一次 nominal target 和 applied
target，并且只对每次新增的命令增量缩放一次：

```text
delta_nominal(t) = q_cmd(t) - q_cmd(t-1)
q_applied(t) = q_applied(t-1) + retention * delta_nominal(t)
```

重复发送不变的绝对目标只会产生零新增运动，因此退化关节不会因 controller 重复更新而逐渐追上
正常目标。E2 施加一个持续但不累加的执行器目标偏置；E3 消耗反向后的增量空行程；E4 裁剪绝对
目标；E5 越过角度坏齿后，在配置的控制更新次数内保持同一个固定物理角度。E1/E3 的命令历史及
E5 的触发只比较相邻两次实际 SAPIEN qpos，不会因为尚未实现的远端 drive target 跨过坏齿而
提前触发。E1/E3 的命令历史及 E5 的全部卡钝状态都会进入 Fault controller 断点状态。

具身 Fault 的规范目标为 `left_arm_joint1`--`left_arm_joint6` 与
`right_arm_joint1`--`right_arm_joint6`，也可以使用无歧义的 SAPIEN 原始关节名。每个有状态
Fault 项只应作用于一条机械臂；如需双臂同时损坏，应组合两个 Fault 项，使回差和周期冻结状态
相互独立。官方评测开关为 `EVALUATION.fault_config`，默认 `null` 时不会导入或调用适配器，
完整保留固定论文基线。expert 可解性检查保持正常仿真；Fault 只在每次 SAPIEN scene 创建后
安装到 policy rollout。

一次生成五张带标注的2×3正常/Fault三相机图，以及每个具身 Fault 的左臂、右臂两段视频：

```bash
CUDA_VISIBLE_DEVICES=0 SAPIEN_RENDER_DEVICE=cuda:0 \
python scripts/resilient/demonstrate_robotwin_fault_catalog.py \
  --catalog configs/fault/robotwin/demo_catalog.yaml \
  --faults all \
  --output-root evaluate_results/robotwin/fault_catalog
```

每段具身对比均从同一恢复后的 SAPIEN 状态开始，只移动标注机械臂的 joint 1，其他关节固定，
正常与 Fault 时间轴完全一致。已提交的演示目录使用 RoboTwin 全局 observer 相机，使两条机械臂
及其运动差异均保持可见；回差与周期卡钝包含方向反转。输出和 manifest 位于被忽略的
`evaluate_results/robotwin/fault_catalog/`。通过维护的官方 checkpoint 启动器评测一个 Fault：

```bash
python scripts/resilient/evaluate_robotwin_baseline.py \
  --mode smoke --gpu-ids 0 --task click_alarmclock \
  --fault-config configs/fault/robotwin/catalog/structure/joint_motion_degradation.yaml
```

解析后的 Fault 路径和 SHA-256 会进入运行协议指纹，避免续跑时混入不同 Fault 定义生成的结果。
具体公式、后端语义、目标别名和扩展规则见 `configs/fault/README.md`。

正式成功率评测配置与剧烈的演示目录分开保存。下面使用左臂 joint 1 运动保留率0.5，按与论文
基线相同的50任务、每任务100次clean加100次randomized协议评测官方checkpoint：

```bash
python scripts/resilient/evaluate_robotwin_baseline.py \
  --mode paper \
  --gpu-ids 0,1,2,3 \
  --run-id fastwam_robotwin_left-joint1-retention-p0p5 \
  --fault-config \
    configs/fault/robotwin/evaluation/structure/left_joint1_motion_retention_0p5.yaml
```

结果写入被Git忽略的
`evaluate_results/robotwin/robotwin_uncond_3cam_384/<run-id>/`。该流程复用现有launcher、manager
与仿真器适配器，不为某一种Fault复制新的评测脚本。

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

### 关节运动退化 Fault 语义

`structure.joint_motion` family 在 MuJoCo simulator 创建时由环境安装，它作用在底层环境
动力学 step 边界，与具体 controller、Fast-WAM、OPSD 或后续任何策略无关。`targets` 可同时指定
一个或多个 MuJoCo 关节名。首个内置实现为 `proportional`；已提供的配置会将每个
环境控制步产生的 `robot0_joint1` 角位移和关节速度仅保留 50%。退化律有独立
注册表，后续可增加确定性非线性函数，无需修改仿真调用器或模型代码。上面的目录演示命令
已经取代旧的 OSC 轨迹演示；新版使用 `JOINT_POSITION` 且只命令关节 1，使观测差异可归因于
所选 Fault，而不是其他关节共同运动。

线性退化定义在每个环境动力学 step 边界：

```text
q_fault_after = q_before + retention * (q_nominal_after - q_before)
```

同时将目标关节速度乘以 `retention`。这是有意设计的运动学位移损失，不是执行器力矩
效率损失。该安装会跨环境 reset 保持，只在环境 close 时拆除；配置中的多个 simulator Fault
按 YAML 顺序组合。由于闭环控制器会在后续控制步持续补偿，Fault 轨迹的最终关节角不应
被预期为 clean 最终角度的严格 50%。未安装 pipeline 时，原始无 Fault 路径不受影响。

已有 checkpoint 无需重新训练即可进行未见状态验证。先通过带 seed 的 LIBERO reset 生成独立
验证状态库；该命令也会从已完成 OPSD run 的 metrics 重建训练实际见过的官方状态及环境 seed：

```bash
python scripts/resilient/generate_libero_validation_states.py \
  --training-run runs/opsd/my_run \
  --output-dir data/libero_state_banks/my_unseen_validation \
  --states-per-task 50 \
  --base-seed 104729
```

随后在原评测命令中同时加入：

```bash
  --state-bank-manifest data/libero_state_banks/my_unseen_validation/validation_manifest.json \
  --training-state-manifest data/libero_state_banks/my_unseen_validation/training_reference_manifest.json
```

状态库属于数据集内容并被 Git 忽略。评测器会逐个验证 simulator state 指纹；只要验证状态
哈希或生成 seed 与训练有一个重合，就会拒绝启动。该开关默认关闭，因此 Fast-WAM 官方基线
路径不变，已有 base/LoRA checkpoint 也不需要重新训练。

相机平移单位为米，沿原始相机局部轴；姿态按局部 X、Y、Z 顺序进行右手系旋转，单位为度。
MuJoCo 相机沿局部 `-Z` 观察，局部 `+X` 对应原始图像右方，局部 `+Y` 对应原始图像上方。
插件在 Teacher 请求特权图像时会临时恢复原始位姿，配对过程不会推进仿真时间。

## 基于 Video DiT latent 的 Fault severity 曲线

`scripts/resilient/evaluate_fault_severity.py` 会对一个固定 LIBERO task 依次测试多个 severity。
默认每个 severity 使用相同的 50 个初始状态。默认模型是官方发布的 Fast-WAM 基座 checkpoint；
也可显式指定其他兼容 checkpoint，或指定 Fast-WAM 基座加 OPSD adapter。不同 severity 会分配给
所指定的物理 GPU，因而可并行运行。

例如，在 Spatial task 0 上测试手腕相机绕局部 +Z 轴旋转 10、20、30 度：

```bash
python scripts/resilient/evaluate_fault_severity.py \
  --fault-config configs/fault/visual/wrist_camera_local_z.yaml \
  --severities 10 20 30 \
  --suite libero_spatial \
  --task-id 0 \
  --gpus 0,1,2
```

默认输出根目录是 Git 已忽略的 `evaluate_results/fault_detection/`，可通过 `--output-dir` 修改。
规范 study 目录名包含 Fault family、目标、severity 字段、checkpoint、suite 和 task。目录内包含
不可变的 `study_manifest.json`、每个 severity 的可复现实验子目录、`metrics_summary.json`、
`metrics_summary.csv`、便于阅读的 `metrics_summary.md`、残差矩阵数据和每项指标的 PNG。默认不保存 episode 视频，以免产生大量
非必要文件；需要时添加 `--save-videos`。断点续跑只会复用同时存在结果 JSON 和残差 prototype
文件的完整 severity。

指标列表与统计参数单独位于 `configs/fault_detection/latent_v1.yaml`；`--metrics ...` 仅覆盖待选
指标。第一版包含：

| 指标 | 定义与聚合方式 |
| --- | --- |
| LPE | 对齐后的未来 `Z_pred` 与 `Z_real` 的均方误差 |
| LCD | 两个对齐 latent 展平后的 1 减余弦相似度 |
| RM | `R = Z_real - Z_pred` 的平均绝对值 |
| RCS | 每个 episode 的残差 prototype 与 clean severity 下同一 episode prototype 的余弦相似度 |
| Residual matrix | 每对 severity 的配对 episode 余弦均值，并为每个 severity 输出 50x50 episode 热图 |
| RTC | 一个 episode 内相邻完整预测窗口残差的平均余弦相似度 |
| Spearman rho | severity 与 episode RM 的等级相关系数，并通过确定性 bootstrap 估计不确定性 |
| FaultScore | 使用 clean run 的 RM 均值和样本标准差对 episode RM 标准化 |
| ROC-AUC | 分别比较 clean 与各非零 severity 的 RM，并通过确定性 bootstrap 估计不确定性 |

LPE、LCD、RM、RCS、RTC 和 FaultScore 先在单个 episode 内聚合，再对多个 episode 统计。JSON/CSV
会记录有效数量、均值、样本方差、标准差和 SEM。误差棒默认使用正负一个标准差；如需 SEM，
在指标 YAML 中设置 `error_bar: sem`。Spearman 与 ROC-AUC 记录 bootstrap 方差。FaultScore 和
ROC-AUC 必须有 clean run；默认配置会在命令未包含时自动补充 severity 0。使用
`--no-auto-clean` 可要求调用者显式提供。

Fast-WAM 的 Video DiT 在每个扩散步输出的是向量场，而不是最终 latent。这里的 `Z_pred` 是原始
video scheduler 完成全部去噪后得到的最终 `latents_video`；`Z_real` 是用同一个冻结 VAE 对实际
观测到的 Fault 图像序列编码而来。发布模型使用与 action step `0,4,...,32` 对齐的 9 帧视频，
所以脚本会完整执行每个 32-action 预测窗口后再重规划，episode 结尾不完整的窗口不计入指标。
两个 latent 都排除时间维第 0 项，因为它是输入条件图像，只比较未来 latent。该完整窗口协议
保证形状和时间严格对齐，但它得到的成功率不能与每 10 个 action 重规划的标准 Fast-WAM 基线
成功率直接比较。

环境实际执行的动作仍来自未改变的 `infer_action` 路径；联合推理只额外用于采集视频预测。
评测开关 `EVALUATION.fault_detection.enabled` 默认关闭；Fast-WAM 新增参数的默认值为
`return_video_latents=false`、`decode_video=true`，保持原来的返回键与解码行为。新的对齐 clip
指标可注册到 `src/resilient/fault_detection/metrics.py`，跨 severity 指标与绘图隔离在
`study.py`。也可像普通 Fault 评测一样传入 `--state-bank-manifest` 和
`--training-state-manifest` 使用未见状态库。

真实集成烟测环境为 Linux、CUDA 12.8、bf16 和一张 NVIDIA RTX 6000 Ada 48 GB；编译完成后
worker 约占用 25.2 GB 显存。每个并行 severity 独占一张 GPU，增加 GPU 数量会提高 severity
级并行度，并不会切分单个 episode。目前尚未计时完整的每 severity 50-episode sweep，因此不
声明完整运行时长。

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
| `configs/opsd/fastwam_libero.yaml` | 多 epoch 参数（默认 1 epoch） |
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

GPU 空闲后，用 4 卡或 8 卡启动训练。默认是 1 epoch，每个 task 使用 50 条独立重置的
轨迹（总计 2,000 条）；长训练需显式覆盖
`opsd.num_epochs`：

```bash
bash scripts/resilient/train_opsd.sh 4
bash scripts/resilient/train_opsd.sh 8 output_dir=runs/opsd/my_run
bash scripts/resilient/train_opsd.sh 4 opsd.num_epochs=20 max_checkpoints=1
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
`resume=auto`。运行时构造不可变的 `(suite, task, rollout, initial-state,
environment-seed, inference-seed)` 描述符，对全局序列进行确定性打乱，再按跨步方式分给各
rank。每个 task 的 LIBERO 初始状态 0--49 恰好各使用一次；seed 由描述符身份而非执行顺序
派生，因此断点恢复后仍会得到相同样本。恢复点可以位于任意完整轨迹边界，且解析后的配置
哈希必须一致。训练状态、LoRA checkpoint、指标和来源记录均写入被忽略的 `runs/`。
每个 epoch 对 40 个标准 task 各训练 50 次，并在 epoch 边界保存完整可恢复状态。
`max_checkpoints` 默认为 2，只会在新状态安全写完后删除更旧状态；参考环境中每个 ZeRO 状态
约占 50 GB，因此长训练应按磁盘空间设置保留数量。

如需在已完成的 LoRA 上开启下一训练阶段，同时设置 `adapter.initial_checkpoint` 与递增后的
`opsd.epoch_offset`。该方式只加载 Student LoRA，重新开始 optimizer/余弦学习率周期，并使用
下一 epoch 的打乱顺序与派生 seed。它不同于用于恢复中断阶段、同时恢复 optimizer/scheduler
状态的 `resume`。

参考硬件仍为 Linux、8×NVIDIA RTX 6000 Ada 48 GB、CUDA 12.8 和 bf16；默认支持 4 卡与 8
卡。由于尚未正式运行或计时 OPSD 训练，目前不声明其最终峰值显存需求。本阶段已验证 CPU
单元测试、Hydra 配置组合、LoRA 插入/
审计，以及真实 LIBERO 相机干净/故障配对渲染，但尚未验证模型载入与多卡 optimizer step。

## 面向具身 Fault 恢复的 Outcome-Guided FPO

该路径不做 Fault representation 预训练，而是直接进行单阶段恢复策略优化。条件流策略梯度估计
参考 [Flow Matching Policy Gradients / FPO](https://github.com/akanazawa/fpo)，具体版本固定为
`manifests/upstream.json` 中的提交
`418c2554f7cd22d52e14c07d951280929d73bf2f`。本项目根据其 Apache-2.0 Playground 目标独立完成
PyTorch 适配，没有引入上游 MuJoCo 或 JAX 运行代码。

冻结的 Fast-WAM 基座同时充当 Teacher 和评估器。Teacher 使用与 Student 相同的当前图像、语言
和 proprio，由 Video DiT 通过发布版 10 步 video scheduler 预测正常状态下的九帧未来，并在
训练 timestep 500 读取 Video DiT 第 19 个 block 的时间变化表示。Student 使用未修改的 10 步
action sampler，从同一状态采样 4 个最终归一化 `32x7` action chunk。每个候选开始前，都把同一
shadow 环境恢复到完全相同的 LIBERO simulator state 与仿真器级 Fault 状态，再完整执行 32 个
动作并在 `0,4,...,32` 步采集九帧。冻结 Teacher 视觉网络以相同层、future token、timestep 和
noise 编码各候选真实未来。唯一 reward 是其 future-token 时间变化与正常目标之间的平均 cosine
similarity；success、progress、collision、Fault metadata 和 simulator reward 都不会加入策略
reward。

FPO 把 Fast-WAM 原生确定性 sampler 当成黑盒，不构造 SDE，也不使用去噪 transition log
probability。对每个最终动作 `A`，独立采样 8 组可复现的连续 action timestep `t` 和高斯噪声
`epsilon`。Fast-WAM 原训练 scheduler 构造 `x_t=(1-t)A+t*epsilon`，action expert 预测 velocity
`v`，打分器计算 `epsilon_hat=x_t+(1-t)v`；每组条件 flow-matching score 是
`epsilon_hat` 与 `epsilon` 的均方误差。old/current 共用完全相同的动作、timestep 和 noise，先对
8 项 loss 求均值，再计算 `ratio=exp(loss_old-loss_current)`。PPO 式 ratio clip 为 0.05，数值
保护将 log-ratio 限制在 `[-3,3]`。

同一状态的 4 个 reward 使用 leave-one-out 标准化 advantage。这只是同状态 control variate，
不是把 10 个去噪步建模为 GRPO；组内 reward 标准差低于阈值时做零更新。每批 on-policy 采集
复用 2 个 FPO update epoch。发布 checkpoint、VAE 和 text encoder 永久冻结；零起点 Recovery
LoRA 覆盖与 Fast-WAM 原训练一致的高层范围：`video_expert`、`action_expert` 和
`proprio_encoder`。默认 rank 64、alpha 128、dropout 0；AdamW 使用学习率 `1e-6`、weight decay
0、bf16 和梯度范数 0.1。

代码按职责解耦：

| 路径 | 作用 |
| --- | --- |
| `src/resilient/outcome_fpo/outcome.py` | 冻结 nominal Video DiT target 与仅含 cosine 的 outcome reward |
| `src/resilient/outcome_fpo/collector.py` | 同状态 shadow 恢复及 32-action/9-frame 对齐 |
| `src/resilient/outcome_fpo/policy.py` | 原生动作采样与 Fast-WAM conditional-flow score |
| `src/resilient/outcome_fpo/advantage.py` | leave-one-out 同状态 advantage 与同分组保护 |
| `src/resilient/outcome_fpo/objective.py` | FPO loss ratio 与裁剪 surrogate objective |
| `src/resilient/outcome_fpo/trainer.py` | 分布式更新、短生命周期 buffer 和完整断点状态 |
| `src/resilient/outcome_fpo/runtime.py` | LIBERO 日程、split 校验、来源记录、日志与调度 |
| `configs/outcome_fpo/fastwam_libero_joint1_half.yaml` | 第一版可复现实验及全部可调参数 |
| `scripts/resilient/train_outcome_fpo.sh` | 严格限制 4/8 卡的 Accelerate/ZeRO-2 入口 |

第一版直接复用 `configs/fault/structure/panda_joint1_half_motion.yaml` 的通用 Fault 管线：MuJoCo
关节 `robot0_joint1` 的每环境步实际位移只保留 nominal 位移的 0.5。Fault 安装在仿真器中，因而
独立于 Fast-WAM 和本优化器；FPO 模块没有复制任何 Fault 专属实现。

训练规模沿用先前视觉 Fault OPSD：1 epoch、全部 40 个标准 LIBERO task、每个 task 使用官方
初始状态 0--49 各一次，共 2,000 个同状态组与 8,000 个候选 rollout。base seed 42 根据 sample
身份确定性派生 schedule、environment、action inference、Teacher 和 Monte Carlo seed。验证继续
使用 `data/libero_state_banks/opsd_step500_unseen_v1/validation_manifest.json`，其 base seed 为
104729。程序启动时强制同时加载 training-reference 与 validation manifest，并在状态哈希或
environment seed 有任何交集时拒绝运行；训练过程不会加载验证状态。

安装常规项目环境并按前文下载 Fast-WAM checkpoint/statistics 与 LIBERO 资产后，先执行不使用
CUDA 的完整 Hydra 配置检查：

```bash
python scripts/resilient/train_outcome_fpo.py outcome_fpo.validate_only=true
```

训练必须使用恰好 4 张或 8 张可见 GPU；除非显式覆盖，所有路径均相对仓库：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/resilient/train_outcome_fpo.sh 4 \
  output_dir=runs/outcome_fpo/joint1_half_seed42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/resilient/train_outcome_fpo.sh 8 \
  output_dir=runs/outcome_fpo/joint1_half_seed42_8gpu
```

默认每次全局采集 8 个 group：4 卡时每 rank 2 个，8 卡时每 rank 1 个。中间 checkpoint 只在
完整采集边界写入 `<output_dir>/checkpoints/state/step_XXXXXXXX/`；每个完整 epoch 另存到
`<output_dir>/checkpoints/epochs/epoch_XXXX_step_XXXXXXXX/`。完整 checkpoint 包含
Accelerate/ZeRO optimizer 与 RNG 状态、`recovery_adapter.pt`、`trainer_state.json`；解析后的配置
与 provenance 位于 run 根目录。保留策略可以删除旧 epoch 的大型 `accelerate/` 目录，同时保留
adapter 与审计元数据；这种 adapter-only epoch 仍可评测，但不能严格恢复 optimizer/RNG。
`resume=auto` 会忽略 adapter-only epoch，在两类目录中选择进度最靠后的完整 checkpoint；显式
指定 adapter-only 路径或配置哈希不一致都会立即拒绝恢复。`rollouts_rank_XX.jsonl` 记录状态、
seed、4 个 reward 以及仅供诊断的成功标志，成功标志不会参与 reward。

训练完成后，使用现有通用 Fast-WAM LoRA loader 和未见状态库评测 Recovery LoRA。参数仍保留
历史 `opsd` 名称，但能加载结构完全相同的本 adapter：

```bash
python scripts/resilient/evaluate_fault.py \
  --fault-config configs/fault/structure/panda_joint1_half_motion.yaml \
  --gpus 0,1,2,3 \
  --checkpoint checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --opsd-adapter runs/outcome_fpo/joint1_half_seed42/checkpoints/epochs/epoch_0001_step_XXXXXXXX/recovery_adapter.pt \
  --opsd-config runs/outcome_fpo/joint1_half_seed42/resolved_config.yaml \
  --state-bank-manifest data/libero_state_banks/opsd_step500_unseen_v1/validation_manifest.json \
  --training-state-manifest data/libero_state_banks/opsd_step500_unseen_v1/training_reference_manifest.json \
  --output-dir evaluate_results/outcome_fpo/joint1_half_seed42
```

参考平台仍为 Linux、CUDA 12.8、bf16 及 4/8 张 RTX 6000 Ada 48 GB。实现复用现有 OPSD
ZeRO-2 启动配置，没有新增 Python 依赖，因此 `requirements*.txt`、环境锁和安装流程保持不变。
当前尚未完成真实多卡模型加载、显存峰值、optimizer step、吞吐量和最终恢复效果烟测，不能将
本轮代码级验证当作实验结果。生成的 run 和 checkpoint 均由 Git 忽略。

## 单任务具身 Fault 两阶段适配

两阶段路径针对每个 `(task, Fault)` 单独训练一个 LoRA。首个正式配置使用 `libero_10` task 7
（把 alphabet soup 与 cream-cheese box 都放进篮子）及仿真器级 `robot0_joint1` 运动保留率
0.5 Fault。两个阶段共用覆盖 `video_expert`、`action_expert`、`proprio_encoder` 的同一套 LoRA；
Stage II 继续训练 Stage-I adapter，而不是再叠加一套 adapter。

Stage I 在 Fault 环境中，使用 Fast-WAM 基座从官方训练状态 0--49 rollout。数据只保存一次完整
轨迹，再用索引表示重叠窗口，因此 12,000 个训练窗口不会重复存储图像。每个状态严格产生 240
个有效窗口；task 成功后立即停止该 episode，如窗口不足则以确定性新 seed 重新 rollout。每个
样本包含 32 个实际执行动作、对应的归一化 proprio，以及 `0,4,...,32` 的双相机九帧。训练直接
调用未修改的 `FastWAM.training_loss()`，video/action loss 权重均为 1。配置特意保持
`model.video_dit_config.action_conditioned=false`：这里复用发布版 Fast-WAM 的 video--action 联合
训练，不宣称存在显式的 Action-to-Video 因果通路。训练参数为 10 epoch、12,000 窗口、四卡
每卡 micro-batch 8、梯度累积 4（全局 batch 128）、AdamW `(0.9,0.95)`、学习率 `1e-4`、
weight decay `1e-2`、5% warm-up 加 cosine、bf16、梯度范数 1.0。

Stage II 复用现有 Outcome-FPO。对同一批 50 个训练状态，当前 Student 在 Fault 下分别到达动作
步 `0,80,160,240` 的四个因果 anchor。每个 anchor 从同一 simulator state 采样四个 32-action
候选，用其真实九帧结果与关闭 LoRA 的冻结基座 Teacher 比较；唯一 reward 仍是 timestep 500、
Video DiT block 19 的时间变化 cosine similarity。每候选使用 8 个 conditional flow-matching
Monte Carlo 对，每批数据更新 2 次，clip 0.05、学习率 `1e-6`、梯度范数 0.1；合计 200 个 group、
800 个候选 Fault rollout 和 50 个 optimizer step。

代码职责如下：

| 路径 | 作用 |
| --- | --- |
| `src/resilient/two_stage_opsd/stage1_collector.py` | 可恢复的基座策略 Fault rollout 采集 |
| `src/resilient/two_stage_opsd/dataset.py` | 轨迹级紧凑存储的 32-action/9-frame 数据集 |
| `src/resilient/two_stage_opsd/stage1_trainer.py` | 原生联合 loss LoRA 训练与 epoch 断点 |
| `src/resilient/outcome_fpo/runtime.py` | 复用的 Stage-II Teacher/reward/FPO 与因果 anchor |
| `configs/two_stage_opsd/fastwam_libero10_task7_joint1_half.yaml` | task、split、采集和两阶段参数 |
| `scripts/resilient/run_two_stage_opsd.sh` | 顺序执行采集与两阶段训练的四卡入口 |

Stage-I 生成数据默认位于
`data/generated/two_stage_opsd/libero10_task7_joint1_retention_p0p5/`，训练产物默认位于
`runs/two_stage_opsd/`，两者均被 Git 忽略。它只依赖前文已经说明的 Fast-WAM checkpoint/统计与
LIBERO 资产；采集阶段会把该 task 的冻结文本 context 和生成数据一起保存，没有新增依赖。数据划分
复用 `opsd_step500_unseen_v1`：训练只读取官方
状态 0--49，评测必须读取与其无交集的 50 个 validation tensor；程序启动时会同时校验两个
manifest，并在存在交集时拒绝运行。

在恰好四张可见 GPU 上顺序跑完：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/resilient/run_two_stage_opsd.sh \
  runs/two_stage_opsd/libero10_task7_joint1_half_seed42
```

采集按完整 state 自动续跑。Stage I 每个 epoch 单独保存完整断点，只保留最近三个；每个断点含
`joint_adapter.pt`、逐 rank optimizer/RNG、scheduler 状态及严格配置哈希，最终 adapter 复制到
`<run>/stage1/final/joint_adapter.pt`。Stage II 使用标准 Outcome-FPO 断点布局
`<run>/stage2/checkpoints/`，最终 epoch adapter 即两阶段策略。
`checkpoint_retention.rolling_keep_last` 控制 epoch 内滚动恢复点数量，
`checkpoint_retention.epoch_keep_last` 控制保留多少个 epoch adapter 目录（`null` 表示全部保留），
`preserve_epochs` 列出额外 adapter 里程碑。独立的 `epoch_full_state_keep_last` 控制其中多少个
epoch 保留大型 Accelerate/ZeRO 严格恢复状态（`null` 表示全部保留），
`preserve_full_state_epochs` 列出额外完整状态里程碑。task 7 当前配置保留每个 epoch adapter，
但仅保留最新一次完整恢复状态：

```yaml
checkpoint_retention:
  rolling_keep_last: 1
  epoch_keep_last: null
  preserve_epochs: []
  epoch_full_state_keep_last: 1
  preserve_full_state_epochs: []
```

只有在新 epoch 完整保存后才会清理旧恢复分片。旧 epoch 目录继续保留
`recovery_adapter.pt`、`trainer_state.json`、`resolved_config.yaml` 以及
`resume_state_pruned.json` 标记，可用于评测但不能严格续训。

如果要从已完成的 Stage-I adapter 全新训练十个 Stage-II epoch，同时保留每个
epoch adapter 且只保留最新一份严格恢复状态，使用：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_opsd_zero2_ds.yaml \
  --num_processes 4 scripts/resilient/train_two_stage_stage2.py \
  --config-name two_stage_opsd/fastwam_libero10_task7_joint1_half_stage2_10ep \
  two_stage_opsd.distributed.num_processes=4
```

这个独立配置将 Stage-II optimizer 固定为学习率 `1e-6`、weight decay `0`、梯度范数
`0.1`和 gradient accumulation `1`；不能误用 Stage-I 的 `1e-4` 学习率。

若要在四张可用GPU上把这条全图奖励训练从epoch 10严格续到epoch 20，使用下面的专用配置。
它继续写入同一个输出目录，只改变`outcome_fpo.num_epochs`；每轮adapter仍保留，只有最新轮
保留完整优化器/RNG状态。该配置必须已有完整的epoch 10断点，不能用于从零启动。

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
  accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_opsd_zero2_ds.yaml \
  --num_processes 4 scripts/resilient/train_two_stage_stage2.py \
  --config-name two_stage_opsd/fastwam_libero10_task7_joint1_half_stage2_continue20
```

若要扩展已经完成的 Stage II，并严格恢复 AdamW、RNG、group 位置和 LoRA 状态，应提高总 epoch
上限并在同一输出目录使用 `resume=auto`。与保存配置相比，只允许更改
`outcome_fpo.num_epochs` 和断点保留参数：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_opsd_zero2_ds.yaml \
  --num_processes 4 scripts/resilient/train_outcome_fpo.py \
  --config-name two_stage_opsd/fastwam_libero10_task7_joint1_half \
  output_dir=runs/two_stage_opsd/libero10_task7_joint1_half_seed42/stage2 \
  two_stage_opsd.distributed.num_processes=4 \
  two_stage_opsd.stage2.initial_adapter=runs/two_stage_opsd/libero10_task7_joint1_half_seed42/stage1/final/joint_adapter.pt \
  learning_rate=1.0e-6 weight_decay=0.0 max_grad_norm=0.1 \
  gradient_accumulation_steps=1 num_epochs=1 \
  outcome_fpo.num_epochs=7 resume=auto
```

参考硬件为 Linux、CUDA 12.8、bf16 与四张 RTX 6000 Ada 48 GB。

### E5 周期坏齿 Fault：全图奖励与任务方向奖励对照

E5 实验沿用相同的 `libero_10` task 7、互斥的训练/验证各 50 状态 manifest、基座权重、
Stage I 的 12,000 窗口/10 epoch，以及 Stage II 每 epoch 200 组/800 个候选。唯一变化是
仿真器底层 Fault：`robot0_joint1` 每跨越一个 10 度的齿轮位置就卡住 35 个控制步。
`expected_embodied_fault` 会在采集和训练前核验 Fault 类型、关节、周期与持续时长；
旧配置不设置此字段时仍保持原先仅允许 E1 半速故障的检查，不影响已有 E1 断点。

先在四卡上采集并训练一次 Stage I，再让两种 Stage II 从**同一个 E5 Stage-I adapter**
独立开始。若有八张 48 GB GPU，两种 Stage II 可以各占四卡并行运行。各训练 20 epoch，
保留每轮 adapter，但完整优化器/RNG 状态只保留最新一次。任务方向奖励使用显式 Gate-4
实验性绕过；E5 尚未通过奖励有效性门控，HQ-SAM oracle 框提示也不是可部署的分割方法。

```bash
export LIBERO_CONFIG_PATH="$PWD/AILOG/libero"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
export MUJOCO_GL=egl
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 \
  --mixed_precision bf16 scripts/resilient/collect_two_stage_opsd.py \
  --config-name two_stage_opsd/fastwam_libero10_task7_e5_periodic_freeze
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml --num_processes 4 \
  scripts/resilient/train_two_stage_stage1.py \
  --config-name two_stage_opsd/fastwam_libero10_task7_e5_periodic_freeze
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_opsd_zero2_ds.yaml --num_processes 4 \
  scripts/resilient/train_two_stage_stage2.py \
  --config-name two_stage_opsd/fastwam_libero10_task7_e5_periodic_freeze_stage2_full_20ep
CONFIG_NAME=two_stage_opsd/fastwam_libero10_task7_e5_periodic_freeze_stage2_task_direction_20ep \
TRAIN_PORT=29501 \
QUEUE_DIR="$PWD/data/.cache/task_decoupling_rpc_samhq_e5" \
LOG_DIR="$PWD/AILOG/processes/e5_task_direction_20ep" \
  bash scripts/resilient/run_task_direction_outcome_fpo.sh 4,5,6,7 4
```

`TRAIN_PORT`用于区分同时运行的全图奖励分布式作业；如果29501已被占用，应改为其他空闲本地
端口。旧实验仍默认使用29500。

采集按完整状态自动续跑。训练沿用前述两阶段严格断点机制；`resume=auto` 用于中断续训，
不用于全新实验。数据与断点分别位于 Git 忽略的
`data/generated/two_stage_opsd/libero10_task7_e5_periodic_freeze/` 和
`runs/two_stage_opsd/libero10_task7_e5_periodic_freeze_seed42/`。本实验未增加依赖或模型资产，
Fast-WAM、LIBERO 与 HQ-SAM 的环境和下载说明沿用前文。相同 50 个留出状态上的基座参照为
无 Fault 49/50、E5 21/50；它不是 Stage-II 效果。后续评测只可读取验证 manifest，
不可混入 Stage-I/Stage-II 训练状态。

## Stage II 任务区域解耦审计

可选的任务区域路径先把任务物体证据与机器人本体运动分离，再决定是否允许它影响 Stage II
目标。当前它严格属于**仅审计的特征提取器**：默认
`task_decoupling.enabled=false`；即使以 `mode=audit` 启用，也只记录特征，不改变现有
Outcome-FPO reward 或优化器。在下面的门控通过前，不能把它宣称为因果表征或 Fault 表征。

实现位于 `src/resilient/task_decoupling/`；固定任务实体与 prompt 位于
`configs/task_entities/`；mask 推理由 `services/sam3/` 下的独立 SAM 3.1 服务完成；
`src/resilient/outcome_fpo/task_features.py` 是唯一的 Stage II 接入边界。生成 mask 时保持两个
相机分离，之后严格复用已发布 Fast-WAM 的裁剪、缩放、相机顺序、拼接、归一化和冻结 VAE
路径。局部 Gaussian blur 反事实只改变任务 mask 内的像素。审计残差定义为：

```text
Delta_task = VAE(完整真实未来视频) - VAE(仅模糊任务区域的真实未来视频)
```

并排除第一个 latent slice。LIBERO instance segmentation 只作为审计 oracle，训练和部署均不可
读取。

外部资产是官方 gated
[`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1) 中的
`sam3.1_multiplex.pt`，适用官方 SAM License。文件大小为3,502,755,717字节，应放到
`checkpoints/sam3/sam3.1_multiplex.pt`，SHA-256为
`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`。源码固定为
`facebookresearch/sam3`提交`2345a4ad109ac29c569da749c91d84f10dc08c40`。权重由Git忽略，
访问申请与下载命令见`checkpoints/sam3/README.md`。

首个维护审计使用 `libero_10` task 7 与仿真器底层 joint-1 retention=0.5 Fault。它复放现有
Stage I 的训练状态 40--49：40--47 用于 probe 拟合，48--49 作为内部留出审计状态，绝不读取
正式验证状态。门控定义如下：

| Gate | 必须提供的证据 |
| --- | --- |
| 0 | 共享预处理以及复放首帧与 Stage I 输入严格一致 |
| 1 | SAM mask 与仿真器 instance mask 对比 IoU/recall、时序 IoU 和机器人污染率 |
| 2 | mask 外像素逐值相同；完整/反事实视频在同一个冻结 VAE batch 中编码 |
| 3 | SAM residual 与 oracle residual 的一致性显著高于等面积平移 mask 对照；在留出状态上分别报告任务进展、joint-1 运动 probe 与有效秩 |

先按 `checkpoints/sam3/README.md` 创建独立环境并取得带 SHA-256 校验的 gated checkpoint。
已验证服务环境使用 uv 0.11.7、CPython 3.12.13、PyTorch 2.10.0/CUDA 12.8 和
`requirements-sam3.txt`：

```bash
bash scripts/resilient/create_sam3_environment.sh .venv-sam3

CUDA_VISIBLE_DEVICES=0 .venv-sam3/bin/python services/sam3/server.py \
  --queue-dir data/.cache/task_decoupling_rpc \
  --checkpoint checkpoints/sam3/sam3.1_multiplex.pt

# 终端 2，Fast-WAM/LIBERO 环境：确定性仿真复放。
python scripts/resilient/audit_task_decoupling.py \
  task_decoupling.audit.phase=collect

# 终端 2，Fast-WAM/LIBERO 环境：冻结 VAE Gate 3 分析。
CUDA_VISIBLE_DEVICES=1 python scripts/resilient/audit_task_decoupling.py \
  task_decoupling.audit.phase=analyze
```

默认输出位于 Git 忽略的
`evaluate_results/task_decoupling/libero10_task7_joint1_half/`；SAM 请求与缓存位于 Git 忽略的
`data/.cache/task_decoupling_rpc/`。服务生命周期见 `services/sam3/README.md`，全部阈值和划分见
`configs/task_decoupling/sam3p1_local_blur.yaml`。

### 临时 HQ-SAM 审计后端

在等待 gated SAM 3.1 权限期间，可以通过同一个 `MaskProvider` 边界，用冻结的 HQ-SAM
ViT-H 跑 Gate 3。这个临时后端不具备语言定位或时序跟踪能力：它把 LIBERO instance mask
转换成逐相机、逐实体、逐帧的 bounding box，再让 HQ-SAM 细化这些框。因此它只是**oracle
提示的上界审计**，不是可部署的 Gate-1/Gate-3 结果。其 summary 中 `gate1_pass` 与
`gate3_residual_specificity_pass` 保持为 `null`，诊断结果写入单独命名的 mask-quality 和
upper-bound 字段；Stage-II runtime 会明确拒绝该 provider。

官方 [HQ-SAM 项目](https://github.com/SysCV/sam-hq) 的 ViT-H 权重可从
[官方下载地址](https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view)取得。
其许可证为 Apache-2.0，大小为 2,570,940,653 字节，SHA-256 为
`a7ac14a085326d9fa6199c8c698c4f0e7280afdbb974d2c4660ec60877b45e35`，规范位置是
`checkpoints/sam_hq/sam_hq_vit_h.pth`；完整资产约定见 `checkpoints/sam_hq/README.md`。
隔离环境使用 uv 0.11.7、CPython 3.12.13、PyTorch 2.10.0/CUDA 12.8 和官方
`segment-anything-hq==0.3`：

```bash
bash scripts/resilient/create_samhq_environment.sh .venv-samhq

CUDA_VISIBLE_DEVICES=0 .venv-samhq/bin/python services/sam_hq/server.py \
  --queue-dir data/.cache/task_decoupling_rpc_samhq \
  --checkpoint checkpoints/sam_hq/sam_hq_vit_h.pth

# 复用已经采集的 40 段 clip；在 Fast-WAM 环境中运行 VAE 分析。
CUDA_VISIBLE_DEVICES=1 python scripts/resilient/audit_task_decoupling.py \
  task_decoupling=samhq_oracle_box_local_blur
```

Git 忽略的输出位于
`evaluate_results/task_decoupling/libero10_task7_joint1_half_samhq_oracle_box/`。后续替换回
SAM3.1 时只需选择 `task_decoupling=sam3p1_local_blur`；反事实、VAE 与指标实现完全复用。

### Gate 4：任务区域解耦 outcome reward 审计

Gate 4 只是离线有效性审计，**不会**修改 Outcome-FPO 的训练 reward。它构造40个同状态组
（训练状态30--39，anchor为0/80/160/240）；Stage-I Student 对每组采样4个action chunk，并从
完全相同的仿真器/Fault状态执行。冻结base Teacher使用相同Video DiT层、噪声和timestep，分别
评价真实完整视频latent及其HQ-SAM任务区域反事实latent。输出包含全图reward、反事实reward、
二者margin，以及预先固定的task-direction reward。候选排序与“物体到篮子的距离改善”比较，
joint-1运动量作为干扰变量单独审计。状态30--34为开发集，35--39为未查看的内部留出集；不会
读取正式验证状态。

临时HQ-SAM路径有意采用单侧分解：只有真实候选被分解，因为box-prompt后端无法为Teacher预测
视频取得仿真器框。因此即使通过，也只能视为oracle提示下的reward上界，不能直接开启正式
Stage-II reward。在SAM3.1及Teacher target分解均验证前，必须保持
`task_decoupling.enabled=false`。

请使用统一管理生命周期的启动器，第一个参数是分割GPU，第二个参数是严格4张审计GPU。无论
审计正常结束、报错还是被中断，它都会自动停止HQ-SAM服务和全部审计worker，避免残留显存：

```bash
bash scripts/resilient/run_task_reward_audit.sh 0 1,2,3,4
```

默认输出到Git忽略的
`evaluate_results/task_decoupling/gate4_libero10_task7_joint1_half_samhq_oracle_box/`。固定划分、
seed、第19层、timestep 500、group size 4、指标和通过阈值均在
`configs/task_reward_audit/samhq_libero10_task7_joint1_half.yaml`中。

Gate 4 未达到预先固定的0.60排序阈值（留出集task-direction pairwise accuracy为0.5526）。
所以下列命令是明确的实验性绕过，而不是已验证默认方案。它从Stage-I adapter开始，用privileged
oracle-box HQ-SAM task-direction reward在4卡上训练一个Stage-II epoch。HQ-SAM与第一张训练卡
共享GPU，生命周期启动器保证训练结束或异常时释放服务和全部worker：

```bash
bash scripts/resilient/run_task_direction_outcome_fpo.sh 0,1,2,3 0
```

固定输出为
`runs/two_stage_opsd/libero10_task7_joint1_half_seed42/stage2_task_direction_gate4_override_epoch1/`。
现有所有配置仍默认使用原始全图reward。

下面的显式实验性绕过会从同一 Stage-I adapter 出发，在物理GPU 0--3上将相同的
task-direction实验扩展到10 epoch。它保留每个epoch adapter，但只保留最新一份完整
Accelerate/ZeRO恢复状态：

```bash
CONFIG_NAME=two_stage_opsd/fastwam_libero10_task7_joint1_half_task_direction_10ep \
LOG_DIR=AILOG/processes/task_direction_outcome_fpo_10ep \
  bash scripts/resilient/run_task_direction_outcome_fpo.sh 0,1,2,3 0
```

输出为
`runs/two_stage_opsd/libero10_task7_joint1_half_seed42/stage2_task_direction_10ep/`。该实验仍是
Gate-4实验性绕过；增加epoch不会改变上文记录的reward有效性未通过状态。

若要把同一条task-direction实验从epoch 10严格续到epoch 20，使用续训配置和HQ-SAM生命周期
启动器。第二个参数须属于四张训练卡之一；必须恢复第10轮完整优化器/RNG状态，仅加载第10轮
推理adapter会变成另一项实验。

```bash
CONFIG_NAME=two_stage_opsd/fastwam_libero10_task7_joint1_half_task_direction_continue20 \
LOG_DIR=AILOG/processes/task_direction_outcome_fpo_continue20 \
  bash scripts/resilient/run_task_direction_outcome_fpo.sh 4,5,6,7 4
```

输出目录仍是`stage2_task_direction_10ep/`，因为其中包含最初十轮。续训并不意味着此前未通过的
Gate-4奖励有效性验证现在通过。

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
| `EVALUATION.fault_detection.enabled` | `false` | 在 LIBERO 评测中采集严格对齐的 Video DiT/VAE latent 指标 | 关闭时不执行联合视频推理、真实 latent 编码或指标输出 |
| `FastWAM.infer_joint` 的 `return_video_latents` / `decode_video` | `false` / `true` | 暴露最终视频 latent，并允许 Fault 指标跳过图像解码 | 默认值保持原来的 `video`/`action` 返回与解码行为 |
| `task_decoupling.enabled` | `false` | 在 Stage II rollout 边界执行 SAM 任务 mask 反事实特征提取 | 关闭时无服务调用、VAE 编码、元数据、reward 或优化器变化 |

## 开发检查

轻量 CI 与完整 GPU 环境分离：

```bash
python -m pip install -r requirements-ci.txt
PYTHONPATH=src python -m pytest -m "not gpu and not libero"
ruff check src/resilient tests scripts/resilient services
```

一次性调试脚本和输出使用后必须删除。可长期复用的 GPU/LIBERO 检查放入 `tests/integration/` 并添加明确标记。

## 可复现性约定

- 只使用仓库相对路径或用户传入路径，禁止提交机器专属的硬编码绝对路径。
- 记录 Git SHA、上游 SHA、命令、Hydra 覆盖参数、随机种子、硬件、环境、资产版本/校验值和结果。
- 依赖变化时同步更新环境文件和中英文 README。
- 数据集、checkpoint、仿真资产、缓存、视频、日志和原始实验结果不得上传 Git。
- Hydra 作业只向标准输出写日志，因此不会在仓库根目录重复生成 `*.log`。需要长期保留的控制台日志应重定向到已忽略的 `AILOG/` 路径或文档指定的单次运行结果目录。
- 源码和代码注释使用英文；`README.md` 使用英文；`README.zh-CN.md` 使用简体中文；被忽略的 `AILOG/WORKLOG.md` 使用中文。

## 上游与许可证

FastWAM 来自 <https://github.com/yuantianyuan01/FastWAM>，保留 MIT License 和完整 Git 历史。来源信息见 `LICENSE` 与 `manifests/upstream.json`。
