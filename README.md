# Resilient

`Resilient` 是一个面向可复现实验与工程开发的 Python 项目骨架。当前版本只提供项目规范、最小可运行包、测试和持续集成；尚未加入具体算法、数据集或预训练权重。

## 快速开始

以下命令均从仓库根目录执行，不依赖任何机器上的绝对路径。

```bash
git clone https://github.com/Morimorerain/Resilient.git
cd Resilient

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==24.0
python -m pip install -r requirements-dev.txt

python -m resilient --version
python -m unittest discover -s tests -v
ruff check .
```

Windows PowerShell 用户可将激活命令替换为：

```powershell
.venv\Scripts\Activate.ps1
```

如果系统没有 Python 3.10.8，可使用 `pyenv` 安装 `.python-version` 中指定的版本。项目要求 Python `>=3.10,<3.11`；本仓库的基线验证版本为 3.10.8。

## 环境与依赖

- 运行时依赖记录在 `requirements.txt`。
- 开发、测试和构建依赖记录在 `requirements-dev.txt`，并固定精确版本。
- 包元数据和工具配置记录在 `pyproject.toml`。
- 修改 Python 版本或依赖时，必须在同一次提交中同步修改相应依赖文件、本节和下方“复现记录”。
- 不要把 `.venv/`、Conda 环境或本机缓存提交到仓库。

## 硬件与系统条件

当前最小功能是 CPU-only，不需要 GPU：

| 项目 | 最低建议 | 本次基线验证环境 |
| --- | --- | --- |
| 操作系统 | Linux、macOS 或 Windows | Ubuntu/Linux 5.4，x86_64 |
| Python | 3.10.8 | CPython 3.10.8 |
| CPU | 1 核 | AMD EPYC 7542 |
| 内存 | 256 MB 可用内存 | 503 GiB 系统内存 |
| GPU | 不需要 | NVIDIA RTX 6000 Ada 48 GB（未被基础测试使用） |

未来若代码依赖 CUDA、特定 GPU 显存或其他硬件，必须同步更新此表，并记录 CUDA、驱动、框架及硬件型号。

## 数据集与模型权重

当前基础版本不需要任何数据集或模型权重，因此暂无下载地址。后续引入外部资产时，必须先更新下表、`data/README.md` 或 `models/README.md`，再提交使用这些资产的代码。

| 资产 | 仓库内目标路径 | 下载地址 | 完整性校验 | 当前状态 |
| --- | --- | --- | --- | --- |
| 数据集 | `data/raw/<dataset_name>/` | 尚未使用 | 必须补充 SHA-256 | 未引入 |
| 处理后数据 | `data/processed/<dataset_name>/` | 由脚本生成 | 必须记录生成命令 | 未引入 |
| 模型权重 | `models/<model_name>/` | 尚未使用 | 必须补充 SHA-256 | 未引入 |

`data/` 和 `models/` 中的实际内容已被 `.gitignore` 屏蔽，只会提交说明文件和空目录占位符。禁止将数据集、权重、检查点或大体积实验产物直接提交到 Git。

## 目录结构

```text
Resilient/
├── .github/workflows/ci.yml   # 自动化质量检查
├── AILOG/                    # 本机工作日志与临时文件，不上传
├── data/                     # 数据目录；实际数据不上传
│   ├── raw/
│   └── processed/
├── models/                   # 模型权重目录；实际权重不上传
├── src/resilient/            # 项目源码
├── tests/                    # 可长期维护的自动化测试
├── AGENTS.md                 # 后续自动化开发约束
├── pyproject.toml            # Python 包与工具配置
├── requirements.txt          # 运行时依赖
└── requirements-dev.txt      # 开发依赖
```

运行产生的结果统一写入 `outputs/`、`runs/` 或 `logs/`。这些目录不会上传。一次性烟测脚本、烟测数据和烟测输出应在验证完成后删除；可复用的行为应改写为 `tests/` 下的正式测试。

## 复现约定

1. 路径必须相对仓库根目录、当前配置文件或通过命令行参数传入；禁止提交开发者机器的硬编码绝对路径。
2. 随机实验必须显式记录随机种子；涉及非确定性算子时，还要记录框架的确定性设置。
3. 每个实验应记录代码提交 SHA、命令、配置、环境版本、输入资产校验值和输出路径。
4. 新增或升级依赖时同步更新依赖文件与 README；不得只修改本机环境。
5. 新增数据或权重时记录官方下载地址、许可证、目标路径、解压结构和 SHA-256。
6. 提交前运行测试与静态检查，并清理临时脚本、缓存和烟测产物。

## 常用命令

```bash
# 运行最小入口
python -m resilient --version

# 运行测试
python -m unittest discover -s tests -v

# 静态检查
ruff check .

# 清理常见本地产物
make clean
```

## 复现记录

| 日期 | 提交/版本 | Python | 依赖 | 资产 | 验证结果 |
| --- | --- | --- | --- | --- | --- |
| 2026-09-03 | 初始化骨架 | 3.10.8 | 见 `requirements*.txt` | 无 | CLI、单元测试、Ruff |

## 许可证

项目许可证尚未确定。在添加或分发第三方代码、数据集或模型前，请先确认并记录其许可证兼容性。
