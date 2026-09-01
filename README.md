# HEPS-xrsabre

面向 HEPS ID33 的统一 X 射线拉曼散射工作台，整合 XRSana 的预测、精细谱
分析和数据浏览能力，以及 XRSlab 的 NXS、ROI、QC 与可追溯导出流程。

## Pixi 部署与唯一入口

```powershell
pixi install
pixi run lab
```

`pixi run lab` 是软件唯一入口。它加载当前 beamtime 的 `xrsabre.toml`，然后把
JupyterLab 根目录固定到配置的 `paths.notebooks`。也可以直接执行：

```powershell
python -m xrsabre
```

不再提供 pipeline、预测、浏览器或传统分析 shell 的命令行入口；这些能力作为
Python API 在 Notebook 中调用。

## 实验工作区

每个 beamtime 使用一份独立的 `xrsabre.toml`：

```toml
schema_version = 1

[workspace]
name = "GID33-260901"

[paths]
raw = "workspace/data/raw"
processed = "workspace/data/processed"
roi = "workspace/data/ROI"
planning = "workspace/planning"
reduced = "workspace/reduced"
notebooks = "workspace/scripts/xrs_script"
scripts = "workspace/scripts"
diagnostics = "workspace/diagnostics"
```

相对路径始终相对 TOML 所在目录解析。Jupyter 启动时按以下顺序查找配置：

1. `XRSABRE_CONFIG`；
2. 从当前目录向父目录查找 `xrsabre.toml`。

Notebook 中也可以显式加载另一份配置：

```python
from xrsabre.paths import check_workspace, load_workspace

workspace = load_workspace("D:/beamtime/GID33-260901/xrsabre.toml")
for check in check_workspace(workspace):
    print(check.level, check.key, check.path, check.message)
```

旧的 `XRSLAB_*`、`XRSA_*` 环境变量会触发迁移错误，不会被静默读取。

## Notebook 工作流

主工作台为 `workspace/scripts/xrs_script/XRS_DataAnalysis.ipynb`，通过以下 API
完成分析：

```python
from xrsabre.paths import load_workspace
from xrslab.workflow import AnalysisConfig, build_qc_report, prepare_analysis

workspace = load_workspace()
config = AnalysisConfig(
    element="Ho",
    elastic_scan_ids=(57,),
    xrs_scan_ids=(59,),
)
prepared = prepare_analysis(config, workspace)
qc = build_qc_report(prepared)
```

`WorkspacePaths` 是不可变强类型对象。pipeline、ROI 编辑器、预测、浏览器和
导出函数都接收该对象；正式导出的 provenance 会记录工作区名称、配置路径、
配置 SHA-256 和规范化后的绝对路径。

`reduced` 中的数据集会被递归发现并使用相对路径作为 ID。Notebook 可导入
`xrsabre.datasets` 和 `xrsana.data_browser` 进行选择与可视化。内置原子数据库
和 ComptonProfiles 使用 Python 包资源，不属于实验路径配置。

## 测试

```powershell
pixi run test-jupyter
pixi run check
```

Jupyter 测试会验证唯一启动入口、工作区根目录、全部 Notebook 的 nbformat 和
Python 语法、无保存输出/绝对路径，以及在真实 Jupyter kernel 中执行工作区
初始化单元。其余测试覆盖科学计算、ROI 编辑、路径安全和正式导出。
