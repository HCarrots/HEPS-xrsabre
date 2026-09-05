# xrs_compton_extraction：Agent 上下文压缩报告

更新时间：2026-09-05  
项目目录：`C:\Users\a1566\Documents\Work\python\xrsabre\xrs-compton-extraction`

这份文档用于把当前任务交给新的 agent。它是状态报告，不等同于科学结果发布声明。

## 模组目标区域平均展示更新

- 用户将Notebook更名为 `notebooks/Compton_Analysis.ipynb` 并修改了说明文字；继续沿用用户版本，新增6.1节，不恢复旧文件名。
- 新增 `module_averages.py`，读取当次成功的逐通道导出，高q residual除以保存的raw_to_hf_scale，还原到输入任意强度单位，再与低q结果作描述性平均。
- 目标显示区默认111–211 eV（TARGET_VIEW=EDGE）。每模组内晶体等权；总体主曲线为模组均值等权，另给全部晶体等权对照。负值保留，缺失点不补0，不推断统计误差。
- 实际参与数：VB14、VU13、HB9、HL7、VD7，共50；HB-E1/E2全零排除。并未完成探测效率或绝对强度校准。
- 图/CSV/元数据保存在 `output/ho-b4-hf-matching/real/module-averages`；之前Notebook备份为before-module-averages.ipynb。全量230 passed，新增单位转换、两种权重、缺失点和异常尺度测试。
- 当前用户参数重跑发现HL-E1/HL-E2在约139–140 eV的Pearson扣除出现百万量级负尖峰，主导总平均。未改拟合、未删除通道；目标残差峰值超过输入峰值10倍时显示诊断提示，并增加symlog总览图。需要后续检查拟合约束，不能把当前总平均当最终谱。

## HF 归一化/目标保留迁移更新（优先于以下历史记录）

- 已按用户授权迁入 XRStools 的方法思路，独立实现 `hf_workflow.py`，没有导入 XRStools、xraylib 或 xrs_processing。
- Ho N4 明确为本地表 Shell_13，4电子，164 eV；之前161 eV对应N5/Shell_14，应按历史误记处理。实验轴不移动。
- HF能量谱结合能以下置零，在独立有限表格支持上做一阶矩 f-sum，不使用实验802.4 eV终点作为理论归一化上限。
- 通过同窗口HF/实验面积比预匹配，再拟合正HF比例+线性背景，替代手填reference_scale/reference_constant。最终只扣其他芯层+价电子+线性背景，保留N4原子连续谱。
- Notebook顶部显式采用 binding<20 eV 的探索价电子划分（边界20归芯层），HoB4对应25电子；这不是实验确认的固态价电子数。保护111–211，拟合20–80/230–700，预匹配20–700同时避开保护区。
- 运行结果：36 Pearson，14高q HF，2全零排除；参考VB-A2，raw-to-HF比例约4.51731e-6，有限支持归一化约1.48972。平滑轮廓1288个负点保留，是模型/窗口警示，不是最终科学结果。
- 输出改为 `output/ho-b4-hf-matching`，执行副本executed.ipynb、旧Notebook备份before-migration.ipynb；提交源Notebook输出清空。旧output/ho-b4-notebook仍保留。
- 全量227 passed（设置XRS_HO_TEST_DATA），ruff通过；Notebook在禁用三个旧包的kernel中执行完成，已检查真实HF匹配、价电子、N4保留及合成多q图。
- API与剩余物理限制见 `docs/hf-matching.md`。模型拟合残差单独存为model_fit_residual，输出residual是保留目标的响应；高q单位HF模型密度/eV，低q仍为任意单位。

## 本地 DABAX 数据源更新（优先于以下历史记录）

- 用户要求不再使用 xraylib。当前 HoB4 Notebook 的真实与合成流程均改为 `DabaxProfileSource`，直接读取项目根目录下 `resources/compton_profiles/ComptonProfiles.dat`。
- 文件日期 2003-01-29；SHA256 为 `db42c6cefad0916886e96acf03b0186b538d01dbde2988d7bdec4c8675d814ba`。不改写原表、不自动下载、不打包外部原子表。
- 占据数直接读取 #UOCCUP：B=2+2+1=5，Ho=67；先前 B=4.333333 是旧后端的历史结果，已不适用于当前流程。
- 保留 Shell_N 原生列名并展示 #UBIND 结合能，不猜 N4 映射。部分轮廓按每电子解释，占据数/化学计量只乘一次；|pz| 线性插值，拒绝外推。
- 本地表有限网格上分壳层加和与 total 相对 L2 差异：Ho 约 0.00271，B 约 0.00086。原子表/目标壳层映射确认和实验比例等参数仍显式保留，不能据占据数一致就生成最终边。
- 新增本地表解析、重复/缺项/维度检查、占据数权重、目标排除和禁用 xraylib 的回归测试。当前设置 Ho 环境变量全量 221 passed。

## 2026-09-05 Notebook 实施更新（优先于下文历史状态）

- 新增 `notebooks/HoB4_Compton_Analysis.ipynb`：中文逐步分析，无工作台 UI 依赖；参数集中在开头。
- 新增 `exploratory.py`：按晶体标签关联输入、原子表审计、参考评分、pz 符号适配、分阶段轮廓和无权探索拟合；保留现有加权接口。
- 样品按 Ho:B=1:4。原能量轴不移动，暂定 N4，保护 111–211 eV；q_ave 作为显式固定 q 近似，9 Å⁻¹ 只作诊断分组。
- 实际运行：52 通道全部保留；36 通道生成 Pearson 探索结果，14 高 q 通道因 HF 输入未齐暂停，HB-E1/HB-E2 标记全零且排除拟合。没有生成最终 Ho N4 科学结果。
- xraylib 4.2.1 的 ElectronConfig 可用壳层占据数：Ho 合计 67，B 合计 4.333333；加权分壳层与总轮廓相对 L2 差异约 0.0030、0.1405。没有用猜测值修补，原子表/占据数核对仍未通过。
- Ho/B 壳层划分、价电子数、目标排除壳层、参考强度比例及常数仍为显式缺项；核对通过记录也必需。真实 HF 步骤在缺参时暂停；合成含氧示例独立跑通 q=5、6、8 a.u.。
- 输出在 `output/ho-b4-notebook/real` 和 `synthetic`；执行副本为 `output/ho-b4-notebook/executed.ipynb`。提交 Notebook 的执行计数及输出保持为空。
- 该 Notebook 的无权结果不导出约化 χ² 或参数协方差；缺失支持为 NaN 加 availability 标记，负结果保留，原始输入哈希未改变。
- 新测试覆盖坐标反射、非对称输入、掩罩、积分、支持范围、误差语义和晶体关联。设置 Ho 环境变量时当前全量 **216 passed**，ruff 通过；Notebook 已完整执行并检查诊断及合成图。

## 1. 项目身份与边界

- 项目显示名：`xrs-compton-extraction`。
- Python 包名：`xrs_compton_extraction`。
- 代码位于 `xrsabre` 工作区，但必须独立于 `xrs_processing`；不得导入、复用或隐式兼容 `xrs_processing` 的对象和算法，该程序是xrsabre报的一个子项。
- 可以继续使用同一 `xrsabre` 工作区中的数据、Pixi 环境和开发工具。
- 当前版本：`0.1.0.dev0`，开发版，尚未达到真实实验生产验证状态。
- 不要擅自选择项目许可证、打包外部 HF 表格或声明实验科学结论。

## 2. 已完成任务

### 软件与数据模型

- `src/xrs_compton_extraction/data.py`：不可变数组、样品/扫描/通道、校正/提取/质量报告模型。
- `config.py`：YAML/JSON 配置往返。
- `geometry.py`、`constants.py`：能量转移、q、pz、原子单位换算。
- `synthetic.py`：确定性合成数据。
- 包边界测试确认不会导入 `xrs_processing`。

### 输入输出与校正

- NeXus/HDF5：`io/nexus.py`。
- 严格 CSV/TSV：`io/text.py`。
- 宽表多通道读取：`load_text_channels(path, mappings)`；每个信号列必须有显式唯一通道标签。
- `XRSWorkbench.load_text(..., mappings=...)` 支持宽表。
- 归一化、弹性/杂散、吸收、自吸收和截面校正都有显式参数、顺序和误差传播。
- 已处理强度标记为 `intensity_kind="processed"` 时，没有显式误差就拒绝自动伪造泊松误差。
- 输出 CSV、metadata、manifest、Markdown 报告和绘图均已实现。

### Compton / XRS 计算路径

- 低 q Pearson 背景拟合，支持边界、权重、窗口、残差和协方差诊断。
- HF core profile：`backgrounds/core_profile.py`。
  - 可选 xraylib 后端：`xraylib>=4.2,<5`。
  - 核壳层必须逐元素显式提供；目标壳层可排除。
  - 默认不会假装 xraylib Python API 提供 `ElectronConfig_Biggs`；4.2.1 中使用 `ElectronConfig` 必须显式 opt-in，并记录 provenance。
  - 不在包内拷贝外部原子表；数据许可证仍需审核。
- 经验价电子 profile：`backgrounds/valence_profile.py`。
  - 参考谱选择、污染 mask、线性填补、对称化、一阶非对称、平滑、有限支持电子数归一化和 q 转移已实现。
  - 未观测尾部不会被估计。
- `profile_pipeline.py`：core + valence 模板共同非负 scale，可选常数项，在显式窗口中拟合并保留负提取值。
- `batch.py`：成功、失败、排除通道分开保存，禁止静默 fallback。
- `multi_q.py`：多 q 对齐、比较、加权平均和图表。
- `q_groups.py`：
  - q < 9 Å⁻¹：`low_q`
  - q > 9 Å⁻¹：`mid_high_q`
  - q = 9 Å⁻¹：`boundary`，不静默归类。

### Jupyter 工作台

- `workbench/app.py` 有 Data、Correction、Background、Extraction、Results 五个页面。
- Pearson 单通道/批处理、配置保存、结果导出和状态日志可交互使用。
- Compton profile、宽表和高级校正可通过 controller Python API 使用；完整高级参数 GUI 尚未完成。
- `notebooks/XRS_Workbench.ipynb` 只保留 UI 入口，已清除执行输出。

### 测试、构建和文档

- 运行时使用 Pixi：

  ```powershell
  & 'C:\Users\a1566\AppData\Local\pixi\bin\pixi.exe' run python -m pytest tests -q
  & 'C:\Users\a1566\AppData\Local\pixi\bin\pixi.exe' run python -m ruff check src tests examples
  ```

- 当前本地完整测试（设置 Ho 环境变量）为：**209 passed**。
- 未设置 Ho 环境变量时：**208 passed, 1 skipped**；这是预期行为，不是失败。
- 源码包和 wheel 已构建并做过隔离导入 smoke test。
- 重要文档：
  - `README.md`
  - `docs/MVP_STATUS.md`
  - `docs/compton-profile-data.md`
  - `docs/ho-data-validation.md`
  - `CHANGELOG.md`

## 3. 已检查的 Ho N4 测试数据

主文件：

`C:\Users\a1566\Documents\Work\python\xrsabre\workspace\Ho\processed\Ho_Comptonscan_standard\Ho_Comptonscan_standard_all_data.txt`

配套文件：

- `Ho_Comptonscan_standard_fit_results.txt`
- `Ho_Comptonscan_standard_run_info.json`

输入事实：

- 52 个通道，4021 个能量点。
- 能量转移范围 −1.6～802.4 eV，步长 0.2 eV。
- 当前表格没有非有限值和负强度。
- `HB-E1`、`HB-E2` 全为零，必须保留并标记，不能静默删除。
- run-info 记录上游已做 I0 归一化、滤波和插值。
- 表格没有传播后的不确定度或协方差。
- fit-results 中的 `q_ave` 是 Å⁻¹ 汇总值，不是能量分辨 q 校准。
- q 汇总范围约为 0.987744～9.859149 Å⁻¹，即约 0.522692～5.217237 a.u.。

已生成：

- `output/ho-input-check/inspection.md`
- `output/ho-input-check/inspection.json`
- `output/ho-input-check/overview.png`
- `output/ho-n4-window-scan/window-scan.json`

## 4. Ho N4 窗口调试结论

- xraylib 的 Ho N4 理论参考能量约 161 eV，但实验能量转移轴在约 112～130 eV 有明显结构。
- 这说明能量校准/参考能量偏移必须先核实；不能把 161 eV 直接当作本数据的提取边中心。
- 弹性峰集中在 0 eV 附近，拟合窗口不能从 −1.6 eV 开始。
- 当前诊断默认排除低于 20 eV 的弹性区，并暂时排除 111～211 eV 作为结构区。
- 低 q 的探索性 Pearson 候选起点：`(20, 80)` 和 `(230, 700)` eV。
- q > 9 Å⁻¹ 的中高 q 汇总谱用 Pearson 会产生结构性巨大残差（约化 χ² 约 5.9×10⁵），不能与低 q 共用 Pearson；应走 HF core + 价电子模板。
- 以上只是窗口诊断，不是 N4 边提取结果，也没有统计置信区间。

复现命令：

```powershell
cd C:\Users\a1566\Documents\Work\python\xrsabre\xrs-compton-extraction
& 'C:\Users\a1566\AppData\Local\pixi\bin\pixi.exe' run python examples\debug_ho_n4_windows.py `
  ..\workspace\Ho\processed\Ho_Comptonscan_standard\Ho_Comptonscan_standard_all_data.txt `
  --fit-results ..\workspace\Ho\processed\Ho_Comptonscan_standard\Ho_Comptonscan_standard_fit_results.txt `
  --output output\ho-n4-window-scan
```

## 5. 尚未完成的任务

这些任务不能被当前报告误标记为“已完成”：

1. **Ho N4 的最终能量校准和目标窗口**：需确认实验参考能量、弹性中心、N4 起止范围和是否存在轴偏移。
2. **Ho 真实边提取**：尚未提供 Ho 样品组成、目标 shell 排除方案、core/valence profile 输入和最终拟合参数，因此尚未生成可信 N4 边。
3. **误差模型**：当前宽表没有误差/协方差；不能声称统计误差、置信区间或严格加权 χ²。需要上游传播误差，或明确标记为探索性无权拟合。
4. **q 的精确定义**：fit-results 只有通道 q 平均值；需确认 q 是否随能量变化、q 的单位/转换和 q>9 的物理分组是否确实按 Å⁻¹。
5. **模型/窗口/平滑 ensemble**：尚未自动运行多模型敏感性实验，也未实现多个 q 之间共享 reference uncertainty/covariance。

## 6. 后续 agent 的执行规则

- 先读本报告、`README.md`、`docs/ho-data-validation.md` 和相关模块测试，再修改代码。
- 所有物理语义必须显式传入；不猜 q、能量种类、归一化状态、误差或窗口。
- 不改写 Ho 原始数据，不把私有数据复制进包，不导入 `xrs_processing`。
- 若缺少目标边/误差/校准信息，只做审计或明确标记的探索性拟合，不生成“最终科学结果”。
- 每次修改后至少运行 ruff、全量 pytest；涉及 Ho 时设置 `XRS_HO_TEST_DATA` 做真实输入回归。
- 修改 Notebook 后确保所有 code cell 的 `execution_count` 为 null、`outputs` 为空。
- 用 `apply_patch` 编辑文件；不要用 destructive git 命令。

- 不使用UI，而是用jupyter，逐步执行代码，尽量不依赖与其他ui代码库。
- 
## 7.本软件的数据处理流程
本软件的数据处理流如下，不要引入过多复杂的语法代码，专注数据处理逻辑。
我将再次详细告诉你数据处理的物理逻辑应该如何做，请把这个作为一个重要目标

第一步：构建芯层 Hartree-Fock 轮廓

对于样品中每一种元素和每一个电子壳层，读取对应的 Hartree-Fock 康普顿轮廓：

$$ J_{\mathrm{core}}(p_z) $$

总芯层轮廓为：

$$ J_{\mathrm{core,total}}(p_z) = \sum_i n_iJ_i(p_z) $$

其中：

\(i\)：元素或电子壳层；
\(n_i\)：该壳层电子数或化学计量权重；
\(J_i(p_z)\)：对应的原子康普顿轮廓。

需要注意：

Hartree-Fock 轮廓主要用于描述远离目标吸收边的原子背景，不能把目标吸收边附近的真实 XRS 精细结构当作 HF 轮廓扣除掉。

因此，目标边附近应设置遮罩区：

edge_mask = (
    energy_loss_eV > edge_energy - 20
) & (
    energy_loss_eV < edge_energy + 50
)

具体范围根据能量分辨率和目标边宽度调整。

四、第二步：选择高 \(q\) 参考谱

如果有多个 \(q\) 数据，应选择最高 \(q\) 或较高 \(q\) 的谱作为价电子康普顿轮廓参考。

选择标准：

\(q\) 较大；
价电子康普顿峰已经明显形成；
目标芯层边没有严重遮挡价电子背景；
统计质量较好；
扣除芯层后仍有足够的高能量损失尾部。

例如：

q_ref = max(q_values)

但不要简单使用最大 \(q\)，应通过自动评分选择：

score = (
    high_q_weight
    + clean_tail_weight
    + signal_to_noise_weight
    - edge_overlap_penalty
)
五、第三步：将参考谱转换到 \(p_z\) 空间

冲量近似下：

$$ p_z=\frac{q}{2}-\frac{\omega}{q} $$

其中 \(q\) 和 \(\omega\) 必须采用一致的原子单位。

如果能量使用 eV，转换时应明确使用：

$$ p_z= \frac{q}{2} - \frac{\omega}{q} $$

这一公式在论文中采用原子单位。

对于每个能量点计算：

pz = q_au / 2.0 - energy_loss_au / q_au

然后将实验谱从：

$$ S(q,\omega) $$

转换为：

$$ J_{\mathrm{exp}}(p_z) $$

在 Sahle 等人的定义下：

$$ J(p_z)=N_{\mathrm{el}}qS(q,\omega) $$

因此：

$$ J_{\mathrm{exp}}(p_z) = N_{\mathrm{el}}qS_{\mathrm{exp}}(q,\omega) $$

如果你的数据还不是绝对归一化的 \(S(q,\omega)\)，则不能直接使用这个公式，需要先引入强度比例因子。

六、第四步：提取价电子康普顿轮廓

高 \(q\) 参考谱中：

$$ J_{\mathrm{exp}} = J_{\mathrm{valence}} + J_{\mathrm{core}} + J_{\mathrm{stray}} $$

因此：

$$ J_{\mathrm{valence}}(p_z) = J_{\mathrm{exp}}(p_z) - J_{\mathrm{core}}(p_z) - B(p_z) $$

其中 \(B(p_z)\) 是残余常数或线性背景。

建议第一版只使用：

$$ B(p_z)=b_0+b_1p_z $$

拟合参数为：

$$ b_0,\quad b_1,\quad C $$

其中 \(C\) 是整体强度比例因子。

拟合目标：

$$ \chi^2= \sum_{p_z\in\Omega} \frac{ \left[ J_{\mathrm{exp}}(p_z) - C J_{\mathrm{core}}(p_z) - B(p_z) - J_{\mathrm{valence}}(p_z) \right]^2 }{ \sigma^2(p_z) } $$

但在实际第一版中，最稳妥的做法是：

先用 HF 芯层轮廓扣除；
在没有目标边遮挡的区域确定常数背景；
提取剩余价电子轮廓；
对其进行对称化和平滑。
七、第五步：对价电子 CP 进行对称化和平滑

实验得到的价电子康普顿轮廓可能存在不对称性：

$$ J_{\mathrm{valence}}(p_z) \neq J_{\mathrm{valence}}(-p_z) $$

建议分为两个结果保存：

J_valence_raw
J_valence_sym
J_valence_corrected

首先得到对称部分：

$$ J_{\mathrm{sym}}(p_z) = \frac{ J(p_z)+J(-p_z) }{2} $$

然后使用高斯卷积进行平滑：

$$ J_{\mathrm{smooth}} = J_{\mathrm{sym}}*G_\sigma $$

不建议一开始就加入复杂的不对称函数。可以把 Sternemann 的经验函数作为可选功能：

$$ A(p_z) = \alpha_1 \tanh\left(\frac{p_z}{\alpha_2}\right) \exp\left[ -\left(\frac{p_z}{\alpha_3}\right)^4 \right] $$

第一版可以设置：

valence_cp:
  symmetrize: true
  smoothing: gaussian
  gaussian_sigma: 0.05
  fit_asymmetry: false

等基本流程稳定后，再启用 fit_asymmetry。

八、第六步：将 CP 映射到每一个 \(q\)

得到统一的：

$$ J_{\mathrm{valence}}(p_z) $$

之后，对任意 \(q_i\) 和能量损失 \(\omega\)，重新计算：

$$ p_z(q_i,\omega) = \frac{q_i}{2} - \frac{\omega}{q_i} $$

再通过插值得到：

$$ J_{\mathrm{valence}}(p_z(q_i,\omega)) $$

最终转换回动态结构因子：

$$ S_{\mathrm{Compton}}(q_i,\omega) = \frac{ J_{\mathrm{valence}}(p_z) }{ N_{\mathrm{valence}}q_i } $$

这一步是软件最核心的映射过程。

不能采用以下错误方法：

把 q_ref 下的康普顿峰沿 energy_loss 轴直接平移

必须采用：

energy_loss → pz → Jvalence(pz) → 每个q下的energy_loss
九、不同 \(q\) 区域的处理逻辑
1. 高 \(q\)

使用实验提取的价电子 CP：

$$ S_{\mathrm{Compton}} = \frac{J_{\mathrm{valence}}(p_z)}{N_{\mathrm{valence}}q} $$

适合：

康普顿背景明显；
目标边与康普顿峰分离；
芯层边提取。
2. 中等 \(q\)

可以继续使用实验 CP，但需要加入模型误差：

S_compton = measured_CP + empirical_correction

中等 \(q\) 下，价电子响应不完全符合冲量近似，目标边附近可能产生系统误差。

3. 低 \(q\)

低 \(q\) 下不建议使用康普顿轮廓扣除，因为背景主要是：

等离激元；
粒子-空穴激发；
集体电子响应；
电子关联效应。

此时应采用 Pearson 或线性背景：

$$ B(\omega)= A \left[ 1+ \left( \frac{\omega-\omega_0}{\Gamma} \right)^2 \right]^{-n} $$

建议软件自动切换：

if q < q_low:
    model = "pearson"
elif q < q_high:
    model = "hybrid"
else:
    model = "measured_compton"

对于 Si \(L\) 边，论文给出的示例是：

q < 2.4 a.u.       Pearson
2.4–3.2 a.u.       hybrid
q > 4.2 a.u.       Compton profile

这些边界应作为默认值，而不是固定物理常数。

十、最终扣除公式

对于高、中 \(q\)：

$$ S_{\mathrm{XRS,target}}(q,\omega) = S_{\mathrm{total}}(q,\omega) - S_{\mathrm{Compton}}(q,\omega) - S_{\mathrm{other\,core}}(q,\omega) - B_{\mathrm{stray}}(q,\omega) $$

如果当前软件只处理一个目标边，可以先实现：

$$ S_{\mathrm{XRS,target}} = S_{\mathrm{total}} - S_{\mathrm{Compton}} - B_{\mathrm{stray}} $$

后续再加入其他芯层轮廓。
## 8.你所测试的数据信息
样品：四硼化钬
各分析晶体信息
