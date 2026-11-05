# 灾前灾后 Agent 搜查路线规划

灾前先验／线索与沿途观测证据组成双层地图，研究移动、观察、规划及返程共同预算下的搜查决策。

**当前尚未建立新方法贡献。** 本仓库保留代码、负结果、真实影像访问诊断和研究交接状态，避免把工程完成度当成算法创新。

## 接手入口

1. [研究目标与最新执行决定](work/disaster_route/研究目标与接手计划.md)
2. [初学者学习与实验路线](work/disaster_route/在线上下文反馈_学习与实验路线.md)／[第一课](work/disaster_route/在线上下文反馈_第一课.md)
3. [项目进展索引](work/disaster_route/README.md)
4. [NOAA实际数据入口审查](output/disaster_route_noaa_entry_v1/README.md)
5. [最小机制筛选及止损](output/disaster_route_verification_screen_v1/README.md)
6. [四区域强对照与观测增量诊断](output/disaster_route_crasar_batch4_v1/README.md)

已完成[固定Pearson算子的区域外同预算检验](output/disaster_route_crasar_pearson_isobudget_v1/README.md)：3区域、54条闭合路线、32次本机先验前向。主预算100代理秒下，Pearson／幅度空间／纯空间／精确静态累计核查阳性为3／3／4／5；未建立稳定方法增量。算子保留为标准对照，不继续调Boone特征。全部源标签、代理成本与配准边界见报告。此前[Boone两步初筛](output/disaster_route_boone_two_step_v1/README.md)及[小VLM结果](output/disaster_route_boone_context_run_v2/README.md)保持原字节。

补充[Iona轨迹与排序—收益诊断](output/disaster_route_iona_ranking_diagnostic_v1/README.md)：已定位第三次阴性反馈后的选点分叉，并提供同首访截面的AP—路线命中图；公开日志可独立重画，无需模型或原始影像。当前只补分析，写稿后置。

补充[四层故事线审定](output/disaster_route_story_review_v1/四层故事线审定.md)：现象与计算机制已核验，配准／区域异构等物理归因未确认；四层作为递进诊断，不视为统一失败定理。

## 四层证据与快速复现

|层次|原实验脚本（work/disaster_route/）|已记录资源与耗时|核心结论|
|---|---|---|---|
|受控机制|`run_verification_screen.py`|CPU精确枚举；原总耗时未记录|强制校准60/108更差，可选11/108改善；非一般理论|
|VLM／低阶表征|`run_boone_context_experiment.py`|本机MPS，276次前向329.61s；总348.71s|完整历史AP0.342低于不更新0.377；低阶核未超幅度控制|
|局部与单点诊断|`run_boone_structure_check.py`、`run_boone_two_step.py`|CPU缓存计算；原总耗时未记录|Top-1 9:1，去B03后3:1；两步8:5非同航程|
|区域外同预算|`run_crasar_pearson_isobudget.py`|32次本机MPS先验14.37s；路线CPU计算|100s合计静态5、纯空间4、Pearson3|

无API费用、无GPU租赁；本机MPS属于本地GPU，不是CPU。运行下列命令只复算已发布表格和两张诊断图，不重新执行模型或原路线实验：

```bash
python -m pip install -r requirements-diagnostics.txt
python work/disaster_route/reproduce_diagnostic_evidence.py --output output/reproduced_diagnostic_evidence
```

仅需表格时可用`--tables-only`，无第三方依赖。输出目录须不存在。本机公开快照完整复现约0.51s、标准库表格约0.02s，不含安装与Python启动；无需影像、权重或网络。见[证据包](output/disaster_route_four_stage_evidence_v1/README.md)与[结论核验](output/disaster_route_four_stage_evidence_v1/结论核验与边界.md)。当前先收敛证据，写稿后置。

## 当前结论（2026-11-03快照）

- 泛化的“路线分歧时考虑复核”在108个受控条件下与同策略空间精确贝叶斯参考收益一致，停止作为独立新方法扩展。
- NOAA影像实际覆盖已有Boone区域，但共同现场真值、可靠校准模型和同状态UAS复访尚未建立；不能替代受控实验中的校准信号。
- 固定VLM先验、付费反馈与Pearson更新已完成Boone外同预算开发检验；当前保留为比较基线，尚无稳定路线增量。5节点教学与VLM上下文路线收益仍未完成，不恢复旧传播或复核规则调参。
- 当前真实区域结果属于档案图像访问回放；运动与服务成本含假设，不宣称真实飞行验证。

## 代码和材料

- `work/disaster_route/`：路线、双层地图、精确参考、数据核验、研究筛选与专项测试。
- `work/disaster_search/`：历史固定感知与数据检查源码；不包含本地环境、下载附件或模型权重。
- `output/disaster_route*/`：报告、协议、摘要、部分结果与来源快照。
- `output/disaster_search*/`：历史感知检查报告与轻量结果。
- `output/感知检查_阶段汇报_20261028/`：保留的感知阶段材料。

## 最小验证

Python3.12。下列6项受控筛选测试不下载影像、不调用模型，也不改封存结果：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m unittest discover -s work/disaster_route -p test_verification_screen.py -v
python tools/verify_snapshot.py
```

影像核验、历史感知推理和报告导出需要额外依赖及原数据，见[环境与复现边界](docs/REPRODUCIBILITY.md)。本快照不是所有实验的一键完整复现包。

## 数据与归档边界

不上传原始影像、逐栋图像切片、模型权重、NumPy预测缓存、逐块下载缓存、部分重建TIFF、本地Python环境及邮件正文。原始数据引用和固定版本保留在实验协议中。选入的旧报告与完成收据保持原字节；它们可能引用未上传的输入或原机器绝对路径。

上传文件由[快照清单](snapshot_manifest.json)逐项记录SHA256。旧完成收据中缺失的原始文件见[归档可用性清单](docs/archive_availability.json)；不能将旧收据当成GitHub端完整数据归档证明。

未替项目统一指定开源许可证。第三方来源和数据许可见[来源说明](docs/THIRD_PARTY.md)；公开托管不改变其许可和引用要求。
