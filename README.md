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

已完成[Boone真实切片与本地小VLM对照](output/disaster_route_boone_context_run_v2/README.md)：60张切片、276次正式前向；UAS视角AP不更新0.377、空间0.460、视觉统计0.436、VLM完整历史0.342。当前小VLM配置没有收益。新增[逐建筑错误分析](output/disaster_route_boone_feature_error_audit_v1/README.md)：视觉统计组净AP差距主要来自B09，已执行[6维差分消融](output/disaster_route_boone_change6_v1/README.md)：重估核宽AP0.363、固定旧核宽0.474；新增[强度匹配对照](output/disaster_route_boone_strength_control_v1/README.md)：6维组0.474，匹配空间控制0.476；新增[归一化结构与Top-3检查](output/disaster_route_boone_structure_check_v1/README.md)：Pearson核AP0.553对匹配控制0.445、NDCG@3提升，Top-3为10对11且B11/B09未修复；下一项冻结候选接路线回放，不继续搜索核宽。

## 当前结论（2026-11-03快照）

- 泛化的“路线分歧时考虑复核”在108个受控条件下与同策略空间精确贝叶斯参考收益一致，停止作为独立新方法扩展。
- NOAA影像实际覆盖已有Boone区域，但共同现场真值、可靠校准模型和同状态UAS复访尚未建立；不能替代受控实验中的校准信号。
- 用户本轮选择新的“在线上下文反馈”学习与筛选路线：固定VLM权重，用已付历史更新未访问建筑判断。5节点教学与VLM路线收益尚未完成；本地小VLM预测试验已完成，见上方实际结果；不恢复旧传播或复核规则调参。
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
