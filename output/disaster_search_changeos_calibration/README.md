# 完整校准观察缓存

已完成 24 对校准影像、1536 个局部区域、3072 次固定规则 ChangeOS 推理。预测路径只读取选中局部的灾前后 RGB，不读取建筑标签内容。本机 MPS 运行，无付费 GPU；未运行验证/测试推理。

- `calibration_protocol.json`：推理前冻结的完整清单、输入规则及拟合方案。
- `observations.json`：固定 float32 派生分数和声明；原图与模拟模糊条件共用 evidence_group_id。
- `cache/`：3072 份规范 bool 严重损伤掩膜，后续评价无需重新推理；本轮不另存完整概率图。
- `scene_records/`：每场景完成后的记录，恢复运行时先核对缓存和固定清单。
- `inference_report.json`、`run_log.txt`：设备、输入/代码/协议哈希、实际运行记录。观察循环约 130 秒，不含启动与模型加载。
- `evaluation_report.json`：质心网格 v1 诊断，保留用于与 v2 对照。

**当前使用的跨格关联、校准结果和下一轮实验计划见 `../disaster_search_calibration_v2/README.md`。** 模型训练暴露预期存在，不能把这些拟合结果当作未见事件测试成绩。
