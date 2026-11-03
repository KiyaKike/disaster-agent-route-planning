# 固定验证集感知缓存

24 对影像、1536 个区域、3072 次局部观察已完成，本机 MPS 运行，观察循环约 124 秒，不含启动/模型加载。没有租卡。模型、输入预处理、四像素声明阈值、合成模糊规则均沿用校准阶段，预测路径不读标签内容。

`validation_protocol.json` 在运行前冻结；`observations.json` 和 `cache/` 保存固定输出与规范二值掩膜；`inference_report.json` 与 `run_log.txt` 记录输入/源码/协议哈希和运行过程。没有 test 推理。

主结果与限制见 `../disaster_search_validation_v1/README.md`。这里的验证隔离仅针对本轮拟合与策略设计，检查点训练暴露预期仍存在。
