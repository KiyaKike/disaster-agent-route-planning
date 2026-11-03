# 72 开发场景的固定感知缓存

9,216 次观察完整，其中复用原六样本 768 次、新增 8,448 次，本机 MPS 完成。RGB-only 预测与标签评价分命令执行；模糊是同一图像的受控退化，没有真实新视角。

原图/模糊全格遍历匹配 1,080/178 栋不同目标建筑，标注目标总数 1,216。这些是全格感知检查，不是预算搜索成绩。逐事件结果、开发拟合及下一步比较见 `../disaster_search_hold_development72_fit_v1/README.md`。完整数据接入记录见 `../disaster_search_hold_development72_v2/README.md`。

代码与源输入在 `perception_protocol.json` 中冻结，复用回执为 `cache_reuse_receipt.json`，运行记录为 `inference_report.json`。`association_report.json` 是评价器私有数据，包含真实实例匹配，不能传入选点函数；后续环境只能在对应观察付费后返回其公开预测。
