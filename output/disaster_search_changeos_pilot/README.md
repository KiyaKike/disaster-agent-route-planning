# ChangeOS 局部观察试跑

已固定 ChangeOS Swin-T，完成 4 对校准影像、32 个固定区域、64 次观察（原始与模拟模糊各 32 次）。未运行验证集或测试集推理，未租用 GPU。

## 运行与模型

- 模型：EVER-Z/torchange_example_changeos_swint_on_xview2_best42k
- 模型版本：4007dd10c5eaee67161e2d4a45a43da3ead6c722
- 代码版本：190b07a2eb4c389a5dc3f72802c67f428cdfd6b7
- 权重约 166 MB；safetensors 哈希与发布元数据匹配，严格加载，没有缺失或意外权重键。
- 本机 MPS；每次前向中位数 32.5 毫秒，均值 35.2 毫秒。
- 按本次单区域耗时线性估算，24 对校准集 × 64 格 × 2 模式的纯前向约 1.8 分钟，含本次平均预处理/缓存开销约 2.3 分钟；不含首次导入、模型初始化和场景读取，不是完整运行的实测总时长。

## 数据与推理边界

- 在模型推理前冻结 pilot_protocol.json。两个校准事件各取清单中的前两对，固定八格，未按目标或模型表现挑选。
- 每次只读取选中格的灾前后 RGB，ImageNet 归一化，输入 [1,6,128,128]。真实建筑轮廓/损伤标签未输入模型。
- 作者训练块为 512×512；本次 128×128 输入存在上下文与边界差异。
- 原始与模拟模糊图来自同一卫星底图；通过 evidence_group_id 关联，不能作为独立证据相乘。
- 四个严重损伤像素的声明规则在推理前固定；原始 soft score 不是校准后的目标概率。

## 诊断结果，不作为性能结论

| 观察 | 已知目标格检出 | 按质心网格规则计出的误报 |
|---|---:|---:|
| 原始 | 7 / 7 | 2 |
| 模拟模糊 | 1 / 7 | 0 |

两个按网格计出的误报均与跨格受损建筑的可见部分有像素重叠：一格预测 743 像素、与已标注目标重叠 707 像素；另一格预测 41 像素、重叠 21 像素。原始网格混淆矩阵保持不变，没有事后改阈值或重算一份更好看的成绩。
这说明按建筑质心定义的网格真值与“局部图像中可见受损部分”并非完全相同。当前误报率不能直接用于传感器似然校准；下一步需明确跨格目标关联、重复发现去重和可观测性。
此批仅有 7 个质心目标格，且模型卡说明在上游 train+tier3 上训练；mini72 全部来自上游 train，预期存在训练暴露，不能据此报告未见事件精度。

## 已保存

- pilot_protocol.json：推理前固定的清单与规则。
- observations.json、cache/：64 次观察及概率/规范二值掩膜缓存，后续策略可复用。
- inference_report.json：设备、依赖、源码/输入哈希及耗时。
- evaluation_report.json：固定规则的校准侧诊断。
- boundary_diagnostics.json：两个误报的跨格关联检查。
- pilot_preview.png：按冻结清单展示前两格，无按结果选图。
- test_log.txt：29 项检查全部通过；dependency_check.txt：依赖一致性通过。
- requirements-inference.lock.txt：本次隔离环境的确切包版本。

## 复现

```bash
work/disaster_search/.venv_inference/bin/python work/disaster_search/run_changeos_pilot.py \
  --data work/disaster_search/attachments/external/huggingface/xbd_mini72_v1 \
  --model work/disaster_search/attachments/external/huggingface/changeos_swint \
  --out output/disaster_search_changeos_pilot --device auto
```

## 下一步

先修正“局部可见目标”和“质心归格真值”的关联协议，保留原协议/成绩作对照；再用校准集估计固定识别器在不同观测条件下的错误率与初始先验映射。随后接入真实观察缓存，比较搜索策略。正式泛化实验需另行处理感知权重的训练数据重叠。当前计算量可先在本机完成。
