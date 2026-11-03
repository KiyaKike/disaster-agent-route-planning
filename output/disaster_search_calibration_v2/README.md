# 首轮真实影像校准已完成

24 对校准影像 × 64 格 × 2 条件，共 3072 次局部观察，来自 1536 个底图区域、两个事件。全部在本机 MPS 运行，没有租用 GPU。观察循环实测约 130 秒，不含启动和模型加载。验证集与测试集没有运行模型或参与本轮拟合。

本文件记录固定感知器的工程校准。后续校准侧首轮搜索已完成，见 `../disaster_search_cached_search_v1/README.md`：分档先验修正尚无稳定增益，简单质量门控与质量条件策略的动作相同。下文“下一轮实验”保留当时安排，执行结果以该新记录为准。

## 主要发现

| 条件 | 已知目标格检出 | 已知阴性格误报 | 可见目标建筑匹配代理（去重） |
|---|---:|---:|---:|
| 原图 | 208 / 243（85.6%） | 10 / 1287（0.78%） | 408 / 454（89.9%） |
| 模拟模糊 | 32 / 243（13.2%） | 0 / 1287 | 73 / 454（16.1%） |

每种条件均排除 6 个真值不确定的区域。模糊条件零误报不等于真实误报率为零，已用 Beta(0.5,0.5) 平滑，避免后续贝叶斯更新产生无限似然比。格检出与建筑匹配是不同指标：前者用于传感器存在性似然，后者用于搜索时对建筑累计去重，不能混用。

事件间差异仍明显：Harvey 原图/模糊目标格检出为 78.2% / 24.1%，Santa Rosa 为 89.7% / 7.1%。质量分层不能自动解决灾害类型、目标尺寸或背景带来的差异；这里只有两个拟合事件，没有证明新事件上的校准可靠性。

## 已拟合的参数

`calibration_parameters.json` 保存按条件分开的敏感度/误报率，以及从粗图 RGB 变化分数到“格内可见目标存在概率”的初始映射。它是由校准集学得的聚合参数，不含真实建筑 UID，可作为后续策略输入；逐建筑匹配表始终属于评价端。

- 原图敏感度、误报率的平滑值分别约 0.8545、0.00815。
- 模拟模糊条件分别约 0.1332、0.000388。
- 粗图变化分数按校准集四分位数划为四档，每档目标比例向总体比例作 8 个伪样本的收缩，拟合概率约为 5.5%、14.1%、21.9%、22.1%。这不是外部验证过的概率，原有 `public_cues.json` 中的空概率字段未被覆盖。
- 推理前的 `calibration_protocol.json` 已写明完整范围、似然平滑和四档映射规则；`fit_protocol.json` 单列拟合定义。没有按效果调 detector 阈值、替换场景或选择最佳建筑关联阈值。

## 边界、重复证据与证据核查

采用 `raster_visibility_v2`：建筑先在整幅影像栅格化，再按网格切片。目标可见面积至少 4 像素才计入目标格；未知标签和极小碎片保留不确定。建筑匹配代理要求至少 4 个重叠像素、覆盖可见轮廓至少 10%，并同时报告严重损伤像素精度。该规则由先前试跑的边界问题触发，是透明记录的事后协议修正，不能宣称预注册或标准实例 AP。

原图建筑匹配发生 544 次，去重后为 408 栋；其中 136 次重复没有再次增加发现数。两种条件合并共有 623 次匹配，去重仍为 408 栋，仅用于检查去重，不能视为独立观察收益。5% / 10% / 25% 覆盖阈值的完整敏感性保存在关联报告，没有按最佳数值更改阈值。

`real_evidence.py` 把同一底图网格的原图和模糊图视为一个证据组；原图可替换模糊代表，不相乘。原图是受控条件中的代表优先级，不是看过 detector 结果后挑分数更高者。该注册表目前未实现搜索器中的后验更新。

核查结果见 `integrity_checks.json`：3072 个缓存哈希与规范掩膜均通过；证据代表共 1536 份；先前 pilot 的 64 个掩膜、声明和分数在完整运行中逐项复现，分数差为零；旧 pilot 报告和 mini72 冻结协议保持不变。39 项单元检查通过，日志为 `test_log.txt`。

## 下一轮实验

1. 接入这些固定观察缓存，先在校准场景调通逐步搜索，日志明确每一步此前可用的信息，避免未来观察或评价标签泄露。
2. 比较固定先验排序、质量条件更新并允许复核、以及增加未访问区域先验修正的策略；补充统一似然和简单质量门控消融。每组使用相同初始信息、可用观察、代价和预算。
3. 主指标按预算绘制“去重目标建筑发现数”，同时报告误报、首次发现成本和无目标场景。先使用 8 / 16 / 32 次基础观察成本；这不是飞行分钟。复核成本和质量条件分配在运行前固定，不把同一底图重复处理当成独立新视角。
4. 校准侧调通后，在验证事件检查概率偏移、动作是否真正变化与新增机制的收益；冻结策略再动测试事件。若质量条件贝叶斯或简单门控已达到相同效果，应保留无增益结论、调整研究机制。

## 解释范围

ChangeOS 检查点预期接触过 mini72 所在的上游训练集，故这是工程代理实验；不能报告为未见灾害感知泛化。128×128 局部输入与训练上下文不同；模拟模糊也不是实际无人机采集。对外研究结论仍需解决权重训练数据重叠，并扩大独立事件和测试目标数量。

## 复现

```bash
work/disaster_search/.venv_inference/bin/python work/disaster_search/run_changeos_calibration.py \
  --data work/disaster_search/attachments/external/huggingface/xbd_mini72_v1 \
  --model work/disaster_search/attachments/external/huggingface/changeos_swint \
  --out output/disaster_search_changeos_calibration
work/disaster_search/.venv_inference/bin/python work/disaster_search/evaluate_changeos_pilot.py \
  --pilot output/disaster_search_changeos_calibration \
  --prepared-data output/disaster_search_xbd_mini72_v1 --full-calibration
work/disaster_search/.venv_inference/bin/python work/disaster_search/associate_buildings.py \
  --data work/disaster_search/attachments/external/huggingface/xbd_mini72_v1 \
  --prepared-data output/disaster_search_xbd_mini72_v1 \
  --pilot output/disaster_search_changeos_calibration \
  --out output/disaster_search_calibration_v2 --full-calibration
work/disaster_search/.venv_inference/bin/python work/disaster_search/fit_real_calibration.py \
  --association output/disaster_search_calibration_v2/association_report.json \
  --observations output/disaster_search_changeos_calibration/observations.json \
  --prepared-data output/disaster_search_xbd_mini72_v1 \
  --out output/disaster_search_calibration_v2
```

预测阶段只读 RGB，标签内容在后续评价/拟合阶段读取。原始文件、pilot 协议及结果均保留。
