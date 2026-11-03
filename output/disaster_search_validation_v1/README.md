# 冻结规则后的验证结果

本轮已经从校准探索推进到预留验证事件：Michael 与 Mexico，共 24 个场景、3072 次局部模型观察，以及 6 策略 × 3 预算 × 7 个质量分配（包括两种纯质量条件）= 3024 个搜索回合。模型、阈值、原始概率分档、似然、邻域和反馈强度均在验证推理前固定，没有在验证数据上重拟合。

后续均匀覆盖、随机搜索与复核拆分已完成，见 `../disaster_search_coverage_controls_v1/README.md`。该补充明确为事后诊断，本轮原报告和轨迹没有覆盖。

**局部正负反馈出现小幅发现增量，但没有形成稳定优势，尤其质量混合、预算 32 时，增量几乎消失且误报增加。当前更需要检查基础先验和传感器可靠性迁移，并补充均匀/随机搜索对照。暂不把这版规则当作论文主方法。**

## 验证搜索结果

下表为跨 24 场景累计去重目标发现数，质量混合为五个固定质量分配的均值。场景成本和观测机会与校准实验相同：初查 1、模糊阴性可花 2 升级为原图。成本是人工代理，不是飞行时间。

| 初查条件 | 预算 | 质量门控/质量条件更新 | 旧全局传播 | 局部正负反馈 | 局部阳性反馈 | 简单阳性邻格 |
|---|---:|---:|---:|---:|---:|---:|
| 原图 | 8 | 6 | 8 | 9 | 6 | 6 |
| 原图 | 16 | 10 | 15 | 13 | 10 | 10 |
| 原图 | 32 | 20 | 20 | 22 | 19 | 19 |
| 质量混合 | 8 | 3.4 | 5.6 | 5.4 | 3.4 | 3.4 |
| 质量混合 | 16 | 6.4 | 9.4 | 7.6 | 6.4 | 6.4 |
| 质量混合 | 32 | 12.8 | 13.8 | 13.0 | 13.0 | 13.0 |
| 全部模糊 | 8 | 2 | 3 | 3 | 2 | 2 |
| 全部模糊 | 16 | 6 | 6 | 7 | 6 | 6 |
| 全部模糊 | 32 | 8 | 10 | 11 | 7 | 7 |

局部正负反馈在这九个合计设置中均比质量门控多发现目标，但增量只有 0.2–3 栋；并不全面优于旧全局传播或简单邻格对照。在质量混合、预算 32 时，门控、局部正负反馈的误报格分别为 0、0.8；简单阳性邻格同样找到 13.0 栋且误报为 0。因此不能仅凭多 0.2 栋宣布复杂反馈胜出。该设置按场景对质量种子取均值后，为 2 胜、21 平、1 负。

![Frozen validation curves](feedback_curves.png)

全部误报、配对场景与事件差值在 `report.json`，逐回合结果在 `runs.json`，完整动作在 `traces.jsonl`。未按预算或事件挑选最佳结果。

## 感知可靠性与先验迁移

按相同 v2 可见性真值规则：

| 观察条件 | 校准集目标格检出 | 验证集目标格检出 | 验证阴性格误报 |
|---|---:|---:|---:|
| 原图 | 208/243（85.6%） | 49/84（58.3%） | 4/1447 |
| 模拟模糊 | 32/243（13.2%） | 0/84 | 0/1447 |

验证每个条件均排除 5 个真值不确定格。固定原图模型遍历全部格时匹配到 35/53 栋目标；模糊条件没有匹配任何目标。搜索实验中“全部模糊初查”仍可通过付费升级获得原图，所以其最终发现数可以大于零。

初始粗图概率映射的场景等权 Brier 误差约为 **0.06873**，使用校准集统一目标比例作为常数概率约为 **0.06330**，更低。Michael 分别约 0.10406/0.10131，Mexico 约 0.03340/0.02530。此结果不能证明先验排序必然无用，但说明已有四档概率映射的验证误差没有优于常数预测；仅相对它改进的策略，尚不能证明信息或方法价值。

上述差异是数据密度、事件、图像与目标特性共同改变时的迁移诊断，不是把下降唯一归因于某一种灾害机制。没有利用验证错误率替换已有校准参数。

## 验证集的实际覆盖

冻结的小样本中，53 栋严重损伤目标全部属于 Michael。Mexico 的 12 个抽中场景没有严重损伤目标，只能提供阴性行为检查。因此这里不是“两个有目标灾害事件上的泛化验证”，Mexico 也不能作为目标发现机制的第二个独立成功事件。

没有在看过标签后替换这些场景。正式扩大样本时应单独冻结扩展选择协议，保留本轮样本与统计，不能通过补挑阳性图来改写这一轮结果。原有 test 事件仍未做模型推理或策略评价。

检查点预期接触过这些场景所在的上游训练集。这里的验证隔离针对本轮先验、似然和策略规则；不代表感知权重未见这些灾害或图像，也没有真实无人机结论。

## 下一步的明确顺序

1. **先补简单对照**：在相同信息、质量安排、动作资格和成本下，加均匀先验的质量门控与预先固定种子的随机覆盖。已有验证结果已看过，这一补充必须标为事后诊断，不重新包装成盲验证。
2. **检查决策目标**：当前选点面向“格内是否有目标”，评价是“发现多少栋不同建筑”，密集场景可能放大少数动作收益。阳性格终止也可能遗漏同格更多建筑。这两点需在增加复杂模型之前澄清。
3. **再决定机制**：若邻域反馈不能稳定超过简单覆盖/门控，停止为它调参。若仍有增量，下一轮先解决不同事件下概率与复核收益的适用范围，再提出可检验的新规则。
4. **扩大独立事件再做结论**：当前有效目标验证只覆盖一个事件，测试小样本也只有 15 栋目标，不能承担稳健论文结论。扩展协议和感知权重暴露问题需另行处理，测试集继续保留。

此处提出的下一步是实验决策，不是已经证明的新研究贡献。无需为当前规模租 GPU 或训练新模型。

## 核查与复现

58 项单元检查通过。独立审计复算 3024 回合、52606 步的动作分数、邻域修正、原始先验后验、相关证据替换、预算和建筑去重，均通过；3072 个验证掩膜及缓存哈希核对通过。原校准参数哈希保持 `db3c3a77647b18009e2e192e187084c5148f39c24fb952a0f7b40420ea3fc600`，旧搜索报告与完整轨迹未改变。

原校准推理脚本及当时评价器在增加显式验证支持前保存到 `../disaster_search_changeos_calibration/source_snapshots/`；推理器不提供 test 选项。旧默认校准计划与冻结文件逐项一致。

```bash
work/disaster_search/.venv_inference/bin/python work/disaster_search/run_changeos_calibration.py \
  --data work/disaster_search/attachments/external/huggingface/xbd_mini72_v1 \
  --model work/disaster_search/attachments/external/huggingface/changeos_swint \
  --out output/disaster_search_changeos_validation --split validation
work/disaster_search/.venv_inference/bin/python work/disaster_search/evaluate_changeos_pilot.py \
  --pilot output/disaster_search_changeos_validation \
  --prepared-data output/disaster_search_xbd_mini72_v1 --validation
work/disaster_search/.venv_inference/bin/python work/disaster_search/associate_buildings.py \
  --data work/disaster_search/attachments/external/huggingface/xbd_mini72_v1 \
  --prepared-data output/disaster_search_xbd_mini72_v1 \
  --pilot output/disaster_search_changeos_validation \
  --out output/disaster_search_validation_v1 --validation
work/disaster_search/.venv_inference/bin/python work/disaster_search/run_validation_search.py \
  --observations output/disaster_search_changeos_validation/observations.json \
  --association output/disaster_search_validation_v1/association_report.json \
  --parameters output/disaster_search_calibration_v2/calibration_parameters.json \
  --inference-plan output/disaster_search_changeos_validation/validation_protocol.json \
  --prepared-data output/disaster_search_xbd_mini72_v1 --out output/disaster_search_validation_v1
```

主要文件：`search_protocol.json` 验证前冻结规则和代码哈希，`association_protocol.json` 沿用 v2，`association_report.json` 感知/建筑评价，`report.json` 搜索与概率迁移结果，`integrity_checks.json` 核查，`test_log.txt` 测试，`feedback_curves.png/.pdf` 可导出图。
