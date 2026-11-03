# 六个固定开发样本：数据接入与感知检查

后续进展：六场景预算搜索工程检查已完成，见 `../disaster_search_cross_cell_smoke_v1/README.md`。简单边缘重复折扣导致发现数下降，保留为失败消融；当前共 116 项测试通过。下文及本目录 JSON 保留本阶段的数据/感知接入记录。

**本轮已完成六样本数据接入、768 次固定模型观察、评价器核对和相关观测/同格新增建筑数效用的组件试跑。没有运行搜索策略评分；216 对候选数据尚未完成全量准入。**

## 样本与标注

样本仍是新版协议预先固定的六个开发场景，没有按标签或模型成绩换图。共 217 栋建筑，其中 8 栋为 major-damage/destroyed，4 栋为未知；四个无严重损伤目标的场景保留。

| 固定样本 | 建筑 | 严重损伤目标 | 未知 |
|---|---:|---:|---:|
| hurricane-harvey_00000363 | 42 | 0 | 0 |
| hurricane-harvey_00000074 | 10 | 0 | 0 |
| santa-rosa-wildfire_00000013 | 6 | 0 | 0 |
| santa-rosa-wildfire_00000266 | 7 | 7 | 0 |
| mexico-earthquake_00000145 | 43 | 1 | 4 |
| mexico-earthquake_00000054 | 109 | 0 | 0 |

旧两个校准场景的 64 个对象除轮廓对应外，逐对象损伤等级也全部与已有原始 JSON 对应。新包的类别顺序是 no-damage/minor/major/destroyed/un-classified，即 **2、3 是目标，4 是未知**；不能直接当作模型像素掩膜的 3、4。未知保留，未填成负例。

每个源多边形生成固定的派生实例 ID，附对象序号和轮廓指纹；没有恢复原始 UID，也没有使用掩膜连通分量代替建筑。六样本内部未发现重复对象多边形。12 张 RGB 与六样本内部、既有 mini72 全部 RGB 的规范像素哈希比较，未发现重复；没有核查整个感知训练池的像素重复。

只生成源中实际具备的灾后建筑标注，没有复制灾后轮廓来伪造灾前标注。已有全部 mini72 结果、模型和拟合参数保留。

## 感知与评价接口

沿用固定 ChangeOS 权重、RGB 归一化、128×128 局部输入、模糊生成与四严重像素声明规则。6 场景 × 64 格 × 2 条件，共 768 次观察；本机 MPS 观察循环约 28.85 秒，单次中位数约 35 毫秒，未租 GPU。预测命令只读 RGB、固定参数和不含损伤统计的接口凭据；标注由另一条评价命令读取。

| 遍历全部 64 格的观察条件 | 匹配的不同目标建筑 | 已知阴性格误报 | 目标格 TP/FN |
|---|---:|---:|---:|
| 原图 | 6/8 | 2 | 5/1 |
| 模拟模糊 | 1/8 | 0 | 1/5 |

每种条件均排除 2 个真值不确定格。**这些是全格遍历的感知检查，不是预算 32 的搜索成绩。** 原图匹配到的 6 栋均来自 Santa Rosa；Mexico 的 1 栋目标未匹配，因此不能称为多个事件上的稳定发现效果。

按预定 5%/10%/25% 可见轮廓覆盖重算，原图都是 6 栋，模糊分别为 1/1/0 栋。模糊的唯一匹配覆盖约 21.8% 可见轮廓，说明这项小样本结果对测量阈值敏感；没有据此更换主阈值。

标注核对图见 [六场景总览](qa/annotation_overview.png)、[两处目标区域放大](qa/annotation_details.png)。轮廓叠加仅用于技术 QA，不是独立专家重新评定损伤。Harvey_00000363 存在明显黑边，灾前/后全零 RGB 像素约 18.2%/19.1%，已记录在 `qa/image_quality_audit.json`；全零不是已验证的 NoData 语义，本轮未修改输入或加入掩膜规则。

## 已实现的基线组件与尚未解决的问题

`paired_count_baseline.py` 实现成对模糊/原图的联合似然及条件后验，避免把同源观测按边际似然独立累乘。另按公开粗图四档线索估计同格新增匹配建筑数；复核的拟合目标是 `native_matched - already_blurred_matched`，模糊阳性也可有新增收益。仅在开发数据拟合，未知格不作为已知负例。

本轮只有 382 对已知真值格，且模糊阳性只有 1 格。组件参数因此只用于接口试跑，不能作为完成 72 开发场景校准或方法泛化的证据。

**跨格重复发现尚未进入效用模型。** 原图累计出现 8 次“观察—建筑”匹配，但只对应 6 栋不同建筑，已有 2 次重复匹配。现有组件只能去掉同格复核的重复，不能宣称已经精确优化全局新增不同建筑数。参数和限制见 `../disaster_search_paired_count_smoke_v1/`。

## 空间筛查与准入边界

另对 216 个候选进行了只含地理轮廓的筛查，没有使用损伤等级。62 个候选的建筑标注包络与公开训练池场景包络相交；19 个缺少可用建筑地理轮廓。可用包络中未发现跨开发/验证/确认分组相交，候选内部另有两对同分组包络相交。

本轮六个样本都未触发训练包络相交，但建筑包络不是完整影像足迹；缺少包络和零相交都不能证明影像独立，也未排除地理邻接。所有九个事件仍在感知器训练池出现。详细记录见 `../disaster_search_hold_spatial_v1/report.json`。候选不自动剔除或替换，后续仍需完整足迹或明确的空间分组记录。

## 数据访问与核查

按上轮修订协议，受控提取器临时解码了全库共享损伤列，物理读取包含保留样本的压缩载荷。只有六个开发样本和两个已使用的旧校验样本进入返回值，其他样本的损伤值、类别统计、模型结果和策略成绩未输出。共享损伤载荷未写入持久范围缓存；范围回执只保存地址、长度和哈希。不能再把本阶段描述为“未物理接触最终确认样本的损伤载荷”。

104 项测试通过，日志为 `test_log.txt`；新增测试覆盖未知类别、样本范围限制、临时损伤读取、RGB-only 预测、成对条件更新、同格去重、未支持条件和空间筛查。首次完整测试的新增测试文件有括号语法错误，修复后完整重跑通过；首次日志保留为 `test_log_first_attempt.txt`。旧实验数据未受影响。

当前可继续扩展的是开发数据接入。下一轮应先记录空间分组、处理全局跨格重复收益与黑边质量线索，再用固定的 72 个开发样本完成基线拟合；确认集继续不进行模型推理或策略评分。

## 复现入口

```bash
work/disaster_search/.venv_inference/bin/python work/disaster_search/materialize_hold_probes.py \
  --design output/disaster_search_v2_design --metadata-audit output/disaster_search_hold_taco_v1 \
  --bridge output/disaster_search_hold_polygon_bridge_v1 \
  --mini-data work/disaster_search/attachments/external/huggingface/xbd_mini72_v1 \
  --out output/disaster_search_hold_probes_v1
work/disaster_search/.venv_inference/bin/python work/disaster_search/run_hold_probe_perception.py \
  --data output/disaster_search_hold_probes_v1 --design output/disaster_search_v2_design \
  --model work/disaster_search/attachments/external/huggingface/changeos_swint \
  --out output/disaster_search_hold_probe_perception_v1 --stage predict
work/disaster_search/.venv_inference/bin/python work/disaster_search/run_hold_probe_perception.py \
  --data output/disaster_search_hold_probes_v1 --design output/disaster_search_v2_design \
  --model work/disaster_search/attachments/external/huggingface/changeos_swint \
  --out output/disaster_search_hold_probe_perception_v1 --stage evaluate
work/disaster_search/.venv_inference/bin/python work/disaster_search/paired_count_baseline.py \
  --perception output/disaster_search_hold_probe_perception_v1 \
  --design output/disaster_search_v2_design --out output/disaster_search_paired_count_smoke_v1
work/disaster_search/.venv_inference/bin/python work/disaster_search/audit_hold_spatial.py \
  --design output/disaster_search_v2_design --metadata-audit output/disaster_search_hold_taco_v1 \
  --identities output/disaster_search_data_admission_v1/author_split_ids.json \
  --out output/disaster_search_hold_spatial_v1
```

预测命令恢复时核对全部缓存，不重复推理；提取器恢复时仍会临时读取约 85 KB 共享损伤列，不缓存这部分数据。所有数据准入/模型/评价/拟合输出分开保存，不能把 `data_interface_passed` 等同于 `whole_216_cohort_admitted`。
