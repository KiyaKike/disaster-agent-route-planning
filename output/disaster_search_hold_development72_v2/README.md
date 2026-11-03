# 固定的 72 个开发样本

72 个预先固定的开发场景已完成数据接入，未按目标或模型成绩删除、替换样本。对应访问修订为 `../disaster_search_v2_design/contract_amendment_02.json`，有效协议为 `experiment_contract_v2_2.json`。本目录只包含开发数据，完整 216 候选的独立准入尚未完成。

| 事件 | 场景 | 标注建筑 | 严重损伤目标 | 未知建筑 | 有目标场景 |
|---|---:|---:|---:|---:|---:|
| Harvey | 24 | 1,751 | 705 | 46 | 13 |
| Santa Rosa | 24 | 1,792 | 508 | 5 | 18 |
| Mexico | 24 | 5,416 | 3 | 25 | 3 |
| 合计 | 72 | 8,959 | 1,216 | 76 | 34 |

目标为 major-damage/destroyed；未知保留，不作为已知阴性。Mexico 的目标很少，不能把三个事件的场景数相同当作三个事件提供同等发现证据。数据仍是卫星图像上的观察代理，没有真实无人机新视角。

## 数据接口与首次停止

144 张 PNG 与 72 个源对象 JSON 构成 216 个数据文件。固定六样本的全部 18 个文件与此前版本哈希一致。开发影像内部及与既有 mini72 的规范 RGB 像素哈希比较未发现重复；未核查完整感知训练池的像素重复。源不含原始 hold UID，沿用“源版本 + 场景名 + 原始对象序号”的派生 ID，并保存多边形指纹。没有生成源中不存在的灾前建筑标注。

首次适配器采用此前的 1e-6 像素边界容差，在新增场景发现越界顶点后停止，尚未下载新增 RGB。首次协议、代码、日志及几何调查保存在 `../disaster_search_hold_development72_v1/`。几何调查只读取坐标和尺寸，没有读取损伤列，也没有按损伤结果选择例外。

72 场景中，34 个场景的 205 个多边形至少有一个顶点落在影像之外，其中 75 个超出旧容差。最大越界 0.35144 像素，仅一个对象超出 0.0001 像素，其余主要为很小的数值误差。

适配器 v2 明确采用半像素容差，**保存原始坐标，不裁剪多边形**；超过容差或质心越界仍停止调查。发现评价继续使用原来的可见轮廓栅格规则，没有改变发现阈值或按结果换图。容差规则、调查哈希与适配器代码在 `materialization_protocol.json` 中冻结。边界检查图见 [subpixel_boundary.png](qa/subpixel_boundary.png)。

## 图像和空间检查

额外抽查固定排序中每事件第三个场景，加最大越界场景，共四个。轮廓叠加未见明显系统错位；密集城市轮廓抽查不等于逐对象专家核验，更不等于独立重新评定损伤。见 [extended_annotation_checks.png](qa/extended_annotation_checks.png) 和 `qa/qa_receipt.json`。

另记录每张图的全零 RGB 比例，仅作图像质量描述，没有修改输入、自动删除场景或启用新选点特征。例如 Santa Rosa_00000112 的灾前/后全零 RGB 约 79.2%/79.6%，实际可见大面积黑边。全零像素尚无已验证的源 NoData 定义；后续简单覆盖对照需要明确图像可用面积的影响，避免把避开黑边的收益混同于复杂决策的收益。

72 开发场景中，20 个与感知训练池的建筑标注包络相交：Harvey 10、Santa Rosa 7、Mexico 3。固定的 216 候选空间组清单仍是 `../disaster_search_cross_cell_smoke_v1/spatial_groups.json`。包络不是完整影像足迹，所有九个事件都在感知训练池中出现，不能声称未见灾害事件的感知泛化或完整空间独立。

## 数据读取范围与费用

只返回固定的 72 个开发行。共享损伤列在受控提取器内临时解码，物理载荷包含其他行；其他行的损伤值和统计没有输出，共享损伤载荷未写入范围缓存。不能称为从未物理读取保留样本的共享压缩字节。

本次成功接入新增范围流量约 209.89 MB；还需加上首次停止时约 85 KB 的临时损伤读取，累计请求仍在修订 02 的 512 MiB 上限内。没有整包下载约 32.4 GB 的源 ZIP。请求范围、校验值和累计网络计数分别见 `data_report.json` 与 `network_attempts/`。

验证、最终确认没有运行模型推理或策略评分；本轮没有租 GPU。固定感知检查与开发拟合已完成，结果另存于 `../disaster_search_hold_development72_perception_v1/` 和 `../disaster_search_hold_development72_fit_v1/README.md`，不能用本目录的 `data_interface_passed` 代替独立评价准入。

## 复现接入

```bash
work/disaster_search/.venv_inference/bin/python work/disaster_search/materialize_hold_development_v2.py \
  --design output/disaster_search_v2_design \
  --metadata-audit output/disaster_search_hold_taco_v1 \
  --mini-data work/disaster_search/attachments/external/huggingface/xbd_mini72_v1 \
  --probes output/disaster_search_hold_probes_v1 \
  --spatial output/disaster_search_cross_cell_smoke_v1/spatial_groups.json \
  --previous-attempt output/disaster_search_hold_development72_v1 \
  --out output/disaster_search_hold_development72_v2
```

恢复会校验已有图像回执，仍临时读取共享损伤列。首次失败不会自动获得新的 512 MiB 额度，累计预算合并首次与恢复的网络尝试。数据、预测、评价与拟合输出分开存放，旧六样本和 mini72 输出继续保留。
