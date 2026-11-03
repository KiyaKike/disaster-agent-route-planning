# 跨格建筑关联诊断 v2

已完成四个校准试跑场景的跨格关联，保留原质心网格报告和冻结推理缓存。此次规则修正由看过的 pilot 边界问题触发，属于事后诊断，不应把新旧计数差异解释成模型进步。

## 规则与计数单位

先将每栋建筑的原始像素轮廓栅格化到整幅影像，再按互不重叠的网格切片。这样格内看得见的建筑部分可参与评价，无须质心也在本格。

- 格内任意严重损伤建筑可见面积至少 4 像素，记为目标存在；只有极小碎片或未知损伤建筑时记为不确定，不强行变成负例。
- 建筑匹配代理要求：固定预测严重损伤掩膜与其可见轮廓重叠至少 4 像素，且覆盖至少 10%。这两个阈值是透明的工程约定，尚未验证为可靠发现标准。
- 累计发现对 `(sample_id, building_uid)` 取并集；跨格重复看到或对同一底图重复处理都不增加建筑数。建筑 UID 和真值始终只在评价端。
- 同时报告严重损伤像素精度，避免仅靠大面积涂抹获得很高的建筑覆盖数。覆盖代理不是标准实例检测 AP。
- 新增 `real_evidence.py`，同一底图网格只保留一个观测代表；原图替换模糊图，不把两者当作独立似然相乘。它尚未实现概率更新。

## 固定 32 个局部区域的结果

| 条件 | 目标存在格检出 | 可见目标建筑的匹配代理 | 已知区域严重损伤像素精度 |
|---|---:|---:|---:|
| 原图 | 9 / 9 | 21 / 25 | 89.9% |
| 模拟模糊 | 1 / 9 | 3 / 25 | 97.2% |

模糊条件像素精度高并不说明更好：它漏掉了大量目标。两种条件合计出现 24 次建筑匹配，去重后仍为 21 栋，3 次重复不额外计数。

原图与模糊图各有两格从质心规则的阴性变成可见性规则的阳性，正是原先两个边界问题。旧结果保存在 `../disaster_search_changeos_pilot/evaluation_report.json`，未覆盖。建筑覆盖阈值分别取 5%、10%、25% 时，原图匹配数为 22、21、18，模糊图均为 3；全部报告，不按最高数值选择阈值。

模型预期接触过该上游训练集，且这里仅有四个校准场景。上述数字只是工程诊断，不能当作未见事件模型精度或搜索策略效果。

## 文件

- `association_protocol.json`：执行 v2 评价前单独冻结的规则。
- `association_report.json`：逐观察匹配、旧报告哈希、输入哈希、阈值敏感性、建筑去重与像素诊断。
- 单元检查覆盖跨格去重、共享边界像素、未知类别、极小碎片、宽泛预测、重复证据与输入字段隔离。完整最新检查见 `../disaster_search_calibration_v2/test_log.txt`。

## 复现

```bash
work/disaster_search/.venv_inference/bin/python work/disaster_search/associate_buildings.py \
  --data work/disaster_search/attachments/external/huggingface/xbd_mini72_v1 \
  --prepared-data output/disaster_search_xbd_mini72_v1 \
  --pilot output/disaster_search_changeos_pilot \
  --out output/disaster_search_boundary_v2
```
