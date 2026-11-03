# hold 替代来源核查

**名称映射和旧样本的图像/逐对象轮廓对应已通过；新数据全量准入尚未通过。**

后续六开发样本的类别/图像接入、768 次固定感知与组件检查已完成，见 `../disaster_search_hold_probes_v1/README.md`；104 项测试通过。下表和本目录 `report.json` 仍描述本次元数据审计自身的访问范围。后续提取器已临时读取共享损伤列，不能将下表“本审计未读损伤”扩大为整个项目从未读取。

来源：[ISP / Universitat de València 的 xView2-TACO 发布包](https://huggingface.co/datasets/isp-uv-es/xview2-taco)，固定版本 `44e219dc1d88c9d0903d6b478d7f2f3c435e574f`。它是 xBD 的再打包，原始依据与声明许可见 [官方数据页](https://xview2.org/dataset)；两处均标示 CC BY-NC-SA 4.0。没有整包下载约 32.4 GB 的 ZIP，也没有验证整包发布哈希。

| 项目 | 核查结果 |
|---|---|
| 原始 hold | 933 个名称，与作者清单完全一致 |
| 本轮预定候选 | 216/216 可映射到原始 hold |
| 名称投影 | 只读 `source_file`、`split_original`、事件和内部行标识 |
| 原始 UID | 未保留，不能声称已恢复 |
| 逐对象轮廓 | XY 坐标和每栋顶点数均保留，另有 WGS84 轮廓列 |
| 已有校准样本的对应核查 | 2 场景，64 个多边形逐一对应，4 张 PNG 字节相同 |
| 本次元数据审计读取损伤等级 | 未进行；范围回执确认未触及损伤列载荷 |
| 本次元数据审计运行模型或策略 | 未进行 |
| 全池地理/像素重复核查 | 未完成 |

对应核查的基准是此前固定的社区镜像文件，不是本轮重新从官方渠道取得的原始发布文件。两个发布包一致不能替代官方文件哈希校验。多边形核对在小数点后六位、忽略环起点和方向的口径下逐一匹配；没有从掩膜连通分量制造建筑。

关键格式差别：该包的 `ml:split` 是重新组织的用途，映射原始分区必须读 `ml:split_original`；它包含原始 hold 和原始 test，不能混用。该包类别表顺序为 no-damage、minor-damage、major-damage、destroyed、un-classified；与模型像素掩膜的背景/损伤编号不同。当前只读了类别表，尚未验证逐对象类别的顺序对应。

由于 UID 未保留，设计修订 01 明确允许以固定版本、场景名、源对象序号及轮廓指纹生成派生实例 ID；它不是原始 UID。下一步只对预先固定的 6 个开发样本检查对象/类别顺序、未知值、图片与轮廓和重复，原评价仍按不同建筑计数。有效协议及样本清单在 `../disaster_search_v2_design/`。

首次身份审计范围读取 281175 字节，首次旧样本对应核查范围读取 63993871 字节，共 64275046 字节；两者缓存重放均无需新增网络。另有探索阶段 16 KiB 前缀探查及 141049 字节 schema 探查：前者含首张 PNG 部分压缩字节而未解码，已在协议中披露。API 和数据卡请求单独保存，不含在以上范围请求合计中。

## 复现

```bash
work/disaster_search/.venv_inference/bin/python work/disaster_search/plan_search_v2.py \
  --identities output/disaster_search_data_admission_v1/author_split_ids.json \
  --out output/disaster_search_v2_design
work/disaster_search/.venv_inference/bin/python work/disaster_search/audit_hold_taco.py \
  --identities output/disaster_search_data_admission_v1/author_split_ids.json \
  --design output/disaster_search_v2_design --out output/disaster_search_hold_taco_v1
work/disaster_search/.venv_inference/bin/python work/disaster_search/verify_taco_polygon_bridge.py \
  --mini-data work/disaster_search/attachments/external/huggingface/xbd_mini72_v1 \
  --metadata-audit output/disaster_search_hold_taco_v1 \
  --out output/disaster_search_hold_polygon_bridge_v1
```

`audit_protocol.json`、`cozip_index.json`、`report.json` 和 `ranges/` 保存固定来源、读取位置、投影字段、保护列范围及每块 SHA256。`../disaster_search_hold_polygon_bridge_v1/` 保存已知原始建筑 UID 的小样本对应证据。新版总核查收据为 `../disaster_search_v2_design/integrity_checks.json`。
