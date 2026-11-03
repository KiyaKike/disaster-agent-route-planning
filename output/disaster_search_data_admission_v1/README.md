# 下一阶段数据与权重准入审计

**找到一个可进一步核查的数据池：作者发布的 hold，共 933 个样本名，与其公开训练池和权重选择池的样本名均不重叠。当前 mini72 的 72 个样本名则全部出现在公开 train 分区中。**

因此，mini72 继续保留为工程与失败诊断集；下一版不应再从同一个 train 池扩图后宣称独立感知验证。hold 是候选，不是已经建好的新基准：原始逐栋标注、像素对应关系和地理重叠仍需核对。

## 已核实的证据

作者模型卡说明使用 train+tier3 训练、test 选择权重、hold 最终报告。这是作者公开说明，不能替代逐训练步骤的日志。[模型卡](https://huggingface.co/EVER-Z/torchange_example_changeos_swint_on_xview2_best42k)

本次固定作者数据仓库 `EVER-Z/torchange_xView2` 的 commit `cda47e44377258222166dac63a85068e8bdca446`，对全部 50 个 Parquet 文件只读取 `image_name` 标量列。请求严格校验 HTTP 206 与 Content-Range，单次最多 1 MiB，总请求预算 16 MiB；实际成功读取 3368933 字节，约 3.37 MB。[作者数据仓库](https://huggingface.co/datasets/EVER-Z/torchange_xView2)

| 作者分区 | 实际清单数 | 作者说明的角色 | 与其他三个分区的样本名交集 |
|---|---:|---|---:|
| train | 2799 | 训练池 | 0 |
| tier3 | 3378 | 训练池 | 0 |
| test | 933 | 权重选择 | 0 |
| hold | 933 | 最终报告 | 0 |

共核对 8043 个不同规范样本名，四个池两两不交。数据服务的 info 响应标记为 partial，部分显示数量并不完整；上表来自所有固定文件的实际行数和名称，没有使用页面估计值推断总量。

当前松散 PNG/JSON 镜像 `aryananand/xBD` 的 train 2799 个名称与作者 train 完全对应，test 933 个名称与作者 test 完全对应。该镜像没有作者 hold 的样本名，所以不能简单把已有镜像 test 改称 hold。

mini72 的 72 个名称均在作者 train 清单中。准确表述是“全部属于公开训练池”；模型卡还说明训练时只保留非空切块，且没有逐优化步骤的样本日志，所以不进一步声称每张图的每个像素都实际进入了训练。

## hold 能解决什么，尚不能解决什么

hold 的 933 个名称未出现在公开训练池，也未出现在权重选择 test 清单。这使它适合继续审计“训练池外场景上的固定感知器与策略”，而不是立即认定为全新的未见灾害基准。

hold 的十个事件名称全部也存在于模型训练池：

| 事件 | hold 样本数 |
|---|---:|
| guatemala-volcano | 5 |
| hurricane-florence | 119 |
| hurricane-harvey | 95 |
| hurricane-matthew | 94 |
| hurricane-michael | 109 |
| mexico-earthquake | 34 |
| midwest-flooding | 86 |
| palu-tsunami | 41 |
| santa-rosa-wildfire | 77 |
| socal-fire | 273 |

因此，使用这份现成权重时仍不能宣称感知器没见过这些灾害事件。策略拟合/验证可以另行按事件隔离，但必须区分策略隔离与感知权重隔离。相邻影像、地理重叠和像素重复尚未审计，名称不重叠不能替代这些检查。

## 当前真正缺的接口

作者 Parquet 的公开 schema 为 `t1_image`、`t2_image`、`t1_mask`、`t2_mask`、`image_name`。它提供影像和像素掩膜，没有原始建筑多边形及 UID 列。

本次只读取样本名，未读取或分析图像、掩膜和损伤结果。尚未确认像素掩膜的值域、未知类别处理及与原始图像的一致性。

如果继续沿用“累计发现多少栋不同建筑”的评价，需要找到对应的原始 hold 建筑 JSON/UID；不能把掩膜连通分量冒充原始建筑实例。若新任务改为“发现目标区域”，像素掩膜可能足够，但那是评价单位调整，必须单独冻结新协议，不能覆盖旧逐栋结果。

作者数据卡没有完整原始数据授权说明，原 xBD 来源、许可与原始文件映射还需核对。没有在本轮批量下载影像、重分组样本、改变评价指标或训练新模型。

## 明确的下一步

1. 优先查找 hold 原始多边形/UID 和可追溯来源，核对少量固定样本的图像与标注对应。如果接口无法恢复，再明确是否把新任务定义为区域发现，而不是暗中替换统计单位。
2. 在读取损伤标签或跑方法前，冻结新版本事件分组、抽样和开发/验证/最终评价用途。保留 mini72 及其全部失败记录。新池的实际目标构成需透明审计，不能按方法成绩换图。
3. 完成图像哈希、地理重复与掩膜语义检查，再决定是否生成本机小批感知缓存。现在没有必要租 GPU。
4. 若论文要求真正未见灾害事件的感知泛化，需要另找模型训练池之外的灾害数据，或对感知器做严格事件排除训练。仅使用 hold 不能满足该主张。

在上述准入完成前，短探查切换器继续保留为未获充分支持的候选机制，不进行追分式调参。早期识别实验的完整结果见 `../disaster_search_early_evidence_v1/README.md`。

## 文件与核查

- `membership_protocol.json`：固定仓库版本、50 个文件的名称/大小/LFS 标识及读取范围限制。
- `shard_names/`：逐文件名称列、schema、范围读取哈希与接收字节数。
- `author_split_ids.json`：8043 个样本名，按作者分区保存。
- `membership_report.json`：分区/事件交集、旧镜像映射、mini72 所属池。
- `integrity_checks.json`：全部清单一致性核查；`test_log.txt`：78 项测试通过。

网络 API 元数据快照和来源 URL 在 `work/disaster_search/attachments/external/huggingface/torchange_xview2_admission/`。只读取名称列，因此 LFS 的整文件 SHA256 是远端标识，没有通过下载全文件重新验证；没有完成图像逐像素哈希核对。本审计不是损伤类别或模型效果报告。

```bash
work/disaster_search/.venv_inference/bin/python work/disaster_search/audit_xbd_membership.py \
  --metadata-root work/disaster_search/attachments/external/huggingface/torchange_xview2_admission \
  --loose-index work/disaster_search/xbd_source_index.json \
  --mini-protocol work/disaster_search/attachments/external/huggingface/xbd_mini72_v1/protocol.json \
  --out output/disaster_search_data_admission_v1
```

缓存恢复会校验名称列哈希与固定版本，不重复读取全套远端元数据；读取器拒绝服务器忽略 Range 后返回整个文件的响应。
