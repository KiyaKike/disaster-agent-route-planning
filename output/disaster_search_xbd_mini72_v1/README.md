# xBD mini72 v1 数据集说明

已固定并下载 6 个事件、72 对灾前后影像，保留 144 个 PNG 和 144 个原始 JSON。原始数据约 222.0 MiB。

| 分组 | 事件 | 影像对 | 有目标场景 | 已知无目标场景 | 无已知目标但仍有未知标签的场景 | 目标建筑 |
|---|---|---:|---:|---:|---:|---:|
| 校准 | hurricane-harvey、santa-rosa-wildfire | 24 | 13 | 10 | 1 | 454 |
| 验证 | hurricane-michael、mexico-earthquake | 24 | 10 | 14 | 0 | 53 |
| 测试 | midwest-flooding、socal-fire | 24 | 7 | 14 | 3 | 15 |

目标定义为原始 major-damage 与 destroyed。建筑按像素多边形面积质心只归入一格；un-classified 保留未知。共 4608 格，其中 23 格为未知。

## 数据位置

- 原始数据与冻结清单：`/Users/bethany/Documents/ChatGPT/复现论文/work/disaster_search/attachments/external/huggingface/xbd_mini72_v1`
- 事件、种子与抽样规则：`/Users/bethany/Documents/ChatGPT/复现论文/work/disaster_search/attachments/external/huggingface/xbd_mini72_v1/protocol.json`
- 不含损伤统计的样本列表：`/Users/bethany/Documents/ChatGPT/复现论文/work/disaster_search/attachments/external/huggingface/xbd_mini72_v1/sample_manifest.csv`
- 下载来源、文件大小与 SHA256：`/Users/bethany/Documents/ChatGPT/复现论文/work/disaster_search/attachments/external/huggingface/xbd_mini72_v1/manifest.json`
- 本目录 `sample_audit.csv`、`adapter_report.json` 为评价侧数据审计；不是方法成绩。
- 每个场景目录中的 `public_cues.json` 为粗图变化线索；`local_observation_index.json` 为环境侧按需裁切坐标；`evaluation_truth.json` 仅供评价。
- `calibration_preview.png` 只显示两个校准事件各自按清单排列的第一对，不按损伤比例挑图。

## 已完成的检查

- 下载前固定事件、抽样算法和种子 20261024；审计后未替换样本。
- 所有 288 个输入哈希已校验，图像完整解码，图像与标签尺寸、文件名和事件一致。
- 每组 24 对、每事件 12 对；事件无交叉；开发诊断样例排除。
- 未发现跨组精确重复图像或重复建筑 UID。
- 原始未知标签未改为负例；公共线索没有目标标签，概率尚未校准，保持 null。
- 20 项代码检查通过，记录见 test_log.txt；数据构建检查见 construction_checks.json。

## 使用边界

这是一批策略预实验数据，不是新的官方基准。事件为人工预先固定，事件内随机抽样保留自然标签分布。测试集仅 15 栋目标建筑、2 个事件，不能据此作强统计或广泛泛化结论。
全部原始文件来自上游 train。本轮按事件分开策略校准/调参/测试；后续冻结感知器若见过这些事件，需要报告该重叠，不能直接宣称感知器未见事件泛化。尚未做地理足迹重叠审计。
128×128 粗图是下采样代理，非真实 PlanetScope 成像；局部图来自同一卫星底图，重复读取不是独立无人机观测。
数据来自固定 commit 的社区镜像，许可参考、署名与原发布入口见原始目录 SOURCES.md；尚未对照官方分发校验和。

## 下一步

先固定感知模型及其训练数据范围，再只用校准集拟合变化分数到目标概率的映射、估计观察错误率；验证集选更新参数；测试集保留最终评价。数据构建阶段未使用 GPU；后续已完成的本机 MPS 感知试跑见相邻的 `disaster_search_changeos_pilot/README.md`。尚未产生真实影像搜索成绩。
