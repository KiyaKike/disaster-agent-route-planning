# 与下一轮实验直接相关的工作：定向核查

核查日期 2026-10-25。这是用于选实验和强对照的定向检索，不是完整领域综述，也不证明任何尚未验证的研究空白。只以作者论文、正式论文集或项目页作为依据。

| 工作 | 已核对内容及边界 | 对本项目的具体约束 |
|---|---|---|
| [Efficient Nonmyopic Active Search，Jiang 等，ICML 2017](https://proceedings.mlr.press/v70/jiang17d.html) | 论文集元信息、摘要及原文检索段落；目标是在有限标注预算下发现尽量多的目标，考虑剩余预算。 | 预算感知探索与利用已有明确基础，不能把普通反馈选点作为独立创新。补充 ENS 类强对照。 |
| [Nonmyopic Multifidelity Active Search，Nguyen、Modiri、Garnett，ICML 2021](https://proceedings.mlr.press/v139/nguyen21f.html) | 论文集页面及 PDF 问题定义；使用不同成本的查询源，低保真源辅助高保真发现。 | “初查还是复核”的多保真分配已有直接相关方法。我们当前固定串行动作与原文设置存在差别，适配必须说明。 |
| [MICRO: Multi-Fidelity Active Search for Severe Error Discovery，2026](https://arxiv.org/html/2609.26025v1) | 已读问题定义、联合反馈模型、采集规则、对照和局限；应用于翻译错误发现。它把弱/强反馈放在共享预算下，并比较 MF-ENS 适配。 | 联合建模相关反馈、付费探查和 rollout 也不是单独的新颖点。其经验收益不直接迁移到图像搜索；该稿是预印本，非已确认录用的方法。 |
| [An Active Search Strategy with Multiple Unmanned Aerial Systems for Multiple Targets，2024](https://arxiv.org/html/2406.16370v1) | 已读摘要、任务与相关工作；由粗先验结合局部/全局信息开展多目标无人机搜索。 | 粗图先验加局部反馈并非空白；我们的二维图像代理不能直接与其连续轨迹结果比较。 |
| [AeroBelief: Dual-Layer Semantic-Spatial Belief Mapping for Aerial Object Goal Navigation，2026](https://arxiv.org/html/2609.08164v1) | 已核对摘要、双层信念、保守证据与覆盖引导章节；目标是空中 ObjectNav。 | 可靠性门控、空间记忆和覆盖探索已有相关设计。逐栋灾损累计发现与寻找指定目标的任务及成功判据需要区分。 |
| [DORA，Can LLM Agents Respond to Disasters?，2026，v2](https://arxiv.org/abs/2605.11633v2) | 已核对最新版本元信息及摘要；灾害应急地理空间工具组合与推理评价。 | 可以借鉴工具输入和失败诊断；它的任务定义不能替代本项目的观察动作、真实成本和发现评价。旧汇报引用应标明版本。 |
| [ESARBench: A Benchmark for Agentic UAV Embodied Search and Rescue，2026](https://arxiv.org/abs/2605.01371v1) | 已核对摘要及作者项目页入口；UE5/AirSim 场景下的具身搜救。未复现。 | 若后续扩大到真实视角、天气和航行安全，需要与该类具身基准对齐。当前不把二维缓存实验称为完整具身搜救验证。 |

## 对方法选择的判断

目前不能把“质量条件贝叶斯 + 邻域传播 + 预算分配”三个模块的组合直接写成创新。下一步先用相关性正确、目标对齐的强基线，测量粗先验失效与复核机会是否留下稳定的可改进空间。

暂定、可被实验否定的候选问题是：**当粗图先验在不同事件上失准、同源初查与复核高度相关时，怎样以新增建筑发现为效用分配有限观察预算？** 是否有区别于既有多保真搜索的实质新问题，仍需完整比较空间重复目标、不可获得真值查询与感知失准条件。此处没有宣称这些条件从未被研究。

## 后续补读项

- [Visual-language active search for wide-area remote sensing imagery](https://www.sciencedirect.com/science/article/abs/pii/S0031320326000695)：检索发现的直接相关候选。出版商页面此次返回 403，尚未据此形成方法或性能结论，需取得作者公开全文后核对。
- MF-ENS 与 ENS 的具体算法、原始实现和可比预算适配，需在新方法实验前完成。当前没有将任何适配标成忠实复现。

网页来源快照中保留官方数据页、TACO 数据卡和 DisasterAdaptiveNet 项目说明，见 `sources/browser_sources.json`。论文链接均为原始出处，未引用聚合网站的技术结论。
