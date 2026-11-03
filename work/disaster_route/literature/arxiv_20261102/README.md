# 灾前灾后搜查路线规划相关论文

检索日期：2026年11月2日。范围为 arXiv 上的主动搜查、带预算的信息路径规划、先验地图修正及双层信念地图。下列论文提供不同环节的建模依据；它们各自的实验不能直接证明本项目在灾损影像上的有效性。

具体接入方式、实验顺序及继续条件见[论文应用与实验推进方案](论文应用与实验推进方案.md)。

2026年11月2日执行更新：[修正后的AIPPMS受控机制参考](../../../../output/disaster_route_aippms_control_v1/README.md)已完成。作者Julia环境与论文原表格尚未复现，固定源码审计、受控Python结果及真实任务的接入边界见报告。

当前优先阅读 Search-TTA、AIPPMS 和 SBO-AIPPMS：前者说明观测怎样修正语义先验，后两者将观察和移动分为需要共同预算的行动。IA-TIGRIS适合补强在线规划基线，AeroBelief用于明确两层记忆与证据接纳规则。

## 与当前问题最直接的工作

### Search-TTA

**Search-TTA: A Multimodal Test-Time Adaptation Framework for Visual Search in the Wild**，Derek Ming Siang Tan 等，2025。本文采用2025年11月7日的v5，避免混用早期摘要中的结果。

[论文](https://arxiv.org/abs/2505.11350v5) · [方法5.3与限制8](https://arxiv.org/html/2505.11350v5) · [作者代码](https://github.com/marmotlab/Search-TTA-RL-VLN)

方法从卫星图像生成搜索分布，按语义特征分组，并用沿途观测更新共享编码器。阴性反馈的影响与对应语义区域的已观察比例有关。其机制提供了观测影响未访问区域的路径，区别于直接给四邻格加减分。

**关键限制是第8节的完美二值传感器假设：主要搜索表述不处理误检和漏检。** 硬件环节是实体小型无人机与Gazebo感知的闭环，不是野外灾后搜救验证。生态目标与灾损巡查也有任务差异。

对本项目的推论：应检验含噪观测会不会把先验越改越错，以及语义关联是否真的有预测力。这个推论尚未建立新颖性或有效性，不能据此恢复已经失败的邻域规则，也不意味着直接移植在线训练即可奏效。

### AIPPMS

**Adaptive Informative Path Planning with Multimodal Sensing**，Shushman Choudhury、Nate Gruver、Mykel J. Kochenderfer，2020，ICAPS。

[论文](https://arxiv.org/abs/2003.09746) · [原文第3至5节](https://arxiv.org/html/2003.09746)

问题将移动和不同成本、精度、范围的传感操作分开，使用POMDP在线规划，并在动作选择中保留抵达终点所需的资源。已访问节点不重复获得任务收益，但仍可在那里购买对其他节点有用的观测。其模拟搜救任务因此具备明确的信息价值。

对本项目的推论：下一步需要验证观察动作能提供什么新增信息，而不只改目标格子的排序。原文有已知随机观测模型和环境关联假设；这些参数不能从我们的未校准损伤分数中直接得到。

### SBO-AIPPMS

**Sequential Bayesian Optimization for Adaptive Informative Path Planning with Multimodal Sensing**，Joshua Ott、Edward Balaban、Mykel J. Kochenderfer。arXiv首发2022，作者仓库对应ICRA 2023论文。

[论文](https://arxiv.org/abs/2209.07660) · [作者代码](https://github.com/sisl/SBO_AIPPMS)

使用Gaussian process表示环境信念，结合任务收益与信息收益，以树搜索选择后续移动或传感动作。作者Julia仓库还包含获许可的2020年AIPPMS实现，便于在同一任务中建立对照。

它的环境包括RockSample和行星车探索；相关性和奖励与灾后建筑任务不同。可借鉴其分离实际任务收益、传感费用与信息收益的实验结构，但需要独立验证灾损空间关联。

## 规划求解与系统接口

### IA-TIGRIS

**IA-TIGRIS: An Incremental and Adaptive Sampling-Based Planner for Online Informative Path Planning**，Brady Moon 等。首发2025，采用2026年4月1日修订的v3；早期版本与v3的作者列表及结果有变化。

[论文](https://arxiv.org/abs/2502.15961v3) · [作者项目与代码入口](https://ia-tigris.github.io/)

在线采样规划器复用已有搜索树，增量细化路径并随信念地图更新，论文包含多旋翼和固定翼实验。它适合作为在线规划求解器的相关工作和候选强基线。

对本项目的推论：可以分别考察地图更新和规划器复用，避免把更强求解器的收益算给双层地图。论文并不保证任何不可靠地图更新都会改善任务结果，也不是灾前灾后的累计建筑巡查实验。

### Shrinking POMCP

**Shrinking POMCP: A Framework for Real-Time UAV Search and Rescue**，Yunuo Zhang 等，2024。

[论文](https://arxiv.org/abs/2411.12967) · [原文第四节](https://arxiv.org/html/2411.12967v1)

针对无人机搜救的有限可见性与决策时限建立POMDP，在2D及AirSim-ROS2模拟器中比较搜索策略。方法尝试复用树中的后续动作，降低反复从头求解的开销。

它可以帮助设计规划延迟、执行与再决策的对照。其搜索时间指标、目标设定和模拟任务不能直接替代本项目的累计损伤覆盖与返程预算评价。

## 先验地图与证据地图

### AeroBelief

**Dual-Layer Semantic-Spatial Belief Mapping for Aerial Object Goal Navigation**，Jianqiang Xiao 等，2026。

[论文](https://arxiv.org/abs/2609.08164) · [证据筛选与双层更新III-C至III-E](https://arxiv.org/html/2609.08164v1)

区分情境线索和符合接纳条件的目标证据，对两层采用不同更新方式，并将语义信念与覆盖引导共同提供给规划器。重复重叠观测不会简单相加为更强证据。

它支持借鉴记忆和证据接纳规则；其目标是找到并接近指定物体，与累计巡查多个受损对象不同。双层地图、门控融合和覆盖引导已有相关方法，不能将这些概念本身当作我们的新贡献。

### Explore From Sketch

**Explore From Sketch: Accelerating UAV Exploration in Large-scale Environments with Prior Maps**，Tiancheng Lai 等，2026。

[论文](https://arxiv.org/abs/2606.11708) · [原文](https://arxiv.org/html/2606.11708v1)

利用不完整、存在形变或未对齐的2D先验地图辅助LiDAR无人机探索。方法保留配准假设，结合风险选择及分层视点路径规划，并报告实地探索实验。

它与灾前底图可能过时的设定有关，但主要解决几何配准和探索，不能直接作为灾损目标先验更新方法。当前项目缺少其LiDAR、3D定位和真实飞行数据，完整复现成本明显高于小规模规划对照。

### HaltNav

**HaltNav: Reactive Visual Halting over Lightweight Topological Priors for Robust Vision-Language Navigation**，Zihui Yu、Pingcong Li、Bichi Zhang、Sören Schwertfeger，2026，v2。

[论文](https://arxiv.org/abs/2603.12696v2)

利用轻量拓扑先验导航；执行中发现障碍后停止局部执行、使相应拓扑连接失效，再规划绕行。在线观测在这里改变路线的可行性，因此对动作有明确影响。

这一点可作为“何时必须重规划”的邻近参考，但本文是室内VLN。地面通行变化不能直接移植成无人机禁飞条件，现有灾损标签也不是道路阻断真值。

## 补充阅读

**DIFF-IPPO: Diffusion-Based Informative Path Planning with Open-Vocabulary Belief Maps**，Sausar Karaf 等，2026，v2。[论文](https://arxiv.org/abs/2606.16780v2)。将开放词汇信念图与扩散轨迹生成结合，评价包含合成信念图和寻找燃烧建筑的模拟任务。可用来了解近期地图到轨迹的接口，但不优先投入扩散模型训练；其评价也没有证明灾前灾后的真实巡查闭环。

## 对下一轮实验的建议

先用AIPPMS或SBO-AIPPMS的明确观测模型验证“额外信息是否值得付费”，建立已有方法对照。再参考Search-TTA，检查真实灾前结构或语义分组能否支持对未访问区域的推断。仅在独立数据支持传感误差与关联模型后，比较静态、在线不传播和候选方法的同预算结果。

一个由文献引出的待验证问题是：**灾前先验存在失配、沿途观测有误检漏检时，Agent何时应该改变搜索分布，何时应该保留原计划？** 双层地图用于分别保存先验与证据；需证明的是更新与行动策略的收益。AIPPMS已处理随机传感，AeroBelief已处理证据接纳，因此“加入噪声”或“增加门控”本身不足以建立新颖性。

本次没有下载大型数据、模型权重，未安装环境或复现上述方法。代码入口已访问不代表运行成功。检索命中和版本记录保存在相邻JSON文件中，现有路线实验与感知归档均未改动。
