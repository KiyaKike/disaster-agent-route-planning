# Boone执行版：本地小VLM与统计更新

2026-11-03。执行入口为`run_boone_context_experiment.py`。模型使用[HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)，固定revision `7e3e67edbbed1bf9888184d9df282b700a323964`，不训练模型，不调用付费API。模型官方许可为Apache-2.0；权重不提交仓库。

本轮将原协议中待固定的设置具体化，先运行一次完整的本地可行性实验，再根据结果决定下一步。

## 执行设置

- 12栋固定建筑，生成灾前卫星、灾后卫星、UAS、NOAA原地理版本和NOAA局部修正版，共60张256×256建筑掩膜切片。
- 主预测只输入目标的灾前／灾后卫星切片；有历史时再输入单个锚点的灾前／灾后卫星、UAS切片和作者UAS视角标签。NOAA图仅作输入检查，不给主模型。
- 图像处理固定最长边512、关闭自动多切片，避免把256px原图放大到2048后重复编码；所有方法共用相同裁剪源。
- 问题统一为二分类A（可见屋顶损伤）／B（未见屋顶损伤）。分数对裸字母及带前导空格的A／B token按类别合并后条件归一化，同时保存这两类共四个token占全词表概率的质量及最高概率token，便于检查输出倾向。它不是模型口述的小数，也未承诺物理概率校准。
- 先跑12次无历史预测作为共同p0，再对每个锚点的其余11栋预测，再做132次仅标签历史消融，总计276次正式前向。统计组使用完全相同的p0与锚点标签。
- S1：50m高斯距离核、0.5残差增益。S2在同式上乘视觉相似度；不根据本轮标签调参。
- 视觉特征固定为灾后RGB直方图24维、灾后均值／标准差6维、灾前灾后像素差均值／标准差6维。标准化只用出发前卫星图，标准差下限0.05，相似度尺度为非零特征距离的中位数。
- UAS视角标签为主要评分参考，NOAA另报；模型预测的“可见屋顶损伤”与作者综合建筑损伤标签可能不完全等价，本轮保留该任务差异。
- 输出逐调用Prompt／输入路径／分数／耗时，逐锚点评分及汇总表。每个案例从空上下文独立前向，不跨案例共享历史。

本轮S2是低阶视觉统计参考，尚不是冻结语义编码器参考；本地256M模型也是小模型可行性试验。先看实际行为，不将单次胜负泛化为全部统计方法或全部VLM的结论。

## 运行

使用独立环境，保留原固定感知环境。正常独立安装示例：

```bash
python3.12 -m venv work/disaster_route/.venv_context
work/disaster_route/.venv_context/bin/python -m pip install torch==2.9.1 transformers==4.57.1 pillow numpy
HF_HUB_DISABLE_XET=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
work/disaster_route/.venv_context/bin/python -u \
  work/disaster_route/run_boone_context_experiment.py \
  --output output/disaster_route_boone_context_run_v2 \
  --revision 7e3e67edbbed1bf9888184d9df282b700a323964
```

首次需下载约513MB权重及少量配置，后续可读缓存。本机复用了既有感知环境的Torch／NumPy包路径，但Transformers安装在新的环境内，不修改原感知环境。脚本读取本机已有区域图像；公共仓库不含原始影像，不能仅靠克隆直接重跑。

## 学习时先看三件事

1. 无历史分数怎样形成初始地图；加入一个真实来源记录后，其他分数是否改变。
2. 空间核会同时影响哪些建筑；再乘视觉相似度后，哪些建筑受影响更小。
3. 分数改变是否改善对留出来源标签的排序，以及模型前向要花多少时间。

最小对比现已完成；下一轮先做较强模型固定锚点复核，再决定是否接路线。

已完成结果：[首轮实际比较](../../output/disaster_route_boone_context_run_v2/README.md)。旧v1只作评分格式调试，正式结果以v2为准。
