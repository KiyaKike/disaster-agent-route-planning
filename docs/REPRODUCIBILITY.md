# 环境与复现边界

最小受控筛选只需Python3.12、NumPy与SciPy。README列出的6项测试已在导出快照中运行；测试验证受控模型逻辑，不证明灾害效果。

历史影像核验通常还需要Pillow、tifffile、rasterio、pyproj、affine、opencv-python；感知推理还需PyTorch与torchvision。报告构建脚本另有python-docx、绘图或本地渲染工具依赖。它们未全部打包，也未在全新跨平台环境逐项验证。

部分脚本仍引用原工作区的`.venv_inference`、`geo_libs`、固定检查点或绝对数据路径。导出保留科学记录原文与源码，不通过静默改写历史路径破坏哈希。再次运行历史实验前，应另建协议与输出目录、补齐所需数据与依赖；不要直接覆盖已有封存目录。

CRASAR固定revision：`4c2732a7ab9658517ca8caa69bfee9e1f87ca532`。数据入口：https://huggingface.co/datasets/CRASAR/CRASAR-U-DROIDs 。卫星／UAS标注是来源视角参考，并非共同现场真值。

原始StrongBaseline检查点SHA256：`fe7ded8c5012dee3d39675226b0aa1fd671d8c19cef5c88080daa8b13645c6b6`，下载与来源见历史acquisition/protocol文件。检查点与外部模型附件不随仓库分发。

`tools/verify_snapshot.py`只核验本次上传文件，`docs/archive_availability.json`披露原封存文件的缺项；历史完成收据验证程序可能因数据未上传而拒绝运行，这是预期边界。
