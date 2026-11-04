"""Create a separate research source/report repository without altering sealed runs."""
import argparse,hashlib,json,shutil,re
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path,required=True);args=parser.parse_args()
    dest=args.destination.resolve()
    if dest.exists():raise RuntimeError('Destination must be new;never overwrite an existing checkout')
    dest.mkdir(parents=True)
    copied=[];skipped=Counter();partial=[]
    def copy(f):
        rel=f.relative_to(ROOT);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
        copied.append({'path':str(rel),'bytes':target.stat().st_size,'sha256':sha(target)})
    for name in ['work/disaster_route','work/disaster_search']:
        folder=ROOT/name
        for f in sorted(folder.rglob('*')):
            if not f.is_file():continue
            parts=f.relative_to(folder).parts
            if any(s in ['__pycache__','attachments'] or s.startswith('.venv') for s in parts):
                skipped['environment_checkpoint_external_attachment_files']+=1;continue
            if f.suffix.lower() not in ['.py','.md','.html','.json','.jl',''] or f.stat().st_size>2_000_000:
                skipped['source_nontext_or_large']+=1;continue
            copy(f)
    for folder in sorted((ROOT/'output').glob('disaster_*')):
        if not folder.is_dir():continue
        route=folder.name.startswith('disaster_route')
        for f in sorted(folder.rglob('*')):
            if not f.is_file():continue
            parts=f.relative_to(folder).parts
            if any(s.startswith('ranges') or s in ['cache','fixed_model_cache','calls','predictions','assets','evidence','qa','renders','__pycache__','证据快照'] for s in parts):
                skipped['raw_cache_media_or_runtime']+=1;continue
            if '.sparse.' in f.name or 'annotations' in f.name or 'alignment.json' in f.name or '_roi.' in f.name:
                skipped['source_data_or_partial_tiff']+=1;continue
            if len(parts)>(2 if route else 1):skipped['historical_nested_artifacts']+=1;continue
            if f.suffix.lower() not in ['.md','.json','.csv','.pdf','.png','.txt','.py','.html'] or f.stat().st_size>2_000_000:
                skipped['nontext_or_result_over_2MB']+=1;continue
            if f.suffix=='.png' and not any(s in str(f.relative_to(folder)).lower() for s in ['figures','panel','blackwhite','boundary','plot']):
                skipped['nonreport_images']+=1;continue
            if not route and f.suffix=='.json' and f.stat().st_size>300_000 and not any(s in f.name for s in ['summary','protocol','audit','receipt','manifest']):
                skipped['large_perception_detail']+=1;continue
            copy(f)
    # Preserve perception work and the latest advisor artifact,without email bodies or bundled raw evidence.
    for name in ['感知检查_阶段汇报_20261028','阶段汇报_灾前灾后Agent路线规划_20261103']:
        folder=ROOT/'output'/name
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix in ['.md','.json','.csv','.pdf','.docx'] and f.stat().st_size<=2_000_000:copy(f)
    # Historical receipts remain verbatim. Record explicitly which sealed payloads are absent.
    for receipt in sorted((dest/'output').rglob('completion_receipt.json')):
        data=json.loads(receipt.read_text());hashes=data.get('files',data.get('files_sha256',{}))
        if not isinstance(hashes,dict):continue
        included=0;missing=[]
        for rel,h in hashes.items():
            if not isinstance(h,str) or len(h)!=64:continue
            target=receipt.parent/rel
            if Path(rel).is_absolute() or not target.is_file():missing.append(rel)
            else:
                assert sha(target)==h,(receipt,rel)
                included+=1
        partial.append({'receipt':str(receipt.relative_to(dest)),'present_verified_entries':included,'absent_original_entries':missing,
          'scope':'Original receipt is historical provenance,not a claim that this Git snapshot contains all original experimental inputs.'})
    copy(ROOT/'requirements-diagnostics.txt')
    (dest/'docs').mkdir(exist_ok=True)
    (dest/'docs/archive_availability.json').write_text(json.dumps(partial,ensure_ascii=False,indent=2)+'\n')
    (dest/'README.md').write_text('''# 灾前灾后 Agent 搜查路线规划

灾前先验／线索与沿途观测证据组成双层地图，研究移动、观察、规划及返程共同预算下的搜查决策。

**当前尚未建立新方法贡献。** 本仓库保留代码、负结果、真实影像访问诊断和研究交接状态，避免把工程完成度当成算法创新。

## 接手入口

1. [研究目标与最新执行决定](work/disaster_route/研究目标与接手计划.md)
2. [初学者学习与实验路线](work/disaster_route/在线上下文反馈_学习与实验路线.md)／[第一课](work/disaster_route/在线上下文反馈_第一课.md)
3. [项目进展索引](work/disaster_route/README.md)
4. [NOAA实际数据入口审查](output/disaster_route_noaa_entry_v1/README.md)
5. [最小机制筛选及止损](output/disaster_route_verification_screen_v1/README.md)
6. [四区域强对照与观测增量诊断](output/disaster_route_crasar_batch4_v1/README.md)

已完成[固定Pearson算子的区域外同预算检验](output/disaster_route_crasar_pearson_isobudget_v1/README.md)：3区域、54条闭合路线、32次本机先验前向。主预算100代理秒下，Pearson／幅度空间／纯空间／精确静态累计核查阳性为3／3／4／5；未建立稳定方法增量。算子保留为标准对照，不继续调Boone特征。全部源标签、代理成本与配准边界见报告。此前[Boone两步初筛](output/disaster_route_boone_two_step_v1/README.md)及[小VLM结果](output/disaster_route_boone_context_run_v2/README.md)保持原字节。

补充[Iona轨迹与排序—收益诊断](output/disaster_route_iona_ranking_diagnostic_v1/README.md)：已定位第三次阴性反馈后的选点分叉，并提供同首访截面的AP—路线命中图；公开日志可独立重画，无需模型或原始影像。当前只补分析，写稿后置。

## 四层证据与快速复现

|层次|原实验脚本（work/disaster_route/）|已记录资源与耗时|核心结论|
|---|---|---|---|
|受控机制|`run_verification_screen.py`|CPU精确枚举；原总耗时未记录|强制校准60/108更差，可选11/108改善；非一般理论|
|VLM／低阶表征|`run_boone_context_experiment.py`|本机MPS，276次前向329.61s；总348.71s|完整历史AP0.342低于不更新0.377；低阶核未超幅度控制|
|局部与单点诊断|`run_boone_structure_check.py`、`run_boone_two_step.py`|CPU缓存计算；原总耗时未记录|Top-1 9:1，去B03后3:1；两步8:5非同航程|
|区域外同预算|`run_crasar_pearson_isobudget.py`|32次本机MPS先验14.37s；路线CPU计算|100s合计静态5、纯空间4、Pearson3|

无API费用、无GPU租赁；本机MPS属于本地GPU，不是CPU。运行下列命令只复算已发布表格和两张诊断图，不重新执行模型或原路线实验：

```bash
python -m pip install -r requirements-diagnostics.txt
python work/disaster_route/reproduce_diagnostic_evidence.py --output output/reproduced_diagnostic_evidence
```

仅需表格时可用`--tables-only`，无第三方依赖。输出目录须不存在。本机公开快照完整复现约0.51s、标准库表格约0.02s，不含安装与Python启动；无需影像、权重或网络。见[证据包](output/disaster_route_four_stage_evidence_v1/README.md)与[结论核验](output/disaster_route_four_stage_evidence_v1/结论核验与边界.md)。当前先收敛证据，写稿后置。

## 当前结论（2026-11-03快照）

- 泛化的“路线分歧时考虑复核”在108个受控条件下与同策略空间精确贝叶斯参考收益一致，停止作为独立新方法扩展。
- NOAA影像实际覆盖已有Boone区域，但共同现场真值、可靠校准模型和同状态UAS复访尚未建立；不能替代受控实验中的校准信号。
- 固定VLM先验、付费反馈与Pearson更新已完成Boone外同预算开发检验；当前保留为比较基线，尚无稳定路线增量。5节点教学与VLM上下文路线收益仍未完成，不恢复旧传播或复核规则调参。
- 当前真实区域结果属于档案图像访问回放；运动与服务成本含假设，不宣称真实飞行验证。

## 代码和材料

- `work/disaster_route/`：路线、双层地图、精确参考、数据核验、研究筛选与专项测试。
- `work/disaster_search/`：历史固定感知与数据检查源码；不包含本地环境、下载附件或模型权重。
- `output/disaster_route*/`：报告、协议、摘要、部分结果与来源快照。
- `output/disaster_search*/`：历史感知检查报告与轻量结果。
- `output/感知检查_阶段汇报_20261028/`：保留的感知阶段材料。

## 最小验证

Python3.12。下列6项受控筛选测试不下载影像、不调用模型，也不改封存结果：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m unittest discover -s work/disaster_route -p test_verification_screen.py -v
python tools/verify_snapshot.py
```

影像核验、历史感知推理和报告导出需要额外依赖及原数据，见[环境与复现边界](docs/REPRODUCIBILITY.md)。本快照不是所有实验的一键完整复现包。

## 数据与归档边界

不上传原始影像、模型权重、NumPy预测缓存、逐块下载缓存、部分重建TIFF、本地Python环境及邮件正文。原始数据引用和固定版本保留在实验协议中。选入的旧报告与完成收据保持原字节；它们可能引用未上传的输入或原机器绝对路径。

上传文件由[快照清单](snapshot_manifest.json)逐项记录SHA256。旧完成收据中缺失的原始文件见[归档可用性清单](docs/archive_availability.json)；不能将旧收据当成GitHub端完整数据归档证明。

未替项目统一指定开源许可证。第三方来源和数据许可见[来源说明](docs/THIRD_PARTY.md)；公开托管不改变其许可和引用要求。
''',encoding='utf-8')
    (dest/'requirements-core.txt').write_text('numpy>=1.26\nscipy>=1.11\n')
    (dest/'docs/REPRODUCIBILITY.md').write_text('''# 环境与复现边界

最小受控筛选只需Python3.12、NumPy与SciPy。README列出的6项测试已在导出快照中运行；测试验证受控模型逻辑，不证明灾害效果。

历史影像核验通常还需要Pillow、tifffile、rasterio、pyproj、affine、opencv-python；感知推理还需PyTorch与torchvision。报告构建脚本另有python-docx、绘图或本地渲染工具依赖。它们未全部打包，也未在全新跨平台环境逐项验证。

部分脚本仍引用原工作区的`.venv_inference`、`geo_libs`、固定检查点或绝对数据路径。导出保留科学记录原文与源码，不通过静默改写历史路径破坏哈希。再次运行历史实验前，应另建协议与输出目录、补齐所需数据与依赖；不要直接覆盖已有封存目录。

CRASAR固定revision：`4c2732a7ab9658517ca8caa69bfee9e1f87ca532`。数据入口：https://huggingface.co/datasets/CRASAR/CRASAR-U-DROIDs 。卫星／UAS标注是来源视角参考，并非共同现场真值。

原始StrongBaseline检查点SHA256：`fe7ded8c5012dee3d39675226b0aa1fd671d8c19cef5c88080daa8b13645c6b6`，下载与来源见历史acquisition/protocol文件。检查点与外部模型附件不随仓库分发。

`tools/verify_snapshot.py`只核验本次上传文件，`docs/archive_availability.json`披露原封存文件的缺项；历史完成收据验证程序可能因数据未上传而拒绝运行，这是预期边界。
''',encoding='utf-8')
    (dest/'docs/THIRD_PARTY.md').write_text('''# 第三方来源与许可

- CRASAR-U-DROIDs：数据集标示CC BY4.0；固定版本及引用见各protocol。原影像与标注数据附件不在本次上传范围，报告中的局部面板用于说明实验。
- `work/disaster_route/vendor/sbo_aippms_24db0a6/`：保留作者Julia参考代码及随附许可；AIPPMS相关输出中的UPSTREAM_LICENSE及协议也保留。
- CrossViewGate：作者仓库https://github.com/rayford295/CrossViewGate ，固定commit`afdd819f6fdbf20ae4563c1ed89ee641859b40f8`；项目标示Apache2.0。本仓库保留的少量源码／说明是近邻方法审查证据，不是本项目原创算法；来源标注见对应sources目录。
- StrongBaseline与其他外部感知模型：保留适配、获取与审计脚本；权重、下载附件和外部模型源码未批量再分发。

该说明不替代各来源的原许可证，也不授予未明确授予的权利。项目未统一开源授权。
''',encoding='utf-8')
    (dest/'.gitignore').write_text('''__pycache__/
*.py[cod]
.pytest_cache/
.venv*/
.env
.env.*
*.pt
*.pth
*.ckpt
*.safetensors
*.npz
*.npy
*.tif
*.tiff
*.bin
*.zip
*.log
.DS_Store
attachments/
geo_libs/
ranges*/
''')
    tools=dest/'tools';tools.mkdir(exist_ok=True)
    (tools/'verify_snapshot.py').write_text('''"""Verify only the files included in this GitHub research snapshot."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'snapshot_manifest.json').read_text())
for row in manifest['files']:
    p=root/row['path']
    assert p.is_file(),row['path']
    assert p.stat().st_size==row['bytes'],row['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['path']
print(f"Verified {len(manifest['files'])} exported files;original large data/caches are intentionally not included.")
''')
    # Validate known credential families without printing matched content.
    patterns=[rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'github_pat_[A-Za-z0-9_]{40,}',rb'AKIA[A-Z0-9]{16}',rb'sk-(?:proj-)?[A-Za-z0-9_-]{40,}',rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
    hits=[]
    for f in dest.rglob('*'):
        if f.is_file() and f.suffix.lower() in ['.py','.json','.md','.txt','.csv','.html','.jl']:
            if any(re.search(p,f.read_bytes()) for p in patterns):hits.append(str(f.relative_to(dest)))
    if hits:raise RuntimeError('Potential credential files require review:'+','.join(hits))
    exported=[{'path':str(f.relative_to(dest)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(dest.rglob('*')) if f.is_file()]
    manifest={'snapshot_date':'2026-11-03','scope':'Disaster before/after Agent route-planning research only;unrelated GR-PTA project excluded',
      'is_full_data_archive':False,'files':exported,'excluded_counts_by_reason':dict(skipped),
      'original_scientific_records_rewritten':False,'credential_pattern_scan':'No known credential patterns found in included text files;not a guarantee for arbitrary secrets',
      'original_copied_files':len(copied),'total_included_bytes':sum(f['bytes'] for f in exported)}
    (dest/'snapshot_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'destination':str(dest),'files':len(exported)+1,'bytes':manifest['total_included_bytes'],'secret_pattern_hits':len(hits),'partial_receipts':len(partial)},ensure_ascii=False))

if __name__=='__main__':main()
