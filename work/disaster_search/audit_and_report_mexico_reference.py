"""Independently replay the fixed-three reference cross-check and source split audit.

Uses existing caches only. Does not import or instantiate a prediction model.
Stage audit writes descriptive reports; stage complete seals all output hashes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'output/disaster_search_mexico_trace3_v1'
PARENT = ROOT / 'output/disaster_search_perception_pair72_v1'
PROSPECTIVE = OUT / 'prospective_sources'
COMMIT = 'fada4e791766f810dc815a4e82d52f2bb27116c6'
REPO = 'https://github.com/SebastianHafner/DisasterAdaptiveNet'
NEW_SOURCES = ['mexico_reference_core.py', 'test_mexico_reference.py',
               'trace_mexico_reference.py', 'audit_and_report_mexico_reference.py']


def read(p): return json.loads(Path(p).read_text())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(p, h):
    if digest(p) != h: raise ValueError('Changed file: ' + str(p))
def write(n, value):
    (OUT/n).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n')


def audit():
    protocol, parent = read(OUT/'protocol.json'), read(PARENT/'completion_receipt.json')
    checked(PARENT/'completion_receipt.json', protocol['parent_receipt_sha256'])
    # Audit every parent output and declared external dependency, not only three rows.
    for n,h in parent['files'].items(): checked(PARENT/n,h)
    for n,h in parent['external_files'].items(): checked(n,h)
    for n,h in parent['source_code_sha256'].items(): checked(HERE/n,h)
    for n,h in protocol['source_code_sha256'].items(): checked(HERE/n,h)
    checked(OUT/'test_log.txt',protocol['test_log_sha256'])
    assert 'Ran 341 tests' in (OUT/'test_log.txt').read_text()
    assert (OUT/'test_log.txt').read_text().rstrip().endswith('OK')
    extraction, ledger = read(OUT/'extraction_report.json'), read(OUT/'transfer_ledger.json')
    metadata = {r['shard_path']:r for r in extraction['metadata']}
    locations = {r['shard_path']:r for r in protocol['selected_locations']}
    for location in locations.values(): checked(location['source_names_file'],location['source_names_sha256'])
    assert set(metadata)==set(locations) and len(ledger)==15
    received = 0
    for row in ledger:
        assert row['status']=='COMPLETE' and row['received']==row['bytes']
        assert re.fullmatch('[a-f0-9]{64}',row['sha256'])
        a,b = row['offset'],row['offset']+row['bytes']
        meta,loc = metadata[row['shard']],locations[row['shard']]
        assert 0<=a<b<=loc['shard_size'] and row['bytes']<=1024*1024
        if row['role']=='parquet_trailer': assert a==loc['shard_size']-8 and row['bytes']==8
        elif row['role']=='parquet_footer_metadata_only':
            assert b==loc['shard_size']-8 and row['bytes']<=256*1024
        else:
            assert row['role']=='transient_selected_group_mask_or_identity'
            assert any(x<=a<b<=y for x,y in meta['allowed_ranges'])
        assert not any(max(a,x)<min(b,y) for x,y in meta['forbidden_ranges'])
        received+=row['received']
    assert received==extraction['received_bytes']==extraction['reserved_bytes']==1898297
    assert received<=protocol['max_network_bytes']
    assert extraction['new_RGB_payloads']==extraction['new_model_calls']==0
    assert not extraction['unselected_mask_payloads_persisted']
    assert not extraction['unselected_outcomes_returned_or_scored']
    assert set(p.name for p in (OUT/'published_masks').iterdir())=={s+'.png' for s in protocol['selected_ids']}
    assert {p.suffix for p in OUT.rglob('*') if p.is_file()} <= {'.json','.txt','.png','.yaml','.md','.py'}
    refs = {r['sample_id']:r for r in read(PARENT/'reference_manifest.json')}
    rgb = {r['sample_id']:r for r in read(PARENT/'rgb_jobs.json')}
    observations = read(PARENT/'observations.json')
    descriptions = {r['sample_id']:r for r in read(OUT/'crosscheck.json')['results']}
    maskrows=read(OUT/'published_mask_manifest.json')
    assert sorted(r['sample_id'] for r in maskrows)==protocol['selected_ids']
    verified=[]; external={str(PARENT/'completion_receipt.json'):digest(PARENT/'completion_receipt.json')}
    for name in ['reference_manifest.json','rgb_jobs.json','observations.json','mexico_failure_description.json']:
        external[str(PARENT/name)]=digest(PARENT/name)
    for source in protocol['selected_locations']:
        external[source['source_names_file']]=source['source_names_sha256']
    for m in maskrows:
        sid=m['sample_id']; ref=refs[sid]; desc=descriptions[sid]
        checked(m['path'],m['sha256'])
        with Image.open(m['path']) as im:
            assert im.format=='PNG' and im.size==(1024,1024) and im.mode in ('L','P')
            published=np.asarray(im).copy()
        assert hashlib.sha256(published.tobytes()).hexdigest()==m['pixel_sha256']
        assert published.dtype==np.uint8 and set(np.unique(published))<={0,1,2,3,4,255}
        for key in ('reference','label'):
            path=ref['reference_file'] if key=='reference' else ref['label_path']
            h=ref['reference_sha256'] if key=='reference' else ref['label_sha256']
            checked(path,h); external[path]=h
        truth=np.asarray(Image.open(ref['reference_file']))
        valid=np.isin(truth,range(5)) & np.isin(published,range(5))
        cm=np.bincount((truth[valid].astype(np.int64)*5+published[valid]),minlength=25).reshape(5,5)
        disagreements=int(cm.sum()-cm.trace()); assert disagreements==0
        a=np.isin(truth,[3,4]) & valid; b=np.isin(published,[3,4]) & valid
        intersection,union=int(np.count_nonzero(a&b)),int(np.count_nonzero(a|b))
        assert union>0 and intersection==union
        assert intersection==desc['correspondence']['severe_intersection_pixels']
        assert desc['correspondence']['severe_iou']==1.0
        doc=read(ref['label_path']); features=[f for f in doc['features']['xy'] if f['properties']['uid']==desc['derived_uid']]
        assert len(features)==1; feature=features[0]
        assert feature['properties']['source_object_index']==desc['source_object_index']
        assert feature['properties']['subtype']==desc['reference_grade']=='major-damage'
        points=np.array([[float(v) for v in s.split()] for s in feature['wkt'][10:-2].split(',')])
        footprint=np.zeros((1024,1024),np.uint8);cv2.fillPoly(footprint,[np.round(points).astype(np.int32)],1)
        footprint=footprint.astype(bool); domain=footprint & valid
        assert int(footprint.sum())==desc['footprint_pixels']
        assert int(domain.sum())==desc['correspondence']['footprint_common_valid_pixels']
        assert np.all(published[domain]==3)
        center=(points.min(0)+points.max(0))/2
        x,y=[max(0,min(864,int(round(v))-80)) for v in center]
        assert desc['crop_bounds_xyxy']==[x,y,x+160,y+160]
        for image in rgb[sid]['images']: checked(image['path'],image['sha256']);external[image['path']]=image['sha256']
        for model in ('swint','r34'):
            rows=[r for r in observations if r['sample_id']==sid and r['role']=='production' and r['model_id']==model]
            assert len(rows)==4 and {r['tile_id'] for r in rows}=={0,1,2,3}
            building=np.zeros((1024,1024),bool); classes=np.zeros((1024,1024),np.uint8)
            for row in rows:
                checked(row['cache_file'],row['cache_sha256']);external[row['cache_file']]=row['cache_sha256']
                with np.load(row['cache_file'],allow_pickle=False) as cache:
                    c=cache['classes'];loc=cache['building'];assert c.shape==loc.shape==(512,512)
                    assert c.dtype==np.uint8 and loc.dtype==bool
                    x0=(row['tile_id']%2)*512;y0=(row['tile_id']//2)*512
                    classes[y0:y0+512,x0:x0+512]=c;building[y0:y0+512,x0:x0+512]=loc
            observed={str(k):int(np.count_nonzero(classes[footprint]==k)) for k in range(5)}
            assert observed==desc['prediction_description'][model]['ungated_class_counts_inside_footprint']
            assert observed['2']==observed['3']==observed['4']==0
            assert float(building[footprint].mean())==desc['prediction_description'][model]['building_fraction_inside_footprint']
        assert desc['human_damage_reassessment'] is None and not desc['original_json_object_identity_verified']
        verified.append({'sample_id':sid,'common_known_pixels':int(valid.sum()),'class_confusion':cm.tolist(),
            'severe_pixels_common_domain':union,'fixed_footprint_pixels':int(footprint.sum()),
            'fixed_footprint_excluded_pixels':int((footprint & ~valid).sum()),'known_class_disagreements':disagreements})
    with Image.open(OUT/'figures/failure_closeups.png') as image: assert image.size==(1600,1100)
    write('audit_report.json',{'status':'PASS','parent_output_files_verified':len(parent['files']),
        'parent_external_files_verified':len(parent['external_files']),
        'old_frozen_sources_verified':len(parent['source_code_sha256']),
        'protocol_sources_verified':len(protocol['source_code_sha256']),
        'transfers_checked':len(ledger),'mask_transfer_bytes':received,'scene_comparisons_replayed':3,
        'cached_model_descriptions_replayed':6,'new_model_calls':0,'results':verified,
        'limitations':'Ledger intervals replayed from recorded metadata; network bodies deliberately not cached. No original JSON/UID or independent human truth recovered.'})
    write('audit_external_files.json',external)
    split_report()
    report(verified)
    print(json.dumps({'audit':'PASS','comparisons':3,'severe_pixels':sum(v['severe_pixels_common_domain'] for v in verified),
                      'old_sources_unchanged':len(parent['source_code_sha256']),'new_model_calls':0}))


def split_report():
    folder_files={}
    for i in (0,1):
        folder=read(PROSPECTIVE/f'drive_folder_{i}.json')
        folder_files.update({f['title']:f for f in json.loads(folder['content'])['files']})
    split_sets={}; rows=[]
    for split in ['train','val','test']:
        name=f'{split}_samples_event_split.txt'; file=PROSPECTIVE/name
        data=file.read_bytes(); meta=folder_files[name]
        assert len(data)==int(meta['size']) and b'\r\n' in data
        paths=data.decode('ascii').splitlines()
        assert paths and len(set(paths))==len(paths)
        assert all(re.fullmatch(r'(train|tier3|test|hold)/images/[a-z-]+_\d{8}',p) for p in paths)
        ids=[p.rsplit('/',1)[1] for p in paths];assert len(set(ids))==len(ids)
        split_sets[split]=set(paths)
        rows.append({'split':split,'file':str(file),'bytes':len(data),'sha256':digest(file),
            'source_metadata':meta,'unique_samples':len(paths),'event_counts':dict(sorted(Counter(s.rsplit('_',1)[0] for s in ids).items()))})
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not split_sets[a]&split_sets[b]
        assert not {p.rsplit('/',1)[1] for p in split_sets[a]} & {p.rsplit('/',1)[1] for p in split_sets[b]}
    cfg=(PROSPECTIVE/'configs_base.yaml').read_text()
    def configured(key):
        body=re.search(r'  '+key+r': \[(.*?)\n  \]',cfg,re.S).group(1)
        return set(re.findall("'([^']+)'",body))
    for split in ('train','val','test'):
        actual=set(next(r for r in rows if r['split']==split)['event_counts'])
        assert actual==configured('TEST_EVENTS' if split=='test' else 'TRAIN_EVENTS')
    devpath=ROOT/'output/disaster_search_hold_development72_v2/manifest.json'
    checked(devpath,'a70646afabf0822b3adea609835ab95c1cfdb2e7f834d6d8173d38fa4b042346')
    membership=[]
    for sample in read(devpath)['samples']:
        expected='hold/images/'+sample['sample_id']
        found=[s for s,v in split_sets.items() if expected in v];assert len(found)==1
        membership.append({'sample_id':sample['sample_id'],'event':sample['event'],
            'source_path':expected,'author_split':found[0],'already_open_project_development':True})
    summary={e:dict(Counter(r['author_split'] for r in membership if r['event']==e)) for e in sorted({r['event'] for r in membership})}
    assert summary=={'hurricane-harvey':{'train':20,'val':4},'mexico-earthquake':{'train':23,'val':1},'santa-rosa-wildfire':{'test':24}}
    write('prospective_event_baseline.json',{'status':'SOURCE_SPLITS_VERIFIED_WEIGHTS_NOT_ADMITTED',
        'repository':REPO,'source_commit':COMMIT,'source_files_sha256':{p.name:digest(p) for p in PROSPECTIVE.iterdir() if p.is_file()},
        'split_files':rows,'split_overlap_by_path_and_id':0,'current72_membership':membership,'current72_summary':summary,
        'candidate':folder_files['strong_baseline_event.pt'],'alternate_metadata_only':folder_files['disasteradaptivenet_event.pt'],
        'rationale':'First independent model-family candidate with author-published event split, without conditioning metadata or training. Architecture and preprocessing still require adapter review.',
        'checkpoint_binary_downloaded':False,'checkpoint_sha256':None,'weights_readable_verified':False,
        'author_checkpoint_selection':'README says median run used for qualitative results; no independent reruns.',
        'scope_next':'Review architecture/normalization/output decoder at pinned commit; download only strong_baseline_event.pt; inspect format/hash and load on CPU/MPS; freeze an inference protocol before scoring existing24SantaRosa. Training/validation overlap rows remain separate descriptive controls, not unseen events.',
        'interpretation':'SantaRosa24 is author-training-unseen according to published lists, but already-used project development. Only one unseen event; not fresh independent or geographic validation. Lists do not independently prove checkpoint training provenance.',
        'future_success_condition':'No threshold/decoder/tiling sweep; report all24 cases plus event failure counts. First establish fixed detector measurement validity, then define a fresh multi-event cohort before policy comparisons.',
        'protocol_not_yet_frozen':'Exact preprocessing, allowed forwards, input tiling and metric gate require architecture review before any new predictions.',
        'training':False,'gpu_rental':False,'new_model_calls':0,'reserved_sets_enabled':False})


def report(verified):
    table='\n'.join(f"| {r['sample_id']} | {r['severe_pixels_common_domain']:,} | 1.00 | 0 |" for r in sorted(verified,key=lambda r:r['sample_id']))
    text=f'''# 墨西哥 3 处漏检核查与下一轮基线准入

2026-10-28。完成固定 3 个失败场景的发布标签交叉核对；下一轮收缩到有明确事件划分的公开预训练基线，当前不租 GPU。

## 已完成的核查

从 `EVER-Z/torchange_xView2` 固定版本 `cda47e44377258222166dac63a85068e8bdca446` 读取这 3 个场景的灾后语义掩膜，和现有 TACO 派生多边形栅格比较。双方都给出已知类别的整幅场景区域中，五类像素标签完全一致；严重区域 IoU 全部为 1.00。

| 固定场景 | 共同有效域严重参考像素 | 严重区域 IoU | 已知类别不一致像素 |
|---|---:|---:|---:|
{table}

共同有效严重参考合计 **2,698 像素**。81 号建筑原轮廓栅格为 1,806 像素，其中 3 像素被现有参考按重叠规则忽略，故共同有效域为 1,803；未删除或补标。3 个目标的有效域发布类别均为 major-damage。

重放现有缓存后，两模型在 3 个完整建筑轮廓内的未门控损伤 argmax 均只有背景或无损，minor/major/destroyed 均为 0；因此不是建筑门控把已预测正确的严重类别全部遮掉。这里没有调整损伤阈值或重新推理。

![固定三处漏检近景](figures/failure_closeups.png)

图中每行是同一目标：灾前、灾后、发布参考、Swin512、R34-512。青色为原派生目标轮廓；同一原生 160×160 窗口放大展示。两预测面板按原合同乘建筑门控；未门控类别计数见 `crosscheck.json`。图仅用于检查，未生成新标注。

这支持“当前多边形到语义类别转换未发现不一致”，**不能证明原始标签正确或损伤一定可由影像观察**。两份表示来自同一 xBD 标注来源，不是两套独立人工真值；原始 post JSON、原始 UID、第二人复核仍未补齐。

## 已核实的下一步

[DisasterAdaptiveNet 作者仓库]({REPO}/tree/{COMMIT}) 发布事件划分清单和预训练权重目录。本地保留完整原字节清单：训练 8,202、验证 912、测试 1,920 对；按完整路径和场景 ID 均无集合重叠。训练/验证含 13 个事件，测试含另 6 个事件，和固定源码配置一致。

| 我们已用的开发数据 | 作者训练集 | 作者验证集 | 作者测试集 |
|---|---:|---:|---:|
| Harvey 24 对 | 20 | 4 | 0 |
| Mexico 24 对 | 23 | 1 | 0 |
| Santa Rosa 24 对 | 0 | 0 | 24 |

先选择作者 `strong_baseline_event.pt`，目录标记 309,003,425 字节；`disasteradaptivenet_event.pt` 仅记录为备选元数据。当前只核实了目录和划分，**未下载、加载或运行这两个权重**；文件可读性、权重哈希、架构及预处理接入仍待验证。作者说明公开的是用于定性展示的 median run，也不能把一个权重当作多次训练结果。

下一轮按顺序做：核对固定源码的架构、BGR/RGB、归一化及输出解码；接入这一个预训练权重并在 CPU/MPS 验证；冻结推理输入和次数后，对全部 24 对 Santa Rosa 做模型训练未见事件的开发诊断。Harvey/Mexico 只作有暴露对照，不混进未见事件结论。当前不增加训练或搜索策略变体。

Santa Rosa 已经被本项目看过，且这里只覆盖一个未见事件，**不能称为全新独立验证或多事件泛化证明**；训练排除是作者清单层面的证据，不能独立还原权重训练全过程。基线接入成立后，再另定新的多事件评估队列和准入条件，旧保留集保持关闭。

## 可复核产物

- `protocol.json`：先固定 3 个失败对象、范围和读取上限。
- `transfer_ledger.json`、`extraction_report.json`：15 次精确范围读取，掩膜提取收到 1,898,297 字节，小于 8 MiB；RGB 和 t1_mask 列未读取。
- `crosscheck.json`、`audit_report.json`：独立重放标签对应、轮廓和 6 份模型描述。
- `prospective_event_baseline.json`、`prospective_sources/`：固定作者源码、完整清单及逐场景成员关系；权重二进制未准入。
- `test_log.txt`：341 项测试通过，含 3 项新增边界与提取测试。

本轮新模型调用 0、新 RGB 0、训练 0、付费 GPU 0、新人工标注 0。父阶段 1,300 份输出、4,833 份外部依赖和 50 份冻结源码核对不变。当前没有新方法收益结论；旧 EBD 输入/规划变体仍停止，512 离线参考不作为旧预算搜索的免费信息。
'''
    (OUT/'README.md').write_text(text)
    (OUT/'ATTRIBUTION.txt').write_text(
        'Source label representations: xBD/xView2; TACO-derived reference already present in project.\n'
        'Published semantic masks: https://huggingface.co/datasets/EVER-Z/torchange_xView2/tree/cda47e44377258222166dac63a85068e8bdca446\n'
        f'Prospective baseline source: {REPO}/tree/{COMMIT}\n'
        'Hafner, Gerard, Sullivan, Ban. DisasterAdaptiveNet. IJAEOG143 (2025),104756.\n'
        'Public Drive folders are linked by author README; retained metadata and split lists only, no model binaries.\n'
        'No original JSON UID reconstruction, new human annotations or redistribution-rights claim.\n')


def complete():
    assert read(OUT/'audit_report.json')['status']=='PASS'
    assert read(OUT/'prospective_event_baseline.json')['status']=='SOURCE_SPLITS_VERIFIED_WEIGHTS_NOT_ADMITTED'
    parent=read(PARENT/'completion_receipt.json')
    sources={**parent['source_code_sha256'],**{n:digest(HERE/n) for n in NEW_SOURCES}}
    snapshots=OUT/'source_snapshots';snapshots.mkdir(exist_ok=True)
    for n,h in sources.items():
        checked(HERE/n,h);dst=snapshots/n
        if dst.exists():checked(dst,h)
        else:shutil.copyfile(HERE/n,dst)
    external=read(OUT/'audit_external_files.json')
    external.update({str(HERE/n):h for n,h in sources.items()})
    for p,h in external.items():checked(p,h)
    receipt={'status':'COMPLETE','client_report_date':'2026-10-28','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'parent_receipt_sha256':digest(PARENT/'completion_receipt.json'),'source_code_sha256':sources,
        'files':{str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='completion_receipt.json'},
        'external_files':external,'tests_passed':341,'new_tests':3,'scenes_crosschecked':3,
        'mask_transfer_bytes':1898297,'mask_requests':15,'figure_viewed':True,'event_split_samples_verified':11034,
        'new_model_calls':0,'new_rgb':0,'new_model_binary_bytes':0,'gpu_rented':False,'training':False,
        'new_human_annotations':0,'original_json_or_uids_recovered':False,'independent_efficacy_claim':False,
        'old_reserved_sets_scored':False,'prospective_baseline_inference_run':False}
    assert receipt['event_split_samples_verified']==sum(r['unique_samples'] for r in read(OUT/'prospective_event_baseline.json')['split_files'])
    write('completion_receipt.json',receipt)
    # Check the actual saved receipt against every included file after writing it.
    for n,h in read(OUT/'completion_receipt.json')['files'].items():checked(OUT/n,h)
    print(json.dumps({'status':'COMPLETE','receipt_sha256':digest(OUT/'completion_receipt.json'),
                      'output_files':len(receipt['files']),'external_files':len(external),'source_snapshots':len(sources)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['audit','complete'],required=True)
    args=parser.parse_args()
    if (OUT/'completion_receipt.json').exists():raise ValueError('Completed output frozen')
    (audit if args.stage=='audit' else complete)()
