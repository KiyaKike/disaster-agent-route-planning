"""Replay all fixed24 inputs, probability decoding and reference scores offline."""
import argparse
import csv
from io import StringIO
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from run_strongbaseline24 import HERE,ROOT,OUT,PARENT,PAIR,SOURCE,read,digest,checked,rawsha,save,verify


def independent_reference(document):
    layers={k:np.zeros((1024,1024),np.uint8) for k in [1,2,3,4,255]}
    mapping={'no-damage':1,'minor-damage':2,'major-damage':3,'destroyed':4,'un-classified':255}
    for feature in document['features']['xy']:
        points=np.array([[float(t) for t in s.split()] for s in feature['wkt'][10:-2].split(',')])
        cv2.fillPoly(layers[mapping[feature['properties']['subtype']]],[np.rint(points).astype(np.int32)],1)
    count=sum(layers[k] for k in [1,2,3,4]);truth=np.zeros((1024,1024),np.uint8)
    for k in [1,2,3,4]:truth[layers[k]!=0]=k
    truth[(count>1)|(layers[255]!=0)]=255
    return truth


def metric(tp,fp,fn):
    return {'tp':tp,'fp':fp,'fn':fn,'reference_severe_pixels':tp+fn,'predicted_severe_pixels':tp+fp,
        'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
        'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def audit():
    protocol=verify();assert read(OUT/'runtime.json')['new_forward_calls']==98
    jobs=read(OUT/'rgb_jobs.json');refs=read(OUT/'references.json');rows=read(OUT/'observations.json')
    manifests=read(OUT/'cache_manifest.json');scores=read(OUT/'scene_scores.json');summary=read(OUT/'summary.json')
    assert len(jobs)==len(refs)==len(manifests)==len(scores)==24 and len(rows)==98
    assert [r['sample_id'] for r in jobs]==protocol['selected_ids']
    expected=[]
    expected.extend([('synthetic_control',None,None,'cpu'),('synthetic_control',None,None,'mps')])
    expected.extend(('production',j['sample_id'],i,'mps') for j in jobs for i in range(4))
    prepared={}
    for job in jobs:
        # Independent official OpenCV BGR path; no adapter preprocessing call.
        images=[cv2.imread(r['path'],cv2.IMREAD_COLOR) for r in job['images']]
        assert all(a is not None and a.shape==(1024,1024,3) for a in images)
        bgr=np.concatenate(images,axis=2).astype('float32');bgr/=127;bgr-=1
        prepared[job['sample_id']]=bgr.transpose(2,0,1)[None].copy()
    for i,(row,wanted) in enumerate(zip(rows,expected),1):
        assert row['index']==i and row['status']=='COMPLETE'
        assert (row['role'],row['sample_id'],row['flip_index'],row['device'])==wanted
        assert row==read(OUT/'calls'/f'{i:03d}.json') and row['protocol_sha256']==digest(OUT/'protocol.json')
        if row['role']=='synthetic_control':array=np.linspace(-1,255/127-1,6*64*64,dtype=np.float32).reshape(1,6,64,64)
        else:
            array=prepared[row['sample_id']]
            axes=[(),(2,),(3,),(2,3)][row['flip_index']]
            if axes:array=np.flip(array,axis=axes).copy()
        assert list(array.shape)==row['input_shape'] and rawsha(array)==row['input_sha256']
    with np.load(OUT/'control_outputs.npz',allow_pickle=False) as cache:
        for i,device in enumerate(['cpu','mps']):
            assert rawsha(cache['logits_'+device])==rows[i]['logit_sha256']
            assert rawsha(cache['probabilities_'+device])==rows[i]['probability_sha256']
            np.testing.assert_allclose(cache['probabilities_'+device],1/(1+np.exp(-cache['logits_'+device])),atol=1e-6,rtol=1e-6)
        error=float(np.max(np.abs(cache['logits_cpu']-cache['logits_mps'])))
        assert error==read(OUT/'control_report.json')['max_abs_logits_difference'] and error<=0.005
    byref={r['sample_id']:r for r in refs};byscore={r['sample_id']:r for r in scores};replayed=[]
    for item in manifests:
        sid=item['sample_id'];ref=byref[sid];checked(item['path'],item['sha256'])
        truth=independent_reference(read(ref['label_path']))
        np.testing.assert_array_equal(truth,np.asarray(Image.open(ref['reference_file'])))
        with np.load(item['path'],allow_pickle=False) as cache:
            probability=cache['mean_probabilities'];assert probability.dtype==np.float32 and probability.shape==(5,1024,1024)
            assert np.isfinite(probability).all() and 0<=probability.min()<=probability.max()<=1
            assert rawsha(probability)==item['mean_probabilities_sha256']
            loc=probability[0]>np.float32(.38);damage=(probability[1:].argmax(0)+1).astype(np.uint8)
            pred=np.where(loc,damage,0).astype(np.uint8)
            for key,a in [('building',loc),('ungated_damage',damage),('prediction',pred)]:np.testing.assert_array_equal(cache[key],a)
        checked(item['prediction_path'],item['prediction_sha256']);np.testing.assert_array_equal(pred,np.asarray(Image.open(item['prediction_path'])))
        good=truth!=255;true_positive=(truth==3)|(truth==4);pred_positive=(pred==3)|(pred==4)
        tp=int(np.count_nonzero(good & true_positive & pred_positive))
        fp=int(np.count_nonzero(good & ~true_positive & pred_positive))
        fn=int(np.count_nonzero(good & true_positive & ~pred_positive))
        values=metric(tp,fp,fn);values.update(evaluated_pixels=int(good.sum()),severe_positive_scene=bool(tp+fn),positive_scene_any_correct_pixel=bool(tp))
        for k,v in values.items():assert byscore[sid][k]==v,(sid,k)
        replayed.append({'sample_id':sid,**values})
    pooled=metric(*[sum(s[k] for s in replayed) for k in ['tp','fp','fn']])
    for k,v in pooled.items():assert summary[k]==v,k
    assert summary['evaluated_pixels']==sum(s['evaluated_pixels'] for s in replayed)
    assert summary['complete_miss_ids']==[s['sample_id'] for s in replayed if s['severe_positive_scene'] and not s['positive_scene_any_correct_pixel']]
    assert summary['severe_positive_scenes']==sum(s['severe_positive_scene'] for s in replayed)
    assert summary['severe_positive_scenes_with_any_tp']==sum(s['positive_scene_any_correct_pixel'] for s in replayed)
    runtime=read(OUT/'runtime.json')
    assert abs(runtime['forward_seconds']-sum(r['elapsed_seconds'] for r in rows))<1e-9
    parent=read(PARENT/'completion_receipt.json')
    for n,h in parent['files'].items():checked(PARENT/n,h)
    for n,h in parent['external_files'].items():checked(n,h)
    for n,h in parent['source_code_sha256'].items():checked(HERE/n,h)
    save(OUT/'audit_report.json',{'status':'PASS','input_tensors_reconstructed':98,'probability_caches_decoded':24,
        'reference_rasters_reconstructed':24,'scene_scores_replayed':24,'pooled_score_replayed':True,
        'synthetic_cpu_mps_max_logit_error':error,'old_frozen_sources_verified':len(parent['source_code_sha256']),
        'parent_output_files_verified':len(parent['files']),'parent_external_files_verified':len(parent['external_files']),
        'new_audit_model_calls':0,'notes':'Replayed inputs, saved mean probabilities and all scores. Per-flip output hashes recorded; model forward outputs are not independently rerun.'})
    csvfile=StringIO();writer=csv.DictWriter(csvfile,fieldnames=list(replayed[0]));writer.writeheader();writer.writerows(replayed)
    (OUT/'scene_scores.csv').write_text(csvfile.getvalue())
    print(json.dumps({'audit':'PASS','inputs':98,'scenes':24,'old_sources_unchanged':len(parent['source_code_sha256'])}),flush=True)


def report():
    summary=read(OUT/'summary.json');runtime=read(OUT/'runtime.json');auditrow=read(OUT/'audit_report.json')
    assert auditrow['status']=='PASS'
    jobs=read(OUT/'rgb_jobs.json');refs={r['sample_id']:r for r in read(OUT/'references.json')}
    scores={r['sample_id']:r for r in read(OUT/'scene_scores.json')}
    (OUT/'figures').mkdir(exist_ok=True)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
    palette=np.array([[0,0,0],[220,224,226],[236,185,65],[243,119,48],[203,43,53],[115,105,158]],np.uint8)
    def color(a):
        ids=a.copy();ids[ids==255]=5
        return Image.fromarray(palette[ids])
    def percent(v):return 'NA' if v is None else f'{v*100:.2f}%'
    for sheet_id in range(4):
        sheet=Image.new('RGB',(1024,1730),'white');draw=ImageDraw.Draw(sheet)
        for col,label in enumerate(['Pre RGB','Post RGB','Post reference','Strong Baseline (author-default)']):draw.text((col*256+4,5),label,font=font,fill='black')
        for i,job in enumerate(jobs[sheet_id*6:sheet_id*6+6]):
            sid=job['sample_id'];s=scores[sid];y=28+i*280
            text=f"{sid} | P={percent(s['precision'])} R={percent(s['recall'])} F1={percent(s['f1'])} | TP/Ref={s['tp']}/{s['reference_severe_pixels']}"
            draw.text((4,y),text,font=font,fill='black')
            panels=[Image.open(row['path']).convert('RGB') for row in job['images']]
            panels.append(color(np.asarray(Image.open(refs[sid]['reference_file']))))
            panels.append(color(np.asarray(Image.open(OUT/'predictions'/(sid+'.png')))))
            for col,panel in enumerate(panels):
                resample=Image.Resampling.BOX if col<2 else Image.Resampling.NEAREST
                sheet.paste(panel.resize((256,256),resample),(col*256,y+20))
        draw.text((4,1710),'All24 fixed scenes; display downsample only. Orange=major,red=destroyed,purple=ignored. NA=undefined ratio.',font=font,fill='black')
        sheet.save(OUT/'figures'/f'all24_sheet_{sheet_id+1}.png')
    positive=summary['severe_positive_scenes'];found=summary['severe_positive_scenes_with_any_tp']
    complete='、'.join(summary['complete_miss_ids']) or '无'
    text=f'''# Strong Baseline：Santa Rosa 固定 24 对开发诊断

2026-10-28。已在本机 MPS 接通并完成作者事件划分预训练权重的默认整图推理。

| 主指标：严重类别 3/4，排除未知参考 | 结果 |
|---|---:|
| 合并严重像素精度 | {percent(summary['precision'])} |
| 合并严重像素召回 | {percent(summary['recall'])} |
| 合并严重像素 F1 | {percent(summary['f1'])} |
| TP / FP / FN 像素 | {summary['tp']:,} / {summary['fp']:,} / {summary['fn']:,} |
| 有严重参考的场景 | {positive}/24 |
| 阳性场景中至少命中一个严重像素 | {found}/{positive} |
| 完全漏检的参考阳性场景 | {positive-found} |

完全漏检场景：{complete}。命中一个像素仅表示存在非零发现信号，不等于该场景已被充分检出，更不等于逐栋召回。全部 24 个场景保留在 `scene_scores.csv` 和 `scene_scores.json`，无严重参考时召回为 null，零预测时精度为 null。

## 模型与输入

使用 [作者固定源码](https://github.com/SebastianHafner/DisasterAdaptiveNet/tree/fada4e791766f810dc815a4e82d52f2bb27116c6) 的 StrongBaselineNet（ResNet34/Siamese U-Net），权重为作者公开的 `strong_baseline_event.pt`，309,003,425 字节，本地 SHA256 `fe7ded8c5012dee3d39675226b0aa1fd671d8c19cef5c88080daa8b13645c6b6`。这是本地内容校验，Drive 连接器未返回作者预先公布的校验和。检查点 epoch36；236 个网络状态张量严格加载，25,728,597 个模型参数；优化器未恢复。

按作者实现读取原生 1024×1024 BGR 灾前、灾后影像，拼成 6 通道，float32 `像素/127-1`。4 种固定翻转后分别 sigmoid、逆翻转并等权平均；通道0为建筑概率，固定 `>0.38` 门控，通道1–4取 argmax+1 后乘门控。`USE_TRICKS=False`。没有裁块、缩放或阈值搜索。

为控制内存，作者每场景 batch4 改成 eval 模式 batch1 串行 4 次前向，保留相同输入与平均顺序。源码适配仅删除配置类导入并以注解占位，以及关闭随后会被完整检查点覆盖的 ImageNet 初始化下载。使用 `torch.load(weights_only=True)`。运行环境 PyTorch {runtime['packages']['torch']}，不同于作者训练环境；记录为本地复现实验，不声明与作者论文数值完全相等。

作者清单核对：这 24 对全部在作者事件测试集，未出现在其训练或验证清单。它们已在本项目多次分析，属于一个已用开发事件；**不是新的独立确认集，也不是跨事件泛化证明**。作者仅提供 median run，本轮不代表重复训练统计。

## 时间、核验与范围

本机 98 次前向，包括 2 次确定性 64×64 CPU/MPS 接入检查、96 次正式整图前向。前向累计 {runtime['forward_seconds']:.2f} 秒；含缓存保存等的运行循环 {runtime['loop_seconds']:.2f} 秒。CPU/MPS 合成对照最大 logit 差 {auditrow['synthetic_cpu_mps_max_logit_error']:.8g}，小于预先固定的 0.005。

349 项测试通过，含 8 项新增接口测试。独立用 OpenCV BGR 路径重建 98 个输入、重建 24 份参考栅格、从缓存平均概率重新解码 24 个结果并重放逐场景和合并评分；54 份旧冻结源码核对不变。没有新 RGB、训练、付费 GPU 或旧保留集评分。

该参考由既有灾后多边形栅格化，未知及冲突重叠忽略；缺少原始灾前定位标签，不是官方 mixedF1、实例 AP 或新人工真值。整图和四翻转均消耗信息与计算，当前结果不能写成预算搜索收益。以前 Swin512/R34-512 的输入范围、TTA 和训练暴露不同，不据此宣布模型优胜或新增方法成立。

## 全部 24 个固定场景

图像仅展示时降采样；推理输入保持 1024。黑=背景，灰=无损，黄=轻损，橙=严重，红=摧毁，紫=忽略参考。

![场景 1–6](figures/all24_sheet_1.png)
![场景 7–12](figures/all24_sheet_2.png)
![场景 13–18](figures/all24_sheet_3.png)
![场景 19–24](figures/all24_sheet_4.png)

## 下一项实验

本阶段只完成整图测量。下一项先单独冻结“按原图观察面积计费”的局部输入可行性协议，在同一感知器、同一信息和预算下比较随机、静态变化排序及理想上界；先确认局部观察能提供非零且有分布范围的发现信号，再考虑反馈策略。新多事件队列必须先核对作者训练排除与项目使用历史，并按身份规则锁定；不得依据损伤或预测挑样本。Santa Rosa 的本轮结果不会升级为最终确认集，旧 EBD 已停止的输入/规划变体不恢复。
'''
    (OUT/'README.md').write_text(text)
    save(OUT/'prospective_next_scope.json',{'status':'NOT_EXECUTED_PROTOCOL_PENDING','next':'Cost-compatible local observation feasibility before any adaptive controller',
        'fixed_baseline':'StrongBaseline event checkpoint fe7ded8c5012dee3d39675226b0aa1fd671d8c19cef5c88080daa8b13645c6b6',
        'requirements':['Count purchased native image area and all quality/flip costs explicitly','Freeze input contract,budgets,metrics,all sample IDs and all inference calls before prediction','Use all eligible fixed scenes; no positive mining','Separate old used development from prospectively selected multi-event confirmation','Report per-event positive prevalence,detectability,random/static/oracle spread before method claims'],
        'training':False,'gpu_rental':False,'new_data_payloads_enabled':False,'new_inference_enabled':False,'reserved_sets_enabled':False})
    (OUT/'ATTRIBUTION.txt').write_text('DisasterAdaptiveNet official implementation and StrongBaseline event checkpoint.\nhttps://github.com/SebastianHafner/DisasterAdaptiveNet/tree/fada4e791766f810dc815a4e82d52f2bb27116c6\nHafner, Gerard, Sullivan, Ban (2025), IJAEOG143,104756.\nRGB and reference: existing xBD/TACO-derived project development24SantaRosa.\nUpstream root LICENSE returned404. No redistribution-rights claim.\n')
    print(json.dumps({'report':'WRITTEN','figures':4,'all_scenes':24}),flush=True)


def complete():
    protocol=verify();assert read(OUT/'audit_report.json')['status']=='PASS'
    # A separate visual inspection record is created after viewing all four sheets.
    visual=read(OUT/'visual_inspection.json');assert visual['all_four_sheets_viewed']
    assert set(visual['files'])=={f'figures/all24_sheet_{i}.png' for i in range(1,5)}
    for n,h in visual['files'].items():checked(OUT/n,h)
    snapshots=OUT/'source_snapshots';snapshots.mkdir(exist_ok=False)
    for n,h in protocol['source_code_sha256'].items():checked(HERE/n,h);shutil.copyfile(HERE/n,snapshots/n)
    external={**protocol['external_files'],**{str(HERE/n):h for n,h in protocol['source_code_sha256'].items()},
        **{str(SOURCE/n):h for n,h in protocol['upstream_files_sha256'].items()}}
    for p,h in external.items():checked(p,h)
    receipt={'status':'COMPLETE','client_report_date':'2026-10-28','parent_receipt_sha256':protocol['parent_receipt_sha256'],
        'source_code_sha256':protocol['source_code_sha256'],'external_files':external,
        'files':{str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='completion_receipt.json'},
        'tests_passed':349,'new_tests':8,'new_forward_calls':98,'production_full1024_calls':96,'synthetic_control_calls':2,
        'new_model_binary_bytes':309003425,'new_rgb':0,'scenes':24,'input_tensors_reconstructed':98,'reference_rasters_reconstructed':24,
        'scores_replayed':24,'all24_preview_viewed':True,'old_frozen_sources_verified':54,
        'gpu_rented':False,'training':False,'old_reserved_sets_scored':False,'new_human_annotations':0,
        'official_mixedF1_claim':False,'independent_efficacy_claim':False,'budget_search_gain_claim':False}
    save(OUT/'completion_receipt.json',receipt)
    for n,h in receipt['files'].items():checked(OUT/n,h)
    print(json.dumps({'status':'COMPLETE','receipt_sha256':digest(OUT/'completion_receipt.json'),
        'output_files':len(receipt['files']),'external_files':len(external),'source_snapshots':len(protocol['source_code_sha256'])}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['audit','report','complete'],required=True)
    args=parser.parse_args()
    if (OUT/'completion_receipt.json').exists():raise ValueError('Completed output frozen')
    {'audit':audit,'report':report,'complete':complete}[args.stage]()
