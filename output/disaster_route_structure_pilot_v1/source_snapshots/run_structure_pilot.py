"""Freeze/run/audit/report a four-identity structural-map integration pilot.

Geometry failure is retained without substitution. No new model inference.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/disaster_search'))
from structure_map_core import (build_public_case, execute, raster, ring, POLICIES,
                               suffix_cost, RETURN_MARGIN, HOME, PLAN, OBSERVE, leg)

OUT=ROOT/'output/disaster_route_structure_pilot_v1'
RAW=ROOT/'work/disaster_search/attachments/external/huggingface/xbd_mini72_v1'
SOURCE=Path(__file__).parent
BUDGETS=(300,600,900)


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+'\n')


def ensure_open():
    if (OUT/'completion_receipt.json').exists():raise ValueError('Completed output frozen; use a new version')


def selected():
    by={}
    for s in read(RAW/'manifest.json')['samples']:
        if s['split']!='test':by.setdefault(s['event'],s)
    return list(by.values())


def source_observations():
    rows=[]
    for split in ('calibration','validation'):
        rows+=read(ROOT/f'output/disaster_search_changeos_{split}/observations.json')
    return [r for r in rows if r['mode']=='native']


def plan():
    ensure_open()
    if (OUT/'protocol.json').exists():raise ValueError('Protocol already fixed')
    samples=selected(); inputs={str(RAW/'manifest.json'):sha(RAW/'manifest.json')}
    sources=['structure_map_core.py','run_structure_pilot.py','test_route_structure.py','structure_viewer.html']
    for s in samples:
        for p in s['files'].values():inputs[str(RAW/p)]=sha(RAW/p)
    ids={s['sample_id'] for s in samples}; obs=[r for r in source_observations() if r['sample_id'] in ids]
    if len(obs)!=256:raise ValueError('Expected64 native caches per selected identity')
    for r in obs:inputs[r['cache_file']]=r['cache_sha256']
    for split in ('calibration','validation'):
        p=ROOT/f'output/disaster_search_changeos_{split}/inference_report.json';inputs[str(p)]=sha(p)
    for p in [ROOT/'work/disaster_search/inferred_scene_geometry.py',ROOT/'work/disaster_search/audit_hold_taco.py']:
        inputs[str(p)]=sha(p)
    receipts={}
    for name in ['disaster_route_loop_v1','disaster_route_factor_v2','disaster_route_snapshot_v3','disaster_route_trigger_v4',
                 'disaster_route_dual_map_demo_v1','disaster_route_real_xbd36_v1']:
        p=ROOT/'output'/name/'completion_receipt.json';receipts[str(p)]=sha(p)
    p=ROOT/'output/感知检查_阶段汇报_20261028/归档校验.json';receipts[str(p)]=sha(p)
    protocol={'client_date':'2026-11-02','stage':'STRUCTURAL_MAP_GEOGRAPHIC_INTEGRATION_PILOT',
      'selection':'First existing identity-ranked mini72 sample per calibration/validation event; four predetermined development identities; no test events, no damage/outcome selection or substitution.',
      'selected':samples,'geometry_failure':'Retain and exclude from metric routing only if unchanged public geometry quality gate fails; never replace.',
      'geometry':'Original pre XY/WGS84 matched by UID; existing frozen affine consistency gate; WGS84 ECEF to local ENU. Reconstructed approximate distances, not official georeferencing/field accuracy.',
      'continuous_scope':'Each1024px image is its own small contiguous area; never stitch separate chips or call them one disaster-wide mission.',
      'motion':{'platform_assumption':'Idealized UAV integration case, pending user/advisor platform decision',
                'speed_m_s':5,'yaw_rate_deg_s':45,'observation_s':20,'solve_s':1,'unspent_return_margin_s':20,
                'start':'Bottom-middle of reconstructed chip boundary, initial yaw north',
                'constraints':'Inside convex reconstructed image extent; all yaw/move/observe/plan/return costed; terrain,obstacles,airspace,altitude and camera validity unknown.'},
      'budgets_s_model':list(BUDGETS),'primary_budget_s_model':600,'policies':list(POLICIES),
      'prior':'Fractional pre-building footprint mass per128px tile; coarse64RGB change rank r=.5+.5rank. Pre mass; post mass*r. Utility not damage probability.',
      'structure_update':'Only same pre-building UID, no general neighbor transfer. c=paid observed footprint fraction, s=predicted severe fraction on those pre pixels; tile reward sum_b share_bi*((1-strength*c_b)*r_i+strength*c_b*s_b).',
      'guard':'strength=.5 vs structure strength1; retains at least half original post score. A fixed exposure-coverage floor, not calibrated reliability or information gain. No tuning.',
      'observer':'Actual frozen native128 cached severe masks revealed after travel+20s observation; repeated same tile forbidden. Disjoint footprint pixels counted once; no independent-repeat assumption.',
      'registration':'Source pre/post chip raster alignment is assumed; pre-mask to post-prediction association can be wrong. Not manually validated or corrected.',
      'solver':'Same reward/(next-leg modeled time+observe) greedy then max8 deterministic best2opt passes, including yaw and return; heuristic, not global optimum.',
      'metrics':['pre-building fractional footprint coverage','published severe pixel coverage and correct recall','reference-relative FP','first target arrival/detection plus failure','reconstructed meters','modeled time/yaw/planning calls'],
      'claim_scope':'Engineering connection and candidate ablations only; four reused development identities, checkpoint exposure expected, not independent method efficacy. SantaRosa is a preselected integration case only, not pooled into prior36 or a rescue score.',
      'input_sha256':inputs,'source_sha256':{f:sha(SOURCE/f) for f in sources},'old_receipt_sha256':receipts,
      'new_model_calls':0,'new_imagery_bytes':0,'gpu_rented':False,'new_annotations':0}
    save(OUT/'protocol.json',protocol)
    for f in sources:shutil.copyfile(SOURCE/f,OUT/'source_snapshots'/f)
    print('PROTOCOL_FROZEN',len(samples),'identities',len(obs),'existing native masks',flush=True)


def verify_inputs():
    p=read(OUT/'protocol.json')
    for path,h in p['input_sha256'].items():
        if sha(path)!=h:raise ValueError('Changed input '+path)
    for f,h in p['source_sha256'].items():
        if sha(SOURCE/f)!=h:raise ValueError('Changed frozen source '+f)
    for path,h in p['old_receipt_sha256'].items():
        if sha(path)!=h:raise ValueError('Changed old receipt '+path)
    return p


def load_public(s):
    images={}
    for phase in ('pre','post'):
        with Image.open(RAW/s['files'][phase+'_image']) as im:
            images[phase]=np.asarray(im.convert('RGB').resize((64,64),Image.Resampling.BOX))
    return build_public_case(read(RAW/s['files']['pre_label']),images['pre'],images['post'],s['sample_id'])


def load_masks(s):
    rows={r['cell_id']:r for r in source_observations() if r['sample_id']==s['sample_id']}
    inference=read(ROOT/f"output/disaster_search_changeos_{s['split']}/inference_report.json")
    for phase in ('pre','post'):
        path=RAW/s['files'][phase+'_image']
        if inference['input_image_sha256'][str(path)]!=sha(path):raise ValueError('Cache/image identity mismatch')
    masks=[]
    for j in range(64):
        r=rows[f'r{j//8:02d}c{j%8:02d}']
        if sha(r['cache_file'])!=r['cache_sha256']:raise ValueError('Changed detector cache')
        with np.load(r['cache_file']) as z:m=z['severe_mask'].copy()
        if m.shape!=(128,128) or m.dtype!=bool or int(m.sum())!=r['predicted_severe_pixels']:
            raise ValueError('Detector mask does not reproduce fixed summary')
        masks.append(m)
    return masks


def reference(s):
    # Evaluator only. Neither execute nor map builder receives this object.
    label=read(RAW/s['files']['post_label'])
    classes={'no-damage':1,'minor-damage':2,'major-damage':3,'destroyed':4,'un-classified':5}
    bits=np.zeros((1024,1024),np.uint8)
    for f in label['features']['xy']:
        c=classes.get(f['properties'].get('subtype','un-classified'))
        if c is None:raise ValueError('Unknown published damage subtype')
        bits[raster(ring(f['wkt']))] |= 1<<c
    out=np.zeros((1024,1024),np.uint8)
    for c in range(1,5):out[bits==(1<<c)]=c
    out[(bits!=0)&((bits&(bits-1))!=0) | ((bits&(1<<5))!=0)]=255
    return out


def score(public,route,predictions,ref):
    s=route['state'];chosen=s['visited'];target=(ref==3)|(ref==4);valid=ref!=255
    counts=[]
    for j,m in enumerate(predictions):
        r,c=divmod(j,8);t=target[r*128:(r+1)*128,c*128:(c+1)*128];v=valid[r*128:(r+1)*128,c*128:(c+1)*128]
        counts.append({'reference':int(t.sum()),'tp':int((m&t).sum()),'fp':int((m&~t&v).sum())})
    total=int(target.sum());tp=sum(counts[j]['tp'] for j in chosen)
    first_arrival=next((h['arrival_s_model'] for h in s['history'] if counts[h['tile_id']]['reference']>0),None)
    first_detect=next((h['observation_complete_s_model'] for h in s['history'] if counts[h['tile_id']]['tp']>0),None)
    pre_cov=sum(public['building_mass'][j] for j in chosen)
    return {'pre_building_fractional_coverage':pre_cov/public['pre_building_count'],
            'published_reference_pixels':total,'covered_reference_pixels':sum(counts[j]['reference'] for j in chosen),
            'correct_severe_pixels':tp,'severe_pixel_recall':tp/total if total else None,
            'reference_coverage':sum(counts[j]['reference'] for j in chosen)/total if total else None,
            'false_positive_pixels':sum(counts[j]['fp'] for j in chosen),
            'first_target_arrival_s_model':first_arrival,'first_correct_detection_s_model':first_detect,
            'failed_correct_detection':first_detect is None,'first_detection_or_budget_s_model':first_detect if first_detect is not None else s['budget_s_model'],
            'all_tiles_detector_tp_ceiling':sum(c['tp'] for c in counts),'observations':len(chosen),'planning_calls':len(s['plans']),
            'distance_m_reconstructed':s['distance_m_reconstructed'],'elapsed_s_model':s['elapsed_s_model'],'turn_s_model':s['turn_s_model']}


def run():
    ensure_open();p=verify_inputs()
    if (OUT/'routes.json').exists():raise ValueError('Do not overwrite completed routes')
    publics=[];routes=[];metrics=[];gates=[]
    for s in p['selected']:
        pub,pre_masks=load_public(s);publics.append(pub)
        gates.append({'sample_id':s['sample_id'],'pre_building_count':pub['pre_building_count'],**pub['geometry_quality']})
        if pre_masks is None:
            print('GEOMETRY_GATE_REJECTED',s['sample_id'],pub['geometry_quality']['status'],flush=True);continue
        preds=load_masks(s)
        for budget in BUDGETS:
            for policy in POLICIES:
                paid=[]
                def observe(j,end):paid.append((j,end));return preds[j].copy()
                r=execute(pub,pre_masks,policy,budget,observe)
                assert paid==[(h['tile_id'],h['observation_complete_s_model']) for h in r['state']['history']]
                key=f"{s['sample_id']}:{budget}:{policy}";routes.append({'key':key,'sample_id':s['sample_id'],**r})
        # All route decisions for this case are fixed before evaluator labels are read here.
        ref=reference(s)
        for r in routes:
            if r['sample_id']==s['sample_id']:
                metrics.append({'key':r['key'],'sample_id':s['sample_id'],'budget_s_model':r['state']['budget_s_model'],
                                'policy':r['state']['policy'],**score(pub,r,preds,ref)})
        for phase in ('pre','post'):
            with Image.open(RAW/s['files'][phase+'_image']) as im:
                im.convert('RGB').resize((512,512)).save(OUT/'assets'/f"{s['sample_id']}_{phase}.jpg",quality=86)
        print('CASE_COMPLETE',s['sample_id'],sum(r['sample_id']==s['sample_id'] for r in routes),'routes',flush=True)
    save(OUT/'public_cases.json',publics);save(OUT/'geometry_gates.json',gates)
    save(OUT/'routes.json',routes);save(OUT/'metrics.json',metrics)
    print('ROUTES_COMPLETE',len(routes),flush=True)


def audit():
    ensure_open();p=verify_inputs();routes=read(OUT/'routes.json');metrics={m['key']:m for m in read(OUT/'metrics.json')}
    publics={c['sample_id']:c for c in read(OUT/'public_cases.json')};n_obs=0
    for s in p['selected']:
        rebuilt,masks=load_public(s);assert rebuilt==publics[s['sample_id']]
        if masks is None:
            assert not any(r['sample_id']==s['sample_id'] for r in routes);continue
        xy=np.array(rebuilt['xy_m_reconstructed']);preds=load_masks(s);ref=reference(s)
        corners=np.array(rebuilt['corners_xy_m_reconstructed'])
        for r in (r for r in routes if r['sample_id']==s['sample_id']):
            state=r['state'];pos=HOME;yaw=math.pi/2;clock=0.;distance=0.;turns=0.
            # Independent chronological accounting: paid plan events before each observation.
            plan_cost=len(state['plans'])*PLAN
            for h in state['history']:
                j=h['tile_id'];d=xy[j]-xy[pos];length=float(np.hypot(*d));heading=float(np.arctan2(d[1],d[0])) if length>1e-12 else yaw
                turn=abs(math.atan2(math.sin(heading-yaw),math.cos(heading-yaw)))/(math.pi/4)
                n_before=sum(e['after_observations']<=state['visited'].index(j) for e in state['plans'])
                expected=clock+length/5+turn+n_before*PLAN
                assert abs(h['arrival_s_model']-expected)<1e-7
                assert abs(h['observation_complete_s_model']-h['arrival_s_model']-OBSERVE)<1e-8
                clock+=length/5+turn+OBSERVE;distance+=length;turns+=turn;pos=j;yaw=heading
            d=xy[HOME]-xy[pos];length=float(np.hypot(*d));heading=float(np.arctan2(d[1],d[0])) if length>1e-12 else yaw
            turn=abs(math.atan2(math.sin(heading-yaw),math.cos(heading-yaw)))/(math.pi/4)
            assert abs(state['elapsed_s_model']-(clock+plan_cost+length/5+turn))<1e-7
            assert abs(state['distance_m_reconstructed']-(distance+length))<1e-7
            assert abs(state['turn_s_model']-(turns+turn))<1e-7
            assert state['elapsed_s_model']+RETURN_MARGIN<=state['budget_s_model']+1e-7
            assert len(set(state['visited']))==len(state['visited']) and state['position']==HOME and state['closed']
            for point in xy:
                cross=np.cross(np.roll(corners,-1,axis=0)-corners,point-corners)
                assert np.all(cross>=-1e-6) or np.all(cross<=1e-6)
            replay=execute(rebuilt,masks,state['policy'],state['budget_s_model'],lambda j,end:preds[j].copy())
            assert replay=={k:r[k] for k in ('state','evidence','frames')}
            assert metrics[r['key']]=={'key':r['key'],'sample_id':s['sample_id'],'budget_s_model':state['budget_s_model'],
                                     'policy':state['policy'],**score(rebuilt,r,preds,ref)}
            n_obs+=len(state['visited'])
    save(OUT/'audit.json',{'status':'PASS','selected_cases':len(p['selected']),
         'geometry_passed_cases':sum(c['geometry_quality']['usable_inferred_range'] for c in publics.values()),
         'retained_geometry_failures':sum(not c['geometry_quality']['usable_inferred_range'] for c in publics.values()),
         'routes_replayed':len(routes),'paid_observations_replayed':n_obs,'budget_or_return_violations':0,
         'new_model_calls':0,'post_damage_labels_in_planner':False,'official_geotransform_verified':False,
         'independent_field_accuracy_verified':False,'new_independent_efficacy_validation':False})
    print(json.dumps(read(OUT/'audit.json')),flush=True)


def report():
    ensure_open();verify_inputs()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=[c for c in read(OUT/'public_cases.json') if c['geometry_quality']['usable_inferred_range']]
    routes=read(OUT/'routes.json');metrics=read(OUT/'metrics.json')
    labels={'pre_static':'Pre static','post_static':'Post static','online_frozen':'Online frozen',
            'online_neighbor':'Neighbor','online_structure':'Same-building','online_structure_guard':'Same-building + floor'}
    fig,axes=plt.subplots(len(cases),3,figsize=(15,4.1*len(cases)),squeeze=False)
    for row,c in enumerate(cases):
        sid=c['sample_id'];xy=np.array(c['xy_m_reconstructed'])
        im=Image.open(OUT/'assets'/f'{sid}_post.jpg')
        ax=axes[row,0];ax.imshow(im,extent=[0,1024,1024,0])
        for poly in c['pre_pixel_rings']:
            a=np.array(poly);ax.plot(a[:,0],a[:,1],color='#49d9dd',lw=.6,alpha=.8)
        r=next(r for r in routes if r['sample_id']==sid and r['state']['policy']=='online_structure_guard' and r['state']['budget_s_model']==600)
        pix=np.array([[j%8*128+64,j//8*128+64] for j in [HOME]+r['state']['visited']+[HOME]],float)
        pix[[0,-1]]=[512,1024]
        ax.plot(pix[:,0],pix[:,1],'-o',color='#ffae42',lw=1.3,ms=3)
        ax.set_title(sid+'\nPre footprints / paid route on post image',fontsize=10);ax.set_axis_off()
        ms=[next(m for m in metrics if m['sample_id']==sid and m['policy']==p and m['budget_s_model']==600) for p in POLICIES]
        axes[row,1].barh(range(6),[m['pre_building_fractional_coverage']*100 for m in ms],color=['#94a3b8','#3b82f6','#64748b','#e87966','#a78bfa','#14b8a6'])
        axes[row,1].set_yticks(range(6),[labels[p] for p in POLICIES],fontsize=8)
        axes[row,1].set_xlim(0,100);axes[row,1].set_xlabel('Pre-building footprint coverage (%)')
        for p in ('post_static','online_neighbor','online_structure_guard'):
            mm=[next(m for m in metrics if m['sample_id']==sid and m['policy']==p and m['budget_s_model']==b) for b in BUDGETS]
            axes[row,2].plot(BUDGETS,[m['severe_pixel_recall']*100 if m['severe_pixel_recall'] is not None else np.nan for m in mm],'-o',label=labels[p])
        axes[row,2].set_xlabel('Modeled budget (s)');axes[row,2].set_ylabel('Published severe-pixel recall (%)');axes[row,2].set_ylim(0,100)
        axes[row,2].legend(fontsize=7);axes[row,2].grid(alpha=.2)
    fig.suptitle('Development integration pilot | reconstructed geometry | assumed UAV time | no efficacy claim',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.975]);fig.savefig(OUT/'figures/structure_pilot.png',dpi=150);plt.close(fig)
    small_routes=[]
    for r in routes:
        fs=[]
        for f in r['frames']:
            fs.append({k:v for k,v in f.items() if k!='records'})
        small_routes.append({'key':r['key'],'sample_id':r['sample_id'],'state':r['state'],'frames':fs})
    payload={'cases':cases,'routes':small_routes,'metrics':metrics,'gates':read(OUT/'geometry_gates.json'),
             'counts':{r['key']:[e['predicted_severe_pixels'] for e in r['evidence']] for r in routes}}
    html=(SOURCE/'structure_viewer.html').read_text().replace('__REPLAY_DATA__',json.dumps(payload,ensure_ascii=False).replace('</','<\/'))
    (OUT/'demo.html').write_text(html)
    lines=['# 灾前建筑结构＋沿途证据：地理接入试验','',
      '2026-11-02。完成一个新的接入里程碑：利用原始灾前建筑XY/WGS84轮廓重建小区域位置，接入实际本地观测、转向与返程费用，并试运行同建筑证据传播及覆盖保底。**本轮是开发工程试验，不是新方法有效性确认。**','',
      '## 数据与地理检查','',
      '从既有mini72的四个开发事件各取原身份排序第一例。原test事件不打开；不按损伤、识别成功或路线成绩换样。三个案例通过既有地理一致性门槛，hurricane-michael_00000341因建筑点分布不足以覆盖影像而保留拒绝记录。SantaRosa只作预先指定的接入案例，不并入上轮36对或用于挽救其结果。','',
      '每个1024像素影像分别是一个小型连续区域，三者不拼接。匹配原灾前建筑UID及对应XY/WGS84顶点，依冻结仿射检查重建坐标，再转WGS84局部ENU计算近似米数；没有使用传感器GSD作为PNG像元间距。低拟合残差仅说明同一来源的内部一致性，不是独立绝对定位精度或官方地理变换。','',
      '## 双层地图的具体改动','',
      '第一层使用真实发布的灾前建筑轮廓。每栋建筑可见像素总权重为1，跨格按面积拆分，避免跨格建筑被重复计数；灾后粗图变化仅调节排序效用，非损伤概率。灾前结构属于发布参考注释，代表假设已获得的底图，不等于真实业务中免费可得。','',
      '第二层只有付费后取得的固定native128严重预测。新增规则仅在同一灾前建筑UID跨格时传播，强度受已观察建筑面积比例限制。coverage-floor对照把传播强度固定减半，保留至少一半原分数；该面积比例不是观测可靠性概率，没有新校准或质量识别。预后影像及灾前轮廓到灾后预测的配准仍是假设。','',
      '## 主预算600：逐案例结果','',
      '| 案例 | 路线 | 灾前建筑面积份额覆盖 | 严重像素参考覆盖 | 严重像素正确召回 | 观察格 / 规划次数 | 重建距离m / 模型时间s |','|---|---|---:|---:|---:|---:|---:|']
    cn={'pre_static':'灾前静态','post_static':'灾后静态','online_frozen':'在线不传播','online_neighbor':'邻格传播','online_structure':'同建筑传播','online_structure_guard':'同建筑＋覆盖保底'}
    pct=lambda v:'null' if v is None else f'{100*v:.2f}%'
    for c in cases:
        for p in POLICIES:
            m=next(m for m in metrics if m['sample_id']==c['sample_id'] and m['policy']==p and m['budget_s_model']==600)
            lines.append(f"| {c['sample_id']} | {cn[p]} | {pct(m['pre_building_fractional_coverage'])} | {pct(m['reference_coverage'])} | {pct(m['severe_pixel_recall'])} | {m['observations']} / {m['planning_calls']} | {m['distance_m_reconstructed']:.1f} / {m['elapsed_s_model']:.1f} |")
    lines += ['', '完整54条路线覆盖300/600/900三档模型预算与六个对照。正确召回按全部发布严重像素计算，未访问区域仍在分母；覆盖与正确识别分开，零严重参考时召回为null。建筑覆盖是灾前轮廓的分数面积份额，不是找到受灾建筑或人员的数量。发布灾后参考只进入独立评分器。','',
      '## 能说明什么、还缺什么','',
      '本轮验证真实灾前结构可以接入两层地图和预算路线。四个已使用身份、三个可路由案例不能用于普遍排名；保留每个失败和零目标场景，不以结果挑方法、改阈值或改传播强度。局部native128缓存来自更早固定检查点，观察粒度与上轮256不同，绝对分数不可直接对比；没有新增模型调用、训练、人工标注、影像下载或GPU费用。','',
      '运动约束现为5m/s、45度/s转向、20s观察、1s求解、20s未花费返程余量；每次规划及返回均计费，路径限重建影像凸四边形内。这些都是显式运动假设，尚缺地形、障碍、禁飞、相机覆盖、海拔和实测航时，不能称为现场可执行航线。','',
      '下一步先核实独立地理参考和影像配准，明确执行平台，再固定新区域与观察合同。只有获得实际可靠性依据后才加入可靠性门控；目前面积覆盖保底仅是候选机制，不能替代可靠性校准。保留全部静态/在线基线及同信息充分统计量对照，不把双层显示本身作为收益。','',
      '[打开回放](demo.html)；[费用及重放核验](audit.json)；[固定协议](protocol.json)。','',
      '原数据：[xBD论文](https://arxiv.org/abs/1911.09296)；[固定镜像](https://huggingface.co/datasets/aryananand/xBD/tree/b98724026e844e1fccd879c95e9f09f03c56e6f8)。CC-BY-NC-SA-4.0，原影像Maxar，原数据xBD作者/DIU/CMU SEI；本地研究使用。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'prospective_next_scope.json',{'status':'NOT_EXECUTED','next':'Independent georeferencing/registration and platform-specific real constraints before new efficacy protocol; no tuning on four development cases.','new_training':False,'gpu_rental':False})
    print('REPORT_READY',flush=True)


def seal():
    ensure_open();p=verify_inputs();a=read(OUT/'audit.json');assert a['status']=='PASS'
    assert read(OUT/'browser_checks.json')['status']=='PASS'
    assert read(OUT/'visual_inspection.json')['status']=='PASS'
    files={str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='completion_receipt.json'}
    save(OUT/'completion_receipt.json',{'status':'COMPLETE_DEVELOPMENT_INTEGRATION_PILOT','files_sha256':files,
         'protocol_sha256':sha(OUT/'protocol.json'),'audit_sha256':sha(OUT/'audit.json'),'new_model_calls':0,'claim':'Not independent efficacy or field flight validation.'})
    print('SEALED',len(files),'files',sha(OUT/'completion_receipt.json'),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['plan','run','audit','report','seal']);args=ap.parse_args()
    globals()[args.command]()
