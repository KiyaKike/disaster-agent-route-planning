"""Publish compact evidence with explicit failed gates and frozen provenance."""
import sys,json,hashlib,shutil,subprocess,datetime
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from probe_crasar_region import ROOT,O,dump

def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def run():
    assert O.name=='disaster_route_crasar_region_rank2_v1'
    if (O/'completion_receipt.json').exists():raise RuntimeError('Already sealed; use new revision')
    summary=read(O/'experiment_summary_any_damage.json');scores=read(O/'route_scores_any_damage.json')
    # Repeat independent replay checks and paid-evidence isolation on every prefix.
    prefixes=0
    for target in ['severe','any_damage']:
        p=read(O/f'public_planning_inputs_{target}.json');c=np.array(p['window_building_fraction'])
        s=np.array(read(O/f'private_paid_observations_{target}.json')['damage_model_pixel_fraction_per_window'])
        ledgers=read(O/f'dual_map_paid_evidence_{target}.json')
        for route in read(O/f'routes_{target}.json'):
            logs=[l for l in ledgers if l['policy']==route['policy'] and l['budget_proxy_s']==route['budget_proxy_s']]
            assert len(logs)==len(route['route'])
            for k,l in enumerate(logs,1):
                visited=route['route'][:k]
                assert l['step']==k and l['accessed_window']==visited[-1] and not l['reference_labels_in_state']
                np.testing.assert_allclose(l['evidence']['observed_building_fraction'],c[visited].sum(axis=0))
                np.testing.assert_allclose(l['evidence']['observed_damage_model_pixel_fraction'],s[visited].sum(axis=0))
                prefixes+=1
        for budget in [60,100,140]:
            results=[r for r in read(O/f'routes_{target}.json') if r['budget_proxy_s']==budget]
            exact=next(r for r in results if r['policy']=='static_exact')
            greedy=next(r for r in results if r['policy']=='value_cost_proxy')
            assert exact['public_utility_proxy']+1e-8>=greedy['public_utility_proxy']
    calls=read(O/'inference_log.json')
    assert len(calls)==32 and sum(r['forward_calls'] for r in calls)==128
    for r in calls:assert sha(O/'fixed_model_cache'/f"{r['mode']}_{r['window']:02d}.npz")==r['cache_sha256']
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'work/disaster_route'),'-p','test_crasar*.py','-v'],capture_output=True,text=True)
    (O/'tests.txt').write_text(tests.stdout+tests.stderr);assert tests.returncode==0
    # Verify earlier completed research via its existing content receipts.
    previous={}
    for name in ['disaster_route_real_xbd36_v1','disaster_route_structure_pilot_v1','disaster_route_headroom_v1',
       'disaster_route_aippms_control_v1','disaster_route_inspection_information_v1',
       'disaster_route_feedback_increment_audit_v2','disaster_route_crasar_feasibility_20261103']:
        folder=ROOT/'output'/name;receipt=read(folder/'completion_receipt.json')
        hashes=receipt.get('files',receipt.get('files_sha256',{}))
        assert hashes,name
        for key,value in hashes.items():
            if not isinstance(value,str) or len(value)!=64:raise ValueError(f'Unexpected receipt entry:{key}')
            path=Path(key) if Path(key).is_absolute() else folder/key
            assert sha(path)==value,(name,key)
        previous[name]=len(hashes)
    audit={'status':'PASS_WITH_EXPLICIT_SCIENTIFIC_LIMITS','tests_passed':9,'routes_replayed':36,'paid_prefixes_verified':prefixes,
      'budget_violations':0,'fixed_model_cache_files':32,'new_forward_calls':128,'device':'local MPS','training':False,'gpu_rental':False,
      'old_receipt_files_verified':previous,'registration':'Published local vector corrections and visual review; automated independent matching insufficient',
      'primary_severe_gate':'FAIL:NO_TARGETS in both fixed regions','secondary_endpoint':'Exploratory any damage,added after inspecting prevalence,before model predictions',
      'method_contribution':'NOT_ESTABLISHED','actual_flight_validation':'NOT_ESTABLISHED'}
    dump(O/'audit.json',audit)
    # Black/white route diagram, source image panel remains natural colors for interpretation.
    im=Image.new('RGB',(1200,435),'white');d=ImageDraw.Draw(im)
    main=summary['main_budget_results'];selected=['distance','static_exact','value_cost_proxy']
    titles=['Distance first:0/3','Exact static:2/3','Value/cost proxy:1/3']
    for panel,(policy,title) in enumerate(zip(selected,titles)):
        off=panel*400;d.text((off+12,10),title,fill='black')
        def pos(x,y):return (off+35+x*2.4,355-y*2.4)
        for j,(x,y) in enumerate(read(O/'public_planning_inputs_any_damage.json')['xy_m_relative'][:-1]):
            cx,cy=pos(x,y);d.ellipse((cx-3,cy-3,cx+3,cy+3),fill='black');d.text((cx+5,cy-8),str(j),fill='black')
        r=next(r for r in main if r['policy']==policy);xy=read(O/'public_planning_inputs_any_damage.json')['xy_m_relative']
        path=[16]+r['route']+[16];points=[pos(*xy[j]) for j in path]
        d.line(points,fill='black',width=2)
        hx,hy=pos(64,0);d.rectangle((hx-5,hy-5,hx+5,hy+5),outline='black',width=2);d.text((hx+9,hy-4),'home',fill='black')
        for k,(x,y) in enumerate(points[1:-1],1):d.text((x-8,y+4),f'({k})',fill='black')
        d.text((off+12,385),f"{r['distance_m']:.1f}m | {r['elapsed_proxy_s']:.1f}s proxy | FP={r['false_positive_checked_buildings']}",fill='black')
    d.text((12,416),'128m square,32m windows,100s common proxy budget. Image access replay; not measured flight paths.',fill='black')
    im.save(O/'main_routes_blackwhite.png')
    names={'distance':'距离优先','damage':'损伤线索优先','uncertainty':'不确定性优先','static_exact_damage':'精确静态（损伤）','static_exact':'精确静态（候选同目标）','value_cost_proxy':'价值／成本候选'}
    table='|路线|有效检出／目标|误报建筑|首次正确核查（代理秒）|距离m|总耗时（代理秒）|\n|---|---:|---:|---:|---:|---:|\n'
    for r in main:
        first='未发生' if r['first_correct_check_proxy_s'] is None else f"{r['first_correct_check_proxy_s']:.1f}"
        table+=f"|{names[r['policy']]}|{r['correctly_checked_target_buildings']}/3|{r['false_positive_checked_buildings']}|{first}|{r['distance_m']:.1f}|{r['elapsed_proxy_s']:.1f}|\n"
    text=f'''# 真实小区域与强对照：2026-11-03

结论：已完成真实三时相影像接入、四类强对照和价值／成本候选的初次回放。**候选未超过精确静态，方法贡献尚未成立。**

## 数据核验

按训练集文件名排序，依次检查，不按路线得分选例。固定区域为共同建筑几何中心的128m方形；边界与覆盖不足的建筑按预先定义规则排除，失败案例保留。

|区域|灾前卫星／灾后卫星／无人机|区域内建筑中心／完整可用|严重损伤目标|判断|
|---|---|---:|---:|---|
|Sienna Village，Harvey|2017-02-16／08-30／09-04|9／1|0|约29%影像缺失，拒绝用于主效益比较|
|Ft. Myers Beach Boone，Ian|2022-01-21／09-30／10-01|16／12|0|影像覆盖近100%；有3栋轻度损伤，可作探索性核查回放|

日期来自作者目录。已统一UTM网格并应用作者局部配准向量；内部坐标换算残差中位数约0.8原生像素，**它不代表独立配准精度**。可视核对见[三时相叠加图](region_audit_panel.png)，自动跨源匹配不足以提供独立精度证书。当前覆盖是正射影像可读范围，32m窗口是图像访问单元；飞行姿态、实际相机视场、能耗和服务时间仍缺失。建筑轮廓的灾前可用时间也未独立核实。

## 强对照与候选结果

主指标仍为严重损伤核查，但本区域无阳性，召回记为未定义。下表是**查看标注数量后、模型推理前另行明确的探索性“任意损伤”指标**，不可充当确认性结果。

同一模型、同一窗口与起点。移动、观察、一次规划和返程全部收费：假设速度5m/s，每窗观察10秒，规划1秒；预算60／100／140代理秒，主比较100。精确静态枚举65,536个子集，并为每个子集求最短闭合访问路径；分别优化损伤线索和候选相同效用。候选效用为“损伤分数×不确定性分数×新增建筑覆盖”，除以含返程变化的增量代价，仍是未校准启发式。

{table}
有效检出定义：累计访问至少80%建筑轮廓，且对应损伤类别预测像素至少占已访问轮廓10%，再由评估器核对作者无人机视角损伤标签。该阈值是固定代理规则，不等于人工确认。误报单独计数，不将纯访问覆盖当作发现损伤。

在60／100／140预算下，精确静态有效检出为1／2／2，候选为1／1／2。评估器使用全部未来预测与标签求得的同预算事后上限为1／2／3，**只用于诊断，不是可执行对照**。主预算下精确静态已达到这个上限；本例不适合证明在线更新能增加主预算收益。

固定感知器在本机MPS完成128次前向调用，形成32份独立窗口缓存；未训练、未租GPU。规划器只读公开先验数组，标签只进入评估器；双层地图记录固定先验与逐次付费证据，尚未验证在线信息增益。共36条路线、{prefixes}个证据前缀核验，预算违规0；9项专项测试通过。

## 下一步

先按[固定扩展协议](next_batch_protocol.json)采集后续四个训练场景，保留负例及拒绝原因，确认存在可评价目标、空间配准可靠且强静态仍有同预算收益空间，再开发在线观测价值估计。当前候选不调阈值追分；任何新更新器均需独立校准和“只用先验／加入已付反馈”的信息增量检验。真实执行仍需补相机视场或任务平台约束。

来源：[CRASAR数据集](https://huggingface.co/datasets/CRASAR/CRASAR-U-DROIDs)，[作者论文](https://arxiv.org/abs/2407.17673)。数据固定revision为4c2732a7ab9658517ca8caa69bfee9e1f87ca532；训练开发案例，不宣称独立泛化。原感知检查与历次冻结结果已按完成收据核验保留。
'''
    (O/'README.md').write_text(text,encoding='utf-8')
    dump(O/'next_batch_protocol.json',{'status':'PLANNED_NOT_EXECUTED','scope':'Training development,not held-out evaluation',
      'selection':'Next four scenes at zero-based eligibility ranks2,3,4,5 in the same geometry/metadata ordering; one fixed128m center ROI each; include all outcomes and failures; no replacement on performance.',
      'scene_names':['1001-Ft-Myers-Beach-DIRT.geo.tif','1001-Iona-Point.geo.tif','1001-Kennedy-Green-Mobile-Homes.geo.tif','1001-McGregor-College-Pkwy-South.1.geo.tif'],
      'primary_endpoint':'UAS-view major/destroyed; zero targets remain undefined and retained',
      'secondary_endpoint':'UAS-view minor/major/destroyed; exploratory across current and next batch,not independent confirmation',
      'input_contract':'Pinned dataset and detector,same footprints/windows,explicit masks,prior/paid evidence isolation; no training or thresholds tuned on this batch',
      'gates':['Published time order and sensor relation','Independent image alignment check','Footprint support and target prevalence','Strong static versus evaluator ceiling gap','Paid feedback increment established before fitting update'],
      'flight_costs':'Proxy sensitivity only until measured telemetry/camera contract available','gpu_rental':False,
      'method_success_criterion':'Consistent paired effective-check gain against sufficiently solved static across admissible regions/budgets,with false positives,computation,return and service costs reported; one case insufficient'})
    # Sources are snapshotted; sparse TIFFs are explicitly excluded from sealed artifacts.
    src=O/'source_snapshots';src.mkdir(exist_ok=True)
    for n in ['probe_crasar_region.py','extract_crasar_region.py','qualify_crasar_region.py','crasar_alignment_core.py',
      'crasar_route_controls.py','run_crasar_fixed_perception.py','evaluate_crasar_routes.py','finish_crasar_region.py',
      'test_crasar_alignment.py','test_crasar_route_controls.py','route_headroom_core.py']:
        shutil.copyfile(ROOT/'work/disaster_route'/n,src/n)
    shutil.copyfile(ROOT/'work/disaster_search/strongbaseline24_core.py',src/'strongbaseline24_core.py')
    files={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and '.sparse.tif' not in p.name and p.name!='completion_receipt.json'}
    dump(O/'completion_receipt.json',{'status':'COMPLETE_EXPLORATORY_REPLAY_NO_METHOD_GAIN','files':files,
      'excluded_ephemeral':'*.sparse.tif are partial logical-size reconstruction files,never full imagery; original range bytes are sealed instead',
      'primary_failed_gate':'No severe targets','method_contribution':False,'flight_validation':False})
    print(json.dumps({'sealed_files':len(files),'audit':audit},ensure_ascii=False))

if __name__=='__main__':run()
