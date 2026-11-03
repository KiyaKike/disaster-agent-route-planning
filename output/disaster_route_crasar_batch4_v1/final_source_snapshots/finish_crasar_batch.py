"""Audit and freeze the fixed four-region expansion,including every rejection."""
import json,hashlib,shutil,re,math,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from run_crasar_batch import ROOT,B,NAMES,dump

def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def run():
    if (B/'completion_receipt.json').exists():raise RuntimeError('Batch is sealed')
    execution=read(B/'execution_status.json');assert len(execution)==4
    assert [r['scene'] for r in execution]==NAMES
    rows=[];routes_total=0;prefixes=0;calls=0;return_checks=0;cross_target_comparisons=0
    for status in execution:
        folder=Path(status['folder']);row={'case':folder.name,'scene':status['scene'],'status':status['status']}
        if (folder/'qualification.json').exists():
            q=read(folder/'qualification.json');row.update(dates=q['capture_dates_author_catalog'],eligible_buildings=q['geometry_eligible'],severe_targets=q['eligible_primary_targets'],
              any_damage_targets=sum(r['eligible'] and r['labels']['uas'] in ['minor damage','major damage','destroyed'] for r in read(folder/'private_reference.json')),
              support=q['rgb_nonzero_support_fraction'],registration=q['image_feature_diagnostics'])
        if status['status']=='PROVISIONAL_REPLAY_COMPLETE':
            inputs=read(folder/'public_planning_inputs_any_damage.json');xy=inputs['xy_m_relative']
            logs=read(folder/'inference_log.json');assert len(logs)==32
            for log in logs:assert sha(folder/'fixed_model_cache'/f"{log['mode']}_{log['window']:02d}.npz")==log['cache_sha256']
            calls+=sum(l['forward_calls'] for l in logs)
            for target in ['severe','any_damage']:
                summary=read(folder/f'experiment_summary_{target}.json');row[target]=summary
                route_list=read(folder/f'routes_{target}.json');ledgers=read(folder/f'dual_map_paid_evidence_{target}.json')
                c=np.array(read(folder/f'public_planning_inputs_{target}.json')['window_building_fraction'])
                paid=np.array(read(folder/f'private_paid_observations_{target}.json')['damage_model_pixel_fraction_per_window'])
                for route in route_list:
                    path=route['route'];assert len(path)==len(set(path));nodes=[16]+path+[16]
                    dist=sum(math.dist(xy[a],xy[b]) for a,b in zip(nodes,nodes[1:]));elapsed=1+dist/5+len(path)*10
                    assert abs(route['distance_m']-dist)<1e-7 and abs(route['elapsed_proxy_s']-elapsed)<1e-7 and elapsed<=route['budget_proxy_s']+1e-7
                    return_checks+=1
                    states=[l for l in ledgers if l['policy']==route['policy'] and l['budget_proxy_s']==route['budget_proxy_s']]
                    assert len(states)==len(path)
                    for k,state in enumerate(states,1):
                        assert state['step']==k and state['accessed_window']==path[k-1] and not state['reference_labels_in_state']
                        np.testing.assert_allclose(state['evidence']['observed_building_fraction'],c[path[:k]].sum(axis=0))
                        np.testing.assert_allclose(state['evidence']['observed_damage_model_pixel_fraction'],paid[path[:k]].sum(axis=0));prefixes+=1
                routes_total+=len(route_list)
                for budget in [60,100,140]:
                    select={r['policy']:r for r in route_list if r['budget_proxy_s']==budget}
                    assert select['static_exact']['public_utility_proxy']>=select['value_cost_proxy']['public_utility_proxy']-1e-8
                    cross_target_comparisons+=1
            pp=read(folder/'perception_route_protocol.json')
            assert pp['localization_threshold']==.38 and pp['budgets_proxy_s']==[60,100,140]
        rows.append(row)
    diagnostic=read(B/'feedback_increment_summary.json')
    # Independently enforce regional/spatial purges used in the prediction experiment.
    from crasar_feedback_diagnostic import load_case,box_distance
    case_data={r['case']:load_case(B/r['case']) for r in rows if r['status']=='PROVISIONAL_REPLAY_COMPLETE'}
    for fold in diagnostic['per_region']:
        test=case_data[fold['held_out']]
        for name in fold['training_regions']:
            train=case_data[name];assert not train['all_ids']&test['all_ids']
            if train['crs']==test['crs']:assert box_distance(train['bounds'],test['bounds'])>128
    # Earlier results are preserved byte-for-byte via their own sealed receipts.
    parent_checks={}
    for name in ['disaster_route_crasar_region_v1','disaster_route_crasar_region_rank2_v1','disaster_route_crasar_feasibility_20261103']:
        folder=ROOT/'output'/name;receipt=read(folder/'completion_receipt.json')
        for file,h in receipt['files'].items():assert sha(folder/file)==h,(name,file)
        parent_checks[name]=len(receipt['files'])
    tests=(B/'tests.txt').read_text();match=re.search(r'Ran (\d+) tests',tests)
    assert match and tests.rstrip().endswith('OK')
    audit={'status':'PASS_FOR_PROVISIONAL_REPLAY_ONLY','fixed_regions':4,'provisional_replay_regions':len(case_data),'routes_verified':routes_total,
      'return_budget_checks':return_checks,'paid_evidence_prefix_checks':prefixes,'exact_control_dominance_checks':cross_target_comparisons,
      'new_detector_forward_calls':calls,'tests_passed':int(match[1]),'parent_sealed_files_verified':parent_checks,
      'prediction_folds_evaluated':diagnostic['regions_evaluated'],'human_truth_prediction_test':False,
      'regions_passing_independent_registration_gate':0,
      'registration_absolute_accuracy_certified':False,'flight_validation':False,'method_gain_established':False,
      'gpu_rental':False,'new_detector_training':False}
    dump(B/'region_results.json',rows);dump(B/'audit.json',audit)
    # Compact scientific comparison: no decorative background colors.
    selected=[r for r in rows if r['status']=='PROVISIONAL_REPLAY_COMPLETE']
    fig=Image.new('RGB',(1120,100+max(1,len(selected))*180),'white');draw=ImageDraw.Draw(fig)
    draw.text((20,15),'Fixed-region replay:main budget100 proxy seconds',fill='black')
    for i,row in enumerate(selected):
        top=55+i*180;draw.text((20,top),row['scene'],fill='black')
        for col,target in enumerate(['severe','any_damage']):
            main=row[target]['main_budget_results'];draw.text((20+col*550,top+23),f'{target}:targets={row[target]["eligible_target_buildings"]}',fill='black')
            for k,policy in enumerate(['distance','damage','uncertainty','static_exact','value_cost_proxy']):
                r=next(r for r in main if r['policy']==policy);draw.text((20+col*550,top+43+k*21),
                    f'{policy:18s} TP={r["correctly_checked_target_buildings"]} FP={r["false_positive_checked_buildings"]} distance={r["distance_m"]:.1f}m',fill='black')
    fig.save(B/'main_comparison_blackwhite.png')
    short={'case_rank03':'Ft. Myers Beach DIRT','case_rank04':'Iona Point','case_rank05':'Kennedy Green','case_rank06':'McGregor South.1'}
    data_table='|区域|完整可用建筑|严重／任意损伤目标|处理|\n|---|---:|---:|---|\n'
    for row in rows:
        handling={'PROVISIONAL_REPLAY_COMPLETE':'完成临时影像访问回放','RGB_SENSOR_GATE_FAILED':'灾前单通道，不进入固定RGB模型比较','NO_ELIGIBLE_BUILDINGS':'无完整可用建筑','TEMPORAL_GATE_FAILED':'时序不满足'}.get(row['status'],row['status'])
        data_table+=f"|{short[row['case']]}|{row.get('eligible_buildings','未核实')}|{row.get('severe_targets','—')}／{row.get('any_damage_targets','—')}|{handling}|\n"
    result_table='|区域／目标|距离优先|线索优先|不确定性优先|精确静态|价值／成本候选|事后上限|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for row in selected:
      for target,label in [('severe','严重'),('any_damage','任意损伤')]:
        s=row[target];by={r['policy']:r for r in s['main_budget_results']};n=s['eligible_target_buildings']
        vals=[f"{by[p]['correctly_checked_target_buildings']}/{n}" if n else '未定义' for p in ['distance','damage','uncertainty','static_exact','value_cost_proxy']]
        ceiling=next(o['correctly_checked_target_buildings'] for o in s['hindsight_same_budget_ceiling'] if o['budget_proxy_s']==100)
        result_table+=f"|{short[row['case']]}／{label}|"+'|'.join(vals)+f'|{ceiling}/{n}'+('（无目标）' if not n else '')+'|\n'
    if diagnostic['regions_evaluated']:
        info=f"完成{diagnostic['regions_evaluated']}个区域留出折；仅先验／加入已付观测的区域宏平均MSE为{diagnostic['macro_mse_prior']:.6f}／{diagnostic['macro_mse_prior_plus_paid']:.6f}，{diagnostic['regions_improved']}折改善。"
    else:info='可用且空间分离的区域／候选不足，本轮不拟合观测增量模型。'
    report=f'''# 四区域固定扩展：数据、路线和观测增量检验

已按此前固定名单检查四处训练区域，未按损伤数量或方法得分替换。完成{len(case_data)}处临时回放、{routes_total}条闭合路线。**尚未建立新方法收益，独立配准精度与实际飞行条件仍未确立。**

{data_table}
时间来自作者目录；每处128m方形区域由共享建筑几何中心固定，90%轮廓完整性与三源影像支持规则一致。RGB非零是影像支持代理，不是可见性保证。配准使用作者局部修正；内部坐标换算检查不能证明独立影像精度，跨源图像特征诊断见各案例qualification.json与region_audit_panel.png。单通道仅重复灰度用于查看，未补造颜色或输入固定RGB模型。

Iona灾后卫星／UAS特征匹配显示约5.57m中位残差（16个一致匹配），McGregor约1.38m（20个一致匹配）；灾前跨源匹配均不足。**本批通过独立配准门槛的区域数为0，以下均属临时诊断，不能作为配准合格后的方法验证。**

同预算100代理秒下有效核查数量如下。严重为作者UAS视角major/destroyed；任意损伤包含minor，两种目标均在本批查看标签前明确。它们属于同一事件的训练开发资料，不是独立确认性结果。

{result_table}
有效核查沿用固定规则：至少80%轮廓被访问，且模型对应损伤预测像素达到已访问轮廓10%，最后由独立评估步骤核对作者标签。全部路线均含移动、观察、一次规划及返程；速度5m/s、观察10秒、规划1秒均是假设。精确静态对65,536子集求最短闭合路径，分别优化损伤线索和与候选相同的效用；事后上限使用全部未来预测与标签，仅用于诊断。误报、首次正确核查、距离、耗时与其余预算结果见各案例route_scores文件及[完整汇总](region_results.json)。

评估修正：未分类／遮挡标签不作负例，单独报告已核查但标签未知的建筑数；全窗口访问结果也不称为一般检出上限，因为比例阈值会被新增无损像素稀释，上限以逐子集求得的事后参考为准。旧封存材料保留，当前修正写入本批。

## 付费观测能否帮助预测未访问区域

{info}协议在本批结果检查前固定。距离优先路线前缀定义已付观测；排除与前缀共享建筑的候选，区域留出时进一步排除重复建筑或128m以内邻近区域。固定岭回归、训练折标准化、按区域等权，无跨折置换借用。

三折结果方向不一致：DIRT的MSE从0.120280变为0.152991，Iona从0.173243变为0.128647，McGregor从0.208161变为0.204830。宏平均约改善3%，其中Iona存在上述配准残差，不能据此建立可靠的信息增量或路线收益结论。

**预测目标是固定模型在未访问UAS窗口的损伤像素比例，衡量模型观测的一致性，不是人工损伤真值，也不是路线净收益。** 重复前缀行不是独立案例；不据此宣称跨事件泛化。即使预测改善，也须再证明它能改变行动并在计入成本后超过强静态。

## 验证与下一步

本机完成{calls}次固定感知器前向调用，未训练感知器、未租GPU。{return_checks}条路线返程预算核验、{prefixes}个付费证据前缀检查及{audit['tests_passed']}项专项测试通过；前两区域与最初可行性包共{sum(parent_checks.values())}个封存文件哈希保持一致。

目前不推进在线更新器调参。DIRT任意损伤的精确静态为2/9，同预算事后上限5/9，说明存在空间，但距离优先反而达3/9，当前代理价值排序尚不可靠；Iona有配准疑点，McGregor严重目标在各预算下事后可检数均为0。下一步先在已下载数据上补灾前／灾后／UAS的空间控制点核验，再隔离评估先验能否预测“付费后可正确核查的目标”，并与均匀、距离和已充分求解静态比较。仅在这两关通过后开发在线更新器，另留区域检验收益。无需为本轮问题租GPU。当前“语义先验＋空间证据”的接口保留，生成图像只作为假设，不能补记为实际观测。

来源固定为CRASAR-U-DROIDs revision4c2732a7ab9658517ca8caa69bfee9e1f87ca532；详见[批次协议](batch_protocol.json)、[预测检验协议](feedback_diagnostic_protocol.json)与[audit.json](audit.json)。
'''
    (B/'README.md').write_text(report,encoding='utf-8')
    # Keep original acquisition snapshots and the final code revision separately.
    final=B/'final_source_snapshots';final.mkdir(exist_ok=True)
    scripts=['run_crasar_batch.py','probe_crasar_region.py','extract_crasar_region.py','qualify_crasar_region.py','crasar_alignment_core.py',
      'run_crasar_fixed_perception.py','evaluate_crasar_routes.py','crasar_route_controls.py','route_headroom_core.py','crasar_feedback_diagnostic.py','finish_crasar_batch.py',
      'test_crasar_alignment.py','test_crasar_route_controls.py','test_crasar_feedback_diagnostic.py','test_crasar_range_cache.py','test_crasar_reference_evaluation.py']
    for script in scripts:shutil.copyfile(ROOT/'work/disaster_route'/script,final/script)
    shutil.copyfile(ROOT/'work/disaster_search/strongbaseline24_core.py',final/'strongbaseline24_core.py')
    dump(B/'infrastructure_changes.json',{'selection_or_model_threshold_changes':False,'changes':['Coalesced adjacent validated byte ranges to reduce network requests','Bounded transport retries and resumed exact byte cache','Raised annotation file cap to12MB','Single-band preview repetition for display only;RGB inference rejects it','Existing sealed output write guard','Fetched margin derived from published local alignment vector magnitudes','Iona UAS cached-source cap64MiB instead of32MiB:see transport amendment','Unknown reference labels excluded from false positive/precision denominators','All-window outcome renamed:only exact subset hindsight result is an upper bound'],
       'scientific_contract':'Fixed scene list,geometric ROI,perception weights,thresholds,costs and route controls retained. Failures and retries remain in logs and attempts.'})
    files={str(p.relative_to(B)):sha(p) for p in sorted(B.rglob('*')) if p.is_file() and '.sparse.tif' not in p.name and p.name!='completion_receipt.json'}
    dump(B/'completion_receipt.json',{'status':'COMPLETE_FIXED_BATCH_PROVISIONAL_ANALYSIS','files':files,'method_gain_established':False,
        'excluded_ephemeral':'Sparse TIFFs are partial reconstruction containers,not complete imagery;source byte-range caches and derived ROIs are sealed instead'})
    print(json.dumps({'audit':audit,'sealed_files':len(files),'feedback':diagnostic},ensure_ascii=False,indent=2))

if __name__=='__main__':run()
