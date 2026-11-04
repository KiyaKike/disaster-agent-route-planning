"""Read frozen JSON logs to explain Iona's fork and ranking/route disagreement.

No model calls, new policy execution, parameter fitting, or original imagery.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'iona-ranking-diagnostic-v1'})

REGIONS = {'case_rank03':'DIRT', 'case_rank04':'Iona', 'case_rank06':'McGregor'}
METHODS = ('no_update','spatial','matched_spatial','pearson')
NAMES = {'no_update':'Fixed prior','spatial':'Spatial','matched_spatial':'Matched spatial','pearson':'Pearson'}
MARKERS = {'no_update':'o','spatial':'s','matched_spatial':'^','pearson':'D'}


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def ranked_metrics(scores, labels):
    """Unknowns removed by caller. AP groups ties; NDCG uses ID-stable ranking."""
    p,y=np.asarray(scores,float),np.asarray(labels,int)
    if not y.sum():
        return {'ap':None,'ndcg_at3':None,'top1':int(y[np.argmax(p)])}
    order=np.argsort(-p,kind='stable')
    ordered_p,ordered_y=p[order],y[order]
    boundaries=np.r_[np.flatnonzero(np.diff(ordered_p)),len(p)-1]
    tp=np.cumsum(ordered_y)[boundaries]
    ap=float(np.sum(np.diff(np.r_[0,tp])*tp/(boundaries+1))/y.sum())
    k=min(3,len(y));discounts=1/np.log2(np.arange(2,k+2))
    ndcg=float(np.dot(ordered_y[:k],discounts)/discounts[:min(k,int(y.sum()))].sum())
    return {'ap':ap,'ndcg_at3':ndcg,'top1':int(ordered_y[0])}


def iona_case(base):
    folder=base/'case_rank04'
    public=read(folder/'public_inputs.json');xy=np.array(public['xy_m'])
    ids=public['building_ids'];sim=np.array(public['similarity'])
    routes={r['policy']:r for r in read(folder/'routes.json') if r['budget_proxy_s']==100}
    pearson,spatial=routes['pearson'],routes['spatial']
    prefix=[]
    for a,b in zip(pearson['route'],spatial['route']):
        if a!=b:break
        prefix.append(a)
    assert prefix==[4,5,7]
    fork=len(prefix)+1
    decision=next(d for d in pearson['decisions'] if d['decision_number']==fork)
    update=next(u for u in pearson['updates'] if u['decision_number']==fork)
    anchor=prefix[-1]
    assert update['anchor_index']==anchor and update['feedback']==0
    candidates=decision['feasible_indices']
    rows=[]
    for i in candidates:
        distance=math.dist(xy[anchor],xy[i])
        before,after=update['scores_before'][i],update['scores_after'][i]
        rows.append({'local_id':f'I{i+1:02d}','index':i,'building_id':ids[i],
                     'distance_from_I08_m':distance,'feature_similarity_to_I08':float(sim[anchor,i]),
                     'spatial_weight':math.exp(-distance**2/(2*50**2)),
                     'score_before':before,'score_after':after,'score_change':after-before,
                     'priority_before':before/distance,'priority_after':after/distance,
                     'observe_and_return_end_s':decision['elapsed_proxy_s']+distance/5+10+math.dist(xy[i],xy[-1])/5})
    before_choice=max(rows,key=lambda r:(r['priority_before'],-r['index']))['index']
    after_choice=max(rows,key=lambda r:(r['priority_after'],-r['index']))['index']
    assert before_choice==6 and after_choice==0==decision['target']
    next_decision=next(d for d in pearson['decisions'] if d['decision_number']==fork+1)
    late_end=next_decision['elapsed_proxy_s']+math.dist(xy[0],xy[6])/5+10+math.dist(xy[6],xy[-1])/5
    assert 6 not in next_decision['feasible_indices'] and late_end>100
    comparative=[]
    for method in ('spatial','matched_spatial','pearson'):
        state=next(d for d in routes[method]['decisions'] if d['decision_number']==fork)
        for i in (6,0):
            distance=math.dist(xy[anchor],xy[i])
            comparative.append({'method':method,'target':f'I{i+1:02d}',
                                'score':state['scores'][i],'distance_m':distance,
                                'priority':state['scores'][i]/distance})
    return {'rows':rows,'common_prefix_indices':prefix,'fork_decision':fork,
            'paid_negative_anchor':'I08','fork_elapsed_s':decision['elapsed_proxy_s'],
            'choice_without_this_last_update':'I07','actual_choice':'I01',
            'late_I07_observe_and_return_end_s':late_end,'late_I07_budget_excess_s':late_end-100,
            'pearson_route':[f'I{i+1:02d}' for i in pearson['route']],
            'spatial_route':[f'I{i+1:02d}' for i in spatial['route']],
            'pearson_hits':pearson['source_positive_checks'],'spatial_hits':spatial['source_positive_checks'],
            'pearson_elapsed_s':pearson['elapsed_proxy_s'],'spatial_elapsed_s':spatial['elapsed_proxy_s'],
            'pearson_distance_m':pearson['distance_m'],'spatial_distance_m':spatial['distance_m'],
            'matched_spatial_same_route':routes['matched_spatial']['route']==spatial['route'],
            'comparative_branch_scores':comparative,
            'static_exact_100s_hits':routes['static_exact']['source_positive_checks'],
            'scope':'Within-log one-update decision lookback and feasibility arithmetic; not a new counterfactual route run'},xy,routes


def ranking_data(workspace,base):
    rows=[]
    for case,name in REGIONS.items():
        folder=base/case;public=read(folder/'public_inputs.json');ids=public['building_ids']
        refs={r['building_id']:r['labels']['uas'] for r in read(workspace/'output/disaster_route_crasar_batch4_v1'/case/'private_reference.json')}
        valid_names={'no damage','minor damage','major damage','destroyed'}
        region_routes={r['policy']:r for r in read(folder/'routes.json') if r['budget_proxy_s']==100 and r['policy'] in METHODS}
        first={tuple(r['route'][:1]) for r in region_routes.values()}
        assert len(first)==1
        first_index=next(iter(first))[0]
        for method in METHODS:
            route=region_routes[method]
            state=next(d for d in route['decisions'] if d['decision_number']==2)
            assert state['observed_indices']==[first_index]
            candidates=[i for i in range(len(ids)) if i!=first_index and refs[ids[i]] in valid_names]
            ys=[int(refs[ids[i]]!='no damage') for i in candidates]
            ps=[state['scores'][i] for i in candidates]
            m=ranked_metrics(ps,ys)
            rows.append({'case':case,'region':name,'method':method,'common_first_index':first_index,
                         'common_first_building_id':ids[first_index],
                         'first_observation_complete_s':route['observations'][0]['observation_complete_proxy_s'],
                         'decision2_elapsed_s':state['elapsed_proxy_s'],
                         'remaining_known_targets':len(candidates),'remaining_positive_targets':sum(ys),
                         'excluded_unknown_targets':len(ids)-1-len(candidates),
                         **m,'route_hits_100s':route['source_positive_checks'],
                         'route_hits_after_common_first':route['source_positive_checks']-int(route['observations'][0]['source_label']==1),
                         'route_visits':len(route['route']),'closed_distance_m':route['distance_m'],
                         'closed_elapsed_s':route['elapsed_proxy_s']})
    comparisons=[]
    for case,name in REGIONS.items():
        indexed={r['method']:r for r in rows if r['case']==case}
        for a,b in [('pearson','no_update'),('pearson','spatial'),('spatial','no_update'),('pearson','matched_spatial')]:
            comparisons.append({'region':name,'method_a':a,'method_b':b,
                                'ap_delta':indexed[a]['ap']-indexed[b]['ap'],
                                'route_hit_delta':indexed[a]['route_hits_100s']-indexed[b]['route_hits_100s']})
    return rows,comparisons


def save_figure(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=200,facecolor='white')
    svg=path.with_suffix('.svg')
    fig.savefig(svg,facecolor='white',metadata={'Date':None})
    # Matplotlib's SVG path lines carry optional trailing spaces; normalize before sealing.
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def plot_iona(out,case,xy,routes):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.5,6.2),gridspec_kw={'width_ratios':[1,1.25]})
    local=xy-xy[-1];home=len(xy)-1
    for i,(x,y) in enumerate(local[:-1]):
        ax.scatter(x,y,marker='*' if i==6 else 'o',s=150 if i==6 else 42,
                   facecolor='black' if i==6 else 'white',edgecolor='black',zorder=5)
        ax.annotate(f'I{i+1:02d}'+(' (+)' if i==6 else ''),(x,y),xytext=(6,3),textcoords='offset points',fontsize=9)
    ax.scatter(*local[-1],marker='^',s=65,color='black',zorder=5)
    ax.annotate('Home',local[-1],xytext=(-26,-14),textcoords='offset points',fontsize=9)
    prefix=[home]+case['common_prefix_indices']
    ax.plot(local[prefix,0],local[prefix,1],'-',color='black',lw=2.4,label='Shared visits 1-3',zorder=2)
    for policy,style,label,offset in [('spatial','--','Spatial suffix (1 hit)',(-11,4)),('pearson','-','Pearson suffix (0 hits)',(6,-7))]:
        indices=[prefix[-1]]+routes[policy]['route'][3:]+[home]
        ax.plot(local[indices,0],local[indices,1],style,color='black',lw=1.2,label=label,zorder=1)
        for a,b in zip(indices,indices[1:]):
            start=local[a]+.54*(local[b]-local[a]);end=local[a]+.64*(local[b]-local[a])
            ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'black','lw':1.1})
    ax.annotate('Fork after I08 (-)',local[7],xytext=(-2,26),textcoords='offset points',fontsize=9,
                ha='center',arrowprops={'arrowstyle':'-','lw':.6})
    ax.set(xlabel='Easting from home (m)',ylabel='Northing from home (m)',title='(a) Iona, 100 proxy seconds')
    ax.set_aspect('equal');ax.set_xlim(-24,60);ax.set_ylim(-8,131)
    ax.legend(loc='upper left',bbox_to_anchor=(-.1,-.15),fontsize=8,frameon=False)
    detail={r['local_id']:r for r in case['rows']}
    for uid,marker,style in [('I07','*','-'),('I01','o','--')]:
        row=detail[uid];ys=[row['priority_before'],row['priority_after']]
        bx.plot([0,1],ys,style,color='black',marker=marker,markersize=9 if uid=='I07' else 5,
                markerfacecolor='black' if uid=='I07' else 'white',label=uid+(' (+)' if uid=='I07' else ' (-)'))
        for x,y in enumerate(ys):
            shift=10 if (uid=='I07' and x==0) or (uid=='I01' and x==1) else -15
            bx.annotate(f'{y:.5f}',(x,y),xytext=(0,shift),textcoords='offset points',ha='center',fontsize=9)
    bx.set(xticks=[0,1],xticklabels=['Before I08 feedback','After I08 negative'],
           xlim=(-.28,1.28),ylim=(.00695,.0101),ylabel='Selection priority: current score / distance',
           title='(b) One negative feedback reverses priority')
    bx.ticklabel_format(axis='y',style='plain',useOffset=False)
    bx.legend(frameon=False,loc='upper right',fontsize=9)
    bx.text(.02,-.19,'I07 immediately + observation + return: 80.08 s\nI01 first, then I07 + return: 103.68 s > 100 s',
            transform=bx.transAxes,fontsize=9,va='top')
    fig.subplots_adjust(left=.075,right=.98,top=.88,bottom=.24,wspace=.43)
    save_figure(fig,out/'iona_fork')


def plot_scatter(out,rows):
    fig,axes=plt.subplots(1,3,figsize=(11.2,4.2),sharex=True,sharey=True)
    offsets={('DIRT','no_update'):(-58,-19),('DIRT','spatial'):(-22,11),
             ('DIRT','matched_spatial'):(-92,9),('DIRT','pearson'):(5,10),
             ('Iona','no_update'):(-8,10),('Iona','pearson'):(5,11)}
    for ax,(case,name) in zip(axes,REGIONS.items()):
        region=[r for r in rows if r['case']==case]
        for method in METHODS:
            r=next(x for x in region if x['method']==method)
            ax.scatter(r['ap'],r['route_hits_100s'],s=85 if method=='matched_spatial' else 45,
                       marker=MARKERS[method],facecolors='black' if method=='pearson' else 'none',
                       edgecolors='black',linewidths=1,zorder=3)
            if name=='McGregor' or (name=='Iona' and method in ('spatial','matched_spatial')):continue
            ax.annotate(NAMES[method],(r['ap'],r['route_hits_100s']),xytext=offsets[name,method],
                        textcoords='offset points',fontsize=8)
        if name=='Iona':
            ax.annotate('Spatial / Matched',(.333333333333,1),xytext=(4,-18),textcoords='offset points',fontsize=8)
            ax.annotate('vs fixed prior:\nAP +0.057; hits -1',(.2,0),xytext=(.46,1.8),fontsize=8,
                        arrowprops={'arrowstyle':'->','lw':.7})
        if name=='McGregor':ax.annotate('All four coincide',(.5,0),xytext=(7,12),textcoords='offset points',fontsize=8)
        ax.set(title=name,xlim=(0,1),ylim=(-.35,3.65),xticks=[0,.25,.5,.75,1],yticks=[0,1,2,3])
        ax.set_xlabel('AP after the common first observation')
        ax.spines[['right','top']].set_visible(False)
    axes[0].set_ylabel('Total source-positive checks at 100 s')
    handles=[plt.Line2D([],[],ls='',marker=MARKERS[m],color='black',markerfacecolor='black' if m=='pearson' else 'white',label=NAMES[m]) for m in METHODS]
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.065,right=.98,top=.87,bottom=.24,wspace=.14)
    save_figure(fig,out/'ranking_route_scatter')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root,out=args.workspace.resolve(),args.output.resolve()
    if out.exists():raise FileExistsError('Use a new diagnostic directory')
    started=time.perf_counter();base=root/'output/disaster_route_crasar_pearson_isobudget_v1'
    case,xy,routes=iona_case(base)
    rows,comparisons=ranking_data(root,base)
    out.mkdir(parents=True)
    dump(out/'iona_fork.json',case)
    write_csv(out/'iona_candidates_at_fork.csv',case['rows'])
    dump(out/'ranking_route_points.json',rows)
    write_csv(out/'ranking_route_points.csv',rows)
    dump(out/'within_region_comparisons.json',comparisons)
    dump(out/'iona_local_id_map.json',[{'local_id':f'I{i+1:02d}','building_id':u}
                                     for i,u in enumerate(read(base/'case_rank04/public_inputs.json')['building_ids'])])
    plot_iona(out,case,xy,routes);plot_scatter(out,rows)
    dump(out/'execution_recipe.json',{'source':'Frozen region logs only; no original imagery, inference or policy replay',
          'budget_proxy_s':100,'scatter_state':'Decision2, after exactly one common paid observation; remaining known-label targets',
          'scatter_y':'Total route hits including identical first-visit credit within each region',
          'methods':list(METHODS),'static_exact_omission':'Different initial route/history; not a matched first-feedback comparison',
          'unknowns':'Excluded from ranking metric, retained and charged in original routes',
          'ap_ties':'Grouped thresholds','ndcg':'Binary,log2 discounts,unvisited-reference ideal; stable ID tie-break',
          'local_lookback':'Remove only last I08 update at logged state; no alternative multi-step route generated',
          'script_sha256':sha(Path(__file__)),'elapsed_seconds':time.perf_counter()-started,
          'input_hashes':{str(p.relative_to(root)):sha(p) for p in
            [base/'summary.json']+[base/c/f for c in REGIONS for f in ('routes.json','public_inputs.json')]+
            [root/'output/disaster_route_crasar_batch4_v1'/c/'private_reference.json' for c in REGIONS]}})
    print(json.dumps({'iona_hits':[case['pearson_hits'],case['spatial_hits']],
                      'fork_decision':case['fork_decision'],'late_return_end_s':case['late_I07_observe_and_return_end_s'],
                      'scatter_points':len(rows),'elapsed_seconds':time.perf_counter()-started},indent=2))


if __name__=='__main__':main()
