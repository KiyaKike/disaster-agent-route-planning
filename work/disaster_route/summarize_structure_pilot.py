"""Outcome interpretation and null-safe plot for the frozen integration pilot."""
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

import run_structure_pilot as run


def main():
    run.ensure_open();run.verify_inputs()
    cases=[c for c in run.read(run.OUT/'public_cases.json') if c['geometry_quality']['usable_inferred_range']]
    routes=run.read(run.OUT/'routes.json');metrics=run.read(run.OUT/'metrics.json')
    comparisons=[]
    for c in cases:
        for b in run.BUDGETS:
            rs={r['state']['policy']:r for r in routes if r['sample_id']==c['sample_id'] and r['state']['budget_s_model']==b}
            ms={m['policy']:m for m in metrics if m['sample_id']==c['sample_id'] and m['budget_s_model']==b}
            comparisons.append({'sample_id':c['sample_id'],'budget_s_model':b,
                'structure_changes_actions_vs_frozen':rs['online_structure']['state']['visited']!=rs['online_frozen']['state']['visited'],
                'guard_changes_actions_vs_structure':rs['online_structure_guard']['state']['visited']!=rs['online_structure']['state']['visited'],
                'structure_minus_frozen_recall':None if ms['online_frozen']['severe_pixel_recall'] is None else ms['online_structure']['severe_pixel_recall']-ms['online_frozen']['severe_pixel_recall']})
    out={'scope':'Nine case-budget comparisons, development integration only. No pooled success claim.',
         'comparisons':comparisons,'structure_changes_actions':sum(r['structure_changes_actions_vs_frozen'] for r in comparisons),
         'guard_changes_actions':sum(r['guard_changes_actions_vs_structure'] for r in comparisons),
         'mechanism_conclusion':'Same-building update sometimes changes actions, but no consistent advantage; the guarded variant has identical actions to unguarded in all nine matched comparisons. No measured added benefit of this floor in this pilot.',
         'decision':'Do not tune guard/propagation on these cases. Obtain independently referenced/registered continuous case and platform constraints; retain these as integration fixtures.'}
    run.save(run.OUT/'interpretation.json',out)
    p=run.OUT/'README.md';text=p.read_text()
    insert='''## 候选机制是否真的改变了路线

三个案例×三档预算共9组配对中，同建筑传播相对在线不传播改变了3组实际访问顺序；保底版本相对未保底版本在9组中全部动作相同。因此不能宣称保底带来额外收益。主预算600下，Harvey的在线不传播/同建筑均为44.81%严重像素召回；SantaRosa由36.77%降到34.92%；墨西哥这一固定案例没有严重参考，召回保留null。当前没有稳定收益，停止在这些案例上调传播强度追分。

本轮的实质进展是灾前真实结构、地理坐标重建、付费局部证据及带转向/返程预算的路线被接通。可靠性校准、真实配准和真实飞行约束仍未解决。

'''
    text=text.replace('## 主预算600：逐案例结果',insert+'## 主预算600：逐案例结果')
    text+='\n审计修正说明：当前NumPy移除了二维叉积支持，原费用审计在边界环节中断后由单独适配器续验。ENU投影的影像边界曲线与角点直线近似产生最大3.28毫米差异，边界核验明确采用1厘米几何容差；这不代表定位精度。原协议、坐标、路线和分数未修改，见`audit_compatibility_amendment.json`。\n'
    p.write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Pre static','Post static','Online frozen','Neighbor','Same-building','Same-building + floor']
    colors=['#94a3b8','#3b82f6','#64748b','#e87966','#a78bfa','#14b8a6']
    fig,axes=plt.subplots(3,3,figsize=(15,12.3))
    for row,c in enumerate(cases):
        sid=c['sample_id'];ax=axes[row,0]
        with Image.open(run.OUT/'assets'/f'{sid}_post.jpg') as im:ax.imshow(im,extent=[0,1024,1024,0])
        for poly in c['pre_pixel_rings']:
            a=np.array(poly);ax.plot(a[:,0],a[:,1],color='#49d9dd',lw=.6,alpha=.8)
        r=next(r for r in routes if r['sample_id']==sid and r['state']['policy']=='online_structure_guard' and r['state']['budget_s_model']==600)
        px=np.array([[j%8*128+64,j//8*128+64] for j in [64]+r['state']['visited']+[64]],float);px[[0,-1]]=[512,1024]
        ax.plot(px[:,0],px[:,1],'-o',color='#ffae42',lw=1.3,ms=3)
        ax.set_title(sid+'\nPre footprints / paid route on post image',fontsize=10);ax.set_axis_off()
        mm=[next(m for m in metrics if m['sample_id']==sid and m['policy']==p and m['budget_s_model']==600) for p in run.POLICIES]
        axes[row,1].barh(range(6),[m['pre_building_fractional_coverage']*100 for m in mm],color=colors)
        axes[row,1].set_yticks(range(6),labels,fontsize=8);axes[row,1].set_xlim(0,100);axes[row,1].set_xlabel('Pre-building footprint coverage (%)')
        ax=axes[row,2]
        if all(m['severe_pixel_recall'] is None for m in mm):
            ax.text(.5,.5,'No published severe reference\nRecall undefined (null)',transform=ax.transAxes,ha='center',va='center',color='#526779')
        else:
            for p in ('post_static','online_neighbor','online_structure_guard'):
                ms=[next(m for m in metrics if m['sample_id']==sid and m['policy']==p and m['budget_s_model']==b) for b in run.BUDGETS]
                ax.plot(run.BUDGETS,[m['severe_pixel_recall']*100 for m in ms],'-o',label=labels[list(run.POLICIES).index(p)])
            ax.legend(fontsize=7)
        ax.set_xlim(280,920);ax.set_xticks(run.BUDGETS);ax.set_ylim(0,100);ax.set_xlabel('Modeled budget (s)');ax.set_ylabel('Published severe-pixel recall (%)');ax.grid(alpha=.2)
    fig.suptitle('Development integration pilot | reconstructed geometry | assumed UAV time | no efficacy claim',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.975]);fig.savefig(run.OUT/'figures/structure_pilot.png',dpi=150);plt.close(fig)
    shutil.copyfile(__file__,run.OUT/'source_snapshots'/Path(__file__).name)
    print(json.dumps({k:v for k,v in out.items() if k!='comparisons'},ensure_ascii=False))


if __name__=='__main__':main()
