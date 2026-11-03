"""Separate exact information-value results from finite POMCP sample results."""
from pathlib import Path
import shutil

import run_aippms_control as run


def main():
    run.open_output();run.verify()
    summary=run.read(run.OUT/'summary.json')
    interpretations=[]
    for b in run.BUDGETS:
        rows={(r['sensor_model'],r['policy']):r for r in summary if r['budget']==b}
        gain=rows['informative','exact_one_sense']['expected_or_sample_mean_reward']-rows['informative','exact_no_sense']['expected_or_sample_mean_reward']
        null=rows['uninformative','exact_one_sense']['expected_or_sample_mean_reward']-rows['uninformative','exact_no_sense']['expected_or_sample_mean_reward']
        assert gain>0 and null==0
        interpretations.append({'budget':b,'exact_paid_information_gain':gain,'exact_null_information_gain':null,
                                'pomcp_gcb_informative_sample_mean':rows['informative','pomcp_gcb']['expected_or_sample_mean_reward'],
                                'pomcp_gcb_uninformative_sample_mean':rows['uninformative','pomcp_gcb']['expected_or_sample_mean_reward']})
    run.save(run.OUT/'interpretation.json',{'role':'Controlled mechanism check,not published-table reproduction or real disaster result.',
        'conditions':interpretations,'decisive_mechanism':'Paid remote measurement directly observes unvisited candidates;independent rocks need no spatial propagation assumption.',
        'limitation':'Perfectly known source sensor likelihood and synthetic remote footprint do not exist as validated inputs in current xBD local-observation task.',
        'null_pomcp_sample':'Small positive/negative sample means on the uninformative model are not information value;exact expected gain is zero. Finite-search sampling effects and unnecessary sensing retained.',
        'objective_mismatch':'Rock visitation reward is not the same as inspection detection reward. Real inspection may finish a task at image acquisition,so do not copy separate visit reward without task justification.',
        'next':'Verify actual observation footprint and task reward contract before adapting a policy to disaster data.',
        'gpu_rental':False,'new_model_calls':0})
    f=run.OUT/'README.md';text=f.read_text()
    text+='''

## 机制判断与接入边界

预算12/20下，最多一次感知的精确期望收益分别为2.8717/9.1217，不感知均为0；无信息传感器下，两者精确期望均为0。该对照验证的是已知远距观测模型中的信息价值，不是新方法的灾害任务增益。

主预算20下，有限计算POMCP-GCB的样本平均收益为3.75，低于最多一次感知的精确参考9.1217；POMCP可重复感知，所以这不是严格的策略集合上界比较，也没有测得实用的高效最优策略。无信息条件下GCB样本均值在预算12/20为+0.3125/-0.3125，不能当作观测收益；精确零收益结论不受这组有限样本波动影响。没有根据这些结果重选参数或种子。

另一个必须解决的任务差异：岩石模型通过到访获得任务奖励；建筑损伤巡查可能在拍摄判读时已经完成任务。因此后续不能原样复制“观察一次，再到目标位置领奖”的结构，须先确定真实观察范围、可判读程度及任务完成条件。

对应双层地图，先验层可保存任务结构与初始概率，证据层保存已付观测及来源；二者融合后的信念交给规划器。双层组织本身仍未被证明带来收益。
'''
    f.write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    policies=('exact_no_sense','exact_one_sense','pomcp_random','pomcp_gcb')
    labels=('Exact no sensing','Exact <=1 sensing','POMCP random (sample)','POMCP GCB (sample)')
    for ax,model in zip(axes,run.MODELS):
        for k,(policy,label) in enumerate(zip(policies,labels)):
            rs=[next(r for r in summary if r['sensor_model']==model and r['budget']==b and r['policy']==policy) for b in run.BUDGETS]
            values=[r['expected_or_sample_mean_reward'] for r in rs]
            bars=ax.bar(np.arange(2)+(k-1.5)*.19,values,.19,label=label)
            for bar,value in zip(bars,values):
                if value:ax.text(bar.get_x()+bar.get_width()/2,value+(.13 if value>0 else -.15),f'{value:.2f}',ha='center',va='bottom' if value>0 else 'top',fontsize=8)
        ax.axhline(0,color='#334155',lw=.8);ax.set_title(model+' sensor');ax.set_xticks([0,1],[str(b) for b in run.BUDGETS])
        ax.set_xlabel('Energy budget (source units)');ax.set_ylabel('Task reward (+10 / -10)');ax.set_ylim(-1,10.3);ax.grid(axis='y',alpha=.2)
    axes[1].legend(fontsize=8,loc='upper left');fig.suptitle('Corrected paper-based ISRS mechanism reference | not native Julia/table replication')
    fig.savefig(run.OUT/'figures/control_results.png',dpi=160);plt.close(fig)
    shutil.copyfile(__file__,run.OUT/'source_snapshots'/Path(__file__).name)
    print('INTERPRETATION_READY',interpretations)


if __name__=='__main__':main()
