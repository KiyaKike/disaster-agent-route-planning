"""One-page, monochrome advisor update incorporating audited feedback results."""
import importlib.util
import json
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'output/阶段汇报_灾前灾后Agent路线规划_20261102_务实版'
OUT=ROOT/'output/阶段汇报_灾前灾后Agent路线规划_20261103'
QA=OUT/'qa'
NAME='灾前灾后Agent路线规划_阶段汇报_20261103'
spec=importlib.util.spec_from_file_location('concise',Path(__file__).with_name('build_advisor_brief_concise.py'))
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
base=mod.base


def build():
    QA.mkdir(parents=True,exist_ok=True)
    audit_dir=ROOT/'output/disaster_route_feedback_increment_audit_v2'
    receipt=base.read_json(audit_dir/'completion_receipt.json')
    assert all(base.sha(audit_dir/n)==h for n,h in receipt['files'].items())
    s=base.read_json(audit_dir/'summary.json')
    old_receipt=base.read_json(OLD/'qa/交付核验.json')
    assert all(base.sha(OLD/n)==h for n,h in old_receipt['files'].items())
    doc=Document(OLD/'灾前灾后Agent路线规划_阶段说明.docx')
    for el in list(doc._element.body):
        if el.tag!=qn('w:sectPr'):doc._element.body.remove(el)
    p=doc.add_paragraph('灾前灾后 Agent 搜查路线规划','Title')
    p.paragraph_format.space_after=Pt(5);p.paragraph_format.line_spacing=Pt(26)
    for r in p.runs:base.set_font(r,20,True,'000000')
    para=lambda t,**kw:mod.paragraph(doc,t,**kw)
    heading=lambda t:mod.heading(doc,t)
    para('阶段进展摘要  2026年11月3日',size=10,after=9)
    para('已完成路线基线及真实缓存反馈诊断，当前反馈尚未显示稳定增益。现有数据仍不足以验证具体无人机巡查。',bold=True,after=7)
    heading('一 已有基础')
    para('已在36对xBD灾前灾后影像上搭建双层地图，分别保存先验线索与沿途证据；固定感知器，完成静态／在线路线回放。移动、观察、规划及返程统一计费，费用为代理设定。前期感知检查和失败记录已归档。')
    heading('二 新增反馈增量诊断')
    para('固定同一批付费观察前缀，用两个灾害事件拟合、第三个事件评分，检查已付预测能否帮助判断未访问格是否存在可正确检出的损伤。主比较结果如下。')
    values=[]
    for label,key,fmt in [('AUROC ↑','auroc','.4f'),('AP ↑','average_precision','.4f'),('Log loss ↓','log_loss','.4f'),('Brier ↓','brier','.5f')]:
        values.append([label,format(s['metrics']['prior'][key],fmt),format(s['metrics']['prior_feedback'][key],fmt)])
    values.append(['下一格命中数','18/42','18/42'])
    base.table(doc,['指标','仅线索','线索＋已付反馈'],values,[2.5,2.2,2.2])
    para('↑越高越好，↓越低越好。命中数仅统计仍有可检出格的42个前缀状态，不代表42个独立场景。',size=9.5)
    para('AP和Log loss未改善，Brier略有改善，下一格命中未提升；目前未建立稳定的增量收益，暂不据此设计新更新器。指标与信息隔离问题已复核，原始结果保留。')
    heading('三 数据与结论边界')
    para('36对影像均为旧开发数据，按事件交叉验证也不等于独立验证。目前仍缺少可对应真实巡查的观测范围、多视角、可信定位配准、飞行轨迹及航时／能耗数据，尚不能据此验证现场航线或实际巡查收益。')
    heading('四 下一步与请教问题')
    para('近期保留预算搜索基线、受控机制分析及本轮负结果，优先落实新的区域／事件或多视角数据。想请老师指导：先做现有条件下的规划方法研究是否合适？若推进无人机巡查验证，是否有可落实的数据或合作渠道？')
    p=doc.add_paragraph();p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=Pt(12)
    base.set_font(p.add_run('数据来源：'),9,False,'000000');base.hyperlink(p,'xBD','https://arxiv.org/abs/1911.09296')
    mod.bw.monochrome_styles(doc)
    doc.core_properties.title='灾前灾后Agent搜查路线规划阶段汇报 2026年11月3日'
    doc.save(OUT/f'{NAME}.docx')
    email='''主题：灾前灾后Agent路线规划进展与数据限制

老师您好！

在已完成双层地图和路线基线的基础上，我新增了真实缓存的反馈增量检查。按灾害事件隔离拟合与评分后，当前反馈特征尚未显示稳定收益，暂不足以支持新的在线更新方法。

现有影像主要支持网格层面的规划比较；真实无人机的观测、定位、轨迹和航时数据仍未落实，暂不能验证具体巡查方案。

想请您指导：是否先做现有条件下的预算搜索与受控方法研究，或先落实数据合作，再推进无人机巡查验证？

附件为11月3日更新的一页汇报，感谢老师指导！
'''
    (OUT/'邮件正文.txt').write_text(email,encoding='utf-8')
    config=(OLD/'qa/fonts.conf').read_text(encoding='utf-8').replace(str(OLD/'qa'),str(QA))
    (QA/'fonts.conf').write_text(config,encoding='utf-8')
    (QA/'来源核验.json').write_text(json.dumps({'date':'2026-11-03','audited_results_receipt_sha256':base.sha(audit_dir/'completion_receipt.json'),'previous_report_preserved':True,'email_sent':False},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'docx':str(OUT/f'{NAME}.docx'),'email_characters':len(email)},ensure_ascii=False))


if __name__=='__main__':build()
