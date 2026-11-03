"""Create a teacher-facing brief without changing any frozen experiment artifact."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/感知检查_阶段汇报_20261028'
STAGES=[
 ('external_area18','外部18对发布参考核查','仅1个场景可发现，停止在该缓存上调搜索'),
 ('context18','同18对裁块与整图对照','正确覆盖915→4796像素，但不是预算内策略收益'),
 ('information18','局部放大与粗图上下文','免费粗图加局部未通过筛查，保留失败结果'),
 ('footprint18','同面积128与256原图','小样本出现正向结果，随后单独复验'),
 ('quality256','256模糊输入','模糊严重预测全零，停止该混合质量候选'),
 ('native36','新增36对原图复验','256未保持优势，仅1个场景可发现，不并入旧18救分'),
 ('perception_pair72','既有72对双检查点','增大上下文改善召回；Mexico3个严重目标仍漏检'),
 ('mexico_trace3','Mexico失败目标标签对应','另一份发布掩膜与现有参考共同已知域一致；非独立人工真值'),
 ('strongbaseline24','事件划分基线整图接入','SantaRosa24对F1为82.85%，属于已用开发事件'),
 ('strongbaseline_local24','SantaRosa局部面积预算','25%面积时随机21.48%、静态36.47%、Oracle51.52%召回'),
 ('multievent36','新三事件36对局部复验','事件间迁移不稳定，暂停当前跨灾害合同规划调参')]

def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def pct(x):return 'NA' if x is None else f'{100*x:.2f}%'


def collect():
    from reportlab.graphics.shapes import Drawing,Line,String,Rect,Circle,PolyLine
    from reportlab.graphics import renderSVG
    from reportlab.lib.colors import HexColor
    OUT.mkdir(exist_ok=False);(OUT/'证据快照').mkdir();(OUT/'图表').mkdir()
    sources={};rows=[]
    for key,title,conclusion in STAGES:
        src=ROOT/'output'/f'disaster_search_{key}_v1';receipt=read(src/'completion_receipt.json')
        target=OUT/'证据快照'/key;target.mkdir()
        copied=[]
        for name in ['README.md','completion_receipt.json','summary.json','audit_report.json','data_admission.json','crosscheck.json']:
            file=src/name
            if not file.exists():continue
            if name!='completion_receipt.json' and name in receipt.get('files',{}):
                assert digest(file)==receipt['files'][name],str(file)
            shutil.copyfile(file,target/name);sources[str(file)]=digest(file);copied.append(name)
        rows.append({'阶段':title,'日期':receipt.get('client_report_date','2026-10-28'),'样本与阶段结论':conclusion,
            '原始目录':str(src),'汇报包内快照':'证据快照/'+key,'完成收据SHA256':digest(src/'completion_receipt.json')})
    with (OUT/'实验索引.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    facts={}
    for key in ['perception_pair72','strongbaseline24','strongbaseline_local24','multievent36']:
        file=ROOT/'output'/f'disaster_search_{key}_v1/summary.json';facts[key]=read(file);sources[str(file)]=digest(file)
    local=ROOT/'output/disaster_search_strongbaseline_local24_v1/totals.json';facts['local_totals']=read(local);sources[str(local)]=digest(local)
    multi=ROOT/'output/disaster_search_multievent36_v1'
    for n in ['runtime.json','RGB_quality_report.json','reference_prevalence_diagnostic.json','inference_protocol.json']:
        facts[n]=read(multi/n);sources[str(multi/n)]=digest(multi/n)
    route=ROOT/'output/disaster_route_scope_v1/decision.json';sources[str(route)]=digest(route)
    shutil.copyfile(route,OUT/'路线主线与感知支线.json')
    dump(OUT/'汇报数据.json',facts);dump(OUT/'来源校验.json',sources)
    # Report-sized vector figures; all values read from frozen result JSON.
    colors=['#6B7280','#2563EB','#B45309'];policies=['random_exact_expected_counts','static_coarse_change','oracle_true_gain'];labels=['Exact random','Coarse-change static','Oracle bound']
    d=Drawing(468,190);left,bottom,w,h=36,35,418,125
    for value in [0,25,50,75,100]:
        y=bottom+h*value/100;d.add(Line(left,y,left+w,y,strokeColor=HexColor('#E5E7EB'),strokeWidth=.5));d.add(String(left-5,y-3,str(value),fontName='Helvetica',fontSize=8,textAnchor='end'))
        x=left+w*value/100;d.add(String(x,bottom-13,str(value),fontName='Helvetica',fontSize=8,textAnchor='middle'))
    for policy,label,color in zip(policies,labels,colors):
        values=sorted([r for r in facts['local_totals'] if r['policy']==policy],key=lambda r:r['budget'])
        points=[left,bottom]
        for r in values:points.extend([left+w*r['budget']/64,bottom+h*r['recall']])
        d.add(PolyLine(points,strokeColor=HexColor(color),strokeWidth=1.6))
        for x,y in zip(points[::2],points[1::2]):d.add(Circle(x,y,2.4,fillColor=HexColor(color),strokeColor=None))
    d.add(Line(left+w*.25,bottom,left+w*.25,bottom+h,strokeColor=HexColor('#666666'),strokeDashArray=[2,2],strokeWidth=.7))
    d.add(String(245,5,'Acquired native area (%)',fontName='Helvetica',fontSize=9,textAnchor='middle'))
    d.add(String(36,177,'Santa Rosa severe-reference recall (%)',fontName='Helvetica-Bold',fontSize=10))
    for i,(label,color) in enumerate(zip(labels,colors)):
        x=42+i*145;d.add(Line(x,166,x+13,166,strokeColor=HexColor(color),strokeWidth=2));d.add(String(x+17,163,label,fontName='Helvetica',fontSize=8))
    renderSVG.drawToFile(d,str(OUT/'图表/局部预算.svg'))
    d=Drawing(468,197);left,w=110,310
    for value in [0,20,40,60,80,100]:
        x=left+w*value/100;d.add(Line(x,34,x,172,strokeColor=HexColor('#E5E7EB'),strokeWidth=.5));d.add(String(x,20,str(value),fontName='Helvetica',fontSize=8,textAnchor='middle'))
    for j,(event,name) in enumerate(zip(['nepal-flooding','sunda-tsunami','tuscaloosa-tornado'],['Nepal flood','Sunda tsunami','Tuscaloosa tornado'])):
        ybase=154-j*46;d.add(String(left-8,ybase-4,name,fontName='Helvetica',fontSize=9,textAnchor='end'))
        for i,(policy,color) in enumerate(zip(policies,colors)):
            r=next(x for x in facts['multievent36']['primary_per_event'] if x['event']==event and x['policy']==policy);value=r['recall']*100;y=ybase+8-i*11
            d.add(Rect(left,y-4,w*value/100,7,strokeColor=None,fillColor=HexColor(color)));d.add(String(left+w*value/100+4,y-3,f'{value:.2f}',fontName='Helvetica',fontSize=8))
    for i,(label,color) in enumerate(zip(labels,colors)):
        x=22+i*149;d.add(Rect(x,184,9,7,strokeColor=None,fillColor=HexColor(color)));d.add(String(x+13,183,label,fontName='Helvetica',fontSize=8))
    d.add(String(265,4,'Severe-reference recall at 25% acquired area (%)',fontName='Helvetica',fontSize=9,textAnchor='middle'))
    renderSVG.drawToFile(d,str(OUT/'图表/跨事件复验.svg'))
    oral='''老师，这一阶段我把感知输入作为灾前灾后路线规划的前置条件做了检查，相关代码、数据清单、预测缓存和失败记录都保留下来了。

主要完成了模型接入、局部与整图输入对照、面积预算对照，以及三个新事件的复验。Santa Rosa上，公开基线整图的严重像素F1为82.85%；改成256局部后，全网格召回仍为85.92%。在只观察四分之一面积时，粗图变化排序的召回36.47%，高于随机期望21.48%。但这个结果没有稳定迁移到其他事件：海啸看完全部局部块仍零命中，龙卷风的静态排序低于随机，尼泊尔误报较多。

因此，我现在不把这些结果当成路线规划已经有效的证据。它们的作用是明确感知输入有误差、局部观察有成本，后续不能假设每次观察都正确。下一阶段回到原来的路线目标，把起终点、移动、观察和返程全部计入预算，比较灾前静态、灾后静态以及沿途获得信息后的重规划。感知部分先固定，作为观测不确定性输入。仿真与真实案例会分开报告；已有未标注区域不补成假真值。

目前没有租GPU、没有新训练，也没有把旧保留集用于评分。最新测试套件363项通过，关键结果已独立重放。前面的工作作为前置检查与失败分析保留，后续主线恢复为路线规划。
'''
    (OUT/'口头汇报要点.txt').write_text(oral)
    print(json.dumps({'evidence_stages':len(rows),'source_files_bound':len(sources),'folder':str(OUT)},ensure_ascii=False))


def docx():
    from docx import Document
    from docx.shared import Inches,Pt,RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    facts=read(OUT/'汇报数据.json');doc=Document();section=doc.sections[0]
    section.page_width=Inches(8.5);section.page_height=Inches(11)
    section.top_margin=section.bottom_margin=Inches(.82);section.left_margin=section.right_margin=Inches(1)
    section.header_distance=section.footer_distance=Inches(.35)
    for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
        style=doc.styles[name];style.font.name='Arial Unicode MS';rf=style._element.get_or_add_rPr().get_or_add_rFonts()
        for attr in list(rf.attrib):
            if 'theme' in attr.lower():del rf.attrib[attr]
        for attr in ['ascii','hAnsi','eastAsia','cs']:rf.set(qn('w:'+attr),'Arial Unicode MS')
        style.font.color.rgb=RGBColor.from_string('111111')
        for node in list(style._element.xpath('.//w:pBdr'))+list(style._element.xpath('.//w:numPr')):node.getparent().remove(node)
        lang=OxmlElement('w:lang');lang.set(qn('w:val'),'zh-CN');lang.set(qn('w:eastAsia'),'zh-CN');style._element.get_or_add_rPr().append(lang)
    normal=doc.styles['Normal'];normal.font.size=Pt(11);normal.paragraph_format.line_spacing=1.22;normal.paragraph_format.space_after=Pt(7)
    normal.paragraph_format.widow_control=True
    for name,size in [('Title',20),('Heading 1',15),('Heading 2',12)]:
        s=doc.styles[name];s.font.size=Pt(size);s.font.bold=True;s.paragraph_format.space_before=Pt(12);s.paragraph_format.space_after=Pt(6);s.paragraph_format.keep_with_next=True
    doc.styles['Title'].paragraph_format.space_before=Pt(0)
    doc.styles['Subtitle'].font.size=Pt(10);doc.styles['Subtitle'].font.italic=False;doc.styles['Subtitle'].font.color.rgb=RGBColor.from_string('666666')
    doc.styles['Caption'].font.size=Pt(9);doc.styles['Caption'].font.italic=False;doc.styles['Caption'].paragraph_format.space_after=Pt(8)
    def para(text,bold=False):
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY;r=p.add_run(text);r.bold=bold;return p
    def heading(text,new_page=False):
        p=doc.add_paragraph(text,'Heading 1');p.paragraph_format.page_break_before=new_page;return p
    def table(headers,rows,widths):
        t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
        for cell,width in zip(t.columns,widths):cell.width=Inches(width)
        for values,row in [(headers,t.rows[0])]+[(v,t.add_row()) for v in rows]:
            is_header=row is t.rows[0]  # identity is not stable; style by row index below
            for i,(cell,text,width) in enumerate(zip(row.cells,values,widths)):
                cell.width=Inches(width);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER;p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3);p.paragraph_format.line_spacing=1.12;p.alignment=WD_ALIGN_PARAGRAPH.LEFT if i==0 else WD_ALIGN_PARAGRAPH.CENTER
                r=p.add_run(str(text));r.font.size=Pt(10.5)
            no=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(no)
        for j,row in enumerate(t.rows):
            for cell in row.cells:
                props=cell._tc.get_or_add_tcPr();m=OxmlElement('w:tcMar')
                for side in ['top','left','bottom','right']:
                    e=OxmlElement('w:'+side);e.set(qn('w:w'),'85');e.set(qn('w:type'),'dxa');m.append(e)
                props.append(m)
                fill='1F3864' if j==0 else ('F4F6F8' if j%2 else 'FFFFFF');shd=OxmlElement('w:shd');shd.set(qn('w:fill'),fill);props.append(shd)
                for run in cell.paragraphs[0].runs:
                    run.bold=j==0
                    if j==0:run.font.color.rgb=RGBColor(255,255,255)
            if j==0:row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
        borders=OxmlElement('w:tblBorders')
        for side in ['top','left','bottom','right','insideH','insideV']:
            e=OxmlElement('w:'+side);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D1D5DB');borders.append(e)
        t._tbl.tblPr.append(borders)
        doc.add_paragraph().paragraph_format.space_after=Pt(0)
        return t
    def fig(name,caption,width=6.35):
        p=doc.add_paragraph();p.paragraph_format.space_after=Pt(3);p.add_run().add_picture(str(OUT/'图表'/name),width=Inches(width));p.paragraph_format.keep_with_next=True;doc.add_paragraph(caption,'Caption')
    footer=section.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=footer.add_run();f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');r._r.addnext(f)
    # Page1
    doc.add_paragraph('灾前灾后路线规划感知检查阶段汇报','Title')
    doc.add_paragraph('阶段工作整理  2026年10月28日','Subtitle')
    para('本阶段围绕路线规划的感知输入，完成了模型接入、输入尺度与上下文对照、标签对应核查及跨事件测量。结果表明，局部观测能否提供可靠信息随事件变化，不能默认建筑损伤识别可直接作为稳定反馈。后续主线恢复为含移动与返程约束的路线规划，感知检查作为前置条件和失败分析保留。')
    heading('已经完成的工作')
    para('建立了固定样本清单、数据范围读取、预训练权重校验、预测缓存、发布参考评分和独立重放链路。原始结果、图像与代码继续保存在项目中，本汇报包另存11个近期阶段的证据快照和实验索引。')
    heading('既有72对双检查点诊断')
    totals={r['model_id']:r for r in facts['perception_pair72']['totals'] if r['decoder']=='raw_pixel'}
    table(['条件','合并召回','合并F1','事件宏F1'],[[name,pct(totals[key]['recall']),pct(totals[key]['f1']),pct(totals[key]['equal_event_macro_f1'])] for key,name in [('legacy_swint128','Swin128 旧缓存'),('swint','Swin512'),('r34','ChangeOS R34 512')]], [2.55,1.25,1.25,1.45])
    para('同一Swin从128局部改为512上下文，严重像素召回由76.92%升至86.59%；两个检查点的结果接近，但不能据此宣称统计等效。72对包含三个开发事件，训练暴露与样本复用限制保留；合并分数不能替代逐事件结果。')
    para('Mexico的3个严重参考目标在两模型中均漏检。另一个发布语义掩膜与现有多边形参考在共同已知区域完全一致，未发现类别转换不一致的证据；两份表示来自同一标注来源，不能证明原始标签正确或损伤一定可观察。')
    # Page2
    heading('整图识别与局部观察预算',new_page=True)
    sb=facts['strongbaseline24'];local=facts['strongbaseline_local24'];lg=local['all_grid_native256']
    para('接入作者公开的Strong Baseline事件划分权重，在现有Santa Rosa24对上运行。24对均未列入作者训练或验证清单，但已经被本项目用于开发，且只覆盖一个事件，不作为全新独立确认集。')
    table(['观察条件','严重像素精度','严重像素召回','严重像素F1'],[['1024整图 四翻转',pct(sb['precision']),pct(sb['recall']),pct(sb['f1'])],['256局部 全部16块',pct(lg['precision_ratio_of_counts']),pct(lg['recall']),pct(lg['f1_ratio_of_counts'])]], [2.45,1.35,1.35,1.35])
    para('局部全网格保留了召回信号，但精度与F1下降。18个参考阳性场景均有正确像素，不等于全部建筑已检出。')
    para('面积合同固定为128×128原图配对面积成本1，256块成本4；1024场景共16块，主预算16只观察4块，即25%面积。粗图排序仅使用免费64×64粗图；每个购买块运行四翻转，计算次数单独记录。')
    primary=local['primary'];names=[('random_exact_expected_counts','随机精确期望'),('static_coarse_change','粗图变化排序'),('oracle_true_gain','Oracle诊断上界')]
    para('主预算16时，严重参考召回为：随机精确期望21.48%，粗图变化排序36.47%，Oracle诊断上界51.52%。随机不选择有利种子，Oracle预先知道全部真实动作收益。')
    fig('局部预算.png','图1  同一感知器下的面积预算对照  虚线为主预算25%面积')
    para('静态到上界相差15.06个百分点，只表示潜在决策空间，尚未运行反馈算法。以上按获取面积选块，未纳入起终点、连续移动和返程，不能写成路线规划收益。')
    # Page3
    heading('新三个事件的复验与失败结果',new_page=True)
    multi=facts['multievent36']
    para('按身份哈希从完整作者清单和TACO来源中各选12对，新增尼泊尔洪水、巽他海啸和塔斯卡卢萨龙卷风共36对。未根据损伤或预测挑样本，旧保留队列未评分，未将Santa Rosa并入救分。所有新样本永久作为开发测量复验。')
    table(['事件','阳性到可发现','全观察精度','全观察召回','全观察F1'],[[name,f"{e['reference_positive_scenes']} → {e['local_detectable_scenes']}",pct(e['all_grid']['precision_ratio_of_counts']),pct(e['all_grid']['recall']),pct(e['all_grid']['f1_ratio_of_counts'])] for e,name in zip(multi['events'],['尼泊尔洪水','巽他海啸','塔斯卡卢萨龙卷风'])],[1.65,1.25,1.1,1.2,1.3])
    doc.add_paragraph('每事件12个场景。可发现表示全部局部块中至少有一个真阳性严重像素。此表为观察100%面积；下图为主预算25%面积。','Caption')
    fig('跨事件复验.png','图2  新三个事件在25%面积预算下的召回  Oracle仅用于诊断')
    para('跨事件结果不稳定。海啸虽有2个严重参考场景，全部局部块仍零命中；龙卷风的粗图排序召回2.32%，低于随机期望11.75%；尼泊尔有发现信号，但全网格精度仅23.63%。不能把单事件正向结果推广为跨灾害有效。')
    para('新72张原图通过完整性和精确影像对重复检查；2张黑色区域超过25%的图仍保留，严重参考没有落在零RGB像素中。云状遮挡与未知标注另有记录，未生成新的人工真值。')
    # Page4
    heading('阶段价值与结论边界',new_page=True)
    para('后续路线实验已有固定感知输入、可重放的观测缓存、发布参考评价器，以及误检、漏检和图像质量变化案例，可用于检查决策对不可靠观测的敏感性。')
    para('本阶段尚未建立新路线算法、独立方法收益或真实航线验证。发布像素参考不等于独立人工逐栋真值；原始UID、灾前定位真值及全训练池地理/像素隔离未补齐。Oracle不能作为可部署策略。')
    heading('后续恢复路线规划主线')
    para('沿用原有PS巡查设定：灾前信息形成初始路线，灾后变化更新优先级，沿途获得新信息后在剩余预算内重规划。以明确位置、移动、观察和返程成本为基础，先建立可验证的路线闭环。')
    table(['比较条件','公平约束与评价'],[['灾前静态路线','同一起终点与总时间预算'],['灾后一次性规划','移动 观察 返程全部计入成本'],['沿途反馈后重规划','比较同预算覆盖 首次到达 总距离与耗时']],[2.2,4.3])
    para('感知模块先固定，误检和漏检作为观测不确定性。仿真与真实案例分开报告；真实PS未标注目标维持unknown，不用变化分数或合成目标冒充真值。已有理想化线段尚未经过地形、障碍及飞行验证。')
    heading('复核与资料保留')
    para('最新测试套件363项通过；新三事件2308次MPS前向及缓存循环约125秒，关键输入与评分已独立重放。未租GPU、未新训练、未新增人工标注。测试数不按阶段相加，复用样本也不累计为独立样本量。')
    para('随附11个阶段的实验索引、完成收据、关键结果快照和口头汇报要点。原始数据、代码、完整缓存与失败图保留原目录，汇报包不重复复制大型影像或权重。后续主线为路线规划，感知检查独立归档。')
    doc.core_properties.title='灾前灾后路线规划感知检查阶段汇报';doc.core_properties.author='';doc.core_properties.subject='前置感知检查  关键结果  失败分析  后续路线规划'
    doc.save(OUT/'感知检查阶段汇报.docx')
    print(json.dumps({'docx':str(OUT/'感知检查阶段汇报.docx'),'tables':len(doc.tables),'figures':2,'planned_pages':4},ensure_ascii=False))


def seal():
    sources=read(OUT/'来源校验.json')
    for file,h in sources.items():assert digest(file)==h,file
    from docx import Document
    doc=Document(OUT/'感知检查阶段汇报.docx');text='\n'.join(p.text for p in doc.paragraphs)
    assert all(t in text for t in ['路线规划','363','2308','25%','unknown'])
    assert len(doc.tables)==4 and len(doc.inline_shapes)==2
    qa=read(OUT/'qa/visual_review.json');assert qa['all_pages_viewed'] and qa['pages']==4
    files={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and 'qa' not in p.relative_to(OUT).parts and p.suffix not in ['.svg','.zip'] and p.name!='归档校验.json'}
    dump(OUT/'归档校验.json',{'status':'COMPLETE','date':'2026-10-28','original_source_files_verified':len(sources),'original_experiments_modified':False,'evidence_stages':11,'docx_pages':4,'new_model_calls':0,'new_training':False,'gpu_rental':False,'files':files})
    archive=OUT/'感知检查汇报包.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and 'qa' not in p.relative_to(OUT).parts and p.suffix not in ['.svg','.zip']:
                z.write(p,arcname='感知检查阶段汇报/'+str(p.relative_to(OUT)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'status':'COMPLETE','package':str(archive),'bytes':archive.stat().st_size,'source_files_unchanged':len(sources)},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['collect','docx','seal'],required=True);a=p.parse_args();{'collect':collect,'docx':docx,'seal':seal}[a.stage]()
