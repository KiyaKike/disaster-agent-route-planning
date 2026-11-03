"""Build a one-page advisor update and a short email from frozen results."""
import importlib.util
import json
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/阶段汇报_灾前灾后Agent路线规划_20261102_精简版"
QA = OUT / "qa"
NAME = "灾前灾后Agent路线规划_进展摘要"
spec = importlib.util.spec_from_file_location("bw_brief", Path(__file__).with_name("restyle_advisor_brief_bw.py"))
bw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bw)
base = bw.base


def paragraph(doc, text, *, size=11, bold=False, after=5):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = Pt(15.7 if size == 11 else size*1.4)
    p.paragraph_format.space_after = Pt(after)
    base.set_font(p.add_run(text), size, bold, "000000")
    return p


def heading(doc, text):
    p = doc.add_paragraph(text, "Heading 1")
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = Pt(18)
    for run in p.runs:
        base.set_font(run, 12.5, True, "000000")


def build():
    QA.mkdir(parents=True, exist_ok=True)
    sources = base.check_frozen()
    real = base.read_json(ROOT / "output/disaster_route_real_xbd36_v1/event_summary.json")
    selected = {(r["event"],r["policy"]):r for r in real if r["budget_s_proxy"] == 240}
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(.8)
    sec.top_margin = sec.bottom_margin = Inches(.65)
    for grid in list(sec._sectPr.findall(qn("w:docGrid"))):
        sec._sectPr.remove(grid)
    for name in ("Normal", "Title", "Heading 1"):
        style = doc.styles[name]
        style.font.name = base.FONT
        style.font.color.rgb = RGBColor(0,0,0)
        rf = style.element.get_or_add_rPr().get_or_add_rFonts()
        for key in ("ascii","hAnsi","eastAsia","cs"):
            rf.set(qn("w:"+key),base.FONT)
    doc.styles["Normal"].font.size = Pt(11)
    doc.styles["Normal"].paragraph_format.line_spacing = Pt(15.7)
    title = doc.add_paragraph("灾前灾后 Agent 搜查路线规划", "Title")
    title.paragraph_format.space_after = Pt(5)
    title.paragraph_format.line_spacing = Pt(26)
    for run in title.runs:
        base.set_font(run,20,True,"000000")
    paragraph(doc, "阶段进展摘要  2026年11月2日", size=10, after=9)
    paragraph(doc, "目前已跑通双层地图与预算路线闭环，但在线更新尚无稳定提升。下一步先验证沿途观测能否改善后续选点。", bold=True, after=7)

    heading(doc,"一 已完成")
    paragraph(doc, "双层地图分别保存“先验／线索”和“沿途观测证据”，移动、观察、规划、返程统一计费。已固定感知模型，在36对xBD灾前灾后影像上比较静态与在线路线；前期感知检查及失败记录已归档。")
    rows=[]
    for event,label in [("nepal-flooding","尼泊尔洪水"),("sunda-tsunami","巽他海啸"),("tuscaloosa-tornado","塔斯卡卢萨龙卷风")]:
        rows.append([label]+[f'{selected[(event,policy)]["recall"]*100:.2f}%' for policy in ("pre_static","post_static","online_frozen","online_neighbor")])
    base.table(doc,["事件","灾前静态","灾后静态","在线不传播","在线邻域反馈"],rows,[1.7,1.22,1.22,1.34,1.42])
    paragraph(doc, "主预算240代理单位；指标为严重损伤像素正确召回率，各事件12对。未访问预测和参考标签不进入决策。",size=9.5)

    heading(doc,"二 目前判断")
    paragraph(doc, "简单邻域和同建筑传播均未获得稳定收益。尼泊尔原灾后静态、同线索精确求解、事后完美知识的召回分别为32.11%、32.17%、40.91%：仅换求解器改善有限，仍需验证可用观测能否帮助选点。")
    paragraph(doc, "AIPPMS受控参考验证了已知有用观测的价值，尚不能证明真实灾害效果。上述36对属于开发数据，当前结果为影像回放，尚无独立验证或实际飞行结论。",size=10)

    heading(doc,"三 下一步与请教问题")
    paragraph(doc, "1. 明确无人机建筑巡查的观察范围、判读完成条件和返程要求。")
    paragraph(doc, "2. 在隔离场景中检验观测价值，继续固定感知器；先用CPU和缓存推进。")
    paragraph(doc, "3. 再做同预算路线对比，评价覆盖、首次发现、距离和耗时。")
    paragraph(doc, "想请老师指导：近期是否收敛为“无人机辅助建筑损伤巡查”？是否有合适的多视角或带定位、配准信息的数据来源？",after=5)
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = Pt(12)
    p.paragraph_format.space_after = Pt(0)
    base.set_font(p.add_run("数据与机制参考："),9,False,"000000")
    base.hyperlink(p,"xBD","https://arxiv.org/abs/1911.09296")
    base.set_font(p.add_run("  ·  "),9,False,"000000")
    base.hyperlink(p,"AIPPMS","https://arxiv.org/abs/2003.09746")
    bw.monochrome_styles(doc)
    doc.core_properties.title = "灾前灾后Agent搜查路线规划进展摘要"
    doc.core_properties.author = ""
    doc.save(OUT/f"{NAME}.docx")

    email = """主题：灾前灾后Agent路线规划阶段汇报

老师您好！

目前已跑通“双层地图＋预算路线”闭环，并在36对真实影像上比较静态与在线规划。在线更新尚未出现稳定提升，当前重点是验证沿途观测能否帮助后续选点。

下一步计划固定感知器，明确巡查观察范围，再做场景隔离验证。想请您指导近期是否收敛为无人机建筑损伤巡查，以及是否有合适的数据来源。

附件为一页进展摘要，感谢老师指导！
"""
    (OUT/"邮件正文.txt").write_text(email,encoding="utf-8")
    original = ROOT / "output/阶段汇报_灾前灾后Agent路线规划_20261102_黑白版"
    config = (original/"qa/fonts.conf").read_text(encoding="utf-8").replace(str(original/"qa"),str(QA))
    (QA/"fonts.conf").write_text(config,encoding="utf-8")
    text = "\n".join(p.text for p in doc.paragraphs)
    audit = {"source_receipts":sources,"report_prose_characters":len(text),"email_characters":len(email),"new_experiments":False,"email_sent":False}
    (QA/"来源核验.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"report_prose_characters":len(text),"email_characters":len(email),"output":str(OUT)},ensure_ascii=False))


if __name__ == "__main__":
    build()
