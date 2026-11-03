"""Create monochrome copies of the advisor materials without changing experiments."""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "output/阶段汇报_灾前灾后Agent路线规划_20261102"
OUT = ROOT / "output/阶段汇报_灾前灾后Agent路线规划_20261102_黑白版"
QA = OUT / "qa"
SUP = OUT / "补充材料"
MAIN = "阶段汇报_灾前灾后Agent搜查路线规划_黑白版"
PERCEPTION = "前期感知检查阶段汇报_20261028_黑白版"
spec = importlib.util.spec_from_file_location("brief_source", Path(__file__).with_name("build_advisor_brief.py"))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def monochrome_styles(doc):
    roots = [doc.element, doc.styles.element]
    for sec in doc.sections:
        roots.extend([sec.header._element, sec.footer._element])
    for root in roots:
        for item in list(root.iter(qn("w:shd"))):
            item.getparent().remove(item)
        for item in list(root.iter(qn("w:highlight"))):
            item.getparent().remove(item)
        for item in root.iter(qn("w:color")):
            item.attrib.clear()
            item.set(qn("w:val"), "000000")
        for item in list(root.iter(qn("w:pBdr"))):
            item.getparent().remove(item)
    for table in doc.tables:
        for col in range(1, len(table.columns)):
            values = [row.cells[col].text.strip() for row in table.rows[1:]]
            numeric = [v for v in values if v and all(ch in "0123456789.%+−- /→NA" for ch in v)]
            if len(numeric) >= max(1, len(values)/2):
                for p in table.rows[0].cells[col].paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        # White paper, three horizontal rules, no filled header or row banding.
        for border in list(table._tbl.iter(qn("w:tblBorders"))):
            border.getparent().remove(border)
        for row_index, row in enumerate(table.rows):
            for cell in row.cells:
                tcpr = cell._tc.get_or_add_tcPr()
                for border in list(tcpr.findall(qn("w:tcBorders"))):
                    tcpr.remove(border)
                borders = OxmlElement("w:tcBorders")
                for edge_name in ("top", "left", "bottom", "right"):
                    edge = OxmlElement("w:" + edge_name)
                    visible = ((edge_name == "top" and row_index == 0)
                               or (edge_name == "bottom" and row_index in (0, len(table.rows)-1)))
                    edge.set(qn("w:val"), "single" if visible else "nil")
                    edge.set(qn("w:sz"), "6" if row_index in (0, len(table.rows)-1) else "4")
                    edge.set(qn("w:color"), "000000")
                    borders.append(edge)
                tcpr.append(borders)
                for p in cell.paragraphs:
                    # Center numeric fields, keep explanations left aligned.
                    text = p.text.strip()
                    if text and all(ch in "0123456789.%+−- /→NA" for ch in text):
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs:
                        if row_index == 0:
                            run.bold = True
        following = table._tbl.getnext()
        if following is not None and following.tag == qn("w:p") and following.xpath(".//w:t"):
            from docx.text.paragraph import Paragraph
            paragraph = Paragraph(following, doc._body)
            before = paragraph.paragraph_format.space_before
            if before is None or before < Pt(4):
                paragraph.paragraph_format.space_before = Pt(4)
    return doc


def replace_image(doc, paragraph, image_path):
    embed = paragraph._p.xpath(".//a:blip")[0].get(qn("r:embed"))
    doc.part.related_parts[embed]._blob = image_path.read_bytes()


def drop_image(doc, paragraph):
    for blip in paragraph._p.xpath(".//a:blip"):
        doc.part.drop_rel(blip.get(qn("r:embed")))
    paragraph._p.getparent().remove(paragraph._p)


def main_report():
    doc = Document(OLD / "阶段汇报_灾前灾后Agent搜查路线规划.docx")
    image_paras = [p for p in doc.paragraphs if p._p.xpath(".//w:drawing")]
    # Rebuild the existing code-defined diagram using black strokes and type.
    import inspect
    code = inspect.getsource(base.draw_loop)
    for color in ("#18354A", "#111111", "#364D5E", "#315C78"):
        code = code.replace(color, "#000000")
    base.FIG = QA / "figures"
    exec(compile(code, "monochrome_loop_diagram", "exec"), base.__dict__)
    base.draw_loop()
    replace_image(doc, image_paras[0], base.FIG / "研究闭环.png")
    target = image_paras[1]
    rows = base.read_json(ROOT / "output/disaster_route_headroom_v1/event_summary.json")
    chosen = {r["event"]: r for r in rows if r["budget_s_proxy"] == 240}
    names = [("nepal-flooding", "尼泊尔洪水"), ("sunda-tsunami", "巽他海啸"), ("tuscaloosa-tornado", "塔斯卡卢萨龙卷风")]
    for fields, caption in [
        (("legacy_post_coverage", "public_exact_coverage", "reference_coverage_oracle_coverage"),
         "表2  同预算严重参考覆盖率，衡量访问范围，不要求模型正确识别。"),
        (("legacy_post_recall", "public_exact_recall", "detector_tp_oracle_recall"),
         "表3  同预算严重像素正确召回率。Oracle提前知道结果，仅用于事后诊断。"),
    ]:
        values = [[label] + [f"{chosen[event][key]*100:.2f}%" for key in fields] for event, label in names]
        t = base.table(doc, ["事件", "原灾后静态", "同线索精确求解", "事后Oracle"], values, [1.7, 1.65, 1.9, 1.65])
        target._p.addprevious(t._tbl)
        p = base.para(doc, caption, size=9, color="000000", after=7)
        p.paragraph_format.space_before = Pt(4)
        target._p.addprevious(p._p)
    drop_image(doc, target)
    for p in list(doc.paragraphs):
        if p.text.startswith("图2  左："):
            p._p.getparent().remove(p._p)
        elif p.text.startswith("表2  严重损伤"):
            p.runs[0].text = p.runs[0].text.replace("表2", "表4", 1)
        elif p.text.startswith("表3  岩石任务"):
            p.runs[0].text = p.runs[0].text.replace("表3", "表5", 1)
    monochrome_styles(doc).save(OUT / f"{MAIN}.docx")


def perception_curve(facts):
    scale = 4
    im = Image.new("RGB", (1872, 760), "white")
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype("/Users/bethany/Library/Fonts/msyh.ttc", 30)
    heading_font = ImageFont.truetype("/Users/bethany/Library/Fonts/msyhbd.ttc", 35)
    left, bottom, width, height = 145, 620, 1650, 455

    def dashed(p0, p1, pattern):
        x0, y0 = p0
        x1, y1 = p1
        length = math.hypot(x1-x0, y1-y0)
        if not length:
            return
        if pattern is None:
            d.line([p0, p1], fill="black", width=5)
            return
        distance = 0
        index = 0
        while distance < length:
            end = min(length, distance + pattern[index % len(pattern)])
            if index % 2 == 0:
                a = distance/length
                b = end/length
                d.line([(x0+(x1-x0)*a, y0+(y1-y0)*a), (x0+(x1-x0)*b, y0+(y1-y0)*b)], fill="black", width=5)
            distance = end
            index += 1

    d.text((left, 15), "Santa Rosa severe-reference recall (%)", font=heading_font, fill="black")
    for value in (0, 25, 50, 75, 100):
        y = bottom-height*value/100
        d.text((left-20, y), str(value), font=font, fill="black", anchor="rm")
        d.line([(left-8, y), (left, y)], fill="black", width=2)
        x = left+width*value/100
        d.text((x, bottom+30), str(value), font=font, fill="black", anchor="mt")
        d.line([(x, bottom), (x, bottom+8)], fill="black", width=2)
    d.line([(left, bottom-height), (left, bottom), (left+width, bottom)], fill="black", width=3)
    policies = [("random_exact_expected_counts", "Exact random", (8, 9)),
                ("static_coarse_change", "Coarse-change static", (28, 13)),
                ("oracle_true_gain", "Oracle bound", None)]
    for i, (policy, label, dash) in enumerate(policies):
        legend_x = 180 + i*555
        dashed((legend_x, 100), (legend_x+70, 100), dash)
        d.text((legend_x+85, 100), label, font=font, fill="black", anchor="lm")
        values = sorted([v for v in facts["local_totals"] if v["policy"] == policy], key=lambda r:r["budget"])
        points = [(left, bottom)] + [(left+width*v["budget"]/64, bottom-height*v["recall"]) for v in values]
        for a, b in zip(points, points[1:]):
            dashed(a, b, dash)
        for x, y in points:
            d.ellipse((x-7, y-7, x+7, y+7), fill="white", outline="black", width=3)
    dashed((left+width*.25, bottom), (left+width*.25, bottom-height), (5, 9))
    d.text((left+width/2, 705), "Acquired native area (%)", font=font, fill="black", anchor="mm")
    result = QA / "figures/感知局部预算_黑白.png"
    im.save(result)
    return result


def perception_report():
    source = ROOT / "output/感知检查_阶段汇报_20261028"
    facts = base.read_json(source / "汇报数据.json")
    doc = Document(source / "感知检查阶段汇报.docx")
    image_paras = [p for p in doc.paragraphs if p._p.xpath(".//w:drawing")]
    replace_image(doc, image_paras[0], perception_curve(facts))
    rows = []
    for event, label in [("nepal-flooding", "尼泊尔洪水"), ("sunda-tsunami", "巽他海啸"), ("tuscaloosa-tornado", "塔斯卡卢萨龙卷风")]:
        rows.append([label] + [f"{next(r['recall'] for r in facts['multievent36']['primary_per_event'] if r['event']==event and r['policy']==policy)*100:.2f}%"
                              for policy in ("random_exact_expected_counts", "static_coarse_change", "oracle_true_gain")])
    target = image_paras[1]
    t = base.table(doc, ["事件", "随机精确期望", "粗图变化排序", "Oracle诊断上界"], rows, [1.65, 1.6, 1.6, 1.65])
    target._p.addprevious(t._tbl)
    drop_image(doc, target)
    for p in doc.paragraphs:
        for run in p.runs:
            if "下图为主预算25%面积" in run.text:
                run.text = run.text.replace("下图为主预算25%面积", "下表为主预算25%面积")
            if run.text.startswith("图2  新三个事件"):
                run.text = run.text.replace("图2  新三个事件", "新三个事件", 1)
    monochrome_styles(doc).save(SUP / f"{PERCEPTION}.docx")


def build():
    for folder in (OUT, QA, SUP, QA / "figures"):
        folder.mkdir(parents=True, exist_ok=True)
    frozen = base.check_frozen()
    old_receipt = base.read_json(OLD / "qa/交付核验.json")
    for name, expected in old_receipt["files"].items():
        assert base.sha(OLD/name) == expected, name
    main_report()
    perception_report()
    for name in ("邮件正文.txt",):
        shutil.copy2(OLD/name, OUT/name)
    for name in ("论文与实验对应.md", "真实影像路线示例.png", "真实影像示例说明.txt"):
        shutil.copy2(OLD/"补充材料"/name, SUP/name)
    text = (OLD/"发送说明.md").read_text(encoding="utf-8")
    text = text.replace("阶段汇报_灾前灾后Agent搜查路线规划.pdf", MAIN+".pdf")
    text = text.replace("前期感知检查阶段汇报_20261028.pdf", PERCEPTION+".pdf")
    text += "\n本版统一为白底黑字、无底色表头与细线表格；彩色柱状图改用数据表，流程图与感知预算曲线以黑色线条区分。主报告与感知补充报告均提供Word及PDF。\n"
    (OUT/"发送说明.md").write_text(text, encoding="utf-8")
    config = (OLD/"qa/fonts.conf").read_text(encoding="utf-8").replace(str(OLD/"qa"), str(QA))
    (QA/"fonts.conf").write_text(config, encoding="utf-8")
    (QA/"来源核验.json").write_text(json.dumps({"frozen_inputs": frozen, "previous_package_unchanged": True,
        "previous_package_files":old_receipt["files"], "style":"black_text_white_background_no_shading"},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"main_docx":str(OUT/f"{MAIN}.docx"), "supplement_docx":str(SUP/f"{PERCEPTION}.docx"), "source_files_verified":sum(v["files_checked"] for v in frozen)},ensure_ascii=False))


if __name__ == "__main__":
    build()
