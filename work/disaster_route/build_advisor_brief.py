"""Build a teacher-facing brief from frozen research outputs; no new experiments."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/阶段汇报_灾前灾后Agent路线规划_20261102"
QA = OUT / "qa"
SUP = OUT / "补充材料"
FIG = QA / "figures"
FONT = "Microsoft YaHei"
NAVY = "18354A"
GRAY = "536270"
real_dir = ROOT / "output/disaster_route_real_xbd36_v1"
head_dir = ROOT / "output/disaster_route_headroom_v1"
pilot_dir = ROOT / "output/disaster_route_structure_pilot_v1"
control_dir = ROOT / "output/disaster_route_aippms_control_v1"
perception_dir = ROOT / "output/感知检查_阶段汇报_20261028"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen():
    results = []
    for folder, receipt_name in [
        (real_dir, "completion_receipt.json"),
        (head_dir, "completion_receipt.json"),
        (pilot_dir, "completion_receipt.json"),
        (control_dir, "completion_receipt.json"),
        (perception_dir, "归档校验.json"),
    ]:
        receipt = read_json(folder / receipt_name)
        files = receipt.get("files", receipt.get("files_sha256"))
        mismatches = [name for name, expected in files.items()
                      if not (folder / name).is_file() or sha(folder / name) != expected]
        if mismatches:
            raise RuntimeError(f"Frozen artifact mismatch: {folder}: {mismatches}")
        results.append({"source": str(folder.relative_to(ROOT)),
                        "receipt_sha256": sha(folder / receipt_name),
                        "files_checked": len(files), "mismatches": []})
    return results


def set_font(run, size=None, bold=None, color=None):
    run.font.name = FONT
    rf = run._element.get_or_add_rPr().get_or_add_rFonts()
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        rf.set(qn("w:" + key), FONT)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def para(doc, text, *, lead=None, size=11, after=6, color=None, keep=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = Pt(size * 1.45)
    p.paragraph_format.keep_with_next = keep
    if lead and text.startswith(lead):
        set_font(p.add_run(lead), size, True, color)
        set_font(p.add_run(text[len(lead):]), size, False, color)
    else:
        set_font(p.add_run(text), size, False, color)
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph(text, style=f"Heading {level}")
    p.paragraph_format.line_spacing = Pt(20 if level == 1 else 17)
    for r in p.runs:
        set_font(r, 14 if level == 1 else 12, True, "000000")
    return p


def table(doc, headers, rows, widths):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, w in enumerate(widths):
        t.columns[i].width = Inches(w)
    for idx, values in enumerate([headers, *rows]):
        cells = t.rows[0].cells if idx == 0 else t.add_row().cells
        trpr = t.rows[idx]._tr.get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
        if idx == 0:
            trpr.append(OxmlElement("w:tblHeader"))
        for col, (cell, value) in enumerate(zip(cells, values)):
            cell.width = Inches(widths[col])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr = cell._tc.get_or_add_tcPr()
            margins = OxmlElement("w:tcMar")
            for name, val in (("top", 80), ("bottom", 80), ("left", 100), ("right", 100)):
                edge = OxmlElement("w:" + name)
                edge.set(qn("w:w"), str(val))
                edge.set(qn("w:type"), "dxa")
                margins.append(edge)
            tcpr.append(margins)
            borders = OxmlElement("w:tcBorders")
            for name in ("top", "left", "bottom", "right"):
                edge = OxmlElement("w:" + name)
                edge.set(qn("w:val"), "single")
                edge.set(qn("w:sz"), "4")
                edge.set(qn("w:color"), "D9D9D9")
                borders.append(edge)
            tcpr.append(borders)
            if idx == 0:
                fill = OxmlElement("w:shd")
                fill.set(qn("w:fill"), NAVY)
                tcpr.append(fill)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = Pt(14)
            set_font(p.add_run(str(value)), 10, idx == 0, "FFFFFF" if idx == 0 else "000000")
    return t


def hyperlink(p, label, url):
    rel = p.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), rel)
    r = OxmlElement("w:r")
    prop = OxmlElement("w:rPr")
    rf = OxmlElement("w:rFonts")
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        rf.set(qn("w:" + key), FONT)
    prop.append(rf)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "185B85")
    prop.append(color)
    size = OxmlElement("w:sz")
    size.set(qn("w:val"), "19")
    prop.append(size)
    r.append(prop)
    text = OxmlElement("w:t")
    text.text = label
    r.append(text)
    link.append(r)
    p._p.append(link)


def draw_loop():
    im = Image.new("RGB", (1800, 480), "white")
    d = ImageDraw.Draw(im)
    font_path = "/Users/bethany/Library/Fonts/msyh.ttc"
    regular = ImageFont.truetype(font_path, 33)
    small = ImageFont.truetype(font_path, 28)
    bold = ImageFont.truetype("/Users/bethany/Library/Fonts/msyhbd.ttc", 34)

    def box(rect, title, line):
        d.rounded_rectangle(rect, radius=14, outline="#18354A", width=3, fill="white")
        x = (rect[0] + rect[2]) / 2
        y = (rect[1] + rect[3]) / 2
        d.text((x, y-23), title, font=bold, fill="#111111", anchor="mm")
        d.text((x, y+27), line, font=regular, fill="#364D5E", anchor="mm")

    def arrow(points):
        d.line(points, fill="#315C78", width=5, joint="curve")
        x, y = points[-1]
        px, py = points[-2]
        if x > px:
            tri = [(x, y), (x-17, y-11), (x-17, y+11)]
        elif x < px:
            tri = [(x, y), (x+17, y-11), (x+17, y+11)]
        elif y > py:
            tri = [(x, y), (x-11, y-17), (x+11, y-17)]
        else:
            tri = [(x, y), (x-11, y+17), (x+11, y+17)]
        d.polygon(tri, fill="#315C78")

    box((10, 20, 485, 170), "先验／线索层", "灾前结构＋灾后粗线索")
    box((10, 255, 485, 405), "沿途观测证据层", "已付观测＋来源与覆盖")
    box((610, 125, 980, 295), "更新信念与规划", "当前位置＋剩余预算")
    box((1100, 125, 1430, 295), "移动＋观察", "费用逐次扣除")
    box((1530, 125, 1790, 295), "返回起点", "返程计费")
    arrow([(485, 95), (550, 95), (550, 165), (610, 165)])
    arrow([(485, 330), (550, 330), (550, 250), (610, 250)])
    arrow([(980, 210), (1100, 210)])
    arrow([(1430, 210), (1530, 210)])
    arrow([(1265, 295), (1265, 435), (250, 435), (250, 405)])
    d.text((815, 405), "新增证据写回，再决定下一步", font=small, fill="#315C78", anchor="mm")
    im.save(FIG / "研究闭环.png")


def build():
    for folder in (OUT, QA, SUP, FIG):
        folder.mkdir(parents=True, exist_ok=True)
    preservation = check_frozen()
    real = read_json(real_dir / "event_summary.json")
    head = read_json(head_dir / "event_summary.json")
    control = read_json(control_dir / "summary.json")
    events = [("nepal-flooding", "尼泊尔洪水"), ("sunda-tsunami", "巽他海啸"), ("tuscaloosa-tornado", "塔斯卡卢萨龙卷风")]
    selected_real = {(r["event"], r["policy"]): r for r in real if r["budget_s_proxy"] == 240}
    selected_head = {r["event"]: r for r in head if r["budget_s_proxy"] == 240}
    selected_control = {(r["sensor_model"], r["budget"], r["policy"]): r for r in control}
    draw_loop()
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(.8)
    sec.top_margin = Inches(.68)
    sec.bottom_margin = Inches(.65)
    sec.header_distance = Inches(.28)
    sec.footer_distance = Inches(.28)
    for grid in list(sec._sectPr.findall(qn("w:docGrid"))):
        sec._sectPr.remove(grid)
    for border in list(doc.styles.element.iter(qn("w:pBdr"))):
        border.getparent().remove(border)
    for name in ("Normal", "Title", "Subtitle", "Heading 1", "Heading 2"):
        style = doc.styles[name]
        style.font.name = FONT
        style.font.color.rgb = RGBColor(0, 0, 0)
        rf = style.element.get_or_add_rPr().get_or_add_rFonts()
        for key in ("ascii", "hAnsi", "eastAsia", "cs"):
            rf.set(qn("w:" + key), FONT)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(11)
    normal.paragraph_format.line_spacing = 1.2
    normal.paragraph_format.space_after = Pt(6)
    for name in ("Heading 1", "Heading 2"):
        doc.styles[name].paragraph_format.space_before = Pt(10)
        doc.styles[name].paragraph_format.space_after = Pt(6)
    hp = sec.header.paragraphs[0]
    set_font(hp.add_run("研究阶段进展  /  灾前灾后 Agent 搜查路线规划"), 8.5, color=GRAY)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_font(fp.add_run("2026.11.02  ·  "), 8.5, color=GRAY)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    fp._p.append(fld)

    # Page 1: question, system, concrete progress.
    p = doc.add_paragraph(style="Title")
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = Pt(29)
    set_font(p.add_run("灾前灾后 Agent 搜查路线规划\n阶段进展与下一步实验"), 21, True, "000000")
    para(doc, "汇报日期：2026年11月2日  |  当前阶段：闭环原型与瓶颈验证", size=9.5, color=GRAY, after=10)
    para(doc, "当前判断：已完成双层地图、付费观测、在线决策与返程预算的闭环，并在真实灾损影像上回放；现有更新规则尚未带来稳定提升。下一阶段应先验证观测能否改善后续选点，再设计新的地图更新与规划方法。", lead="当前判断：")
    heading(doc, "1  研究问题与任务边界")
    para(doc, "研究问题是：灾前先验可能失配、灾后观测存在误检漏检时，Agent 怎样利用有限预算决定去哪里、何时观察、何时改变路线，并完成返程？")
    para(doc, "当前以灾后建筑损伤巡查为数据载体，拟按无人机辅助巡查建模；具体平台与任务仍待确定。双层地图指“搜索先验／线索层＋沿途观测证据层”，灾前、灾后是时间阶段。当前未开展人员搜救或地面道路通行验证。")
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(FIG / "研究闭环.png"), width=Inches(6.9))
    para(doc, "图1  两层地图分别记录线索与已付证据；标签仅供独立评分，不进入 Agent 决策。", size=9, color=GRAY, after=7)
    heading(doc, "2  已完成的工作")
    table(doc, ["模块", "已完成", "当前边界"], [
        ["路线闭环", "移动、观察、规划、返程统一计费；保留静态与在线对照。", "费用仍含代理值或平台假设。"],
        ["真实数据回放", "36对 xBD 前后影像；固定576块模型预测；432条预算路线。", "已使用的开发数据，非独立验证。"],
        ["结构与地理接入", "灾前建筑轮廓、近似坐标、转向和局部证据接通。", "3个可路由案例，尚无独立定位核验。"],
        ["机制与诊断", "精确选点空间诊断；AIPPMS受控信息价值参考。", "未证明双层地图或新方法有效。"],
    ], [1.13, 3.52, 2.25])
    para(doc, "前期感知检查完整保留为补充材料；当前固定感知器，将识别误差作为观测不确定性，暂停追识别精度。", size=10, after=0)

    # Page 2: real replay and exact headroom, kept separate.
    doc.add_page_break()
    heading(doc, "3  真实影像：当前路线表现")
    para(doc, "使用3个事件各12对影像，每幅1024像素图像划分为16格，独立执行任务，不拼接为连续灾区。主预算240，另保留120／360；移动一格宽、观察一格分别计20代理单位，每次规划计1，返程计费。这里的单位不是实测秒或米。")
    para(doc, "预测预先离线缓存，但Agent只能在付费后读取到达格。未访问格预测与发布参考均不进入决策；粗图线索只是排序效用，不是校准后的损伤概率。", size=10)
    real_rows = []
    for event, label in events:
        real_rows.append([label] + [f'{selected_real[(event, policy)]["recall"]*100:.2f}%'
                                    for policy in ("pre_static", "post_static", "online_frozen", "online_neighbor")])
    table(doc, ["事件", "灾前静态", "灾后静态", "在线不传播", "在线邻域反馈"], real_rows, [1.7, 1.22, 1.22, 1.34, 1.42])
    para(doc, "表1  主预算下严重损伤像素正确召回率：全部正确检出参考像素／全部严重参考像素，按事件汇总。不是受灾建筑数、人员发现率或单幅平均。", size=9, color=GRAY)
    para(doc, "结果：简单邻域反馈没有稳定收益。海啸固定感知器即使查看全部格，真阳性仍为0，路线无法补救完全漏检；但其参考覆盖仍可改善，因此“访问覆盖”与“正确识别”需要分别报告。", lead="结果：")
    heading(doc, "4  是求解器不够强，还是线索不够有用？")
    para(doc, "在相同16格任务和预算下，精确枚举全部65,536个访问子集的最短闭合路线。分别比较原灾后静态、仅使用原公开效用的精确选点，以及提前知道结果的事后 Oracle。Oracle 只作诊断，不是可执行策略。")
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(head_dir / "figures/headroom.png"), width=Inches(6.9))
    para(doc, "图2  左：严重参考覆盖；右：正确检出。绿色使用事后完美知识，不能作为在线方法成绩。", size=9, color=GRAY)
    nepal = selected_head["nepal-flooding"]
    para(doc, f'尼泊尔主预算下，原静态、同线索精确求解、同预算检出 Oracle 的正确召回分别为{nepal["legacy_post_recall"]*100:.2f}%、{nepal["public_exact_recall"]*100:.2f}%、{nepal["detector_tp_oracle_recall"]*100:.2f}%。仅换求解器提升不足0.1个百分点；理想选点仍有空间，但尚未证明沿途观测能识别这些更好的点。')
    para(doc, "证据强度：36对均属反复使用的开发数据，三个事件只有9／2／3例含严重参考。全部失败与零目标案例均保留；当前不作泛化、显著性或现场可执行性结论。", lead="证据强度：", size=10, after=0)

    # Page 3: two independent follow-up tests and their meaning.
    doc.add_page_break()
    heading(doc, "5  灾前结构地图接入：工程已通，收益未证实")
    para(doc, "按预先固定的身份顺序选取4个旧开发案例：3个通过内部地理一致性检查，1个因建筑分布不足被拒绝并保留。利用匹配的灾前建筑XY／WGS84轮廓重建局部坐标，试跑54条路线；这些坐标尚未经独立地理参考验证。")
    table(doc, ["模型预算600的案例", "在线不传播", "同建筑传播", "解读"], [
        ["Harvey", "44.81%", "44.81%", "无提升"],
        ["Santa Rosa", "36.77%", "34.92%", "下降"],
        ["Mexico", "无严重参考", "无严重参考", "召回记为空值"],
    ], [2.1, 1.38, 1.38, 1.64])
    para(doc, "表2  严重损伤像素正确召回。预算600为假设飞行与观察成本下的模型秒，非实测航时；与36对回放的观测粒度、费用不同，不合并绝对分数。", size=9, color=GRAY)
    para(doc, "3案例×3预算的9组对照中，传播改变了3组访问顺序；覆盖保底相对普通同建筑传播，9组动作全部相同。接通真实结构不等于形成算法收益，当前停止在这些案例上调整传播强度。")
    heading(doc, "6  AIPPMS：验证信息为什么值得付费")
    para(doc, "基于 AIPPMS／SBO-AIPPMS 文献及固定源码，实现小规模 Python 机制参考：5×5网格、3个目标，移动与近／远观察共同消耗能源，必须返程。对全部8种隐藏状态穷举，并设置“有信息观测”与正确率0.5的“无信息观测”对照。")
    control_rows = []
    for model, label in (("informative", "有信息"), ("uninformative", "无信息")):
        for budget in (12, 20):
            vals = [selected_control[(model, budget, pol)]["expected_or_sample_mean_reward"]
                    for pol in ("exact_no_sense", "exact_one_sense", "pomcp_gcb")]
            control_rows.append([label, str(budget), f"{vals[0]:.4f}", f"{vals[1]:.4f}", f"{vals[2]:+.4f}"])
    table(doc, ["观测", "能源预算", "不观察\n精确期望", "至多一次观察\n精确期望", "POMCP-GCB\n样本均值"], control_rows, [1.0, 1.12, 1.44, 1.77, 1.57])
    para(doc, "表3  岩石任务首次到达好目标奖励+10，坏／重复目标−10，观察本身无奖励；数值不是灾损召回率。POMCP每条件8种状态×4个种子，非32个独立场景；无信息条件的正负波动不视为观测收益。", size=9, color=GRAY)
    para(doc, "得到的启示：观测只有提供后续行动可利用的信息，才可能值得付费。受控任务的信标直接观察未访问目标；现有xBD回放只提供已到达格的局部预测，尚缺对其他候选点的可靠信息通道。", lead="得到的启示：")
    para(doc, "适用边界：这是修正观测接口后的机制参考，未运行作者Julia环境或复现论文原表格；至多一次观察不是可重复观察策略的全局上界。建筑巡查还可能在拍摄判读时即完成任务，不能照搬岩石“到访领奖”的奖励。", lead="适用边界：", size=10, after=0)

    # Page 4: a concrete next experiment, advisor decisions, literature.
    doc.add_page_break()
    heading(doc, "7  下一阶段：先验证真实信息通道，再做路线方法")
    para(doc, "首要实验问题：付费获得一次观测后，是否能在未参与拟合的场景中，更好地决定下一处去哪里？先固定感知模块，不在旧36对上追分。", lead="首要实验问题：")
    table(doc, ["顺序", "具体工作与交付物", "继续条件"], [
        ["① 固定任务", "明确巡查平台、相机可观察范围、判读完成条件和返程要求；形成任务与观测定义。", "现有影像能支持定义，不能把缓存重读当新视角。"],
        ["② 检验信息", "检验真实视场是否覆盖未访问候选点，或灾前语义特征能否支持跨点推断；按场景隔离。", "已付观测对预测或选点有可重复的增量价值。"],
        ["③ 接入更新", "分开保存先验、证据及来源；比较不更新、候选更新与保留同等信息的对照。", "收益来自有效信息与更新机制，非显示成两层。"],
        ["④ 公平评测", "比较灾前静态、灾后静态、在线不传播及适配的成熟方法；固定预算、感知器和求解器。", "任务覆盖、首次有效发现、距离与总耗时一起报告。"],
    ], [.72, 3.78, 2.4])
    para(doc, "评价约定：首次有效发现的失败案例保留并明确截尾；像素参考覆盖、模型正确召回和任务完成率分开。规划实际耗时单独记录；代理规划费用不直接作为实时性证据。")
    para(doc, "资源安排：前两步可先用CPU和既有缓存，暂不租GPU。优先缺口是真实观测范围、独立场景与地理／配准依据；新增训练需有明确实验理由后再安排。当前也没有新增人工标注支持。")
    heading(doc, "8  希望老师帮助确定的问题")
    para(doc, "① 是否将近期任务收敛为“无人机辅助建筑损伤巡查”，并固定拍摄／判读的完成标准？\n② 是否有可用的多视角、定位或配准灾害数据，或适合合作获取的数据来源？\n③ 是否先以“先验失配与含噪证据下的预算决策”为研究问题，在独立数据验证信息价值后，再确定具体方法贡献？", after=4)
    heading(doc, "主要参考与应用位置")
    refs = [
        ("[1] xBD", "https://arxiv.org/abs/1911.09296", "：当前灾前灾后影像与发布参考。"),
        ("[2] AIPPMS / SBO-AIPPMS", "https://arxiv.org/abs/2003.09746", "：移动、观察和返程的共同预算；后续工作 arXiv:2209.07660。"),
        ("[3] Search-TTA v5", "https://arxiv.org/abs/2505.11350v5", "：语义关联修正先验；需补含噪观测检验。"),
        ("[4] AeroBelief", "https://arxiv.org/abs/2609.08164", "：先验与证据的职责及接纳规则，任务需适配。"),
        ("[5] IA-TIGRIS v3", "https://arxiv.org/abs/2502.15961v3", "：增量在线求解候选基线，尚未接入。"),
    ]
    for label, url, desc in refs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = Pt(13)
        hyperlink(p, label, url)
        set_font(p.add_run(desc), 9.5)
    para(doc, "复核说明：本报告由既有冻结输出整理，未新增训练、推理或数据选样。真实回放、结构试验与受控奖励分别报告；双层地图的独立收益与方法新颖性均待建立。", size=9, color=GRAY, after=0)

    path = OUT / "阶段汇报_灾前灾后Agent搜查路线规划.docx"
    doc.core_properties.title = "灾前灾后Agent搜查路线规划：阶段进展与下一步实验"
    doc.core_properties.subject = "教师阶段汇报；真实数据结果、信息价值诊断与后续实验"
    doc.core_properties.author = ""
    doc.core_properties.keywords = "灾前灾后,Agent,搜查路线规划,双层地图,信息价值"
    doc.save(path)

    email = """主题：灾前灾后Agent搜查路线规划阶段进展及下一步实验计划

老师您好！

我整理了目前“灾前灾后Agent搜查路线规划”的阶段进展，想向您汇报，并请您指导下一步的任务收敛和实验安排。

目前已将“搜索先验／线索层＋沿途观测证据层”接入路线闭环，移动、观察、规划和返程统一计入预算；使用36对真实灾前灾后影像及固定感知模型，比较了灾前静态、灾后静态和在线重规划。前期感知检查也已归档。

现阶段还没有得到在线更新的稳定提升。进一步诊断发现，仅换成更强的求解器改善有限，关键是沿途观测能否为后续选点提供有效信息。AIPPMS受控小实验验证了这一机制，但它尚不能作为真实灾害任务的有效性证据。

下一步我计划固定感知模块，先明确巡查平台、观察范围和任务完成条件，再用场景隔离的数据检验观测价值，之后开展同预算路线对比。这一步先用CPU与已有缓存推进，暂不租GPU。

想请您重点指导：近期是否先收敛为“无人机辅助建筑损伤巡查”；是否有适合的多视角或带定位、配准信息的数据；以及上述实验顺序是否合适。

附件为阶段汇报，包含主要数据、当前限制和下一步计划。前期感知检查另附为补充材料。

感谢老师指导！
2026年11月2日
"""
    (OUT / "邮件正文.txt").write_text(email, encoding="utf-8")

    supplement_md = """# 论文与下一轮实验的对应关系

整理日期：2026年11月2日。以下为应用计划，不表示完成全部论文复现。

| 论文 | 拟借鉴内容 | 本项目当前状态与前提 |
|---|---|---|
| [AIPPMS](https://arxiv.org/abs/2003.09746)、[SBO-AIPPMS](https://arxiv.org/abs/2209.07660) | 移动、不同观察动作和返程共同预算 | 已完成修正后Python受控机制参考；未运行原生Julia，未复现论文表格。需适配建筑巡查的观察范围和任务收益。 |
| [Search-TTA v5](https://arxiv.org/abs/2505.11350v5) | 用语义相关区域的反馈修正搜索先验 | 尚未接入。先检验独立场景中的真实关联；原文主要搜索设定采用完美二值传感器，不能直接照搬到有误检漏检的模型输出。 |
| [AeroBelief](https://arxiv.org/abs/2609.08164) | 区分情境线索与可接纳证据，处理重叠观测 | 尚未复现。其单物体导航与累计建筑巡查不同，双层表示本身不能直接作为贡献。 |
| [IA-TIGRIS v3](https://arxiv.org/abs/2502.15961v3) | 地图更新后增量复用路线求解 | 尚未接入。先证明地图更新有价值，再分别评价更新与求解器的贡献。 |

执行顺序：固定任务与观测定义 → 验证实际信息通道 → 设计有依据的证据更新 → 与静态、在线不传播、成熟方法公平比较。

继续条件：已付观测在未参与开发的场景上改善预测或后续行动。没有真实新视角时，重复读取同一缓存不计为新观察；没有关联证据时，不继续调相邻格传播规则。

所有旧36对属于开发材料；当前还没有独立真实数据上的新方法有效性结论。新增训练或GPU不是下一步的前置条件。
"""
    (SUP / "论文与实验对应.md").write_text(supplement_md, encoding="utf-8")
    shutil.copy2(perception_dir / "qa/final/感知检查阶段汇报.pdf", SUP / "前期感知检查阶段汇报_20261028.pdf")
    shutil.copy2(real_dir / "figures/fixed_real_examples.png", SUP / "真实影像路线示例.png")
    (SUP / "真实影像示例说明.txt").write_text(
        "示例来源：既有36对xBD开发影像，每事件固定选择身份排序第一例，未按结果换例。\n"
        "图从左至右为灾前影像、灾后影像与发布严重参考、灾后静态路线、在线邻域反馈路线。\n"
        "红色参考仅用于事后评价，不进入Agent状态；青色是已付完整路线。\n"
        "直线网格路线不代表现场可执行航线；影像不拼接为连续灾区，代理费用不是实测航时。\n"
        "本图保留失败和零目标案例，用于说明数据与流程，不证明方法有效。\n"
        "数据来源：https://arxiv.org/abs/1911.09296；固定数据版本：https://huggingface.co/datasets/isp-uv-es/xview2-taco/tree/44e219dc1d88c9d0903d6b478d7f2f3c435e574f\n"
        "影像归Maxar Open Data，原数据归xBD作者／DIU／CMU SEI；衍生展示遵循CC-BY-NC-SA-4.0，用于学术研究交流。\n",
        encoding="utf-8")
    (OUT / "发送说明.md").write_text("""# 邮件材料使用说明

建议将 `邮件正文.txt` 复制到邮箱正文，补上自己的署名。

建议附件：

1. `阶段汇报_灾前灾后Agent搜查路线规划.pdf`：当前研究主线、关键结果、瓶颈、实验安排和拟请教问题。
2. `补充材料/前期感知检查阶段汇报_20261028.pdf`：前期感知工作留档，按需附上。

Word为可编辑留存版本，与主PDF内容一致。补充文件中的真实影像示例及论文对应关系供老师需要细看时使用；可以整体发送本材料包，也可以只发上述两个PDF。

报告不将36对旧开发数据写成独立验证，不将AIPPMS受控奖励写成灾害效果，也不宣称双层地图已有稳定收益。

本材料包已在本地生成，未创建邮箱草稿或发送邮件。原始代码、缓存、模型权重与全部实验归档仍保留在原项目内；本包不包含这些大文件。内部核验和渲染预览位于本目录的qa中，不打入邮件材料包。
""", encoding="utf-8")
    numeric = {
        "date": "2026-11-02", "real_budget_proxy": 240,
        "real_results": [selected_real[(event, pol)] for event, _ in events for pol in ("pre_static", "post_static", "online_frozen", "online_neighbor")],
        "headroom": [selected_head[event] for event, _ in events],
        "controlled_results": control,
        "frozen_inputs": preservation,
        "supplement_pdf_sha256": sha(SUP / "前期感知检查阶段汇报_20261028.pdf"),
        "source_supplement_pdf_sha256": sha(perception_dir / "qa/final/感知检查阶段汇报.pdf"),
        "new_inference": False, "new_training": False, "email_sent": False,
    }
    (QA / "数据与来源核验.json").write_text(json.dumps(numeric, ensure_ascii=False, indent=2), encoding="utf-8")
    config = f'''<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig><dir>/Users/bethany/Library/Fonts</dir><dir>/Library/Fonts</dir><dir>/System/Library/Fonts/Supplemental</dir><dir>/System/Library/Fonts</dir><dir>/Users/bethany/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype</dir><cachedir>{QA}/fontcache</cachedir></fontconfig>'''
    (QA / "fonts.conf").write_text(config, encoding="utf-8")
    print(json.dumps({"docx": str(path), "source_files_verified": sum(x["files_checked"] for x in preservation)}, ensure_ascii=False))


if __name__ == "__main__":
    build()
