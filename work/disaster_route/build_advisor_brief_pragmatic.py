"""Restate the advisor update around the actual data limits and feasible scope."""
import importlib.util
import json
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "output/阶段汇报_灾前灾后Agent路线规划_20261102_精简版"
OUT = ROOT / "output/阶段汇报_灾前灾后Agent路线规划_20261102_务实版"
QA = OUT / "qa"
NAME = "灾前灾后Agent路线规划_阶段说明"
spec = importlib.util.spec_from_file_location("concise_brief", Path(__file__).with_name("build_advisor_brief_concise.py"))
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
base = source.base


def build():
    QA.mkdir(parents=True, exist_ok=True)
    receipt = base.read_json(OLD / "qa/交付核验.json")
    for name, expected in receipt["files"].items():
        assert base.sha(OLD / name) == expected, name
    frozen = base.check_frozen()
    doc = Document(OLD / "灾前灾后Agent路线规划_进展摘要.docx")
    replacements = {
        "目前已跑通双层地图与预算路线闭环":
            "现有数据能支持影像层面的预算搜查基线，尚不足以验证具体无人机巡查。当前主要限制是任务数据。",
        "一 已完成": "一 已有基础",
        "双层地图分别保存":
            "已在36对xBD灾前灾后遥感影像上搭建双层地图，区分先验线索与沿途证据；固定感知器，完成静态／在线路线回放及预算核验。前期感知检查和失败记录已保留，现有在线更新尚未取得稳定提升。",
        "主预算240代理单位":
            "各事件12对，指标为严重损伤像素正确召回率。主预算240为移动、观察、规划和返程的代理费用，非实测航时。",
        "二 目前判断": "二 当前数据限制",
        "简单邻域和同建筑传播":
            "目前尚未找到合适的无人机巡查数据来源。手头资料缺少可对应真实任务的观测范围、多视角、可信定位配准、飞行轨迹及航时／能耗依据；已有局部坐标重建也未经过独立核验。",
        "AIPPMS受控参考":
            "因此，现阶段只能在假设网格和成本下比较基线，不能据此说明真实无人机如何观察、如何飞行或能取得多少巡查收益。36对均为开发数据，也不能作为独立方法验证。",
        "三 下一步与请教问题": "三 建议推进方式",
        "1. 明确无人机建筑巡查":
            "1. 先把研究范围限定为现有影像支持的预算搜查基线与受控仿真，明确假设和适用边界。",
        "2. 在隔离场景中检验":
            "2. 无人机巡查验证以落实数据或合作条件为前提；条件未具备时，不继续增加缺乏依据的飞行细节。",
        "想请老师指导：":
            "想请老师指导：先做这一层面的规划方法研究是否合适？若需推进无人机巡查验证，是否有可落实的数据或合作渠道？",
    }
    replaced = []
    for p in list(doc.paragraphs):
        if p.text.startswith("3. 再做同预算路线对比"):
            p._p.getparent().remove(p._p)
            continue
        if p.text.startswith("数据与机制参考："):
            p.clear()
            base.set_font(p.add_run("数据来源："), 9, False, "000000")
            base.hyperlink(p, "xBD", "https://arxiv.org/abs/1911.09296")
            continue
        for prefix, text in replacements.items():
            if p.text.startswith(prefix):
                p.runs[0].text = text
                for run in p.runs[1:]:
                    run.text = ""
                replaced.append(prefix)
                break
    assert len(replaced) == len(replacements), set(replacements) - set(replaced)
    source.bw.monochrome_styles(doc)
    doc.core_properties.title = "灾前灾后Agent路线规划阶段说明"
    doc.save(OUT / f"{NAME}.docx")
    email = """主题：灾前灾后Agent路线规划进展与数据限制

老师您好！

目前已用36对灾前灾后影像搭建双层地图，跑通静态／在线路线基线，但尚未取得稳定提升。

主要困难是数据：现有材料能做影像网格上的规划比较，但真实无人机的观测、定位、轨迹和航时数据还没有合适来源，暂不能验证具体巡查方案。

我倾向先做现有数据支持的预算搜查与受控仿真。想请您指导：这样收敛是否合适，还是应先通过课题组或合作方落实数据条件，再推进无人机任务？

附件为一页阶段说明，感谢老师指导！
"""
    (OUT / "邮件正文.txt").write_text(email, encoding="utf-8")
    config = (OLD / "qa/fonts.conf").read_text(encoding="utf-8").replace(str(OLD / "qa"), str(QA))
    (QA / "fonts.conf").write_text(config, encoding="utf-8")
    audit = {"source_receipts": frozen, "previous_materials_preserved": True,
             "scope": "image_grid_baselines_and_controlled_simulation_pending_task_data",
             "data_gap_from_user": True, "new_experiments": False, "email_sent": False}
    (QA / "来源核验.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"docx": str(OUT / f"{NAME}.docx"), "email_characters": len(email)}, ensure_ascii=False))


if __name__ == "__main__":
    build()
