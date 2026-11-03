"""Render fixed comparison and predeclared qualitative examples after audit."""
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from run_perception_pair72 import ROOT, read, assemble, load_pair, digest

OUT = ROOT / 'output/disaster_search_perception_pair72_v1'


def percent(x): return 'null' if x is None else f'{100*x:.2f}%'


def main():
    if (OUT / 'completion_receipt.json').exists(): raise ValueError('Completed output frozen')
    audit = read(OUT / 'audit.json')
    if audit['status'] != 'PASS': raise ValueError('Audit required')
    summary = read(OUT / 'summary.json'); events = read(OUT / 'event_scores.json')
    protocol = read(OUT / 'protocol.json'); inference = read(OUT / 'inference_report.json')
    primary = {r['model_id']: r for r in summary['totals'] if r['decoder'] == 'raw_pixel'}
    voted = {r['model_id']: r for r in summary['totals'] if r['decoder'] == 'common_object_vote'}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (OUT / 'figures').mkdir(exist_ok=True)
    labels = ['Harvey', 'Mexico', 'Santa Rosa']; names = ['hurricane-harvey', 'mexico-earthquake', 'santa-rosa-wildfire']
    colors = {'legacy_swint128': '#9aa4b2', 'swint': '#1b728a', 'r34': '#b55a3c'}
    title = {'legacy_swint128': 'Swin128 cached', 'swint': 'Swin512', 'r34': 'R34-512'}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)
    x = np.arange(3)
    for axis, metric in zip(axes, ['precision', 'recall', 'f1']):
        for i, m in enumerate(['legacy_swint128', 'swint', 'r34']):
            values = [next(r[metric] for r in events if r['model_id'] == m and r['decoder'] == 'raw_pixel' and r['event'] == e) for e in names]
            axis.bar(x + (i-1)*.24, [100*v if v is not None else np.nan for v in values], width=.22, label=title[m], color=colors[m])
            for j, value in enumerate(values):
                if value is None: axis.text(j+(i-1)*.24, 2, 'n/a', ha='center', fontsize=7, color=colors[m])
        axis.set_xticks(x, labels); axis.set_title('Severe pixel ' + metric); axis.set_ylim(0, 105)
        axis.spines[['top', 'right']].set_visible(False); axis.grid(axis='y', alpha=.18); axis.set_axisbelow(True)
    axes[0].set_ylabel('Percent'); axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle('Already-open xBD development72 | raw pixels | post-reference diagnostic', fontsize=12)
    fig.text(.5, .01, '3 events,24 scenes/event; training exposure; no official mixed-F1 or independent generalization claim.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, .92])
    fig.savefig(OUT / 'figures/comparison.png', dpi=180); fig.savefig(OUT / 'figures/comparison.pdf'); plt.close(fig)
    rows = read(OUT / 'observations.json'); jobs = {r['sample_id']: r for r in read(OUT / 'rgb_jobs.json')}
    references = {r['sample_id']: r for r in read(OUT / 'reference_manifest.json')}
    palette = np.asarray([[0, 0, 0], [220, 224, 226], [236, 185, 65], [243, 119, 48], [203, 43, 53]], np.uint8)
    def colorize(a):
        result = palette[np.minimum(a, 4)]; result[a == 255] = [115, 105, 158]
        return Image.fromarray(result).resize((240, 240), Image.Resampling.NEAREST)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 13)
    small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)
    sheet = Image.new('RGB', (1200, 40 + len(protocol['preview_ids'])*274 + 30), 'white'); draw = ImageDraw.Draw(sheet)
    for i, label in enumerate(['Pre RGB', 'Post RGB', 'Post reference', 'Swin512 raw', 'R34-512 raw']): draw.text((i*240+10, 10), label, fill='black', font=font)
    for row, sid in enumerate(protocol['preview_ids']):
        y = 40 + row*274; pair = load_pair(jobs[sid])
        panels = [im.resize((240, 240), Image.Resampling.BILINEAR) for im in pair]
        panels.append(colorize(np.asarray(Image.open(references[sid]['reference_file']))))
        for model_id in ('swint', 'r34'):
            selected = [r for r in rows if r['role'] == 'production' and r['sample_id'] == sid and r['model_id'] == model_id]
            loc, damage = assemble(selected); panels.append(colorize(damage*loc.astype(np.uint8)))
        for col, panel in enumerate(panels): sheet.paste(panel, (col*240, y+23))
        draw.text((8, y+3), sid + ' | ' + ('severe reference present' if references[sid]['pixels_by_id']['3'] + references[sid]['pixels_by_id']['4'] else 'no severe reference'), fill='black', font=small)
    draw.text((10, sheet.height-22), 'Black=background; gray=no damage; yellow=minor; orange=major; red=destroyed; violet=ignored. Fixed reference-based examples, not selected by model output.', fill='black', font=small)
    sheet.save(OUT / 'figures/fixed_examples.png')
    # Post-hoc failure description, after all primary scores were fixed.
    failures = []
    for ref in references.values():
        if ref['event'] != 'mexico-earthquake': continue
        document = read(ref['label_path'])
        truth = np.asarray(Image.open(ref['reference_file']))
        predictions = {}
        for model_id in ('swint', 'r34'):
            selected = [r for r in rows if r['role'] == 'production' and r['sample_id'] == ref['sample_id'] and r['model_id'] == model_id]
            predictions[model_id] = assemble(selected)
        for feature in document['features']['xy']:
            if feature['properties']['subtype'] not in ('major-damage', 'destroyed'): continue
            points = np.asarray([[float(v) for v in p.split()] for p in feature['wkt'][10:-2].split(',')])
            mask = np.zeros((1024, 1024), np.uint8); cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
            selected = (mask != 0) & (truth != 255)
            result = {'sample_id': ref['sample_id'], 'derived_uid': feature['properties']['uid'],
                      'reference_grade': feature['properties']['subtype'], 'valid_footprint_pixels': int(selected.sum()),
                      'diagnostic_only_not_instance_detection': True, 'models': {}}
            for model_id, (loc, classes) in predictions.items():
                result['models'][model_id] = {'localization_coverage_fraction': float(loc[selected].mean()),
                    'ungated_damage_class_counts': {str(k): int((classes[selected] == k).sum()) for k in range(5)},
                    'severe_pixels_after_localization_gate': int((selected & loc & (classes >= 3)).sum())}
            failures.append(result)
    if len(failures) != 3: raise ValueError('Expected all3 severe Mexico source objects; do not select failures')
    (OUT / 'mexico_failure_description.json').write_text(json.dumps({'status': 'POST_HOC_DESCRIPTIVE_ONLY',
        'selection': 'All3 severe source polygons in the failed event, no replacement or primary metric changes', 'objects': failures}, indent=2))
    lines = ['# 72 对 xBD：感知器与裁块诊断', '', '日期：2026-10-28。全部已有开发场景保留；没有训练、租用 GPU、增加影像或开启保留集。', '',
             '**当前建议：先不租GPU。** 同一Swin从128改为512时，合并严重像素F1增加4.70个百分点、召回增加9.68个百分点；同为512输入的两个检查点合并F1仅相差0.27个百分点，等权事件宏F1相差0.12个百分点，各事件优劣方向不同。这个描述不代表统计等效或显著性结论。', '',
             '维持现有Swin作为已接通的工作感知器，512作为离线参考，R34保留对照。三栋墨西哥严重参考在两模型下均无正确严重像素；其参考轮廓被预测建筑覆盖约80%–100%，但未门控损伤类别全部是背景或无损。下一项优先核验这些源标签的对应关系与损伤可观察性，再确定可独立评估的数据；暂不启动训练或继续增加搜索变体。', '',
             '本轮完成两个固定公开检查点的统一评估：现有 ChangeOS Swin 与官方 v0.2 ChangeOS ResNet34。两者同属 ChangeOS 家族，不能称为跨架构强基线。', '',
             '## 结果', '', '| 条件 | 严重像素精度 | 召回 | 合并 F1 | 等权事件宏 F1 | 有任意正确严重像素的场景 |',
             '|---|---:|---:|---:|---:|---:|']
    for m in ['legacy_swint128', 'swint', 'r34']:
        r = primary[m]
        lines.append(f"| {title[m]} | {percent(r['precision'])} | {percent(r['recall'])} | {percent(r['f1'])} | {percent(r['equal_event_macro_f1'])} | {r['scenes_with_any_reference_overlap']}/{r['reference_positive_scenes']}参考阳性 |")
    lines += ['', '前三项按全部有效像素合并；宏 F1 为三个事件 F1 等权平均。有任意像素重叠只作诊断，不能解释成逐栋成功检出。旧128只缓存严重二值掩膜，不能用于完整五类混淆分析。', '',
              '| 事件 | Swin512 精度／召回／F1 | R34-512 精度／召回／F1 | 参考阳性场景 |', '|---|---:|---:|---:|']
    for e in names:
        a = next(r for r in events if r['model_id'] == 'swint' and r['decoder'] == 'raw_pixel' and r['event'] == e)
        b = next(r for r in events if r['model_id'] == 'r34' and r['decoder'] == 'raw_pixel' and r['event'] == e)
        fmt = lambda r: '／'.join(percent(r[k]) for k in ('precision', 'recall', 'f1'))
        lines.append(f"| {e} | {fmt(a)} | {fmt(b)} | {a['reference_positive_scenes']}/24 |")
    lines += ['', '共同对象投票作为预先声明的次要条件：Swin512 宏 F1 ' + percent(voted['swint']['equal_event_macro_f1']) + '，R34-512 宏 F1 ' + percent(voted['r34']['equal_event_macro_f1']) + '。投票仅用预测建筑连通域及公开固定权重，不使用真值对象，不据此调整阈值。', '',
              '## 墨西哥失败描述', '',
              '以下是在主评分固定后检查全部3栋严重参考对象的描述，不是新的实例评价，也不改变主结论。定位覆盖仅指预测建筑掩膜与参考轮廓相交比例，不能单独解释成定位正确。', '',
              '| 场景 | Swin预测建筑覆盖 | R34预测建筑覆盖 | Swin／R34严重像素 |', '|---|---:|---:|---:|']
    for failure in failures:
        a, b = failure['models']['swint'], failure['models']['r34']
        lines.append(f"| {failure['sample_id']} | {percent(a['localization_coverage_fraction'])} | {percent(b['localization_coverage_fraction'])} | {a['severe_pixels_after_localization_gate']}／{b['severe_pixels_after_localization_gate']} |")
    lines += ['', '完整未门控损伤类别计数保存在 `mexico_failure_description.json`，用于区分定位掩膜损失与损伤分类错误。没有凭肉眼重写原标签。固定社区镜像目录中未找到这3份原始JSON，原始标签逐对象对应仍待核验；目录查找记录见 `original_label_lookup.json`，目录未命中不代表原始xBD标签不存在。', '',
              '## 评估口径与限制', '',
              '- 每张1024原图分四个512块，不缩放、不增强，两模型看相同像素。512离线感知不能作为旧128预算搜索的免费上下文。旧128与新512的差异是上下文／裁块诊断；两个512检查点的差异才是本轮模型对照。',
              '- 严重损伤合并 major-damage 与 destroyed。由现有灾后多边形统一栅格化；未知以及冲突类别覆盖区记255并排除，不填成负例。假阳性同时含背景和非严重建筑像素；细分在 JSON 中。',
              '- 这里只有灾后多边形，没有原始灾前定位标签。因此不报告官方 xView2 mixed F1，不把像素 F1 当实例 AP、建筑检出率或实际救灾效果。',
              '- 原参考来自已有 TACO 重打包并保留源多边形，使用派生对象身份，没有新增人工标注。既有两场景桥接核验不等于全部标注逐栋人工复核。',
              '- 72对已反复用于开发，Swin存在训练事件暴露；R34发布物没有完整训练样本身份清单，不能排除暴露。两个模型均不能在此建立未见事件泛化结论。墨西哥仅3栋严重参考，事件分数尤其不稳定。', '',
              '## 执行与复核', '',
              f"本机 {inference['device']} 共578次记录前向（2个旧阳性对照＋288个Swin512块＋288个R34-512块），调用记录累计 {inference['sum_recorded_call_seconds']:.2f} 秒，含I/O的循环 {inference['loop_seconds']:.2f} 秒。338项测试通过，两个旧对照逐像素一致。",
              f"独立复核重建{audit['references_independently_rasterized']}份参考、{audit['input_tensors_reconstructed']}个输入张量、{audit['legacy128_masks_verified']}个旧掩膜和{audit['scene_scores_independently_replayed']}项场景分数；对象投票用第二种连通域实现复算。旧{audit['old_frozen_sources_verified']}份冻结源码核对不变。", '',
              '## 来源与文件', '',
              '- [官方 ChangeOS 代码](https://github.com/Z-Zheng/ChangeOS/tree/7936b84aceb9c4b89e959690f3da28749b38d10c)；[v0.2 权重](https://github.com/Z-Zheng/ChangeOS/releases/tag/v0.2)。本轮142,087,496字节权重SHA256为 `' + protocol['models']['r34']['changeos_r34.pt'] + '`。原作者发布物只本地使用；固定根目录LICENSE未找到，不作再分发授权结论。',
              '- [现有 Swin 模型卡](https://huggingface.co/EVER-Z/torchange_example_changeos_swint_on_xview2_best42k)。模型、配置、源代码与全部RGB输入均固定哈希。',
              '- `protocol.json`：推理前固定的口径、调用上限和文件身份；`summary.json`、`event_scores.json`、`scene_scores.json`：总计／事件／场景结果。',
              '- `audit.json`、`calls/`、`cache/`：复核与原始预测缓存；`figures/comparison.png`：事件对比；`figures/fixed_examples.png`：预定6个正／负示例。', '',
              '本轮没有建立新方法，未恢复EBD已停止的输入或搜索变体。下一步选择依据必须同时包含事件失败和数据暴露限制，不能仅凭本开发集的胜出分数申请训练预算。', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'report': str(OUT / 'README.md'), 'figures': 3, 'renderer_sha256': digest(Path(__file__))}), flush=True)


if __name__ == '__main__': main()
