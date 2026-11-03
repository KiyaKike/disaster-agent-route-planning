"""Inspect the first two fixed cells; never select pictures by prediction quality."""
import argparse
import json
from pathlib import Path

from run_changeos_pilot import crop_pair, degrade_image


def main():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.pilot / "pilot_protocol.json").read_text())
    observations = json.loads((args.pilot / "observations.json").read_text())
    sample = plan["samples"][0]
    with Image.open(sample["pre_image"]) as im:
        pre = im.convert("RGB")
    with Image.open(sample["post_image"]) as im:
        post = im.convert("RGB")
    truth = json.loads((args.prepared_data / sample["sample_id"] / "evaluation_truth.json").read_text())["cells"]
    font_path = "/System/Library/Fonts/Hiragino Sans GB.ttc"
    font = ImageFont.truetype(font_path, 16)
    title_font = ImageFont.truetype(font_path, 22)
    canvas = Image.new("RGB", (1010, 710), "#f7f6f2")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 10), "ChangeOS｜校准集局部观察试跑", font=title_font, fill="#18202b")
    draw.text((20, 44), sample["sample_id"], font=font, fill="#34414c")
    draw.text((20, 70), "按冻结清单展示前两格；红色为模型疑似严重受损像素，不是真值。", font=font, fill="#34414c")
    index = {(r["sample_id"], r["cell_id"], r["mode"]): r for r in observations}
    for row_index, cell_id in enumerate(plan["cells"][:2]):
        y = 110 + row_index * 295
        _, cell = crop_pair(pre, post, cell_id)
        truth_value = truth[cell_id]["target_present"]
        draw.text((20, y), f"{cell_id}｜网格真值：{truth_value}（按建筑质心归属）", font=font, fill="#18202b")
        panels = []
        for mode, label in (("native", "原始"), ("blurred", "模拟模糊")):
            record = index[(sample["sample_id"], cell_id, mode)]
            image = degrade_image(cell, mode)
            with np.load(record["cache_file"]) as cache:
                mask = cache["severe_mask"]
            array = np.array(image, dtype=float)
            array[mask] = array[mask] * .35 + np.array([255, 40, 25]) * .65
            overlay = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            panels.extend([(label + "灾后", image),
                           (label + f"预测：{record['predicted_severe_pixels']} 像素", overlay)])
        for column, (label, image) in enumerate(panels):
            x = 20 + column * 245
            draw.text((x, y + 27), label, font=font, fill="#34414c")
            canvas.paste(image.resize((225, 225), Image.Resampling.NEAREST), (x, y + 52))
    canvas.save(args.pilot / "pilot_preview.png")
    print(str(args.pilot / "pilot_preview.png"))


if __name__ == "__main__":
    main()
