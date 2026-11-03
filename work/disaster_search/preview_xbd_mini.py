"""Calibration-only input/label preview; never selects examples by damage count."""
import argparse
import json
from pathlib import Path

from prepare_xbd_sample import grid_truth


def main():
    from PIL import Image, ImageDraw, ImageFont

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.data / "protocol.json").read_text())
    examples, seen = [], set()
    for sample in protocol["samples"]:
        if sample["split"] == "calibration" and sample["event"] not in seen:
            examples.append(sample)
            seen.add(sample["event"])
    font_path = "/System/Library/Fonts/Hiragino Sans GB.ttc"
    font = ImageFont.truetype(font_path, 17)
    title_font = ImageFont.truetype(font_path, 22)
    canvas = Image.new("RGB", (840, 105 + 340 * len(examples)), "#f7f6f2")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), "xBD mini72｜校准集输入检查", font=title_font, fill="#17212b")
    draw.text((20, 47), "仅展示每个校准事件按清单排列的第一对；不按损伤数量挑图。", font=font, fill="#34414c")
    draw.text((20, 74), "红框：目标格  黄框：仅有未知损伤标签的格子；框来自真值，不是预测。", font=font, fill="#34414c")
    for index, sample in enumerate(examples):
        base_y = 105 + index * 340
        files = sample["files"]
        with Image.open(args.data / files["pre_image"]) as im:
            pre = im.convert("RGB")
        with Image.open(args.data / files["post_image"]) as im:
            post = im.convert("RGB")
        features = json.loads((args.data / files["post_label"]).read_text())["features"]["xy"]
        truth = grid_truth(features, *post.size, 8)
        overlay = post.resize((256, 256), Image.Resampling.LANCZOS)
        od = ImageDraw.Draw(overlay)
        for cell_id, target in truth.items():
            row, col = int(cell_id[1:3]), int(cell_id[4:6])
            color = "#ff403b" if target["target_present"] is True else ("#ffd54a" if target["target_present"] is None else None)
            if color:
                od.rectangle((col * 32 + 1, row * 32 + 1, (col + 1) * 32 - 2, (row + 1) * 32 - 2), outline=color, width=3)
        draw.text((20, base_y), sample["sample_id"], font=font, fill="#17212b")
        for column, (label, image) in enumerate((("灾前", pre), ("灾后", post), ("灾后＋评价网格", overlay))):
            x = 20 + column * 275
            draw.text((x, base_y + 26), label, font=font, fill="#34414c")
            canvas.paste(image.resize((256, 256), Image.Resampling.LANCZOS), (x, base_y + 53))
        positives = sum(c["target_present"] is True for c in truth.values())
        unknowns = sum(c["target_present"] is None for c in truth.values())
        draw.text((20, base_y + 313), f"64 格中：目标格 {positives}；未知格 {unknowns}。", font=font, fill="#34414c")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(str(args.out))


if __name__ == "__main__":
    main()
