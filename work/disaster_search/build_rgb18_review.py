"""Build an offline RGB-only first-review tool; it creates no human annotations."""
import argparse
import json
from pathlib import Path

from build_xbd_mini import atomic_json, sha
from plan_search_v2 import freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    index = root/"review_packet_index.json"
    template = Path(__file__).with_name("rgb18_review_template.html")
    freeze(root/"review_tool_protocol.json", {"stage": "PREDICTION_BLIND_FIRST_REVIEW_TOOL_ONLY",
        "index_sha256": sha(index.read_bytes()), "template_sha256": sha(template.read_bytes()),
        "builder_sha256": sha(Path(__file__).read_bytes()),
        "initial_human_records": 0, "completed_gold_records": 0,
        "images_only_no_masks_or_predictions": True,
        "export_role": "First-review draft, never final gold. Distinct second review/adjudication still required."})
    rows = json.loads(index.read_text())
    data = {"cohort_sha256": sha(index.read_bytes()), "rows": rows,
        "attribution": (root/"ATTRIBUTION.txt").read_text()}
    text = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = template.read_text().replace("__PACKET_DATA_JSON__", text)
    (root/"review.html").write_text(html)
    atomic_json(root/"review_tool_build.json", {"review_html_sha256": sha(html.encode()),
        "selected_pairs": len(rows), "human_records_created": 0,
        "no_inference_or_annotation_model": True})
    print("Prediction-blind first-review tool built; no human labels generated.")


if __name__ == "__main__":
    main()
