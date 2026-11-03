"""Prospective acquisition-footprint comparison after a stopped input candidate."""
import json
from pathlib import Path

from build_xbd_mini import sha
from plan_search_v2 import freeze


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);args=p.parse_args();root=args.root
    read=lambda name:json.loads((root/name).read_text());current=read("protocol.json");decision=read("next_step_decision.json")
    if decision["primary_screen_passed"]:raise ValueError("This plan only follows the stopped free-coarse+128 candidate")
    blocks=[]
    for r in range(2):
        for c in range(2):
            cells=[f"r{rr:02d}c{cc:02d}" for rr in range(2*r,2*r+2) for cc in range(2*c,2*c+2)]
            blocks.append({"block_id":f"b{r}{c}","bounds_xyxy":[c*256,r*256,(c+1)*256,(r+1)*256],"paid_cell_ids":cells,"cost":4})
    protocol={"status":"PROSPECTIVE_NOT_EXECUTED","client_report_date":"2026-10-26",
        "question":"At the same purchased native-pixel area, does acquiring contiguous256px blocks recover more published severe reference than independent128px cells?",
        "research_scope":"Measurement/acquisition-interface feasibility, no new search method or independent efficacy claim.",
        "stopped_candidate":"free_coarse512_plus_native_cell at128px remains stopped; do not resume its mixed-quality extension.",
        "basis_protocol_sha256":sha((root/"protocol.json").read_bytes()),"basis_evidence_sha256":sha((root/"evidence_summary.json").read_bytes()),
        "basis_decision_sha256":sha((root/"next_step_decision.json").read_bytes()),"generator_sha256":sha(Path(__file__).read_bytes()),
        "selected_ids":current["selected_ids"],"checkpoint":current["checkpoint"],"blocks":blocks,
        "primary_condition":"native256_contiguous","secondary_condition":"free_coarse512_plus_native256",
        "conditions":[{"name":"native256_contiguous","calls":72,"input":"Unresized256x256 paired RGB from each of4 fixed nonoverlapping2x2-cell blocks.","output":"Canonical256 boolean severe output, no resize; all prediction pixels belong to four purchased128px cells."},
            {"name":"free_coarse512_plus_native256","calls":72,"input":"Free64 BOX coarse ->512 BILINEAR canvas per date; paste only purchased256 block at exact source position.","output":"Crop output to exact256 purchased ROI. Transform receives only64 coarse plus256 block, never the full native512 scene."}],
        "locked_positive_controls":"Same2 old128 native controls first; any unequal pixel stops before new conditions.","max_new_forward_calls":146,"no_hidden_warmup":True,
        "area_cost_contract":"Controlled purchased native-pixel area: one128 cell costs1, contiguous256 block costs4, full native512 scene costs16. This is a NEW acquisition-cost proxy, not measured field/flight latency or physical sensor resolution.",
        "budgets":[4,8,16],"primary_budget":8,"quality":"Native only; do not compare directly against previous mixed-quality scores or label it the old primary experiment.",
        "cached_controls":"Old128 native predictions (288 masks), free-coarse-only18 masks, offline full512 native18 masks reused unchanged. Full512 ineligible at budget4/8, available only at16; never credit its free context at budget8.",
        "free_baseline":"Report each scene's free-coarse-only severe mask at budget0 and count only new reference-pixel union beyond it as paid gain. Report total union and false pixels separately, including any harmful additions.",
        "policies_before_outputs":["Uniform random without replacement over16cells or4blocks; evaluate exact expected incremental pixel area since units are disjoint (no neural calls or MonteCarlo seeds needed).",
            "Fixed rawRGB-change rank: original coarse-only cell75th-percentile cue; block utility = arithmetic mean of its4 cell cues (equivalent totalcue/cost); tie ranks inherited from10 frozen cell-order seeds, block rank = minimum member rank. No feedback adaptation or new utility fit."],
        "primary_contrast":"Budget8 exact-random expected additional severe-reference area for primary native256 vs cached native128, pooled reference precision including free baseline, each event and3-event macro. Random precision is the ratio of expected pooled covered counts to expected pooled predicted counts, not mean trajectory precision. Ranked strategy is a separately reported secondary contrast.",
        "engineering_followup_rule":"Advance only primary native256 if both reference-positive events have nonzero incremental opportunity, budget8 exact-random event-macro incremental coverage strictly exceeds128, and pooled reference precision at budget8 does not fall below128. Secondary condition cannot replace failed primary by a posthoc win. Report every budget/event/condition.",
        "domains":["all_published_pixels","both_dates_nonzero"],"fixed_cohort":"All18 retained, all4blocks scored, no reference-guided placement, offset selection, resizing, fitting or threshold tuning.",
        "required_preflight":"Lock implementation SHA before any new predictions; check four-block partition, exact256 ROI, source pixel ownership,4/16 cost accounting and full512 budget ineligibility, independent random-area identity and equality with enumeration, and no reference/future-feedback access.",
        "stop_scope":"If primary fails, stop measurement-interface variants on these18; assess new data/reference or fixed detector replacement as a separately scoped study before any planner development.",
        "human_annotation_resources":"Unavailable","new_training_or_paid_gpu_or_download":False,"old_reserved_sets_scored":False}
    freeze(root/"prospective_footprint_diagnostic.json",protocol)
    print(json.dumps({"next":"NATIVE_AREA_COST_FOOTPRINT_DIAGNOSTIC","status":protocol["status"],"max_new_forward_calls":146,"new_inference":0}))


if __name__=="__main__":main()
