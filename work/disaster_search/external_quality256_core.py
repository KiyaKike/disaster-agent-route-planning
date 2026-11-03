"""Fixed synthetic blur on purchased256 pixels and full-area quality measurements."""
import numpy as np

from external_footprint_core import BLOCKS,block_bounds,score_union
from external_information_core import require_pair
from external_context_core import summarize,paired_changes
from run_changeos_pilot import degrade_image

QUALITIES=("native256","blurred256")


def blurred_inputs(patch_pair):
    require_pair(patch_pair,(256,256))
    return tuple(degrade_image(im,"blurred") for im in patch_pair)


def canonical_summary(mask):
    if mask.dtype!=np.bool_ or mask.shape!=(256,256):raise ValueError("Canonical boolean256 required")
    count=int(mask.sum());return {"predicted_severe_pixels":count,"declared_target":count>=4}


def mosaic(masks):
    if set(masks)!=set(BLOCKS):raise ValueError("Exactly four fixed256 blocks required")
    result=np.zeros((512,512),dtype=np.bool_)
    for block,mask in masks.items():
        canonical_summary(mask);x,y,xx,yy=block_bounds(block);result[y:yy,x:xx]=mask
    return result


def measure_pair(native,blurred,reference,free,domain):
    rows={quality:score_union(reference,free,paid,domain) for quality,paid in [(QUALITIES[0],native),(QUALITIES[1],blurred)]}
    changes=paired_changes(free|native,free|blurred,reference,domain)
    return rows,changes


def quality_summary(rows):
    result=summarize(rows)
    keys=["free_covered_reference_pixels","free_false_positive_pixels","incremental_covered_reference_pixels","incremental_false_positive_pixels"]
    for row in result["event_rows"]:
        part=[r for r in rows if all(r[k]==row[k] for k in ["domain","input_mode","event"])]
        row.update({k:sum(r[k] for r in part) for k in keys})
    for row in result["totals"]:
        part=[r for r in rows if all(r[k]==row[k] for k in ["domain","input_mode"])]
        row.update({k:sum(r[k] for r in part) for k in keys})
    return result


def measurement_decision(summary):
    screens=[]
    for domain in ["all_published_pixels","both_dates_nonzero"]:
        rows=[r for r in summary["event_rows"] if r["domain"]==domain and r["input_mode"]==QUALITIES[1]]
        positive=[r for r in rows if r["reference_pixels"]>0]
        checks={"two_reference_positive_events_retained":len(positive)==2,
            "both_positive_events_have_blurred_reference_overlap":len(positive)==2 and all(r["incremental_covered_reference_pixels"]>0 for r in positive)}
        screens.append({"domain":domain,"checks":checks,"reference_positive_events":[r["event"] for r in positive],"passed":all(checks.values())})
    passed=screens[0]["passed"]
    return {"primary_domain":"all_published_pixels","domain_screens":screens,"primary_screen_passed":passed,
        "decision":"REGISTER_SEPARATE_MIXED_QUALITY_FEASIBILITY" if passed else "STOP_THIS_SYNTHETIC_BLUR_MIXED_QUALITY_CANDIDATE",
        "native_only_footprint_result_preserved":True,"blur_not_retuned":True,"secondary_input_not_substituted":True,
        "new_search_runs":0,"new_method_selected":False,"new_fitting_enabled":False,"independent_efficacy_claim":False,
        "budget8_initial_cost":4,"native_upgrade_cost":8,"budget8_revisit_feasible":False}
