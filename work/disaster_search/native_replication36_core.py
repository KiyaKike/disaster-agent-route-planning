"""Predeclared new36-only measurement screens, without adaptive method selection."""
from external_footprint_core import NATIVE128,NATIVE256


def decide_replication(summary,opportunities):
    screens=[]
    for domain in ["all_published_pixels","both_dates_nonzero"]:
        events=[r for r in summary["event_rows"] if r["domain"]==domain and r["budget"]==8 and r["policy"]=="random_exact"]
        totals={r["input_mode"]:r for r in summary["totals"] if r["domain"]==domain and r["budget"]==8 and r["policy"]=="random_exact"}
        by_event={(r["event"],r["input_mode"]):r for r in events};positive=sorted({r["event"] for r in events if r["reference_pixels"]>0})
        available=[r for r in opportunities if r["domain"]==domain and r["input_mode"]==NATIVE256 and r["incremental_covered_reference_pixels"]>0]
        counts={e:sum(r["event"]==e for r in available) for e in positive};a=totals[NATIVE256];b=totals[NATIVE128]
        differences={e:by_event[e,NATIVE256]["incremental_covered_reference_pixels"]-by_event[e,NATIVE128]["incremental_covered_reference_pixels"] for e in positive}
        checks={"at_least_two_reference_positive_events":len(positive)>=2,"at_least_six_detectable_new_scenes":len(available)>=6,
            "at_least_two_detectable_scenes_in_each_of_two_events":sum(n>=2 for n in counts.values())>=2,
            "budget8_random_macro_incremental_coverage_increases":a["event_macro_incremental_covered_pixels_per_scene"]>b["event_macro_incremental_covered_pixels_per_scene"],
            "budget8_pooled_reference_precision_does_not_fall":a["reference_precision"] is not None and b["reference_precision"] is not None and a["reference_precision"]>=b["reference_precision"],
            "every_reference_positive_event_nondecreasing":bool(differences) and all(v>=0 for v in differences.values()),
            "strict_event_gains_in_at_least_two_reference_positive_events":sum(v>0 for v in differences.values())>=2}
        sufficient=all(checks[k] for k in ["at_least_two_reference_positive_events","at_least_six_detectable_new_scenes","at_least_two_detectable_scenes_in_each_of_two_events"])
        screens.append({"domain":domain,"checks":checks,"passed":all(checks.values()),"evidence_sufficient":sufficient,
            "reference_positive_events":positive,"detectable_new_scenes":len(available),"detectable_scenes_by_reference_positive_event":counts,"event_incremental_pixel_differences":differences})
    primary=screens[0]
    return {"primary_domain":"all_published_pixels","primary_budget":8,"primary_policy":"random_exact","primary_condition":NATIVE256,
        "domain_screens":screens,"primary_screen_passed":primary["passed"],"evidence_sufficient":primary["evidence_sufficient"],
        "decision":"NATIVE256_MEASUREMENT_REPLICATION_SCREEN_PASSED" if primary["passed"] else "STOP_SOURCE_CHECKPOINT_VARIANTS_REPLICATION_NOT_ESTABLISHED",
        "cohort_scope":"NEW36_ONLY_OLD18_NOT_POOLED","new_method_selected":False,"new_fitting_enabled":False,
        "independent_geographic_or_human_validation":False,"old18_used_to_rescue_screen":False}
