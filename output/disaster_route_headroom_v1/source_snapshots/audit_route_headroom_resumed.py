"""Resume frozen audit with non-overflowing cardinality arithmetic.

Solver used floating observation cost20.;its emitted routes/costs are unchanged.
The independent auditor used integer20 against uint8 counts,which overflows for
13-16observations. Cast the audit-only count view to int64 before arithmetic.
"""
from pathlib import Path
import shutil

import numpy as np

import run_route_headroom as run


def main():
    run.open_output();run.verify()
    cat=run.catalogue()
    # Reproduce and guard the precise overflow,including unselected large subsets.
    original=cat['cardinality'];safe=original.astype(np.int64)
    assert original.max()==16
    affected=np.flatnonzero(original*20 != safe*20)
    assert len(affected)>0 and np.min(safe[affected])==13
    assert int(safe[-1]*20)==320
    # Fixed solver's float cost has no overflow at any mask.
    np.testing.assert_array_equal(original*20.,safe*20.)
    routes_before=run.sha(run.OUT/'solutions.json')
    run.save(run.OUT/'audit_dtype_amendment.json',{
        'scope':'Independent audit-only integer arithmetic correction;frozen solver and all route/score inputs unchanged.',
        'reason':'uint8 subset cardinality times integer20 wraps for13-16visits;core solver uses float20.and is unaffected.',
        'correction':'Cast audit catalogue cardinality to int64 before its integer cost calculation.',
        'masks_with_wrapped_audit_cost':len(affected),'floating_solver_costs_checked':len(original),
        'solutions_sha256_before':routes_before,'frozen_runner_sha256':run.sha(Path(run.__file__)),
        'resumed_auditor_sha256':run.sha(__file__)})
    def audit_catalogue():
        return {**cat,'cardinality':safe}
    run.catalogue=audit_catalogue
    run.audit()
    assert run.sha(run.OUT/'solutions.json')==routes_before
    shutil.copyfile(__file__,run.OUT/'source_snapshots'/Path(__file__).name)


if __name__=='__main__':main()
