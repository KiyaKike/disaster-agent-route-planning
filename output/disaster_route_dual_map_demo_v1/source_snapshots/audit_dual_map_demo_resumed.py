"""Resume frozen functional audit with JSON-normalized snapshot equality.

The original audit compared an in-memory likelihood tuple to its saved JSON
array. Diagnose/canonicalize exactly that field;no route,fee or map value changes.
"""
import json

import build_dual_map_demo as frozen
from dual_map_core import DualMapAgent
from route_factor_core import Planner


def main():
    assert not (frozen.OUT/'completion_receipt.json').exists()
    frozen.verify()
    routes=frozen.read(frozen.OUT/'routes.json');mm=frozen.maps();ww=frozen.worlds();checked=0
    for meta in frozen.read(frozen.OUT/'agent_snapshots.json'):
        g=mm[meta['region']];w=frozen.private(g,ww[g.region])
        result=DualMapAgent.restore(g,meta['snapshot'],{w.token:w},Planner()).finish()
        reference=next(r for r in routes if all(r[k]==meta[k] for k in ('region','policy','mode')))
        assert json.loads(json.dumps(result))==reference
        comparable=json.loads(json.dumps(result))
        assert isinstance(result['maps']['assumed_likelihood'],tuple)
        assert isinstance(reference['maps']['assumed_likelihood'],list)
        # The saved representation is already the desired canonical JSON schema.
        assert comparable['maps']['assumed_likelihood']==reference['maps']['assumed_likelihood']
        checked+=1
    original_read=frozen.read
    def normalized_read(path):
        result=original_read(path)
        if path==frozen.OUT/'routes.json':
            for r in result:r['maps']['assumed_likelihood']=tuple(r['maps']['assumed_likelihood'])
        return result
    frozen.read=normalized_read
    frozen.audit()
    frozen.save(frozen.OUT/'serialization_audit_note.json',{
        'status':'PASS','diagnosis':'Only /maps/assumed_likelihood is an in-memory tuple and a saved JSON list;JSON-value equality holds.',
        'canonical_json_snapshot_continuations_checked':checked,'original_frozen_source_unchanged':True,
        'route_or_policy_changes':False,'adapter_source_sha256':frozen.digest(__file__),
        'audit_replay':'Original frozen audit with read-time tuple normalization only for that recorded metadata field.'})


if __name__=='__main__':main()
