"""Identity-only cohort selection and prospective RGB quality observations."""
import hashlib
import re
import numpy as np

EVENTS=('nepal-flooding','sunda-tsunami','tuscaloosa-tornado')
SEED='multievent36-v1-20261028'


def select(identities,author_splits,blocked_ids,blocked_events,count=12):
    required={'internal:current_id','ml:source_file','ml:split_original','ml:disaster'}
    if any(set(r)!=required for r in identities):raise ValueError('Identity-only schema required; no damage or counts allowed')
    ids=[r['ml:source_file'] for r in identities]
    if len(set(ids))!=len(ids):raise ValueError('Ambiguous source identity')
    result=[]
    for event in EVENTS:
        if event in blocked_events:raise ValueError('Previously used/reserved event')
        pool=[]
        for row in identities:
            sid=row['ml:source_file']
            if row['ml:disaster']!=event:continue
            if not re.fullmatch(re.escape(event)+r'_\d{8}',sid):raise ValueError('Malformed event ID')
            name=row['ml:split_original']+'/images/'+sid
            if name not in author_splits['test'] or name in author_splits['train']|author_splits['val']:raise ValueError('Checkpoint train/validation exposure or missing author membership')
            if sid not in blocked_ids:pool.append(row)
        if len(pool)<count:raise ValueError('Insufficient untouched identity-only pool; do not substitute')
        ranked=sorted(pool,key=lambda r:(hashlib.sha256((SEED+'|'+event+'|'+r['ml:source_file']).encode()).hexdigest(),r['ml:source_file']))
        for rank,row in enumerate(ranked[:count],1):
            result.append({'sample_id':row['ml:source_file'],'event':event,'source_subset':row['ml:split_original'],
                'source_parent_id':row['internal:current_id'],'selection_rank':rank,
                'selection_sha256':hashlib.sha256((SEED+'|'+event+'|'+row['ml:source_file']).encode()).hexdigest(),
                'purpose':'measurement_replication_development_not_final_confirmation'})
    return result


def rgb_quality(array):
    if array.dtype!=np.uint8 or array.shape!=(1024,1024,3):raise ValueError('Native RGB1024 required; retain ID and stop on invalid payload')
    fraction=float(np.all(array==0,axis=2).mean())
    return {'zero_RGB_fraction':fraction,'zero_RGB_at_least25pct_flag':fraction>=.25,
            'rule':'Flag only;all36 retained,no pixel masking or replacement;exact-zero is not a validated nodata classifier.'}
