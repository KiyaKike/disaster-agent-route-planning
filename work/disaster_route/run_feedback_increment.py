"""Protocol frozen before any fitting. Development diagnostic on already-used xBD36."""
from pathlib import Path
import hashlib
import json
import shutil
import time
import numpy as np

from feedback_increment_core import *

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'output/disaster_route_real_xbd36_v1'
OUT=ROOT/'output/disaster_route_feedback_increment_v1'
SCORERS=('prior','prior_feedback','prior_permuted_feedback','post_utility_unfitted','neighbor_rule_unfitted')


def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def protocol():
    return {'date':'2026-11-03','stage':'FEEDBACK_INCREMENT_DIAGNOSTIC_V1',
      'question':'Given the same paid prefix, do fixed-detector predictions on paid tiles improve prediction of unvisited tiles beyond free public pre/post cues?',
      'data':'Existing xBD36 cache (36 chips x16 tiles, 3 events x12). Already used for development: results are development diagnostics, never independent validation.',
      'free_inputs':list(PRIOR_NAMES),'feedback_inputs':list(FEEDBACK_NAMES),
      'feedback_source':'predicted_severe_pixels of paid prefix tiles only. tp/fp/reference never used as features.',
      'targets':{'primary':'y_tp: unvisited tile contains fixed-model true-positive severe pixels (route opportunity actually realizable by this detector)',
                 'secondary':'y_ref: unvisited tile contains published major/destroyed reference pixels (separates route opportunity from detector misses)'},
      'prefixes':{'primary':'route: prefixes k=1..n of executed post_static budget240 routes (feedback-free policy, not outcome-selected)',
                  'secondary':'random: k in {1,2,4}, 30 uniform draws per chip, seed 20261102'},
      'model':f'L2 logistic regression, lambda={L2} on standardized features, intercept unpenalized. No tuning.',
      'splits':{'primary':'leave-one-event-out (train two events, score third)','secondary':'leave-one-chip-out (event-mixed; weaker isolation)'},
      'controls':['prior_permuted_feedback: same feature count, feedback values borrowed from another chip with same scheme and k',
                  'post_utility_unfitted: existing post-cue utility','neighbor_rule_unfitted: existing untuned 4-neighbor rule from xBD36 replay'],
      'metrics':['pooled out-of-fold AUROC, average precision, log loss, Brier','per-event blocks (Sunda y_tp has no positives: AUROC/AP null)',
                 'next-pick hit rate over states with any positive unvisited tile',
                 'chip-cluster bootstrap 95% CI of deltas vs prior, 2000 reps, predictions held fixed (no refit)'],
      'decision_rule':'Incremental information SUPPORTED only if in primary cell (route, leave-one-event-out, y_tp): AP delta CI lower>0 AND log-loss delta CI upper<0 AND permuted-control AP delta CI lower<=0. Otherwise NOT ESTABLISHED. If not established, do not design a new online updater on this signal; old neighbor/floor/trigger rules stay retired.',
      'power_note':'Fixed-model TP tiles: Nepal 24 (9 chips), Sunda 0, Tuscaloosa 4 (2 chips). Expect wide intervals; a null result is not evidence of absence.',
      'gpu_rental':False,'new_model_inference':0,'new_annotations':0,
      'input_sha256':{n:sha(SRC/n) for n in ('public_features.json','private_tile_scores.json','routes.json')}}


def features_for(name):
    if name=='prior':return lambda r:r['prior']
    if name in ('prior_feedback','prior_permuted_feedback'):return lambda r:r['prior']+r['feedback']
    raise ValueError(name)


def evaluate(rows,perm_rows,label,split):
    scores={'prior':out_of_fold(rows,label,features_for('prior'),split),
            'prior_feedback':out_of_fold(rows,label,features_for('prior_feedback'),split),
            'prior_permuted_feedback':out_of_fold(perm_rows,label,features_for('prior_permuted_feedback'),split),
            'post_utility_unfitted':np.array([r['post_utility'] for r in rows]),
            'neighbor_rule_unfitted':np.array([r['neighbor_rule'] for r in rows])}
    y=np.array([r[label] for r in rows]);out={'pooled':{},'per_event':{},'next_pick':{},'delta_vs_prior':{}}
    for name,s in scores.items():
        out['pooled'][name]=metric_block(y,s)
        out['per_event'][name]={e:metric_block(y[m],s[m]) for e in EVENTS for m in [np.array([r['event']==e for r in rows])]}
        hits=next_pick_hits(rows,s,label)
        out['next_pick'][name]={'opportunity_states':len(hits),'hits':int(sum(h for _,h in hits)),
                                'rate':float(np.mean([h for _,h in hits])) if hits else None}
    for name in ('prior_feedback','prior_permuted_feedback','neighbor_rule_unfitted'):
        base='post_utility_unfitted' if name=='neighbor_rule_unfitted' else 'prior'
        point={k:(out['pooled'][name][k]-out['pooled'][base][k]) if out['pooled'][name][k] is not None and out['pooled'][base][k] is not None else None
               for k in ('auroc','average_precision','log_loss')}
        out['delta_vs_prior'][name]={'baseline':base,'point':point,'bootstrap':chip_bootstrap_delta(rows,label,scores[base],scores[name])}
    return out,scores


def run():
    OUT.mkdir(exist_ok=False);dump(OUT/'protocol.json',protocol())
    snap=OUT/'source_snapshots';snap.mkdir()
    for n in ('feedback_increment_core.py','run_feedback_increment.py','test_feedback_increment.py'):shutil.copy2(Path(__file__).with_name(n),snap/n)
    t0=time.perf_counter()
    public=json.loads((SRC/'public_features.json').read_text());private=json.loads((SRC/'private_tile_scores.json').read_text())
    routes=json.loads((SRC/'routes.json').read_text())
    schemes={'route':route_prefixes(routes),'random':random_prefixes(list(private))}
    results={};table=[]
    for scheme,states in schemes.items():
        rows=build_rows(states,public,private);perm_rows=build_rows(states,public,private,donors=permuted_donors(states))
        assert [(r['state'],r['tile']) for r in rows]==[(r['state'],r['tile']) for r in perm_rows]
        for label in ('y_tp','y_ref'):
            for split in ('event','chip'):
                res,_=evaluate(rows,perm_rows,label,split)
                res.update({'states':len(states),'rows':len(rows),'chips':len({r['sample_id'] for r in rows})})
                results[f'{scheme}|{label}|{split}']=res
                for name in SCORERS:
                    p=res['pooled'][name];nx=res['next_pick'][name]
                    table.append({'scheme':scheme,'label':label,'split':split,'scorer':name,**{k:p[k] for k in ('n','positives','auroc','average_precision','log_loss','brier')},
                                  'next_pick_rate':nx['rate'],'opportunity_states':nx['opportunity_states']})
    prim=results['route|y_tp|event']['delta_vs_prior']
    def ci(name,metric):
        b=prim[name]['bootstrap'][metric];return b['ci95'] if b else None
    ap=ci('prior_feedback','average_precision');ll=ci('prior_feedback','log_loss');pap=ci('prior_permuted_feedback','average_precision')
    supported=bool(ap and ll and pap and ap[0]>0 and ll[1]<0 and pap[0]<=0)
    summary={'primary_cell':'route|y_tp|event','primary':{k:results['route|y_tp|event'][k] for k in ('states','rows','chips')},
             'primary_pooled':{n:results['route|y_tp|event']['pooled'][n] for n in SCORERS},
             'primary_next_pick':results['route|y_tp|event']['next_pick'],'primary_deltas':prim,
             'decision':'SUPPORTED' if supported else 'NOT_ESTABLISHED','elapsed_seconds':time.perf_counter()-t0,
             'scope':'Development diagnostic on already-used36;not validation,not a routing gain,not a new method.'}
    dump(OUT/'results.json',results);dump(OUT/'summary.json',summary);dump(OUT/'table.json',table)
    print(json.dumps({'decision':summary['decision'],'primary':summary['primary'],'deltas':prim},ensure_ascii=False,indent=1))


if __name__=='__main__':run()
