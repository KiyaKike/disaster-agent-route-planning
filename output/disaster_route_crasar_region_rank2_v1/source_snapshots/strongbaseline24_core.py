"""Source-faithful Strong Baseline adapter, without training-time downloads."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

FLIPS=((),(2,),(3,),(2,3))
SOURCE_SHA='ad656c3c78f490fe32a6a67ccaa9d705d02bb58bfedaf0b6efae04723bf663fe'


def prepare_pair(pre_rgb,post_rgb):
    arrays=[np.asarray(im) for im in (pre_rgb,post_rgb)]
    if any(a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 for a in arrays) or arrays[0].shape!=arrays[1].shape:
        raise ValueError('Same-size uint8 RGB pair required')
    joined=np.concatenate([a[:,:,::-1] for a in arrays],axis=2).astype(np.float32)
    joined/=127;joined-=1
    return joined.transpose(2,0,1)[None].copy()


def decode_probabilities(probs):
    if probs.dtype!=np.float32 or probs.ndim!=3 or probs.shape[0]!=5 or not np.isfinite(probs).all() or probs.min()<0 or probs.max()>1:
        raise ValueError('Five finite float32 probability channels required')
    localization=probs[0]>0.38
    damage=(np.argmax(probs[1:],axis=0)+1).astype(np.uint8)
    return localization,damage,damage*localization.astype(np.uint8)


def restore_probability(probability,flip):
    if flip not in FLIPS:raise ValueError('Unplanned flip')
    if probability.dtype!=np.float32 or probability.ndim!=3 or probability.shape[0]!=5:raise ValueError('Five-channel CHW probabilities required')
    return np.flip(probability,axis=tuple(v-1 for v in flip)).copy() if flip else probability.copy()


def source_namespace(path):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SOURCE_SHA:raise ValueError('Pinned model source changed')
    tree=ast.parse(path.read_text());removed=0;changed=0
    body=[]
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module=='utils.experiment_manager':removed+=1
        else:body.append(node)
    tree.body=body
    for node in ast.walk(tree):
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id=='pretrained':
            if not isinstance(node.value,ast.Constant) or node.value.value is not True:raise ValueError('Unexpected encoder constructor')
            node.value=ast.Constant(False);changed+=1
    if (removed,changed)!=(1,1):raise ValueError('Unrecognized source adapter change')
    ns={'CfgNode':object,'__name__':'pinned_strongbaseline_source'}
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),ns)
    return ns


def load_network(source,checkpoint):
    import torch
    ns=source_namespace(source)
    cfg=SimpleNamespace(MODEL=SimpleNamespace(OUT_CHANNELS=5))
    net=ns['StrongBaselineNet'](cfg)
    bundle=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if not isinstance(bundle,dict) or 'network' not in bundle:raise ValueError('Expected author checkpoint network entry')
    state=bundle['network']
    if not state or not all(k.startswith('module.') for k in state):raise ValueError('Expected author DataParallel keys')
    state={k.removeprefix('module.'):v for k,v in state.items()}
    net.load_state_dict(state,strict=True);net.eval()
    info={'checkpoint_keys':sorted(bundle),'epoch':int(bundle['epoch']),'strict_state_dict':True,
        'state_tensors':len(state),'parameters':sum(p.numel() for p in net.parameters()),
        'source_patches':['Remove config-class import; supply annotation-only CfgNode=object',
                          'Disable initialization-only ImageNet download; every tensor is strictly loaded from author checkpoint'],
        'checkpoint_loader':'torch.load(weights_only=True,map_location=cpu); no optimizer restored'}
    del bundle,state
    return net,info


def severe_score(truth,prediction):
    if truth.dtype!=np.uint8 or prediction.dtype!=np.uint8 or truth.shape!=prediction.shape:raise ValueError('Matched uint8 rasters required')
    if not set(np.unique(truth))<={0,1,2,3,4,255} or not set(np.unique(prediction))<={0,1,2,3,4}:raise ValueError('Unexpected label code')
    valid=truth!=255;reference=np.isin(truth,[3,4]) & valid;predicted=np.isin(prediction,[3,4]) & valid
    tp=int(np.count_nonzero(reference & predicted));fp=int(np.count_nonzero(~reference & predicted));fn=int(np.count_nonzero(reference & ~predicted))
    return {'tp':tp,'fp':fp,'fn':fn,'reference_severe_pixels':tp+fn,'predicted_severe_pixels':tp+fp,'evaluated_pixels':int(valid.sum()),
        'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
        'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        'severe_positive_scene':bool(tp+fn),'positive_scene_any_correct_pixel':bool(tp)}
