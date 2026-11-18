"""Full fixed-protocol NN10 + risk105/80 + SITE80 training, with no teacher."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import torch
from . import neural, risk, site, rules
from .data import ROOT, load_rounds
from .helpers import internal_split, round_weights


def train_neural(fit, seed, output, device='cpu'):
    """Exact selected STRUCT optimizer, weighting, batching, loss and 10-epoch schedule."""
    torch.manual_seed(seed)
    esc=neural.escape_law(fit)
    table=np.full(512,len(neural.MASKS)); table[neural.MASKS]=np.arange(len(neural.MASKS))
    probs=np.where(table[fit['marks']]==len(neural.MASKS),esc[fit['marks']],1.)
    stats=[torch.as_tensor(v,device=device) for v in (
        fit['x'],fit['bins'].astype('int64'),table[fit['marks']].astype('int64'),
        np.log(probs.clip(1e-30)).astype('f4'),
        round_weights(fit['counts'],fit['series_ids']).astype('f4')*len(probs))]
    net=neural.Decoder(stats[0].shape[1],True).to(device)
    optim=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.015)
    rng=np.random.default_rng(seed); history=[]
    for epoch in range(10):
        net.train(); total=0.
        permutation=rng.permutation(len(fit['bins']))
        for start in range(0,len(permutation),512):
            ids=permutation[start:start+512]
            x,b,k,e,w=[v[ids] for v in stats]
            loss=(neural.score(*net(x),b,k,e)*w).mean()
            optim.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.,error_if_nonfinite=True)
            optim.step(); total+=loss.item()*len(ids)
        history.append(total/len(permutation))
        for group in optim.param_groups:
            group['lr']=.00015+.00085*(1+np.cos((epoch+1)*np.pi/10))/2
        print(f'NN seed={seed} epoch={epoch+1}/10 loss={history[-1]:.6f}',flush=True)
    net.cpu().eval()
    torch.save(dict(model=net.state_dict(),escape=esc.tolist()),output/f'neural_seed{seed}.pt')
    return history


def train(split='main', output=None, seeds=neural.SEEDS, device='auto', data_root=None):
    output=Path(output or ROOT/'outputs'/'trained'/split)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose an empty training output: {output}')
    output.mkdir(parents=True,exist_ok=True)
    train_rounds=load_rounds('train',data_root)
    fit,held=internal_split(train_rounds) if split=='internal' else (train_rounds,load_rounds('validation',data_root))
    ids=sorted({str(r['match_id']) for r in fit})
    if set(ids)&{str(r['match_id']) for r in held}:
        raise ValueError('Overlapping fit/evaluation matches')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    protocol=dict(model='SPARC-CS2',split=split,fit_series=ids,held_series=sorted({str(r['match_id']) for r in held}),
        seeds=list(seeds),neural_epochs=10,device=device,neural_tree_mix=.5,site_mix=.5,projection_escape=.02,
        risk_hazard_iterations=105,risk_type_iterations=80,site_iterations=80,
        fitting_scope='fit matches only; held data never used for optimization or model selection')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    nd=neural.materialize(fit)
    histories={str(seed):train_neural(nd,seed,output,device) for seed in seeds}
    td=risk.flatten(fit)
    tree=risk.RiskLaw().fit(td,np.arange(len(td['b'])))
    joblib.dump(tree,output/'risk.joblib',compress=3)
    sd=site.materialize(fit)
    for name,shuffled in (('site',False),('shuffle',True)):
        joblib.dump(site.fitted(sd,shuffled),output/f'{name}.joblib',compress=3)
    clocks,support=rules.duration_model(fit)
    clock=dict(train=ids,deadline_rounds=support,offsets=[dict(offsets=o.tolist(),counts=c.tolist()) for o,c in clocks])
    (output/'clock.json').write_text(json.dumps(clock,indent=2),encoding='utf-8')
    versions={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    (output/'training_receipt.json').write_text(json.dumps(dict(protocol=protocol,history=histories,sha256=versions),indent=2),encoding='utf-8')
    return protocol
