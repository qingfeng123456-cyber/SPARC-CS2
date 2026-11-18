"""Complete frozen SPARC-CS2 law and component-deletion evaluations."""
from pathlib import Path
import csv
import json
import sys
import joblib
import numpy as np
import torch
from . import neural, risk, site, rules
from .data import ROOT, load_rounds
from .features import frame_indices

SEEDS = (2026,2027,2028)
VARIANTS = ('NEURAL','TREE','BASE_MIX','SITE','RULES','FULL','SHUFFLE_RULES')
METRICS = ('joint_nll','time_nll','brier_a10','brier_b10','brier_ab10')


def load_legacy_risk(path):
    """Resolve the historical __main__.RiskLaw symbol and restore caller state."""
    main = sys.modules['__main__']
    had = hasattr(main, 'RiskLaw')
    previous = getattr(main, 'RiskLaw', None)
    main.RiskLaw = risk.RiskLaw
    try:
        return joblib.load(path)
    finally:
        if had:
            main.RiskLaw = previous
        else:
            delattr(main, 'RiskLaw')


def neural_law(model, escape, x):
    with torch.inference_mode():
        logp, logtail = model(torch.from_numpy(x))
    pc = logp.double().exp().numpy()
    p = pc[:, :, -1:] * escape[None,None,1:]
    p[:, :, neural.MASKS-1] = pc[:, :, :-1]
    return p, logtail.double().exp().numpy()


def tree_law(tree, x):
    times, types, term = tree.forecast(x)
    primary = rules.MARKS & 255
    cl = risk.MASK_CLASS[primary]
    by_mark = types[:,:,cl] * tree.within[primary]
    by_mark *= np.where(rules.MARKS[None,:] >= 256, term[:,cl], 1-term[:,cl])[:,None,:]
    return times[:,:-1,None] * by_mark[:,risk.AGE_GROUP], times[:,-1]


def site_project(p, guess):
    t = p.sum(2)
    singleton = p[:,:,site.BITS-1]
    rest = t-singleton.sum(-1)
    proposed = guess[:,site.BANDS]*t[:,:,None]
    if not np.all(rest > 0):
        raise ValueError('Positive residual support required')
    q = p*(.5+.5*proposed[:,:,4]/rest)[:,:,None]
    q[:,:,site.BITS-1] = .5*(singleton+proposed[:,:,:4])
    return q


def measure(p, tail, bins, marks):
    pos = np.arange(len(p)); event = bins < 20; j = bins.clip(max=19)
    time_mass = p.sum(2)
    joint = np.where(event,p[pos,j,np.maximum(marks.astype(int)-1,0)],tail)
    tp = np.where(event,time_mass[pos,j],tail)
    integrated = p[:,:10].sum(1)
    ab = np.stack([integrated[:,(rules.MARKS & bit)!=0].sum(1) for bit in (1,2)],axis=1)
    np.testing.assert_allclose(tail+time_mass.sum(1),1.,atol=3e-6)
    if not np.all(joint>0):
        raise ValueError('Observed event has zero support')
    return -np.log(joint), -np.log(tp), ab


def macro(loss, temporal, ab, data):
    y = (data['bins'][:,None]<10) & ((data['marks'][:,None]&[1,2])!=0)
    brier = np.square(ab-y)
    rows = np.column_stack((loss,temporal,brier,brier.mean(1)))
    n, ss = data['counts'], data['series_ids']
    avg = np.add.reduceat(rows,np.r_[0,n.cumsum()[:-1]],axis=0)/n[:,None]
    per = {str(s):avg[ss==s].mean(0).tolist() for s in sorted(set(ss))}
    return dict(mean=dict(zip(METRICS,np.mean(list(per.values()),axis=0).tolist())),
                per_series={s:dict(zip(METRICS,v)) for s,v in per.items()})


class Predictor:
    """One independent seed's complete hybrid; no fitting and no labels as inputs."""
    def __init__(self, seed=2026, weights=None):
        self.weights = Path(weights or ROOT/'weights/main')
        saved = torch.load(self.weights/f'neural_seed{seed}.pt',map_location='cpu',weights_only=True)
        self.net = neural.Decoder(272,True)
        self.net.load_state_dict(saved['model']); self.net.eval()
        self.escape = np.asarray(saved['escape'])
        self.tree = load_legacy_risk(self.weights/'risk.joblib')
        self.site = joblib.load(self.weights/'site.joblib')
        self.shuffle = joblib.load(self.weights/'shuffle.joblib') if (self.weights/'shuffle.joblib').exists() else None
        clock = json.loads((self.weights/'clock.json').read_text(encoding='utf-8'))
        self.clock = [(np.array(v['offsets']),np.array(v['counts'])) for v in clock['offsets']]
        self.training_ids = set(map(str,clock['train']))

    def predict_features(self, x, risk_x, site_x, site_z, variant='FULL', flip=None):
        if variant not in VARIANTS:
            raise ValueError(variant)
        nn, nt = neural_law(self.net,self.escape,x)
        if variant == 'NEURAL':
            return nn, nt
        tp, tt = tree_law(self.tree,risk_x)
        if variant == 'TREE':
            return tp,tt
        p,t = .5*(nn+tp),.5*(nt+tt)
        if variant == 'BASE_MIX':
            return p,t
        if variant in ('SITE','FULL','SHUFFLE_RULES'):
            models = self.site
            if variant == 'SHUFFLE_RULES':
                if self.shuffle is None or flip is None:
                    raise ValueError('SHUFFLE_RULES requires its trained control and fixed per-round flip')
                site_z = site_z.copy(); site_z[flip] = site_z[flip,::-1]
                models = self.shuffle
            p = site_project(p,site.group_pred(models,site_x,site_z))
        if variant == 'SITE':
            return p,t
        return rules.project(p,t,x,self.clock,True,True)

    def predict_round(self, r, indices=None, variant='FULL'):
        # An unlabeled live round can provide explicit cutoff indices. Features never read labels.
        ix = np.flatnonzero(r['cutoff_valid']) if indices is None else np.asarray(indices,dtype=int)
        sx,sz = site.features(r,ix)
        import hashlib
        payload=f"site-control-47/{r['match_id']}/{r['round_uid']}".encode()
        flip=np.full(len(ix),bool(hashlib.sha256(payload).digest()[0]&1))
        return self.predict_features(neural.features(r)[ix],risk.public_state(r,ix),sx,sz,variant,flip)


def evaluate(role='validation', output=None, seeds=SEEDS, weights=None, data_root=None,
             variants=VARIANTS, batch=80, max_rounds=None):
    output=Path(output or ROOT/'outputs'/dict(validation='dev',test='rio',internal='internal')[role])
    if (output/'summary.json').exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    if role=='internal':
        from .helpers import internal_split
        _,rr=internal_split(load_rounds('train',data_root))
    else:
        rr=load_rounds(role,data_root)
    if max_rounds is not None:
        rr=rr[:max_rounds]
    d,s,rx=neural.materialize(rr),site.materialize(rr),risk.flatten(rr)
    meta={k:v for k,v in d.items() if k!='x'}
    meta.update(event_time=np.concatenate([r['event_time'][frame_indices(r)] for r in rr]),
                ingress_first_time=np.concatenate([r['ingress_first_time'][frame_indices(r)] for r in rr]))
    scores={}
    for seed in seeds:
        model=Predictor(seed,weights)
        if set(map(str,d['series_ids'])) & model.training_ids:
            raise ValueError('Evaluation cohort overlaps model training matches; use the corresponding held-out weights')
        for variant in variants:
            values=[[],[],[]]; time_masses=[]; tails=[]
            for i in range(0,len(d['bins']),batch):
                sl=slice(i,i+batch)
                law,tail=model.predict_features(d['x'][sl],rx['X'][sl],s['x'][sl],s['z'][sl],variant,s['flip'][sl])
                for out,value in zip(values,measure(law,tail,d['bins'][sl],d['marks'][sl])):
                    out.append(value)
                time_masses.append(law.sum(2)); tails.append(tail)
            ll,tl,ab=[np.concatenate(v) for v in values]
            dest=output/'runs'/f'{variant}_seed{seed}'; dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'cutoffs.npz',**meta,nll=ll,time_nll=tl,probability_a_b_10s=ab,
                                time_mass=np.concatenate(time_masses),tail=np.concatenate(tails))
            score=macro(ll,tl,ab,d); scores[f'{variant}_seed{seed}']=score
            (dest/'metrics.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
            print(f'{role} {variant} seed={seed}: joint NLL={score["mean"]["joint_nll"]:.8f}',flush=True)
    means={v:{m:float(np.mean([scores[f'{v}_seed{s}']['mean'][m] for s in seeds])) for m in METRICS} for v in variants}
    report=dict(model='SPARC-CS2',role=role,seeds=list(seeds),variants=list(variants),mean=means,by_seed=scores,
                partial_run=max_rounds is not None,series=len(set(d['series_ids'])),rounds=len(rr),cutoffs=len(d['bins']))
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (output/'series_scores.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['variant','seed','series',*METRICS])
        for v in variants:
            for seed in seeds:
                for sid,values in scores[f'{v}_seed{seed}']['per_series'].items():
                    w.writerow([v,seed,sid,*[values[m] for m in METRICS]])
    return report
