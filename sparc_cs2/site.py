"""SITE80 symmetric spatial attribution; three 80-iteration binary models."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from . import neural, risk
HERE = Path(__file__).resolve().parent
CENTRES = np.array([.5, 2.5, 6.5, 12., 17.])
BITS = np.array([1,2,16,32])
BANDS = risk.AGE_GROUP
GEO = json.loads((HERE/'geometry.json').read_text(encoding='utf-8'))

def site_summary(x, side):
    ct, enemy = x[:,:5], x[:,5:]
    alive, known = ct[...,31], enemy[...,31]
    fresh = np.exp(-enemy[...,5]*30/6) * known
    cols = []
    for region in ("site", "stage"):
        box = np.asarray(GEO[region+side]).reshape(3,2)
        center = box[:2].mean(1)
        for team, mask, is_ct in ((ct,alive,True),(enemy,known,False)):
            xy = team[...,:2]*4096.
            toward = center - xy
            dist = np.linalg.norm(toward,axis=-1)
            loc = np.exp(-dist/800)*mask
            inside = (((xy>=box[:2,0]) & (xy<=box[:2,1])).all(-1))*mask
            cols += [loc.max(1),loc.sum(1),np.sort(loc,axis=1)[:,-2],inside.sum(1)]
            if is_ct:
                unit = toward / np.maximum(dist[...,None],1.)
                facing = unit[...,0]*ct[...,9] + unit[...,1]*ct[...,8]
                speed = (unit*ct[...,13:15]).sum(-1)
                cols += [(loc*np.maximum(0.,facing)).sum(1),(loc*speed).sum(1),
                         (loc*ct[...,5]).sum(1),(loc*ct[...,10]).sum(1)]
            else:
                cols += [(loc*fresh).max(1),(loc*fresh).sum(1),(loc*enemy[...,4]).sum(1),
                         (fresh*inside).sum(1)]
    return np.column_stack(cols)


def features(r, ix):
    x, p, t = r["x"], r["pub"], r["time"]
    now = np.stack([site_summary(x,s) for s in ("A","B")],axis=1)
    prior = []
    for delay in (3.,10.):
        j = np.maximum(0,np.searchsorted(t,t[ix]-delay,side="right")-1)
        prior.append(now[ix]-now[j])
    side = np.concatenate([now[ix],*prior],axis=2).astype("f4")
    generic = p[ix][:,[0,1,2,3,4,5,6,14,15,16,17,18,19]]
    return generic,side


def materialize(rounds):
    xx,zz,bb,mm,flips=[],[],[],[],[]
    ids=[neural.frame_indices(r) for r in rounds]
    series=np.array([str(r["match_id"]) for r in rounds])
    counts=np.array([len(k) for k in ids])
    for r,ix in zip(rounds,ids):
        x,z=features(r,ix)
        xx.append(x); zz.append(z)
        bb.append(r["target_bin"][ix]); mm.append(r["target_mask"][ix])
        payload=f"site-control-47/{r['match_id']}/{r['round_uid']}".encode()
        flips.append(np.full(len(ix),bool(hashlib.sha256(payload).digest()[0]&1)))
    return dict(x=np.concatenate(xx),z=np.concatenate(zz),b=np.concatenate(bb),m=np.concatenate(mm),
                counts=counts, series_ids=series,round_ids=np.array([str(r["round_uid"]) for r in rounds]),
                cutoff_index=np.concatenate(ids),flip=np.concatenate(flips),
                w=neural.round_weights(counts,series)*sum(counts))


def booster(x,y,w):
    return HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=9,min_samples_leaf=48,
                l2_regularization=5.,learning_rate=.06,max_bins=127,early_stopping=False,
                random_state=241).fit(x,y,sample_weight=w)


def prepare(x,z,step):
    base=np.column_stack((x,step))
    invariant=np.column_stack((base,z.mean(1),np.abs(z[:,0]-z[:,1])))
    orient=[np.column_stack((base,np.full(len(x),i),z[:,i],z[:,i]-z[:,1-i])) for i in range(2)]
    return invariant,orient


def fitted(t,shuffled):
    ix=t["b"]<20
    z=t["z"][ix].copy()
    if shuffled:
        flip=t["flip"][ix]
        z[flip]=z[flip,::-1]
    inv,ori=prepare(t["x"][ix],z,CENTRES[BANDS[t["b"][ix]]])
    m,w=t["m"][ix],t["w"][ix]
    use=np.isin(m,BITS)
    side_b=(m==2)|(m==32)
    gate=booster(inv,use,w)
    where=booster(np.concatenate([a[use] for a in ori]),np.r_[~side_b[use],side_b[use]],np.tile(w[use]*.5,2))
    picked=np.where(side_b[:,None],ori[1],ori[0])
    what=booster(picked[use],m[use]>=16,w[use])
    return gate,where,what


def group_pred(models,x,z):
    g,d,k=models
    result=[]
    for c in CENTRES:
        v,ori=prepare(x,z,np.full(len(z),c))
        active=g.predict_proba(v)[:,1]
        a=.5*(d.predict_proba(ori[0])[:,1]+d.predict_proba(ori[1])[:,0])
        plant_a=k.predict_proba(ori[0])[:,1]
        plant_b=k.predict_proba(ori[1])[:,1]
        pred=np.column_stack((active*a*(1-plant_a),active*(1-a)*(1-plant_b),
                              active*a*plant_a,active*(1-a)*plant_b,1-active))
        np.testing.assert_allclose(pred.sum(1),1.)
        result.append(pred)
    return np.stack(result,axis=1)

