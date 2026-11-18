"""Frozen 105/80-iteration grouped risk-set boosting and conditional mark law."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from .features import frame_indices, public_features
H=20
SINGLE = [0,1,2,4,8,16,32,64,128]
MASK_CLASS = np.full(256,9)
MASK_CLASS[SINGLE] = np.arange(9)
AGE_GROUP = np.array([0,0,1,1,2,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4])
CENTRES = np.array([.5,2.5,6.5,12.,17.])

def public_state(r,ids):
    u, x, time = r["pub"],r["x"],r["time"]
    p=u[ids,:30]
    players=x[ids]
    ct, foe = players[:,:5], players[:,5:]
    present = ct[:,:,31]>.5
    seen=(foe[:,:,31]>.5) & (foe[:,:,4]>.5)
    known=foe[:,:,31]>.5
    def masked_min(vals, present, fallback=4.):
        return np.where(present,vals,fallback).min(1)
    def mean(z):
        return (z*present).sum(1)/np.maximum(present.sum(1),1)
    extra=[mean(np.linalg.norm(ct[:,:,13:15],axis=-1)), mean(ct[:,:,10]),
           masked_min(ct[:,:,5],present,1.),masked_min(foe[:,:,5],known,4.)]
    for center in ([1175,2550],[-1575,2725]):
        c=np.asarray(center)/4096
        for unit,mask in ((ct,present),(foe,seen),(foe,known)):
            extra.append(masked_min(np.linalg.norm(unit[:,:,:2]-c,axis=-1),mask))
    for dt in (2.,6.):
        prev=np.maximum(0,np.searchsorted(time,time[ids]-dt,side="right")-1)
        extra.extend((p[:,18]-u[prev,18],p[:,15]-u[prev,15],p[:,19]-u[prev,19],
                      np.linalg.norm(p[:,8:10]-u[prev,8:10],axis=1)))
    return np.ascontiguousarray(np.column_stack([p,public_features(p,players),*extra]),dtype=np.float32)


def flatten(rounds):
    ids=[frame_indices(r) for r in rounds]
    sid=np.array([str(r["match_id"]) for r in rounds]); counts=np.array([len(i) for i in ids])
    sw=np.array([1./(len(set(sid))*(sid==s).sum()*n) for s,n in zip(sid,counts)])
    return dict(X=np.concatenate([public_state(r,i) for r,i in zip(rounds,ids)]),
                b=np.concatenate([r["target_bin"][i] for r,i in zip(rounds,ids)]).astype(int),
                m=np.concatenate([r["target_mask"][i] for r,i in zip(rounds,ids)]).astype(int),
                w=np.repeat(sw*counts.sum(),counts), row_series=np.repeat(sid,counts),
                series_ids=sid,round_ids=np.array([str(r["round_uid"]) for r in rounds]),counts=counts,
                cutoff_index=np.concatenate(ids))


def clocks(x,dt):
    dt=np.broadcast_to(dt,(len(x),))
    t=x[:,0]*120+dt
    plant=x[:,3]>.5
    elapsed_bomb=x[:,4]*45+dt
    until=np.where(plant,40-elapsed_bomb,115-t)
    return np.column_stack((x,dt,t,np.where(plant,elapsed_bomb,0),until,
                            x[:,5]*30+dt,x[:,6]*30+dt)).astype("float32")


def booster(minleaf,iterations,leaves):
    return HistGradientBoostingClassifier(max_iter=iterations,learning_rate=.07,max_leaf_nodes=leaves,
                min_samples_leaf=minleaf,l2_regularization=5,max_bins=127,early_stopping=False,random_state=2041)


def terminal_cell(x):
    return ((x[:,3]>.5).astype(int)*4 + (x[:,1]<.3).astype(int)*2 + (x[:,2]<.3).astype(int))


class RiskLaw:
    def fit(self,t,which):
        x,b,m,w=[t[k][which] for k in ("X","b","m","w")]
        ix,j=np.nonzero(np.arange(H)[None,:]<=b[:,None])
        hit=j==b[ix]
        rng=np.random.default_rng(2144)
        keep=hit | (rng.random(len(ix))<.25)
        ix,j,hit=ix[keep],j[keep],hit[keep]
        self.hazard=booster(90,105,15).fit(clocks(x[ix],j+.5),hit,sample_weight=w[ix]*np.where(hit,1.,4.))
        eligible=b<H
        x,b,m,w=x[eligible],b[eligible],m[eligible],w[eligible]
        primary=m & 255
        coarse=MASK_CLASS[primary]
        self.types=booster(95,80,12).fit(clocks(x,CENTRES[AGE_GROUP[b]]),coarse,sample_weight=w)
        pc=np.bincount(primary,weights=w,minlength=256)+.0005
        self.classprior=np.bincount(MASK_CLASS,weights=pc,minlength=10)
        self.within=pc/self.classprior[MASK_CLASS]
        self.classprior/=self.classprior.sum()
        self.ingress=np.stack([np.bincount(MASK_CLASS,weights=self.within*((np.arange(256)&bit)!=0),minlength=10)
                               for bit in (1,2)],axis=1)
        cells=terminal_cell(x)
        hit=m>=256
        denom=np.bincount(coarse,minlength=10,weights=w)
        marginal=(np.bincount(coarse,minlength=10,weights=w*hit)+.2)/(denom+1.)
        counts=np.bincount(cells*10+coarse,weights=w,minlength=80).reshape(8,10)
        positive=np.bincount(cells*10+coarse,weights=w*hit,minlength=80).reshape(8,10)
        self.terminal=(positive+3*marginal[None,:])/(counts+3.)
        self.terminal[:,0]=1.
        return self

    def forecast(self,x):
        n=len(x)
        xi=np.repeat(x,H,axis=0)
        hazard=self.hazard.predict_proba(clocks(xi,np.tile(np.arange(H)+.5,n)))[:,1].reshape(n,H)
        hazard=np.clip(hazard,.0002,.9998)
        survival=np.cumprod(1-hazard,axis=1)
        mass=np.column_stack((np.column_stack((np.ones(n),survival[:,:-1]))*hazard,survival[:,-1]))
        xx=np.repeat(x,len(CENTRES),axis=0)
        classp=self.types.predict_proba(clocks(xx,np.tile(CENTRES,n)))
        classes=np.broadcast_to(.005*self.classprior, (len(xx),10)).copy()
        classes[:,self.types.classes_]+=.995*classp
        classes=classes.reshape(n,len(CENTRES),10)
        return mass,classes,self.terminal[terminal_cell(x)]

    def score(self, x,b,m):
        mass,classes,terminal=self.forecast(x)
        idx=np.arange(len(x))
        pm=m&255
        c=MASK_CLASS[pm]
        time_p=mass[idx,b]
        mark_p=classes[idx,AGE_GROUP[np.minimum(b,H-1)],c]*self.within[pm]*np.where(m>=256,terminal[idx,c],1-terminal[idx,c])
        joint=time_p*np.where(b==H,1.,mark_p)
        ab=(mass[:,:10,None]*(classes[:,AGE_GROUP[:10]]@self.ingress)).sum(1)
        np.testing.assert_allclose(mass.sum(1),1.,atol=1e-10)
        np.testing.assert_allclose(classes.sum(-1),1.,atol=1e-6)
        assert np.all(joint>0) and np.all(joint<=time_p+1e-8)
        return joint,time_p,ab

