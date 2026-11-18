"""Frozen NN10 architecture, features and exact loss; 55,577 parameters."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .helpers import round_weights
from .features import public_features, frame_indices
MASKS = np.array([1,2,4,8,16,32,64,128,256,64+256,128+256])
NCAUSE = len(MASKS)+1
SEEDS=(2026,2027,2028)

def features(r):
    u, players, time = r["pub"][:,:30], r["x"], r["time"]
    obs = public_features(u, players)
    ct, enemy = players[:,:5,:], players[:,5:,:]
    # Symmetric reductions avoid identifying a particular roster slot.
    alive = ct[:,:,31]
    coverage = np.concatenate([ct[:,:,i:i+1]*alive[:,:,None] for i in (5,7,10,11,12)], axis=-1)
    motion = np.linalg.norm(ct[:,:,13:15],axis=-1)*alive
    spread = np.linalg.norm(ct[:,:,:2] - u[:,None,8:10],axis=-1)*alive
    close = np.linalg.norm(ct[:,:,None,:2] - enemy[:,None,:,:2],axis=-1)
    valid = (alive[:,:,None]>.5)&(enemy[:,None,:,31]>.5)
    fresh = np.exp(-enemy[:,:,5]*30/5)
    pressure = (np.exp(-close*5)*valid*fresh[:,None,:]).max(axis=-1)
    base = np.column_stack((u,obs,coverage.sum(1),coverage.max(1),motion.mean(1),motion.max(1),
                            spread.mean(1),spread.max(1),pressure.mean(1),pressure.max(1)))
    histories = []
    for age in (1.5,5.,12.):
        earlier = np.maximum(0,np.searchsorted(time,time-age,side="right")-1)
        histories.append(base - base[earlier])
    return np.ascontiguousarray(np.column_stack([base,*histories]),dtype=np.float32)


def materialize(rounds):
    groups, ids, states, events, marks, times = [], [], [], [], [], []
    for r in rounds:
        take = frame_indices(r)
        states.append(features(r)[take])
        ids.append(take)
        events.append(r["target_bin"][take])
        marks.append(r["target_mask"][take])
        times.append(r["time"][take])
        groups.append((str(r["match_id"]),str(r["round_uid"]),len(take)))
    sids,rids,counts = map(np.asarray, zip(*groups))
    return dict(x=np.concatenate(states), bins=np.concatenate(events),marks=np.concatenate(marks),
                round_ids=rids,series_ids=sids,counts=counts,cutoff_index=np.concatenate(ids),
                time=np.concatenate(times))


def escape_law(fit):
    # One catch-all component gives all other nonempty simultaneous sets support.
    hist=np.bincount(fit["marks"],minlength=512).astype(float)+1/512
    hist[0]=0
    hist[MASKS]=0
    return hist/hist.sum()


class Decoder(nn.Module):
    def __init__(self, dimensions, structured):
        super().__init__()
        self.structured=structured
        self.state=nn.Sequential(nn.Linear(dimensions,128),nn.LayerNorm(128),nn.SiLU(),
                                 nn.Dropout(.10),nn.Linear(128,96),nn.SiLU())
        self.lead=nn.Sequential(nn.Linear(13,48),nn.SiLU(),nn.Linear(48,96))
        self.out=nn.Sequential(nn.LayerNorm(96),nn.SiLU(),nn.Linear(96,NCAUSE+1))
        self.strength=nn.Linear(96,NCAUSE)
        self.register_buffer("step",torch.arange(20,dtype=torch.float32)+.5)
        nn.init.zeros_(self.strength.weight)
        nn.init.zeros_(self.strength.bias)
        with torch.no_grad():
            self.out[-1].bias[0]=2.5

    def forward(self,x):
        h=self.state(x)
        j=self.step.expand(len(x),-1)
        dt=j if self.structured else torch.zeros_like(j)
        elapsed=120*x[:,0:1]+dt
        planted=x[:,3:4]
        age=45*x[:,4:5]+dt
        d_t=30*x[:,5:6]+dt
        d_ct=30*x[:,6:7]+dt
        close=x[:,18:19]*torch.exp(-dt/3)
        def radial(z,mu,width):
            return torch.exp(-.5*((z-mu)/width).square())
        lead=torch.stack((j/20,torch.log1p(j)/3,torch.exp(-j/3),elapsed/120,
                          (1-planted)*radial(elapsed,115,1.5),(1-planted)*radial(elapsed,115,6),
                          planted*radial(age,40,1.5),planted*radial(age,40,6),planted*age/40,
                          torch.exp(-d_t/3),torch.exp(-d_ct/3),close,torch.exp(-elapsed/12)),dim=-1)
        logits=self.out(h[:,None,:]+self.lead(lead))
        last_t=(x[:,2:3]<.3)&(x[:,2:3]>.1)
        last_ct=(x[:,1:2]<.3)&(x[:,1:2]>.1)
        inc=F.softplus(self.strength(h))
        if self.structured:
            # Before the first future event, neither plant status nor living counts
            # can change via another scored event. Use finite odds adjustments.
            t_end=last_t.to(x.dtype)*(1-planted)
            ct_end=last_ct.to(x.dtype)
            zero=torch.zeros_like(planted)
            impossible=torch.cat([zero]*4+[planted]*2+[t_end,ct_end,zero,1-t_end,1-ct_end,zero],dim=-1)
            shift=-inc*impossible
        else:
            shift=-inc
        logp=torch.cat((logits[:,:,:1],logits[:,:,1:]+shift[:,None,:]),dim=-1).log_softmax(-1)
        surv=logp[:,:,0].cumsum(1)
        start=F.pad(surv[:,:-1],(1,0))
        return start[:,:,None]+logp[:,:,1:],surv[:,-1]


def score(mass,tail,bins,classes,extra):
    pos=torch.arange(len(bins),device=bins.device)
    logp=mass[pos,bins.clamp_max(19),classes]+extra
    return -torch.where(bins==20,tail,logp)

