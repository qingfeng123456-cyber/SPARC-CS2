"""Redraw saved scientific comparisons and mechanisms without training or inference."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/core/plot_data'
LABEL={'NEURAL':'Neural','TREE':'Tree','BASE_MIX':'Base mixture','SITE':'Site',
       'RULES':'Rules','FULL':'SPARC-CS2','SHUFFLE_RULES':'Shuffled site + rules'}


def read(name):
    with (DATA/name).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def save(fig, output, name):
    for ext in ('svg','pdf','png'):
        fig.savefig(output/f'{name}.{ext}',bbox_inches='tight',dpi=220)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/figures')
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    rows=read('paired_effects.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    for ax,cohort in zip(axes,('dev','rio')):
        rr=[r for r in rows if r['cohort']==cohort]
        mid=np.array([float(r['mean']) for r in rr]); lo=np.array([float(r['lo']) for r in rr]); hi=np.array([float(r['hi']) for r in rr])
        ax.errorbar(mid,np.arange(len(rr)),xerr=np.vstack((mid-lo,hi-mid)),fmt='o',color='#087f7a',capsize=3)
        ax.set_yticks(np.arange(len(rr)),[LABEL[r['reference']] for r in rr]); ax.invert_yaxis(); ax.axvline(0,color='grey',lw=.8)
        ax.set(title='Development' if cohort=='dev' else 'Rio',xlabel='Joint NLL reduction relative to reference (%)')
    save(fig,args.output,'paired_effects')
    rows=read('time_mark_decomposition.csv'); lead=read('destination_by_lead.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    colors={'dev':'#216897','rio':'#c77537'}
    labels=[]
    for cohort in ('dev','rio'):
        rr=[r for r in rows if r['cohort']==cohort]
        positions=np.arange(len(rr))+(0 if cohort=='dev' else len(rr)+.7)
        temporal=np.array([float(r['time_gain']) for r in rr]); mark=np.array([float(r['conditional_mark_gain']) for r in rr])
        axes[0].bar(positions,temporal,.65,color='#216897',label='Time' if cohort=='dev' else None)
        axes[0].bar(positions,mark,.65,bottom=temporal,color='#087f7a',label='Conditional mark' if cohort=='dev' else None)
        labels.extend((x,f'{cohort}\n{LABEL[r["variant"]]}') for x,r in zip(positions,rr))
        ll=[r for r in lead if r['cohort']==cohort]
        x=np.arange(len(ll)); y=np.array([float(r['gain']) for r in ll])
        low=np.array([float(r['ci_low']) for r in ll]); high=np.array([float(r['ci_high']) for r in ll])
        axes[1].errorbar(x+(0 if cohort=='dev' else .1),y,yerr=np.vstack((y-low,high-y)),fmt='o-',label=cohort,color=colors[cohort],capsize=3)
    axes[0].set_xticks([x for x,_ in labels],[s for _,s in labels],rotation=25,ha='right'); axes[0].set_ylabel('NLL reduction from base mixture'); axes[0].legend()
    axes[1].set_xticks(range(4),['(0,2]','(2,5]','(5,10]','(10,20]']); axes[1].set(xlabel='Realized future-event lead band (s)',ylabel='Conditional destination log-loss gain'); axes[1].legend()
    save(fig,args.output,'mechanisms')
    with np.load(DATA/'deadline_mass_transfer.npz',allow_pickle=False) as a:
        grids=a['change']
    fig,axes=plt.subplots(1,2,figsize=(10,3.2),layout='constrained',sharey=True)
    limit=float(np.abs(grids).max())*100
    for i,ax in enumerate(axes):
        display=np.column_stack((grids[i,:,:20],np.full((4,1),np.nan),grids[i,:,20]))
        im=ax.pcolormesh(np.arange(23)-.5,np.arange(5)-.5,100*display,cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set(ylim=(3.5,-.5),xticks=[0,4,9,14,19,21],xticklabels=['1','5','10','15','20','Tail'],
               yticks=range(4),yticklabels=['0–5','5–10','10–15','15–20'],xlabel='Future first-event second / tail',title=('Development','Rio')[i])
    axes[0].set_ylabel('Public nominal time remaining (s)')
    fig.colorbar(im,ax=list(axes),label='Probability mass change (percentage points)',shrink=.8)
    save(fig,args.output,'deadline_transfer')
    print(f'Redrew three scientific figures in {args.output}')


if __name__=='__main__':
    main()
