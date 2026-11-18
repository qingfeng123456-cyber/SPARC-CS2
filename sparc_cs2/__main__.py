"""SPARC-CS2 public commands; help never starts fitting."""
import argparse
from pathlib import Path
import json


def main(argv=None):
    p=argparse.ArgumentParser(description='SPARC-CS2: spatial attribution and rule-guided next-event forecasting')
    sub=p.add_subparsers(dest='command',required=True)
    stat=sub.add_parser('data-stats',help='Verify fixed match/round/cutoff counts')
    stat.add_argument('--data',type=Path)
    ev=sub.add_parser('evaluate',help='Score frozen pretrained models; no fitting')
    ev.add_argument('--role',choices=('validation','test','internal'),default='validation')
    ev.add_argument('--output',type=Path,required=True)
    ev.add_argument('--weights',type=Path)
    ev.add_argument('--data',type=Path)
    ev.add_argument('--seeds',nargs='+',type=int,default=[2026,2027,2028])
    ev.add_argument('--variants',nargs='+',choices=('NEURAL','TREE','BASE_MIX','SITE','RULES','FULL','SHUFFLE_RULES'),default=['FULL'])
    ev.add_argument('--batch',type=int,default=80)
    ev.add_argument('--max-rounds',type=int,help='Small local check; output is marked partial')
    tr=sub.add_parser('train',help='Fit the complete selected model with the fixed 10-epoch protocol')
    tr.add_argument('--split',choices=('main','internal'),default='main')
    tr.add_argument('--output',type=Path,required=True)
    tr.add_argument('--data',type=Path)
    tr.add_argument('--seeds',nargs='+',type=int,default=[2026,2027,2028])
    tr.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    args=p.parse_args(argv)
    if args.command=='data-stats':
        from .data import statistics
        print(json.dumps(statistics(args.data),indent=2)); return
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.command=='evaluate':
            from .inference import evaluate
            evaluate(args.role,args.output,args.seeds,args.weights,args.data,args.variants,args.batch,args.max_rounds)
        else:
            from .training import train
            train(args.split,args.output,args.seeds,args.device,args.data)


if __name__=='__main__':
    main()
