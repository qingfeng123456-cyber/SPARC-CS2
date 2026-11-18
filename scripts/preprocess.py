"""Rebuild derived archives from independently acquired, hash-verified native DEMOs."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('sparc_original_parser',ROOT/'source/portable/cs2ai/data.py')
parser=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(parser)
KEEP={'x','pub','time','tick','cutoff_valid','target_bin','target_mask','events','event_seconds','event_time','ingress_first_time'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,default=ROOT/'data/raw_demo_manifest.csv')
    ap.add_argument('--demos',type=Path,required=True,help='Root containing manifest file_name paths or flattened basenames')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--matches',nargs='*',help='Optional public match IDs; default all 51')
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(args.manifest.open(encoding='utf-8-sig',newline='')))
    if args.matches:
        rows=[r for r in rows if str(r['match_id']) in args.matches]
        missing=set(args.matches)-{str(r['match_id']) for r in rows}
        if missing:
            raise ValueError(f'Unknown match IDs in manifest: {sorted(missing)}')
    index=[]
    for row in rows:
        path=args.demos/row['file_name']
        if not path.exists():
            path=args.demos/Path(row['file_name']).name
        if not path.is_file() or path.stat().st_size!=int(row['bytes']):
            raise ValueError(f'Missing or wrong-sized native DEMO for match {row["match_id"]}')
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for part in iter(lambda:stream.read(1024*1024),b''):
                h.update(part)
        if h.hexdigest()!=row['sha256']:
            raise ValueError(f'Native DEMO hash mismatch for match {row["match_id"]}')
        dest=args.output/'matches'/f'{row["match_id"]}.npz'
        if dest.exists():
            raise FileExistsError(f'Choose a fresh output directory: {dest}')
        report=parser._parse_match(path,row,dest)
        # Keep the public snapshot's exact anonymous numeric schema.
        with np.load(dest,allow_pickle=False) as a:
            metadata=json.loads(a['metadata'].item())
            blocked=('steam','player','roster','path','local')
            clean=[{k:v for k,v in m.items() if not any(s in k.lower() for s in blocked)} for m in metadata]
            arrays={k:a[k] for k in a.files if k!='metadata' and k.split('_',1)[-1] in KEEP}
            arrays['metadata']=json.dumps(clean)
        np.savez_compressed(dest,**arrays)
        index.append(dict(match_id=str(row['match_id']),role=report['role'],path='matches/'+dest.name,rounds=report['rounds']))
        print(row['match_id'],report['rounds'],'rounds',flush=True)
    parser.save_json(args.output/'index.json',index)


if __name__=='__main__':
    main()
