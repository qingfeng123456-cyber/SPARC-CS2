"""Filtered derived matches, labels and the fixed scoring eligibility contract."""
from pathlib import Path
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
READ_KEYS = ('x','pub','time','tick','cutoff_valid','target_bin','target_mask',
             'events','event_seconds','event_time','ingress_first_time')
_spec = importlib.util.spec_from_file_location('_sparc_parser', ROOT/'source/portable/cs2ai/data.py')
_parser = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parser)
scoring_eligibility = _parser.scoring_eligibility


def load_rounds(role='validation', root=None):
    root = Path(root or ROOT/'data')
    index = json.loads((root/'index.json').read_text(encoding='utf-8'))
    result = []
    for item in index:
        if item['role'] != role:
            continue
        with np.load(root/item['path'], allow_pickle=False) as a:
            for i, meta in enumerate(json.loads(a['metadata'].item())):
                if meta['role'] != role or str(meta['match_id']) != str(item['match_id']):
                    raise ValueError('Index/round role or match mismatch')
                r = {**meta, **{k:a[f'r{i}_{k}'] for k in READ_KEYS}}
                if scoring_eligibility(r)['eligible']:
                    result.append(r)
    return sorted(result, key=lambda r:(str(r['match_id']),str(r['round_uid'])))


def statistics(root=None):
    from .helpers import internal_split
    output = {}
    train = None
    for role in ('train','validation','test'):
        rr = load_rounds(role, root)
        if role == 'train':
            train = rr
        output[role] = _counts(rr)
    for name, rr in zip(('inner_fit','inner_val'), internal_split(train)):
        output[name] = _counts(rr)
    return output


def _counts(rr):
    return dict(series=len({str(r['match_id']) for r in rr}),rounds=len(rr),
                cutoffs=sum(scoring_eligibility(r)['cutoffs'] for r in rr))
