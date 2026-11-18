"""Small checks of the endpoint, data protocol, inference isolation and probability law."""
import unittest
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from sparc_cs2 import rules
from sparc_cs2.data import load_rounds, statistics
from sparc_cs2.features import frame_indices
from sparc_cs2.inference import Predictor, VARIANTS


class ContractTests(unittest.TestCase):
    def test_fixed_split_counts(self):
        s=statistics()
        self.assertEqual([s[k]['series'] for k in ('train','validation','test','inner_fit','inner_val')],[30,10,11,22,8])
        self.assertEqual([s[k]['cutoffs'] for k in ('train','validation','test')],[56704,18210,19655])

    def test_full_law_and_label_independence(self):
        torch.set_num_threads(1)
        r=load_rounds('validation')[0]
        ix=frame_indices(r)[::max(1,len(frame_indices(r))//4)][:4]
        only_public={k:r[k] for k in ('x','pub','time','cutoff_valid','match_id','round_uid')}
        with threadpool_limits(limits=2):
            model=Predictor()
            for variant in VARIANTS:
                p,t=model.predict_round(only_public,ix,variant)
                self.assertEqual(p.shape,(len(ix),20,511))
                self.assertTrue(np.all(p>=0) and np.all(t>=0))
                np.testing.assert_allclose(p.sum((1,2))+t,1.,atol=3e-6)
            p,t=model.predict_round(r,ix)
            q,u=model.predict_round(only_public,ix)
            np.testing.assert_array_equal(p,q); np.testing.assert_array_equal(t,u)

    def test_projection_conserves_mass(self):
        rng=np.random.default_rng(7)
        p=rng.random((3,20,511)); t=rng.random(3)
        total=p.sum((1,2))+t
        p/=total[:,None,None]; t/=total
        x=np.zeros((3,30),dtype='f4'); x[:,1:3]=1; x[:,0]=np.array([114,100,5])/120
        clock=[(np.array([0.]),np.array([1])),(np.array([0.]),np.array([1]))]
        q,u=rules.project(p,t,x,clock)
        np.testing.assert_allclose(q.sum((1,2))+u,1.,atol=1e-12)
        self.assertTrue(np.all(q>0))


if __name__=='__main__':
    unittest.main()
