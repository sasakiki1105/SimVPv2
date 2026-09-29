import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import radaz_transport_channels as tc
from radaz_metrics_v3 import band_pool, local_observables
import run_radaz_transport_channels as runner
from run_radaz_transport_channels import epoch_order


def contract_fixture():
    manifest=tc.read_json(tc.SOURCE)
    return dict(manifest=manifest,pre=10,aft=10,splits={'train':[0,30],'val':[30,60]},
                gamma_rms={'raw':2e23,'corr':1e23},t10={str(b):list(range(1,11)) for b in range(4)})


def frames_fixture(count=20):
    y=np.arange(256)*2*np.pi/256
    data=np.zeros((count,3,260,256),np.float32)
    data[:,0]=.3 + .05*np.sin(3*y)
    data[:,1]=.3
    data[:,2]=.2 + .01*np.cos(3*y)
    data[:,:,257:]=data[:,:,256:257]
    return data


class TransportDefinitionTests(unittest.TestCase):
    def test_periodic_central_difference_sign_and_full_spectral_identity(self):
        n=64; dy=.01; b=.02; mode=3
        theta=2*np.pi*mode*np.arange(n)/n
        p=np.zeros((1,3,257,n)); p[:,0]=10+2*np.sin(theta); p[:,2]=np.cos(theta)
        ey=np.sin(2*np.pi*mode/n)/dy*np.sin(theta)
        raw=tc.transport_image(p,dy,b,'raw')
        corr=tc.transport_image(p,dy,b,'corr')
        np.testing.assert_allclose(raw,-p[:,0]*ey/b,rtol=1e-12,atol=1e-9)
        np.testing.assert_allclose(raw.mean(-1),corr.mean(-1),rtol=1e-12)
        pool=band_pool(np.arange(257)*.00005)
        obs=local_observables(p[None],pool,dy,b,max_mode=32)
        np.testing.assert_allclose(obs['flux_full'][0],raw.mean((1,2))[:,None]*np.ones((1,4)))
        self.assertTrue(np.all(raw.mean(-1)<0))

    def test_gauge_invariance_and_density_mean_removal(self):
        p=np.random.default_rng(7).normal(size=(2,3,7,32))
        before=tc.transport_image(p,.1,.02,'corr')
        shifted=p.copy(); shifted[:,0]+=20; shifted[:,2]+=100
        np.testing.assert_allclose(before,tc.transport_image(shifted,.1,.02,'corr'),atol=2e-11)

    def test_nyquist_derivative_zero_and_B_scaling(self):
        p=np.zeros((1,3,5,32)); p[:,0]=2; p[:,2]=(-1.)**np.arange(32)
        np.testing.assert_array_equal(tc.transport_image(p,.1,.02,'raw'),0)
        p[:,2]=np.sin(np.arange(32)*2*np.pi/32)
        np.testing.assert_allclose(tc.transport_image(p,.1,.04,'raw'),tc.transport_image(p,.1,.02,'raw')/2)

    def test_padding_conditions_channel_order_and_no_mutation(self):
        c=contract_fixture(); case=c['manifest']['cases'][0]
        frames=frames_fixture(10); saved=frames.copy()
        a=tc.make_input(frames,case,c,'raw'); b=tc.make_input(frames,case,c,'none')
        self.assertEqual(a.shape,(10,6,260,256)); self.assertEqual(b.shape,(10,5,260,256))
        np.testing.assert_array_equal(a[:,:3],b[:,:3]); np.testing.assert_array_equal(a[:,-2:],b[:,-2:])
        np.testing.assert_array_equal(frames,saved)
        np.testing.assert_array_equal(a[:,3,257:],np.repeat(a[:,3,256:257],3,axis=1))
        expected=tc.transport_image(tc.physical_fields(frames,c['manifest']),.00005,.02,'raw')/c['gamma_rms']['raw']
        np.testing.assert_allclose(a[:,3,:257],expected,rtol=1e-6)
        # Poison padding: transport on valid nodes must not change.
        frames[:,:,257:]=1e10
        np.testing.assert_array_equal(tc.make_input(frames,case,c,'raw')[:,3],a[:,3])

    def test_future_cannot_modify_past_input(self):
        c=contract_fixture(); c['manifest']['cases']=c['manifest']['cases'][:1]
        frames=frames_fixture(30)
        ds=tc.TransportWindows(c,'train','corr',preloaded=[frames])
        before,y=ds[0]
        frames[10:20]+=30
        after,_=ds[0]
        np.testing.assert_array_equal(before,after)
        self.assertEqual(len(ds),11)
        self.assertEqual(ds.samples[-1],(0,10))

    def test_full_is_not_T10(self):
        c=contract_fixture(); case=c['manifest']['cases'][0]
        frames=frames_fixture(1)
        y=np.arange(256)*2*np.pi/256
        frames[:,0]=.3+.05*np.sin(20*y); frames[:,2]=.2+.01*np.cos(20*y)
        obs=tc.observables(frames,case,c)
        self.assertGreater(np.max(abs(obs['full'])),1e15)
        self.assertLess(np.max(abs(obs['t10']))/np.max(abs(obs['full'])),1e-10)

    def test_unseen_B_uses_fixed_source_scaling(self):
        c=contract_fixture(); case=copy.deepcopy(c['manifest']['cases'][0])
        first=tc.make_input(frames_fixture(10),case,c,'corr')
        case['B_mT']*=1.75
        other=tc.make_input(frames_fixture(10),case,c,'corr')
        np.testing.assert_allclose(first[:,3]/1.75,other[:,3],rtol=2e-6)
        self.assertFalse(np.array_equal(first[:,-2:],other[:,-2:]))

    def test_invalid_inputs_rejected(self):
        c=contract_fixture(); case=c['manifest']['cases'][0]
        c['gamma_rms']['raw']=0
        with self.assertRaises(ValueError):
            tc.make_input(frames_fixture(10),case,c,'raw')
        with self.assertRaises(ValueError):
            tc.transport_image(np.ones((3,3,3)),.1,-1,'raw')


class ModelAndBaselineTests(unittest.TestCase):
    def test_matched_initialization_shapes_and_learning(self):
        def tiny(c,arm):
            return dict(in_shape=(2,5+(arm!='none'),12,8),hid_S=4,hid_T=8,N_S=2,N_T=2,
                model_type='gSTA',spatio_azimuth_downsample=2,translator_norm='group',
                translator_norm_groups=2,condition_dim=2,condition_film=True,out_channels=3,
                aft_seq_length=2,simvp_direct_aft_seq=True,drop_path=0.,drop_path_schedule='zero_to_max')
        torch.set_num_threads(2)
        with patch.object(tc,'model_config',side_effect=tiny):
            base=tc.matched_model({},'none',42)
            augmented=tc.matched_model({},'corr',42)
        x=torch.randn(1,2,5,12,8)
        xa=torch.cat((x[:,:,:3],torch.randn(1,2,1,12,8),x[:,:,3:]),dim=2)
        base.eval(); augmented.eval()
        pb,pa=base(x),augmented(xa)
        self.assertEqual(pa.shape,(1,2,3,12,8))
        torch.testing.assert_close(pb,pa,rtol=1e-5,atol=1e-6)
        loss=pa.square().mean(); loss.backward()
        weights=next(p for p in augmented.parameters() if p.ndim==4 and p.shape[1]==4)
        self.assertGreater(float(weights.grad[:,3].abs().sum()),0)

    def test_disjoint_target_alignment_and_ridge_scalers(self):
        series=np.arange(60.)[:,None]*np.arange(1,5)[None,:]
        x,y=tc.forecast_windows(series[:30])
        xv,yv=tc.forecast_windows(series[30:])
        np.testing.assert_array_equal(y[0],series[10:20].mean(0))
        np.testing.assert_array_equal(x[-1].reshape(10,4)[-1],series[19])
        np.testing.assert_array_equal(xv[0].reshape(10,4)[0],series[30])
        fit=tc.fit_ridge(x,y,xv,yv)
        np.testing.assert_allclose(fit['x_mean'],x.mean(0))
        np.testing.assert_allclose(fit['y_mean'],y.mean(0))
        self.assertLess(np.max(abs(tc.predict_ridge(fit,xv)-yv)),.1)

    def test_epoch_order_independent_of_model_rng(self):
        a=epoch_order(50,42,7)
        torch.manual_seed(99); torch.rand(10000)
        torch.testing.assert_close(a,epoch_order(50,42,7),rtol=0,atol=0)
        self.assertFalse(torch.equal(a,epoch_order(50,42,8)))

    def test_degenerate_null_is_not_a_pass(self):
        y=np.ones((5,4))
        result=tc.score_transport(y,y,y)
        self.assertIsNone(result['skill'])
        self.assertEqual(result['skill_by_band'],[None]*4)

    def test_actual_trainer_epoch_pause_resume_optimizer_scheduler(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer=torch.nn.Linear(3,3)
            def forward(self,x):
                return self.layer(x)
        def factory(contract,arm,seed):
            torch.manual_seed(seed)
            return Tiny()
        dataset=torch.utils.data.TensorDataset(torch.arange(36,dtype=torch.float32).reshape(12,3)/36,
                                               torch.zeros(12,3))
        contract=dict(pre=10,aft=10,training=dict(epochs=2,lr=.001,pct_start=.3))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            full,resumed=root/'full',root/'resumed'
            for folder in (full,resumed):
                folder.mkdir(); tc.atomic_json(folder/'bundle.json',{'same':'identity'})
            with patch.object(runner,'verify_prepared',return_value=(contract,{})), \
                 patch.object(runner,'matched_model',side_effect=factory), \
                 patch.object(runner,'TransportWindows',return_value=dataset):
                runner.train_job(full,'none',42,torch.device('cpu'))
                (resumed/'PAUSE_AFTER_EPOCH').touch()
                self.assertFalse(runner.train_job(resumed,'none',42,torch.device('cpu')))
                (resumed/'PAUSE_AFTER_EPOCH').unlink()
                self.assertTrue(runner.train_job(resumed,'none',42,torch.device('cpu')))
            a=torch.load(full/'jobs/none42/last.pt',weights_only=False)
            b=torch.load(resumed/'jobs/none42/last.pt',weights_only=False)
            self.assertEqual(a['global_step'],24)
            self.assertEqual(a['scheduler'],b['scheduler'])
            for key in a['model']:
                torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
