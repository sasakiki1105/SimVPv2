import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

import radaz_transport_channels as old
import radaz_transport_output as parent
import radaz_transport_velocity as state
import run_radaz_transport_velocity as runner


def fixture():
    contract=dict(manifest=old.read_json(old.SOURCE),pre=10,aft=10,
        splits={'train':[0,30]},gamma_rms={'corr':4e21},velocity_rms=25000.,
        t10={str(b):list(range(1,11)) for b in range(4)})
    contract['manifest']['cases']=contract['manifest']['cases'][:1]
    frames=np.zeros((30,4,260,256),np.float32)
    theta=np.arange(256)*2*np.pi/256
    frames[:,0]=.3+.05*np.sin(3*theta); frames[:,1]=.3
    frames[:,2]=.2+.01*np.cos(3*theta)
    frames[:,3]=np.arange(30)[:,None,None]/10
    return contract,frames


class StateTests(unittest.TestCase):
    def test_order_scale_and_future_separation(self):
        c,a=fixture(); case=c['manifest']['cases'][0]
        ds=state.TransportWindows(c,'train',preloaded=[a])
        x,y=ds[0]
        self.assertEqual(x.shape,(10,7,260,256)); self.assertEqual(y.shape,(10,5,260,256))
        np.testing.assert_array_equal(x[:,4],a[:10,3])
        np.testing.assert_array_equal(y[:,4],a[10:20,3])
        np.testing.assert_array_equal(y[:,:4],parent.make_target(a[10:20,:3],case,c,'corr'))
        a[10:20,3]+=100; a[10:20,0]*=2
        x2,y2=ds[0]
        np.testing.assert_array_equal(x,x2)
        self.assertFalse(np.array_equal(y[:,4],y2[:,4]))
        self.assertFalse(np.array_equal(y[:,3],y2[:,3]))
        self.assertEqual(len(ds),11)

    def test_h5_velocity_units_time_and_grid(self):
        c,_=fixture(); case=dict(c['manifest']['cases'][0])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.h5'; case['path']=str(path)
            with h5py.File(path,'w') as f:
                f['axes/time_s']=np.arange(4)*15e-9
                for name in ('electron_den','ion_den','phi','electron_wd'):
                    data=np.zeros((4,257,257))
                    if name=='electron_wd':
                        data[:]=np.arange(4)[:,None,None]*c['velocity_rms']
                        data[:,:,256]=1e10  # duplicate periodic endpoint must be excluded
                        data[:,256,:256]+=c['velocity_rms']/2
                    f['fields/'+name]=data
            frames=state.load_frames(case,1,3,c)
        self.assertEqual(frames.shape,(2,4,260,256))
        np.testing.assert_array_equal(frames[:,3,0,0],[1,2])
        np.testing.assert_array_equal(frames[:,3,256,0],[1.5,2.5])
        np.testing.assert_array_equal(frames[:,3,257:],np.repeat(frames[:,3,256:257],3,axis=1))

    def test_losses_and_gradients(self):
        p=torch.ones(1,2,5,260,256,requires_grad=True); target=torch.zeros_like(p)
        parts=state.loss_parts(p,target)
        self.assertAlmostEqual(float(parts['total']),5/3,places=6)
        target[:,:,3:,257:]=10000
        changed=state.loss_parts(p,target)
        torch.testing.assert_close(changed['total'],parts['total'])
        changed['total'].backward()
        self.assertGreater(float(p.grad[:,:,4,:257].abs().sum()),0)
        self.assertEqual(float(p.grad[:,:,4,257:].abs().sum()),0)

    def test_initial_shared_fields_match_and_both_heads_learn(self):
        def config(c,arm):
            return dict(in_shape=(2,5+(arm!='none'),12,8),hid_S=4,hid_T=8,N_S=2,N_T=2,
                model_type='gSTA',spatio_azimuth_downsample=2,translator_norm='group',
                translator_norm_groups=2,condition_dim=2,condition_film=True,out_channels=3,
                aft_seq_length=2,simvp_direct_aft_seq=True,drop_path=0.,drop_path_schedule='zero_to_max')
        def five_config(c,arm):
            cfg=config(c,'corr'); cfg.update(in_shape=(2,7,12,8),out_channels=5)
            return cfg
        with patch.object(old,'model_config',side_effect=config), \
             patch.object(state,'model_config',side_effect=five_config):
            base=parent.matched_model({},'corr',42).eval()
            model=state.matched_model({},'corr_uez',42).eval()
        x=torch.randn(1,2,6,12,8)
        xa=torch.cat((x[:,:,:4],torch.randn(1,2,1,12,8),x[:,:,4:]),dim=2)
        pred=model(xa)
        torch.testing.assert_close(pred[:,:,:4],base(x),rtol=1e-5,atol=1e-6)
        self.assertTrue(torch.all(pred[:,:,4]==0))
        state.loss_parts(pred,torch.ones_like(pred))['total'].backward()
        for index in (3,4):
            self.assertGreater(float(model.dec.readout.weight.grad[index].abs().sum()),0)

    def test_diagnostics_do_not_relabel_velocity_as_flux(self):
        c,a=fixture(); case=c['manifest']['cases'][0]
        p=state.make_target(a[:10],case,c)
        metrics,arrays=state.diagnose(p,a[:10],case,c)
        self.assertEqual(metrics['velocity_rmse_m_s'],0)
        p[:,4]+=2
        changed,other=state.diagnose(p,a[:10],case,c)
        self.assertAlmostEqual(changed['velocity_rmse_m_s'],2*c['velocity_rms'],places=2)
        np.testing.assert_array_equal(arrays['direct_full'],other['direct_full'])
        p[:,3]=0
        _,zero=state.diagnose(p,a[:10],case,c)
        np.testing.assert_array_equal(zero['direct_full'],0)
        self.assertNotIn('direct_t10',zero)


class RolloutTests(unittest.TestCase):
    def test_gamma_and_velocity_are_both_predicted_feedback(self):
        class Spy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.inputs=[]
            def forward(self,x):
                self.inputs.append(x.clone())
                p=x[:,:,:5].clone(); p[:,:,3]+=2; p[:,:,4]-=3
                p[:,:,:,4:]=999
                return p
        x=torch.zeros(1,2,7,5,8); x[:,:,3]=7; x[:,:,4]=10
        x[:,:,5]=.25; x[:,:,6]=-.5
        model=Spy().eval(); pred=state.autonomous_rollout(model,x,3,4)
        for i,(gamma,velocity) in enumerate(((7,10),(9,7),(11,4))):
            self.assertTrue(torch.all(model.inputs[i][:,:,3,:4]==gamma))
            self.assertTrue(torch.all(model.inputs[i][:,:,4,:4]==velocity))
            torch.testing.assert_close(model.inputs[i][:,:,-2:],x[:,:,-2:])
        self.assertEqual(pred.shape,(1,6,5,5,8))
        torch.testing.assert_close(pred[:,:,:,4],pred[:,:,:,3])

    def test_invalid_rollouts_fail(self):
        class Bad(torch.nn.Module):
            def forward(self,x):
                return x[:,:,:5]*float('nan')
        x=torch.zeros(1,2,7,4,8)
        with self.assertRaises(RuntimeError):
            state.autonomous_rollout(Bad().eval(),x,2,4)
        x[:,1,6]=1
        with self.assertRaises(ValueError):
            state.autonomous_rollout(Bad().eval(),x,2,4)

    def test_real_trainer_pause_resume(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(.5))
            def forward(self,x):
                return self.weight*x[:,:,:5]
        x=torch.arange(12*7*4*8,dtype=torch.float32).reshape(12,1,7,4,8)/10000
        ds=torch.utils.data.TensorDataset(x,torch.zeros(12,1,5,4,8))
        contract=dict(pre=10,aft=10,training=dict(epochs=2,lr=.001,pct_start=.3,gamma_weight=1/3))
        with tempfile.TemporaryDirectory() as tmp:
            folders=[Path(tmp)/'full',Path(tmp)/'resume']
            for folder in folders:
                old.atomic_json(folder/'bundle.json',{'identity':'test'})
            with patch.object(runner,'verify',return_value=(contract,{})), \
                 patch.object(state,'matched_model',side_effect=lambda *args:Tiny()), \
                 patch.object(state,'TransportWindows',return_value=ds):
                runner.train_job(folders[0],'corr_uez',42,torch.device('cpu'))
                (folders[1]/'PAUSE_AFTER_EPOCH').touch()
                self.assertFalse(runner.train_job(folders[1],'corr_uez',42,torch.device('cpu')))
                (folders[1]/'PAUSE_AFTER_EPOCH').unlink()
                self.assertTrue(runner.train_job(folders[1],'corr_uez',42,torch.device('cpu')))
            a,b=[torch.load(p/'jobs/corr_uez42/last.pt',weights_only=False) for p in folders]
            torch.testing.assert_close(a['model']['weight'],b['model']['weight'],rtol=0,atol=0)
            self.assertEqual(a['scheduler'],b['scheduler'])
            row=old.read_json(folders[1]/'jobs/corr_uez42/status.json')
            self.assertGreater(row['train_loss']['velocity'],0)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
