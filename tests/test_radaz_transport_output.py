import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import radaz_transport_channels as old
import radaz_transport_output as v2
import run_radaz_transport_output as runner


def fixture():
    c=dict(manifest=old.read_json(old.SOURCE),pre=10,aft=10,splits={'train':[0,30]},
           gamma_rms={'raw':8e22,'corr':4e21},t10={str(b):list(range(1,11)) for b in range(4)})
    c['manifest']['cases']=c['manifest']['cases'][:1]
    a=np.zeros((30,3,260,256),np.float32)
    theta=np.arange(256)*2*np.pi/256
    a[:,0]=.3+.05*np.sin(3*theta); a[:,1]=.3; a[:,2]=.2+.01*np.cos(3*theta)
    return c,a


class StateTests(unittest.TestCase):
    def test_future_gamma_uses_input_definition_and_scale(self):
        c,a=fixture(); case=c['manifest']['cases'][0]
        for arm in ('raw','corr'):
            target=v2.make_target(a[10:20],case,c,arm)
            self.assertEqual(target.shape,(10,4,260,256))
            np.testing.assert_array_equal(target[:,:3],a[10:20])
            np.testing.assert_array_equal(target,old.make_input(a[10:20],case,c,arm)[:,:4])

    def test_future_labels_cannot_change_input(self):
        c,a=fixture(); ds=v2.TransportWindows(c,'train','corr',preloaded=[a])
        x,y=ds[0]
        a[10:20,0]*=2
        x2,y2=ds[0]
        np.testing.assert_array_equal(x,x2)
        self.assertFalse(np.array_equal(y[:,3],y2[:,3]))
        self.assertEqual(y.shape[1],4)
        self.assertEqual(v2.TransportWindows(c,'train','none',preloaded=[a])[0][1].shape[1],3)

    def test_output_flux_is_not_recomputed(self):
        c,a=fixture(); case=c['manifest']['cases'][0]
        state=v2.make_target(a[:10],case,c,'corr')
        obs=old.observables(a[:10],case,c)
        np.testing.assert_allclose(v2.direct_full_flux(state,case,c,'corr'),obs['full'],rtol=2e-6)
        state[:,3]=0
        metrics,arrays=v2.diagnose(state,a[:10],case,c,'corr')
        np.testing.assert_array_equal(arrays['direct_full'],0)
        self.assertGreater(np.max(abs(arrays['derived_full'])),1e15)
        self.assertGreater(metrics['direct_derived_local_normalized_mse'],0)
        self.assertEqual(metrics['derived_gamma_normalized_mse'],0)
        self.assertNotIn('direct_t10',arrays)

    def test_loss_preserves_field_weight_and_masks_gamma_padding(self):
        p=torch.ones(1,2,4,260,256,requires_grad=True); t=torch.zeros_like(p)
        parts=v2.loss_parts(p,t)
        self.assertAlmostEqual(float(parts['field']),1)
        self.assertAlmostEqual(float(parts['gamma']),1)
        self.assertAlmostEqual(float(parts['total']),4/3,places=6)
        t[:,:,3,257:]=1000
        changed=v2.loss_parts(p,t)
        torch.testing.assert_close(changed['gamma'],parts['gamma'],rtol=0,atol=0)
        changed['total'].backward()
        self.assertGreater(float(p.grad[:,:,3,:257].abs().sum()),0)
        self.assertEqual(float(p.grad[:,:,3,257:].abs().sum()),0)
        self.assertAlmostEqual(float(v2.loss_parts(p[:,:,:3],t[:,:,:3])['total']),1)

    def test_bad_state_shapes_fail(self):
        with self.assertRaises(ValueError):
            v2.loss_parts(torch.zeros(1,2,4,4,8),torch.zeros(1,2,3,4,8))
        with self.assertRaises(ValueError):
            v2.loss_parts(torch.zeros(1,2,4,4,8),torch.zeros(1,2,4,4,8),float('nan'))

    def test_initial_fields_match_control_and_gamma_head_learns(self):
        def cfg(c,arm):
            return dict(in_shape=(2,5+(arm!='none'),12,8),hid_S=4,hid_T=8,N_S=2,N_T=2,
                model_type='gSTA',spatio_azimuth_downsample=2,translator_norm='group',
                translator_norm_groups=2,condition_dim=2,condition_film=True,out_channels=3,
                aft_seq_length=2,simvp_direct_aft_seq=True,drop_path=0.,drop_path_schedule='zero_to_max')
        torch.set_num_threads(2)
        with patch.object(old,'model_config',side_effect=cfg):
            base=v2.matched_model({},'none',42).eval()
            model=v2.matched_model({},'corr',42).eval()
        x=torch.randn(1,2,5,12,8)
        xa=torch.cat((x[:,:,:3],torch.randn(1,2,1,12,8),x[:,:,3:]),dim=2)
        p=model(xa)
        self.assertEqual(p.shape,(1,2,4,12,8))
        torch.testing.assert_close(p[:,:,:3],base(x),rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(p[:,:,3],torch.zeros_like(p[:,:,3]),rtol=0,atol=0)
        v2.loss_parts(p,torch.ones_like(p))['total'].backward()
        self.assertGreater(float(model.dec.readout.weight.grad[3].abs().sum()),0)
        self.assertGreater(float(model.dec.readout.bias.grad[3].abs()),0)


class RolloutTests(unittest.TestCase):
    def test_predicted_gamma_and_known_conditions_are_reused(self):
        class Spy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.inputs=[]
            def forward(self,x):
                self.inputs.append(x.clone())
                y=x[:,:,:4].clone(); y[:,:,3]+=2
                y[:,:,:,4:]=999  # invalid padding must be restored before feedback
                return y
        x=torch.zeros(1,2,6,5,8); x[:,:,3]=7; x[:,:,4]=.25; x[:,:,5]=-.5
        model=Spy().eval(); p=v2.autonomous_rollout(model,x,3,valid_height=4)
        self.assertEqual(p.shape,(1,6,4,5,8))
        for index,expected in enumerate((7,9,11)):
            torch.testing.assert_close(model.inputs[index][:,:,3,:4],torch.full((1,2,4,8),float(expected)))
            torch.testing.assert_close(model.inputs[index][:,:,-2:],x[:,:,-2:])
        self.assertEqual(float(p[0,-1,3,0,0]),13)
        torch.testing.assert_close(p[:,:,:,4],p[:,:,:,3])
        # ne and phi are zero, so a recomputed gamma would have been zero.
        self.assertTrue(torch.all(p[:,:,3]>0))

    def test_reject_nonfinite_wrong_shape_and_changing_conditions(self):
        class Bad(torch.nn.Module):
            def forward(self,x):
                return x[:,:,:4]*float('nan')
        x=torch.zeros(1,2,6,4,8)
        with self.assertRaises(ValueError):
            v2.autonomous_rollout(Bad(),x,2,4)
        with self.assertRaises(RuntimeError):
            v2.autonomous_rollout(Bad().eval(),x,2,4)
        x[:,1,5]=1
        with self.assertRaises(ValueError):
            v2.autonomous_rollout(Bad().eval(),x,2,4)

    def test_three_field_control_rolls_out_too(self):
        class Copy(torch.nn.Module):
            def forward(self,x):
                return x[:,:,:3]
        x=torch.ones(1,2,5,4,8)
        result=v2.autonomous_rollout(Copy().eval(),x,2,4)
        self.assertEqual(result.shape,(1,4,3,4,8))

    def test_real_training_loop_resume_restores_four_output_loss(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(.5))
            def forward(self,x):
                return self.weight*x[:,:,:4]
        x=torch.arange(12*6*4*8,dtype=torch.float32).reshape(12,1,6,4,8)/10000
        ds=torch.utils.data.TensorDataset(x,torch.zeros(12,1,4,4,8))
        contract=dict(pre=10,aft=10,training=dict(epochs=2,lr=.001,pct_start=.3,gamma_weight=1/3))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); folders=[root/'full',root/'resume']
            for folder in folders:
                old.atomic_json(folder/'bundle.json',{'identity':'test'})
            with patch.object(runner,'verify',return_value=(contract,{})), \
                 patch.object(v2,'matched_model',side_effect=lambda *args:Tiny()), \
                 patch.object(v2,'TransportWindows',return_value=ds):
                runner.train_job(folders[0],'corr',42,torch.device('cpu'))
                (folders[1]/'PAUSE_AFTER_EPOCH').touch()
                self.assertFalse(runner.train_job(folders[1],'corr',42,torch.device('cpu')))
                (folders[1]/'PAUSE_AFTER_EPOCH').unlink()
                self.assertTrue(runner.train_job(folders[1],'corr',42,torch.device('cpu')))
            a,b=[torch.load(p/'jobs/corr42/last.pt',weights_only=False) for p in folders]
            torch.testing.assert_close(a['model']['weight'],b['model']['weight'],rtol=0,atol=0)
            self.assertEqual(a['scheduler'],b['scheduler'])
            row=old.read_json(folders[1]/'jobs/corr42/status.json')
            self.assertGreater(row['train_loss']['gamma'],0)
            self.assertAlmostEqual(row['val_loss']['total'],row['val_loss']['field']+row['val_loss']['gamma']/3,places=7)


if __name__=='__main__':
    unittest.main()
