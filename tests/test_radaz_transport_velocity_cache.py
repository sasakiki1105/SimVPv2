import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

import radaz_transport_channels as old
import radaz_transport_velocity as state
import radaz_transport_velocity_cache as cache
import run_radaz_transport_velocity as parent
import run_radaz_transport_velocity_cached as runner
from test_radaz_transport_velocity import fixture


class CacheTests(unittest.TestCase):
    def test_all_windows_bits_padding_conditions_and_future_separation(self):
        c,raw=fixture(); case=c['manifest']['cases'][0]
        frames=state.make_target(raw,case,c)
        ds=cache.CachedWindows(c,'train',[frames])
        self.assertEqual(len(ds),11)
        for s in range(11):
            x,y=ds[s]
            self.assertTrue(cache.bit_equal(x,state.make_input(raw[s:s+10],case,c)))
            self.assertTrue(cache.bit_equal(y,state.make_target(raw[s+10:s+20],case,c)))
        before=ds[0][0].copy(); frames[10:]+=100
        self.assertTrue(cache.bit_equal(before,ds[0][0]))

    def test_invalid_split_shape_dtype_and_case_count(self):
        c,raw=fixture(); frames=state.make_target(raw,c['manifest']['cases'][0],c)
        with self.assertRaises(ValueError):
            cache.CachedWindows(c,'test',[frames])
        for arrays in ([],[frames[:29]],[frames.astype('f8')]):
            with self.assertRaises(ValueError):
                cache.CachedWindows(c,'train',arrays)

    def test_h5_roundtrip_and_signed_zero_check(self):
        a=np.array([0.,-0.,np.nextafter(np.float32(0),np.float32(1)),1.25],dtype='f4')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.h5'
            with h5py.File(path,'w') as handle:
                handle['state']=a
            digest=cache.stream_digest(path)
            with h5py.File(path,'r') as handle:
                self.assertTrue(cache.bit_equal(a,handle['state'][()]))
            with h5py.File(path,'r+') as handle:
                handle['state'][0]=1
            self.assertNotEqual(digest,cache.stream_digest(path))
        self.assertFalse(cache.bit_equal(a,np.abs(a)))

    def test_unreviewed_checkpoint_and_other_runtime_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'last.pt'; path.write_bytes(b'original')
            manifest={'transition_checkpoint_sha256':cache.stream_digest(path)}
            runner.validate_checkpoint_runtime({},path,manifest,'runtime')
            runner.validate_checkpoint_runtime({'runtime_amendment_sha256':'runtime'},path,manifest,'runtime')
            with self.assertRaises(RuntimeError):
                runner.validate_checkpoint_runtime({'runtime_amendment_sha256':'other'},path,manifest,'runtime')
            path.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                runner.validate_checkpoint_runtime({},path,manifest,'runtime')

    def test_legacy_epoch_to_cached_epoch_exact_optimizer_scheduler_rng(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); torch.manual_seed(42)
                self.weight=torch.nn.Parameter(torch.tensor(.5))
            def forward(self,x):
                # Consume global RNG: equality must include resumed stochastic behavior.
                return self.weight*x[:,:,:5]*(1+.01*torch.rand((),device=x.device))
        x=torch.arange(12*7*4*8,dtype=torch.float32).reshape(12,1,7,4,8)/10000
        ds=torch.utils.data.TensorDataset(x,torch.zeros(12,1,5,4,8))
        contract=dict(pre=10,aft=10,training=dict(epochs=2,lr=.001,pct_start=.3,
            gamma_weight=1/3,velocity_weight=1/3))
        with tempfile.TemporaryDirectory() as tmp:
            full,resumed,amend=[Path(tmp)/n for n in ('full','resumed','amend')]
            for folder in (full,resumed):
                old.atomic_json(folder/'bundle.json',{'identity':'test'})
            old.atomic_json(amend/'bundle.json',{'identity':'runtime'})
            with patch.object(parent,'verify',return_value=(contract,{})), \
                 patch.object(state,'matched_model',side_effect=lambda *a:Tiny()), \
                 patch.object(state,'model_config',return_value={'tiny':True}), \
                 patch.object(state,'TransportWindows',return_value=ds), \
                 patch.object(cache,'CachedWindows',return_value=ds):
                parent.train_job(full,'corr_uez',42,torch.device('cpu'))
                (resumed/'PAUSE_AFTER_EPOCH').touch()
                self.assertFalse(parent.train_job(resumed,'corr_uez',42,torch.device('cpu')))
                path=resumed/'jobs/corr_uez42/last.pt'
                manifest={'transition_checkpoint_sha256':cache.stream_digest(path)}
                (resumed/'PAUSE_AFTER_EPOCH').unlink()
                with patch.object(runner,'verify_amendment',return_value=(contract,manifest)):
                    self.assertTrue(runner.train_job(resumed,'corr_uez',42,torch.device('cpu'),([],[]),amend))
            a,b=[torch.load(p/'jobs/corr_uez42/last.pt',weights_only=False) for p in (full,resumed)]
            for key in ('identity','kind','model','optimizer','scheduler','torch_rng','cuda_rng',
                        'completed_epochs','global_step','model_config'):
                runner.assert_tree_equal(a[key],b[key],key)
            self.assertEqual(b['runtime_amendment_sha256'],old.digest(amend/'bundle.json'))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
