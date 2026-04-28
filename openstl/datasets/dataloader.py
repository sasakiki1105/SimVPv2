def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, dist=False, **kwargs):
    cfg_dataloader = dict(
        pre_seq_length=kwargs.get('pre_seq_length', 10),
        aft_seq_length=kwargs.get('aft_seq_length', 10),
        in_shape=kwargs.get('in_shape', None),
        distributed=dist,
        use_augment=kwargs.get('use_augment', False),
        use_prefetcher=kwargs.get('use_prefetcher', False),
        drop_last=kwargs.get('drop_last', False),
    )

    if dataname == 'taxibj':
        from .dataloader_taxibj import load_data as _load
        return _load(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)

    elif dataname == 'mmnist':
        from .dataloader_moving_mnist import load_data as _load
        return _load(batch_size, val_batch_size, data_root, num_workers, data_name='mnist', **cfg_dataloader)

    elif dataname == 'mfmnist':
        from .dataloader_moving_mnist import load_data as _load
        return _load(batch_size, val_batch_size, data_root, num_workers, data_name='fmnist', **cfg_dataloader)

    elif dataname == 'mmnist_cifar':
        from .dataloader_moving_mnist import load_data as _load
        return _load(batch_size, val_batch_size, data_root, num_workers, data_name='mnist_cifar', **cfg_dataloader)

    elif dataname == 'noisymmnist_dynamic':
        from .dataloader_noisy_moving_mnist import load_data as _load
        return _load(batch_size, val_batch_size, data_root, num_workers, data_name='noisymmnist_dynamic', **cfg_dataloader)

    elif dataname == 'noisymmnist_missing':
        from .dataloader_noisy_moving_mnist import load_data as _load
        return _load(batch_size, val_batch_size, data_root, num_workers, data_name='noisymmnist_missing', **cfg_dataloader)

    elif dataname == 'noisymmnist_perceptual':
        from .dataloader_noisy_moving_mnist import load_data as _load
        return _load(batch_size, val_batch_size, data_root, num_workers, data_name='noisymmnist_perceptual', **cfg_dataloader)

    elif dataname == 'kinetics':
        from .dataloader_kinetics import load_data as _load
        return _load(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)

    elif dataname == 'kitticaltech':
        from .dataloader_kitticaltech import load_data as _load
        return _load(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)

    elif dataname == 'human':
        from .dataloader_human import load_data as _load
        return _load(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)

    elif dataname == 'kth':
        from .dataloader_kth import load_data as _load
        return _load(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)

    elif dataname == 'bair':
        from .dataloader_bair import load_data as _load
        return _load(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)

    elif dataname in [
        'weather',
        'weather_t2m_5_625', 'weather_t2m_1_40625',
        'weather_r_5_625', 'weather_r_1_40625',
        'weather_uv10_5_625', 'weather_uv10_1_40625',
        'weather_tcc_5_625', 'weather_tcc_1_40625'
    ]:
        from .dataloader_weather import load_data as _load
        cfg_dataloader['data_name'] = dataname
        return _load(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)

    elif dataname in ['sevir', 'sevir_ir069', 'sevir_ir107', 'sevir_vil', 'sevir_vis']:
        from .dataloader_sevir import load_data as _load
        cfg_dataloader['data_name'] = dataname
        return _load(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)

    elif dataname == 'pepapic_h5':
        from .dataloader_pepapic_h5 import load_data as _load
        cfg_dataloader['train_ratio'] = kwargs.get('train_ratio', 0.8)
        cfg_dataloader['val_ratio'] = kwargs.get('val_ratio', 0.1)
        cfg_dataloader['test_ratio'] = kwargs.get('test_ratio', 0.1)
        cfg_dataloader['force_test_all'] = kwargs.get('force_test_all', False)
        return _load(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)

    else:
        raise ValueError(f"Invalid dataname: {dataname}")