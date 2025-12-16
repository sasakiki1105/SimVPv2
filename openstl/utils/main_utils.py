# Copyright (c) CAIRI AI Lab. All rights reserved

import cv2
import os
import logging
import subprocess
import sys
import platform
from collections import defaultdict, OrderedDict
from typing import Tuple

import torch
import torchvision
from torch import distributed as dist

import openstl
from .config_utils import Config
import importlib.util
from pathlib import Path



def collect_env():
    """Collect the information of the running environments."""
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    if cuda_available:
        from torch.utils.cpp_extension import CUDA_HOME
        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and os.path.isdir(CUDA_HOME):
            try:
                nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc')
                # Windows でも動く形にして、tail は自前で Python 側で処理
                nvcc_out = subprocess.check_output([nvcc, '-V'])
                lines = nvcc_out.decode('utf-8').splitlines()
                nvcc_str = lines[-1].strip() if lines else 'Not Available'
            except Exception:
                nvcc_str = 'Not Available'
            env_info['NVCC'] = nvcc_str

        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            env_info['GPU ' + ','.join(devids)] = name

    # ★ gcc は Windows ではスキップ、その他でも失敗しても落とさない
    try:
        if platform.system() != "Windows":
            gcc_out = subprocess.check_output(['gcc', '--version'])
            gcc_line = gcc_out.decode('utf-8').splitlines()[0].strip()
            env_info['GCC'] = gcc_line
        else:
            env_info['GCC'] = 'Skipped on Windows'
    except Exception as e:
        env_info['GCC'] = f'GCC not available: {e}'

    env_info['PyTorch'] = torch.__version__
    # __config__.show() は文字列を直接返すので、そのまま入れてOK
    env_info['PyTorch compiling details'] = torch.__config__.show()
    env_info['TorchVision'] = torchvision.__version__
    env_info['OpenCV'] = cv2.__version__

    env_info['openstl'] = openstl.__version__

    return env_info



def print_log(message):
    print(message)
    logging.info(message)


def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    return path


def get_dataset(dataname, config):
    from openstl.datasets import dataset_parameters
    from openstl.datasets import load_data
    config.update(dataset_parameters[dataname])
    return load_data(**config)


def measure_throughput(model, input_dummy):

    def get_batch_size(H, W):
        max_side = max(H, W)
        if max_side >= 128:
            bs = 10
            repetitions = 1000
        else:
            bs = 100
            repetitions = 100
        return bs, repetitions

    if isinstance(input_dummy, tuple):
        input_dummy = list(input_dummy)
        _, T, C, H, W = input_dummy[0].shape
        bs, repetitions = get_batch_size(H, W)
        _input = torch.rand(bs, T, C, H, W).to(input_dummy[0].device)
        input_dummy[0] = _input
        input_dummy = tuple(input_dummy)
    else:
        _, T, C, H, W = input_dummy.shape
        bs, repetitions = get_batch_size(H, W)
        input_dummy = torch.rand(bs, T, C, H, W).to(input_dummy.device)
    total_time = 0
    with torch.no_grad():
        for _ in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            if isinstance(input_dummy, tuple):
                _ = model(*input_dummy)
            else:
                _ = model(input_dummy)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * bs) / total_time
    return Throughput


def load_config(filename: str = None):
    """load and print config"""
    print('loading config from ' + str(filename) + ' ...')

    if filename is None:
        print('warning: no filename given!')
        return {}

    cfg_path = Path(filename)
    if not cfg_path.is_file():
        print('warning: config file not found! ->', cfg_path)
        return {}

    # Config クラス経由だと Windows で PermissionError になるので、
    # ここでは Python モジュールとして直接 import する
    try:
        spec = importlib.util.spec_from_file_location("openstl_user_cfg", str(cfg_path))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        # モジュール内の「先頭が '_' で始まらない名前」だけを dict に落とす
        config = {
            k: getattr(module, k)
            for k in dir(module)
            if not k.startswith("_")
        }
        return config

    except Exception as e:
        print('warning: fail to load the config via importlib!')
        print('  -> exception type :', type(e).__name__)
        print('  -> exception msg  :', e)
        return {}


def update_config(args, config, exclude_keys=list()):
    """update the args dict with a new config"""
    assert isinstance(args, dict) and isinstance(config, dict)
    for k in config.keys():
        if args.get(k, False):
            if args[k] != config[k] and k not in exclude_keys and args[k] is not None:
                print(f'overwrite config key -- {k}: {config[k]} -> {args[k]}')
            else:
                args[k] = config[k]
        else:
            args[k] = config[k]
    return args


def weights_to_cpu(state_dict: OrderedDict) -> OrderedDict:
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    # Keep metadata in state_dict
    state_dict_cpu._metadata = getattr(  # type: ignore
        state_dict, '_metadata', OrderedDict())
    return state_dict_cpu


def get_dist_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size