import os
import sys
import json
import hydra
import wandb
import builtins
from termcolor import colored
from datetime import datetime
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from diffusers import AutoencoderKL

from ldmseg.models import GeneralVAESeg
from ldmseg.trainers import TrainerAE
from ldmseg.utils import prepare_config, Logger, is_main_process
def main_worker(gpu: int, ngpus_per_node: int, cfg_dist: dict, p: dict, name: str = 'segmentation_diffusion') -> None:
    """
    单个训练进程的函数，与原 main_worker 内部逻辑一致
    """
    cfg_dist['gpu'] = gpu
    cfg_dist['ngpus_per_node'] = ngpus_per_node

    if cfg_dist.get('multiprocessing_distributed', False) and cfg_dist['gpu'] != 0:
        # 非主进程时屏蔽标准输出，避免混乱
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    if cfg_dist['gpu'] is not None:
        print("Use GPU: {} for printing".format(cfg_dist['gpu']))

    if cfg_dist['distributed']:
        if cfg_dist['dist_url'] == "env://" and cfg_dist.get('rank', -1) == -1:
            cfg_dist['rank'] = int(os.environ.get("RANK", 0))
        if cfg_dist.get('multiprocessing_distributed', False):
            cfg_dist['rank'] = cfg_dist.get('rank', 0) * ngpus_per_node + gpu
        print(colored("Starting distributed training", "yellow"))
        dist.init_process_group(
            backend=cfg_dist['dist_backend'],
            init_method=cfg_dist['dist_url'],
            world_size=cfg_dist['world_size'],
            rank=cfg_dist['rank']
        )
    print(colored("Initialized distributed training", "yellow"))
    torch.cuda.set_device(cfg_dist['gpu'])

    if p.get('wandb', False) and is_main_process():
        wandb.init(name=name, project="ddm")
    # 重定向标准输出到日志文件，输出目录由配置提供
    sys.stdout = Logger(os.path.join(p['output_dir'], f'log_file_gpu_{cfg_dist["gpu"]}.txt'))

    p['name'] = name
    readable_p = json.dumps(p, indent=4, sort_keys=True)
    print(colored(readable_p, 'red'))
    print(colored(datetime.now(), 'yellow'))

    train_params = p['train_kwargs']
    eval_params = p['eval_kwargs']

    pretrained_model_path = p['pretrained_model_path']
    print('pretrained_model_path', pretrained_model_path)

    vae_encoder = None
    if p.get('shared_vae_encoder', False):
        vae_image = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        vae_encoder = torch.nn.Sequential(vae_image.encoder, vae_image.quant_conv)
    vae = GeneralVAESeg(**p['vae_model_kwargs'], encoder=vae_encoder)
    print(vae)

    if train_params.get('gradient_checkpointing', False):
        vae.enable_gradient_checkpointing()

    vae = vae.to(cfg_dist['gpu'])
    print(colored(f"Number of trainable parameters: {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6 :.2f}M", 'yellow'))

    if cfg_dist['distributed']:
        vae = torch.nn.parallel.DistributedDataParallel(
            vae, device_ids=[cfg_dist['gpu']],
            find_unused_parameters=train_params.get('find_unused_parameters', False),
            gradient_as_bucket_view=train_params.get('gradient_as_bucket_view', False),
        )
        print(colored(f"Batch size per process is {train_params['batch_size']}", 'yellow'))
        print(colored(f"Workers per process is {train_params['num_workers']}", 'yellow'))
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    trainer = TrainerAE(
        p,
        vae,
        args=cfg_dist,
        results_folder=p['output_dir'],
        save_and_sample_every=eval_params['vis_every'],
        cudnn_on=train_params['cudnn'],
        fp16=train_params['fp16'],
    )

    # 断点恢复
    trainer.resume()  # 自动寻找模型断点
    if p.get('load_path', None) is not None:
        trainer.load(model_path=p['load_path'])
    if p.get('eval_only', False):
        trainer.compute_metrics(['miou', 'pq'])
        return
    
    trainer.train_loop()

    if p.get('wandb', False) and is_main_process():
        wandb.finish()