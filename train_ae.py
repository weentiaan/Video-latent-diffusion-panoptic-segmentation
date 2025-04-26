"""
Author: Wouter Van Gansbeke

Main file for training auto-encoders and vaes
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import sys
sys.argv = sys.argv[:1]
import json
import hydra
import wandb
import builtins
from termcolor import colored
from datetime import datetime
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any
from termcolor import colored
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from diffusers import AutoencoderKL

from ldmseg.models import GeneralVAESeg
from ldmseg.trainers import TrainerAE
from ldmseg.utils import prepare_config, Logger, is_main_process

from hydra import initialize, compose
from main_worker_ae import main_worker
if __name__ == "__main__":
    # 可选：mp.set_start_method("spawn", force=True)
    
    # -------------------------------------------------------------------------------
    # Step 0: 清理 Notebook 自动传入的额外命令行参数
    sys.argv = sys.argv[:1]

    # -------------------------------------------------------------------------------
    # Step 1: 使用 Hydra API 加载配置
    # 请根据实际情况修改 config_path，例如你的配置文件存放在 "configs/" 文件夹下
    with initialize(config_path="tools/configs/", job_name="config"):
        cfg = compose(config_name="config")
    # 将 OmegaConf 对象转换为普通字典
    cfg = OmegaConf.to_object(cfg)

    # -------------------------------------------------------------------------------
    # Step 2: 配置分块、组合与预处理
    wandb.config = cfg
    # 这里假设配置文件中存在以下键；如果不存在，请在配置文件中添加或使用 .get() 方法提供默认值
    cfg_dist    = cfg['distributed']
    cfg_dataset = cfg['datasets']
    cfg_base    = cfg['base']
    project_dir = cfg['setup']

    # 合并 base 与数据集专用配置（让数据集配置覆盖 base 中的同名字段）
    cfg_dataset = {**cfg_base, **cfg_dataset}

    root_dir = os.path.join(cfg['env']['root_dir'], project_dir)
    data_dir = cfg['env']['data_dir']

    # 调用 prepare_config 进一步整理数据集配置，返回更新后的配置和项目名称
    cfg_dataset, project_name = prepare_config(cfg_dataset, root_dir, data_dir, run_idx=cfg['run_idx'])
    project_name = f"{cfg_dataset['train_db_name']}_{project_name}"
    #print(colored(f"Project name: {project_name}", 'red'))

    # -------------------------------------------------------------------------------
    # Step 3: 配置分布式训练参数
    # 若配置中 world_size 为 -1 且 dist_url 为 "env://" 则根据环境变量 WORLD_SIZE 更新配置
    if cfg_dist['dist_url'] == "env://" and cfg_dist['world_size'] == -1:
        cfg_dist['world_size'] = int(os.environ.get("WORLD_SIZE", 1))
    cfg_dist['distributed'] = cfg_dist['world_size'] > 1 or cfg_dist['multiprocessing_distributed']

    # -------------------------------------------------------------------------------
    # Step 4: Debug 模式下特殊设置
    if cfg.get('debug', True):
        print(colored("Running in debug mode!", "red"))
        cfg_dist['world_size'] = 1
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cfg_dataset['train_kwargs']['num_workers'] = 0
        cfg_dataset['eval_kwargs']['num_workers'] = 0

    ngpus_per_node = torch.cuda.device_count()
    # 根据 GPU 数量调整世界大小
    cfg_dist['world_size'] = ngpus_per_node * cfg_dist['world_size']
    #print(colored(f"World size: {cfg_dist['world_size']}", 'blue'))

    # -------------------------------------------------------------------------------
    # Step 6: 启动训练
    cfg_dataset['train_kwargs']['train_num_steps']=20000
    cfg_dataset['lr_scheduler_kwargs']['warmup_iters']=200
    if cfg.get('debug', True):
        # Debug 模式下直接调用 main_worker（单 GPU 单进程）
        main_worker(0, ngpus_per_node, cfg_dist, cfg_dataset, project_name)
    else:
        # 非 Debug 模式下使用 mp.spawn 启动分布式训练（注意：Notebook 中多进程可能会有额外问题）
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg_dist, cfg_dataset, project_name))