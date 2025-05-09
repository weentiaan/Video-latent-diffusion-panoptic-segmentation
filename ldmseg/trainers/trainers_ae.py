"""
Author: Wouter Van Gansbeke

File for training auto-encoders and vaes
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import math
import time
import wandb
from pathlib import Path
from datetime import timedelta
from termcolor import colored
from typing import Dict, Any, Optional, Union, List, Tuple
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from .losses import SegmentationLosses
from .optim import get_optim_general
from ldmseg.data.dataset_base import DatasetBase
from ldmseg.evaluations import SemsegMeter
from ldmseg.utils import (
    AverageMeter, ProgressMeter,
    cosine_scheduler, warmup_scheduler, step_scheduler,
    is_main_process, color_map, gpu_gather, collate_fn,
    get_world_size
)

from ldmseg.evaluations import KITTIPanopticEvaluator
from ldmseg.evaluations.cityscapes_pap_eval import compute_cityscapes_pq
from ldmseg.evaluations import PanopticEvaluatorAgnostic
 
def get_color_map(num_colors):
    """
    生成一个包含 num_colors 个随机颜色的映射表。
    """
    np.random.seed(20)  # 固定种子，保证每次生成相同的颜色
    return np.random.randint(0, 256, (num_colors, 3), dtype=np.uint8)

def colorize_panoptic(panoptic_map, colormap):
    """
    根据 panoptic_map 中每个像素的 panoptic_id，从 colormap 中取对应颜色，
    生成彩色图像。
    """

    h,w=panoptic_map.shape

    color_image = np.zeros((h, w,3), dtype=np.uint8)

    unique_ids = np.unique(panoptic_map)

    for uid in unique_ids:
        # 如果 uid 为 0 或 2550000，设定为黑色
        if uid == 0:
            color = np.array([0, 0, 0], dtype=np.uint8)
        else:
            # 使用 modulo 确保 uid 超过颜色数量时仍然可以映射
            color = colormap[uid % len(colormap)]

        color_image[panoptic_map == uid] = color
    return color_image#[h,w,3]



class TrainerAE(DatasetBase):

    def __init__(
        self,
        p: Dict[str, Any],
        vae_model: nn.Module,
        *,
        save_and_sample_every: int = 1000,
        results_folder: str = './results',
        args: Any = None,
        cudnn_on: bool = True,
        fp16: Optional[bool] = False,
    ) -> None:

        """
        Trainer class for autoencoder and vae models
        This class implements the training loop and evaluation

        Args:
            p (Dict[str, Any]): dictionary containing the parameters
            vae_model (nn.Module): vae model
            save_and_sample_every (int, optional): save and sample every n iterations. Defaults to 1000.
            results_folder (str, optional): folder to save results. Defaults to './results'.
            args (Any, optional): distributed training arguments. Defaults to None.
            cudnn_on (bool, optional): turn on cudnn. Defaults to True.
            fp16 (Optional[bool], optional): turn on mixed precision training. Defaults to False.
        """

        # init inherited class
        super(TrainerAE, self).__init__(data_dir=p['data_dir'])

        # save arguments
        self.p = p
        self.args = args

        # handle fp16 scaler
        self.fp16_scaler = torch.cuda.amp.GradScaler() if fp16 else None
        if fp16:
            print(colored('Warning -- Using FP16', 'yellow'))

        # model
        self.vae_model = vae_model

        self.clip_grad = p['train_kwargs']['clip_grad']
        if self.clip_grad > 0:
            print(colored(f'Warning -- Using gradient clipping of {self.clip_grad}', 'yellow'))

        self.use_wandb = p['wandb']
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = p['train_kwargs']['batch_size']
        self.num_workers = p['train_kwargs']['num_workers']
        self.gradient_accumulate_every = p['train_kwargs']['accumulate']
        assert self.gradient_accumulate_every >= 1 and isinstance(self.gradient_accumulate_every, int)
        self.train_num_steps = p['train_kwargs']['train_num_steps']
        self.print_freq = p['eval_kwargs']['print_freq']
        assert self.print_freq % self.gradient_accumulate_every == 0
        if self.gradient_accumulate_every > 1:
            print(colored('Warning -- Accumulating gradients', 'yellow'))
        self.eff_batch_size = self.batch_size * self.gradient_accumulate_every
        self.image_size = p['transformation_kwargs']['size']
        self.latent_size = self.image_size // self.vae_model.module.downsample_factor
        self.mask_th = p['eval_kwargs']['mask_th']
        self.overlap_th = p['eval_kwargs']['overlap_th']
        self.count_th = p['eval_kwargs']['count_th']
        self.prob_inpainting = p['train_kwargs']['prob_inpainting']

        # optimizer
        self.opt, self.save_optim = get_optim_general(vae_model.parameters(),
                                                      p['optimizer_name'],
                                                      p['optimizer_kwargs'],
                                                      zero_redundancy=p['optimizer_zero_redundancy'])
        self.lr = self.opt.param_groups[0]['lr']
        print(self.opt)

        # cudnn / cuda
        cudnn.benchmark = cudnn_on
        torch.backends.cuda.matmul.allow_tf32 = p['train_kwargs']['allow_tf32']

        # dataset and dataloader
        self.transforms = self.get_train_transforms(p['transformation_kwargs'])
        self.transforms_val = self.get_val_transforms(p['transformation_kwargs'])
        print('Train transforms', self.transforms)
        print('Val transforms', self.transforms_val)
        self.ds = self.get_dataset(p['train_db_name'],
                                   split=p['split'],
                                   transform=self.transforms,
                                   tokenizer=None,
                                   remap_labels=p['train_kwargs']['remap_seg'],
                                   encoding_mode=p['train_kwargs']['encoding_mode'],
                                   num_classes=p['num_classes'],
                                   fill_value=p['fill_value'],
                                   ignore_label=p['ignore_label'],
                                   inpainting_strength=p['inpainting_strength'])
        self.ds_val = self.get_dataset(p['val_db_name'],
                                       split='val',
                                       transform=self.transforms_val,
                                       tokenizer=None,
                                       remap_labels=p['train_kwargs']['remap_seg'],
                                       encoding_mode=p['train_kwargs']['encoding_mode'],
                                       num_classes=p['num_classes'],
                                       fill_value=p['fill_value'],
                                       ignore_label=p['ignore_label'],
                                       inpainting_strength=p['inpainting_strength'])
        print(colored('The dataset contains {} samples'.format(len(self.ds)), 'blue'))
        if args['distributed']:
            self.train_sampler = DistributedSampler(self.ds)
            self.val_sampler = DistributedSampler(self.ds_val)
        else:
            self.train_sampler = None
            self.val_sampler = None
        # train loader
        self.dl = DataLoader(self.ds,
                             batch_size=self.batch_size,
                             num_workers=self.num_workers,
                             shuffle=(self.train_sampler is None),
                             sampler=self.train_sampler,
                             pin_memory=True,
                             drop_last=True,
                             collate_fn=collate_fn)

        # val loader
        self.dl_val = DataLoader(self.ds_val,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 shuffle=(self.val_sampler is None),
                                 sampler=self.val_sampler,
                                 pin_memory=True,
                                 drop_last=False,
                                 collate_fn=collate_fn)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.visualization_dir = "visualizations"
        os.makedirs(self.visualization_dir, exist_ok=True)
        print(colored('The dataset train {} samples'.format(len(self.dl)), 'blue'))
        # step counter state
        self.step = 0
        self.start_epoch = 0
        self.num_iters_per_epoch = math.ceil(len(self.dl) / self.gradient_accumulate_every)
        self.epochs = math.ceil(self.train_num_steps / self.num_iters_per_epoch)
        print(colored(f'training for {self.epochs} epochs or {self.num_iters_per_epoch} iters per epoch '
                      f'or {self.epochs * self.num_iters_per_epoch} iterations',
                      'yellow'))

        # learning rate scheduler
        self.lr_scheduler = None
        if 'lr_scheduler_name' in self.p.keys():
            try:
                self.lr_scheduler = self.get_lr_scheduler(p['lr_scheduler_name'], **p['lr_scheduler_kwargs'])
            except NotImplementedError:
                print(colored('Warning -- No learning rate scheduler found', 'yellow'))
                assert self.lr_scheduler is None

        # define losses
        self.segmentation_losses = SegmentationLosses(ignore_label=self.ds.ignore_label, **p['loss_kwargs'])

        # additional params
        self.cmap = color_map()
        self.latent_mask = self.p['train_kwargs']['latent_mask']
        self.fuse_rgb = self.p['vae_model_kwargs']['fuse_rgb']
        self.loss_weights = self.p['loss_weights']
        self.evaluate_trainset = False

    def compute_point_loss(self, outputs, targets, do_matching=False, masks=None):
        posterior = outputs.posterior#概率分布
        outputs = outputs.sample#预测值

        # (1) compute segmentation losses (ce + mask)
        losses = self.segmentation_losses.point_loss(outputs, targets, do_matching, masks)

        # (2) compute kl loss
        losses['kl'] = torch.mean(posterior.kl())

        # linearly combine losses in dict
        total_loss = self.get_weighted_loss(losses)
        return total_loss, losses['ce'], losses['kl'], losses['mask']

    def get_weighted_loss(self, losses):
        """ Linearly combine losses
        """
        total_loss = 0.
        for loss_name, loss in losses.items():
            total_loss += self.loss_weights[loss_name] * loss
        return total_loss

    @torch.no_grad()
    def get_loss_weight_mask(
        self,
        targets: torch.Tensor,
        mode: str = 'nearest',
        size: Tuple[int] = (64, 64),
        device: str = 'cuda',
        ignore_label: int = 0,
    ) -> torch.Tensor:

        """ Returns a mask that can be used to weigh the loss (or as ignore)
        """

        # TODO: move this to the dataset (i.e. cpu)
        targets = F.interpolate(targets[:, None].float(), size=size, mode=mode).squeeze(1)
        mask = (targets != ignore_label).to(device=device, dtype=torch.float32)
        return mask

    def train_single_epoch(self, epoch, meters_dict, progress, semseg_meter):
        """ Train the model for one epoch
        """ 
        self.vae_model.train()
        total_loss = total_ce_loss = total_kl_loss = total_mask_loss = 0.
        for batch_idx, data in enumerate(self.dl):

            assert self.vae_model.training
            
            # move data to gpu
            
            # images = data['image_semseg'].cuda(self.args['gpu'], non_blocking=True)#[bit,h,w]
            # targets = data['semseg'].cuda(self.args['gpu'], non_blocking=True)#[3,h,w]
            # images = 2. * images - 1.
            images = data['image_semseg'].cuda(self.args['gpu'], non_blocking=True)  # [bit,h,w]
            targets = data['semseg'].cuda(self.args['gpu'], non_blocking=True)  # [3,h,w]
            images = 2. * images - 1.
            
            # move rgb to gpu if needed
            rgbs = None
            if self.fuse_rgb:
                rgbs = data['image'].cuda(self.args['gpu'], non_blocking=True)
                rgbs = 2. * rgbs - 1.

            # (optional) corrupt semseg (reduce redundancy)
            masks = None
            if self.prob_inpainting > 0.:
                bs, _, h, w = images.shape
                strenghts = torch.rand((bs, 1, 1, 1), device=self.args['gpu']) * self.prob_inpainting
                masks = torch.rand((bs, 1, 32, 32), device=self.args['gpu']) < strenghts
                masks = F.interpolate(masks.float(), size=(h, w), mode="nearest")
                masks[targets[:, None] == self.ds.ignore_label] = 0  # remove ignore as valid region in mask
                images[~masks.bool().repeat(1, images.shape[1], 1, 1)] = 0.

            latent_mask = None
            if self.latent_mask:
                latent_mask = self.get_loss_weight_mask(
                    data['semseg'],
                    mode='nearest',
                    device=self.args['gpu'],
                    size=(self.latent_size, self.latent_size),
                    ignore_label=self.ds.ignore_label,
                )

            # update loss
            with torch.autocast('cuda', enabled=self.fp16_scaler is not None):
                output = self.vae_model(images, sample_posterior=True, rgb_sample=rgbs, valid_mask=latent_mask)
                loss, ce_loss, kl_loss, mask_loss = self.compute_point_loss(output, targets, masks=masks)
                loss = loss / self.gradient_accumulate_every
                total_loss += loss.detach()
                total_ce_loss += (ce_loss / self.gradient_accumulate_every).detach()
                total_kl_loss += (kl_loss / self.gradient_accumulate_every).detach()
                total_mask_loss += (mask_loss / self.gradient_accumulate_every).detach()
                
            if batch_idx==99:
                print(torch.unique(output.sample[0]),torch.unique(targets[0]))
            # update gradients
            if self.fp16_scaler is None:
                loss.backward()
            else:
                self.fp16_scaler.scale(loss).backward()
            #print("finish_update gradients")
            # accumulate gradients
            if ((batch_idx + 1) % self.gradient_accumulate_every != 0) and (batch_idx + 1 != len(self.dl)):
                continue

            # schedule learning rate
            if self.lr_scheduler is not None:
                for param_group in self.opt.param_groups:
                    param_group["lr"] = self.lr_scheduler[self.step]
                if batch_idx + 1 == self.gradient_accumulate_every:
                    print(f"Learning rate is set to: {self.opt.param_groups[0]['lr']:.3e}")

            # update weights
            dist.barrier()
            if self.fp16_scaler is None:
                if self.clip_grad > 0:
                    clip_grad_norm_(self.vae_model.parameters(), self.clip_grad, norm_type=2)
                self.opt.step()
            else:
                if self.clip_grad > 0:
                    self.fp16_scaler.unscale_(self.opt)  # unscale the gradients in-place
                    clip_grad_norm_(self.vae_model.parameters(), self.clip_grad, norm_type=2)
                self.fp16_scaler.step(self.opt)
                self.fp16_scaler.update()

            # zero gradients after update
            self.opt.zero_grad()
            dist.barrier()

            # update meters
            torch.cuda.synchronize()
            loss_dict = {'loss': total_loss, 'ce': total_ce_loss, 'mask': total_mask_loss, 'kl': total_kl_loss}
            #print(loss_dict)
            meters_dict = self.update_meters(loss_dict, meters_dict)
            # interpolate output for evaluation, slows down training
            if self.evaluate_trainset:
                output.sample = F.interpolate(
                    output.sample, size=targets.shape[-2:], mode='bilinear', align_corners=False)
                semseg_meter.update(torch.argmax(output.sample, dim=1), targets)
            total_loss = total_ce_loss = total_kl_loss = total_mask_loss = 0.0

            # take step
            self.step += 1

            if (batch_idx + 1) % self.print_freq == 0:
                progress.display(batch_idx)

            if self.check_iter(batch_idx, epoch):
                self.save_train_images(output, data, threshold_output=True, stack_images=True)
            
            #print("finish one loop")
    def train_loop(self) -> None:
        """ Train the model for a given number of epochs
        """
        
        start_training_time = time.time()

        # first compute metrics
        self.compute_metrics(['miou', 'pq'], threshold_output=True, save_images=True)

        # put model in train mode
        self.vae_model.train()

        # start training loop
        for epoch in range(self.start_epoch, self.epochs):

            # track epochs
            print(colored('-'*25, 'blue'))
            print(colored(f"Starting epoch {epoch}", "blue"))

            # define progress meters
            losses = AverageMeter("Loss", ":.4e")
            ce_losses = AverageMeter("CE", ":.4e")
            kl_losses = AverageMeter("KL", ":.4e")
            mask_losses = AverageMeter("Mask", ":.4e")
            semseg_meter = SemsegMeter(self.p['num_classes'],  # noqa
                                       self.ds.get_class_names(),
                                       has_bg=False,
                                       ignore_index=self.ds.ignore_label,
                                       gpu_idx=self.args['gpu'])
            progress = ProgressMeter(
                len(self.dl),
                [losses, ce_losses, kl_losses, mask_losses],
                prefix="Epoch: [{}]".format(epoch),
            )
            log_dict = {}
            meters_dict = {"loss": losses, "ce": ce_losses, "mask": mask_losses, "kl": kl_losses}
            
            # randomize sampler
            if self.args['distributed']:
                self.dl.sampler.set_epoch(epoch)

            # start counting time
            start_epoch_time = time.time()

            # start looping over the batches
            self.train_single_epoch(epoch, meters_dict, progress, semseg_meter)

            # aggregate training results
            results_train = None
            if self.evaluate_trainset:
                semseg_meter.synchronize_between_processes()
                results_train = semseg_meter.return_score(verbose=False, name='train set')

            # save model
            dist.barrier()
            if is_main_process():
                self.save(epoch)
                print(colored(f'Model saved for run {self.p["name"]}', 'yellow'))

            # validate model
            self.compute_metrics(['miou', 'pq'], threshold_output=True, save_images=True)

            # log average loss at the end of the epoch
            if self.use_wandb and is_main_process():
                log_dict.update({"average_loss_epoch": losses.avg,
                                 "mIoU reconstruction train set": results_train['mIoU']})
                wandb.log(log_dict)

            # print statements at the end of the epoch
            print(colored(f'Average loss: {losses.avg:.3e}', 'yellow'))
            time_per_epoch = time.time() - start_epoch_time
            print(colored(f'Epoch took {str(timedelta(seconds=time_per_epoch))}', 'yellow'))
            avg_time_per_epoch = (time.time() - start_training_time) / (epoch + 1 - self.start_epoch)
            eta = avg_time_per_epoch * (self.epochs - 1 - epoch)
            print(colored(f'ETA: {str(timedelta(seconds=eta ))}', 'yellow'))
        print(meters_dict)
        # compute metrics at the end of training
        self.compute_metrics(['miou', 'pq'], threshold_output=True, save_images=True)
        print(f"Finished run {self.p['name']} and took {str(timedelta(seconds=time.time()-start_training_time))}")

    def update_meters(self, loss_dict: dict, meter_dict: dict, size: int = 1) -> dict:
        loss_dict = {name: gpu_gather(val.repeat(size)).mean().item() for name, val in loss_dict.items()}
        for name, meter in meter_dict.items():
            meter.update(loss_dict[name], size)
        return meter_dict

    def check_iter(self, batch_idx: int, epoch: int) -> bool:
        return (self.step != 0 and self.step % self.save_and_sample_every == 0) or \
               (epoch == self.epochs - 1 and batch_idx == len(self.dl) - 1)

    def get_lr_scheduler(
        self,
        name: str = 'cosine',
        final_lr: float = 0.0,
        warmup_iters: Optional[int] = None,
        lr_scaling: bool = False,
    ) -> Optional[np.ndarray]:
        """Returns the learning rate scheduler
        """

        # (optional) lr scaling
        if lr_scaling:
            self.eff_lr = self.lr * (self.eff_batch_size * get_world_size()) / 64.  # linear scaling rule
        else:
            self.eff_lr = self.lr

        if name == 'cosine':
            lr_schedule = cosine_scheduler(
                self.eff_lr,
                final_lr,
                self.epochs,
                self.num_iters_per_epoch,
                warmup_iters=warmup_iters,
            )
        elif name == 'warmup':
            lr_schedule = warmup_scheduler(
                self.eff_lr,
                final_lr,  # N/A
                self.epochs,
                self.num_iters_per_epoch,
                warmup_iters=warmup_iters,
            )
        elif name == 'step':
            lr_schedule = step_scheduler(
                self.eff_lr,
                final_lr,  # N/A
                self.epochs,
                self.num_iters_per_epoch,
                decay_epochs=[self.epochs // 2, 3 * self.epochs // 4],
                decay_rate=0.1,
                warmup_iters=warmup_iters,
            )
        else:
            raise NotImplementedError(f'Unknown lr scheduler: {name}')

        print(colored(
            f'Using lr scheduler {name} with '
            f'effective lr: {self.eff_lr:.3e}, '
            f'final lr: {final_lr:.3e}, '
            f'warmup iters {warmup_iters}',
            'yellow'))
        return lr_schedule

    def save(self, epoch: int) -> None:
        """ Saves the model
        """

        if not is_main_process():
            return
        data = {
            'step': self.step,
            'epoch': epoch,
            'vae': self.vae_model.state_dict(),
            'opt': self.opt.state_dict() if self.save_optim else None,
            'p': self.p,
            'scaler': self.fp16_scaler.state_dict() if self.fp16_scaler is not None else None
        }
        torch.save(data, str(self.results_folder / 'model.pt'))

    def resume(self) -> None:
        """ Resumes training from a saved model
        """

        model_path = str(self.results_folder / 'model.pt')
        if not os.path.exists(model_path):
            print(colored(f'No saved model at {model_path} to resume ...', 'blue'))
            return

        # load model
        print(colored(f'Resuming model {model_path} ...', 'blue'))
        data = torch.load(model_path, map_location='cpu')
        self.vae_model.load_state_dict(data['vae'])

        self.start_epoch = data['epoch'] + 1
        self.step = data['epoch'] * self.num_iters_per_epoch + 1
        if data['opt'] is not None:
            self.opt.load_state_dict(data['opt'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if data['scaler'] is not None:
            self.fp16_scaler.load_state_dict(data['scaler'])

    def load(self, model_path: Optional[str] = None) -> None:
        """ Resumes training from a saved model
        """

        print(colored('Start loading function ...', 'blue'))
        if model_path is None:
            model_path = str(self.results_folder / 'model.pt')
        if not os.path.exists(model_path):
            print(colored('No saved model ...', 'blue'))
            return
        print(colored(f'Loading saved model on all gpus {model_path} ...', 'blue'))
        data = torch.load(model_path, map_location='cpu')
        self.vae_model.load_state_dict(data['vae'])

    def compute_metrics(
        self,
        names: Union[List[str], str] = ['miou'],
        threshold_output: bool = False,
        save_images: bool = False
    ):
        """ Compute different metrics on the validation set
            NOTE: some variability is to be expected due to the random assignment of label ids.
        """

        assert isinstance(names, str) or isinstance(names, list)

        if not isinstance(names, list):
            names = [names]

        for name in names:
            if name.lower() == 'miou':
                self.compute_miou(threshold_output=threshold_output, save_images=save_images)
            elif name.lower() == 'pq':
                self.compute_cityscapes_pq(threshold_output=threshold_output, save_images=save_images)
            else:
                raise NotImplementedError(f'Unknown metric {name}')

            dist.barrier()

    def crop_padding(self, prediction: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # TODO handle this in a nicer way by loading the coordinates in the dataloader
        padding_co = padding_mask.nonzero()
        y_min, y_max = padding_co[:, 0].min(), padding_co[:, 0].max()  # make more efficient assuming a square
        x_min, x_max = padding_co[:, 1].min(), padding_co[:, 1].max()
        prediction = prediction[:, y_min:y_max + 1, x_min:x_max + 1]
        return prediction
    

    @torch.no_grad()
    def compute_pq(
        self,
        threshold_output: bool = True,
        save_images: bool = False,
    ):
        """
        Computes the panoptic quality on the validation set
        Currently only class agnostic evaluation is supported
        """

        from ldmseg.evaluations import PanopticEvaluatorAgnostic
        
        meta_data = self.ds_val.meta_data
        evaluator = PanopticEvaluatorAgnostic(meta=meta_data)
        evaluator.reset()

        count_th = self.count_th

        self.vae_model.eval()

        for batch_idx, data in tqdm(enumerate(self.dl_val)):
            file_names = [x["image_file"] for x in data['meta']]
            image_ids = [x["image_id"] for x in data['meta']]
            h, w = [x["im_size"][0] for x in data['meta']], [x["im_size"][1] for x in data['meta']]
            images = data['image_semseg'].cuda(self.args['gpu'], non_blocking=True)
            padding_masks = data['mask'].cuda(self.args['gpu'], non_blocking=True)
            images = 2. * images - 1.
            rgbs = None
            if self.fuse_rgb:
                rgbs = data['image'].cuda(self.args['gpu'], non_blocking=True)
                rgbs = 2. * rgbs - 1.
            masks_logits = self.vae_model(images, sample_posterior=False, rgb_sample=rgbs).sample

            # upsample masks
            masks_logits = F.interpolate(
                masks_logits,
                size=(images.shape[-2], images.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            # postprocess masks
            processed_results = []
            for image_idx, mask_pred_result in enumerate(masks_logits):

                # crop mask to get rid of padding
                mask_pred_result = self.crop_padding(mask_pred_result, padding_masks[image_idx])

                # interpolate to original size
                mask_pred_result = F.interpolate(
                    mask_pred_result[None, ...].float(),  # [1, C, H, W]
                    size=(h[image_idx], w[image_idx]),
                    mode="bilinear",
                    align_corners=False
                )[0]  # [C, H, W]

                # get panoptic prediction
                panoptic_pred = torch.argmax(mask_pred_result, dim=0)
                if threshold_output:
                    probs = F.softmax(mask_pred_result, dim=0)
                    probs = probs.max(dim=0)[0]
                    panoptic_pred[probs < self.mask_th] = -1

                # move to cpu
                panoptic_pred = panoptic_pred.cpu().numpy()

                processed_results.append({})
                segments_info = []
                for panoptic_label, count_i in zip(*np.unique(panoptic_pred, return_counts=True)):

                    # set small segments to void label (later we add 1 to get 0 for void class)
                    if count_i < self.count_th or panoptic_label in {-1, self.ds_val.ignore_label}:
                        panoptic_pred[panoptic_pred == panoptic_label] = -1
                        continue

                    # (optional) also enforce overlap between argmax and thresholded mask
                    original_mask = mask_pred_result[panoptic_label] >= self.mask_th
                    if (panoptic_pred == panoptic_label).sum() / original_mask.sum() < self.overlap_th:
                        panoptic_pred[panoptic_pred == panoptic_label] = -1
                        continue

                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": 1,
                            "isthing": True,
                        }
                    )
                processed_results[-1]["panoptic_seg"] = panoptic_pred + 1, segments_info

            evaluator.process(file_names, image_ids, processed_results)

            if is_main_process() and save_images and batch_idx == 0:
                self.overlay_predictions(file_names=file_names,
                                         processed_results=processed_results,
                                         meta_data=meta_data)

        evaluator.evaluate()

        self.vae_model.train()
        return
    
    @torch.no_grad()
    def compute_cityscapes_pq(self, threshold_output=True, save_images=False):
        """
        计算 Cityscapes 数据集的全景分割质量（PQ）
        
        此方法与 compute_pq_kitti 类似，但专门针对 Cityscapes 数据格式：
        1. 将模型置于评估模式
        2. 迭代验证集中的每个 batch：
           - 获取模型预测并进行后处理
           - 对于每个图像，提取预测的全景分割图和真实语义分割标签
           - 使用 compute_cityscapes_pq 函数评估质量
        3. 汇总并输出评估结果
        """
        from ldmseg.evaluations.cityscapes_pap_eval import compute_cityscapes_pq
        
        self.vae_model.eval()
        all_results = []
        
        for batch_idx, data in tqdm(enumerate(self.dl_val)):
            # 获取输入图像
            images = data['image_semseg'].cuda(self.args['gpu'], non_blocking=True)
            images = 2. * images - 1.
            
            # 获取RGB图像（如果需要）
            rgbs = None
            if self.fuse_rgb:
                rgbs = data['image'].cuda(self.args['gpu'], non_blocking=True)
                rgbs = 2. * rgbs - 1.
            
            # 进行模型推理
            output = self.vae_model(images, sample_posterior=False, rgb_sample=rgbs)
            
            # 对输出进行上采样到原始大小
            target_size = data['semseg'][0].shape
            output = F.interpolate(output.sample, size=target_size, mode="bilinear", align_corners=False)
            
            # 使用argmax获取预测的单通道全景分割图
            pred_panoptic = torch.argmax(output, dim=1)
            
            # 阈值处理（如果需要）
            if threshold_output:
                probs = F.softmax(output, dim=1)
                max_probs = probs.max(dim=1)[0]
                pred_panoptic[max_probs < self.mask_th] = 0
            
            # 评估每个样本
            batch_size = pred_panoptic.shape[0]
            for i in range(batch_size):
                # 获取当前样本的预测和真实标签
                pred = pred_panoptic[i].cpu().numpy()
                gt_semantic = data['semseg'][i].cpu().numpy().astype(np.int32)
                
                # 计算评估指标
                result = compute_cityscapes_pq(
                    panoptic_pred=pred,
                    gt_semantic=gt_semantic,
                    count_th=self.count_th,
                    mask_th=self.mask_th,
                    overlap_th=self.overlap_th
                )
                all_results.append(result)
            
            # 可选：保存第一个batch的可视化结果
            if save_images and batch_idx == 0 and is_main_process():
                self.save_panoptic_visualization(pred_panoptic, data, identifier='cityscapes')
        
        # 汇总评估结果
        if all_results:
            avg_pq = np.mean([r['pq'] for r in all_results])
            avg_sq = np.mean([r['sq'] for r in all_results])
            avg_rq = np.mean([r['rq'] for r in all_results])
            
            # 输出结果
            print(f"Cityscapes Panoptic Evaluation Results:")
            print(f"PQ: {avg_pq:.4f}, SQ: {avg_sq:.4f}, RQ: {avg_rq:.4f}")
            
            if len(all_results) > 0 and 'per_class' in all_results[0]:
                # 计算每个类别的平均PQ
                class_ids = set()
                for r in all_results:
                    class_ids.update(r['per_class'].keys())
                
                class_metrics = {}
                for class_id in class_ids:
                    class_pq = np.mean([r['per_class'].get(class_id, {'pq': 0})['pq'] for r in all_results if class_id in r['per_class']])
                    class_metrics[class_id] = class_pq
                
                # 打印类别级别的结果
                print("\n类别级别的PQ:")
                for class_id, pq in sorted(class_metrics.items()):
                    print(f"类别 {class_id}: PQ = {pq:.4f}")
            
            final_result = {
                'pq': avg_pq,
                'sq': avg_sq,
                'rq': avg_rq,
                'details': all_results
            }
            
            if self.use_wandb and is_main_process():
                wandb.log({
                    "cityscapes_pq": avg_pq,
                    "cityscapes_sq": avg_sq,
                    "cityscapes_rq": avg_rq
                })
            
            return final_result
        
        self.vae_model.train()
        return None
    
    def save_panoptic_visualization(self, pred_panoptic, data, identifier=''):
        """保存全景分割可视化结果"""
        if not is_main_process():
            return
            
        # 将预测结果转换为彩色图像
        predictions = pred_panoptic.cpu().numpy()
        colormap = get_color_map(256)  # 使用之前定义的颜色映射函数
        
        # 获取GT和RGB图像
        targets = data['semseg'].numpy()
        rgbs = (data['image'].numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        
        # 设置可视化参数
        size = predictions.shape[2]  # 宽度
        height = predictions.shape[1]  # 高度
        offset = int(0.02 * size)
        max_size = 6
        bs = min(predictions.shape[0], max_size)
        
        # 创建画布
        pred_array = np.zeros((height, bs * (size + offset), 3), dtype=np.uint8)
        gt_array = np.zeros((height, bs * (size + offset), 3), dtype=np.uint8)
        rgb_array = np.zeros((height, bs * (size + offset), 3), dtype=np.uint8)
        
        ptr = 0
        for j, (pred, target, rgb) in enumerate(zip(predictions, targets, rgbs)):
            # 为预测的全景分割添加颜色
            colored_pred = colorize_panoptic(pred, colormap)
            # 为GT添加颜色
            colored_gt = self.encode_seg(target[None, ...])[0]
            
            # 填充画布
            pred_array[:, ptr:ptr+size, :] = colored_pred
            gt_array[:, ptr:ptr+size, :] = colored_gt
            rgb_array[:, ptr:ptr+size, :] = rgb
            ptr += size + offset
            if j == max_size - 1:
                break
                
        # 保存结果
        stacked_image = np.vstack([rgb_array, gt_array, pred_array])
        filename = f'panoptic_vis_{identifier}.jpg'
        self.write_images(stacked_image, filename)
        print(f'save {filename}')

    @torch.no_grad()
    def save_train_images(self, output: dict, data: dict, threshold_output: bool = False, stack_images: bool = False):
        """ Saves images during training
        """
        if not is_main_process():
            return

        targets = data['semseg'].numpy()
        pap_gts = data['image_semseg'].numpy()
        rgbs = (data['image'].numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        output.sample = F.interpolate(output.sample, size=targets.shape[-2:], mode='bilinear', align_corners=True)
        preds = torch.argmax(output.sample, dim=1)

        if threshold_output:
            probs = F.softmax(output.sample, dim=1)
            probs = probs.max(dim=1)[0]
            preds[probs < self.mask_th] = self.ds.ignore_label

        predictions = preds.cpu().numpy()
        predictions = self.encode_seg(predictions).astype(np.uint8)
        targets = self.encode_seg(targets).astype(np.uint8)
        pap_gts = pap_gts.astype(np.uint8)
        masks = (data['mask'].unsqueeze(1).repeat(1, 3, 1, 1).numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        size = predictions.shape[1]
        size_2 = predictions.shape[2]
        offset = int(0.02 * size)
        max_size = 6
        bs = min(predictions.shape[0], max_size)
        pred_array = np.zeros((size, bs * (size_2 + offset), 3), dtype=np.uint8)
        gt_array = np.zeros((size, bs * (size_2 + offset), 3), dtype=np.uint8)
        rgb_array = np.zeros((size, bs * (size_2 + offset), 3), dtype=np.uint8)
        mask_array = np.zeros((size, bs * (size_2 + offset), 3), dtype=np.uint8)
        panpotic_array= np.zeros((size, bs * (size_2 + offset), 3), dtype=np.uint8)
        ptr = 0
        colormap = get_color_map(20)
        max_pool = nn.MaxPool2d(kernel_size=2, stride=1,padding=1)
        for j, (semseg, target, rgb, mask,pap_gt) in enumerate(zip(predictions, targets, rgbs, masks,pap_gts)):
            seg = self.ds.decode_bitmap(data['image_semseg'][j,:,:,:],n = 16).cpu().numpy().astype(np.uint8)  # [3, H, W], 已经是 0～1 之间
            
            pop=seg
            color_image = colorize_panoptic(pop, colormap)
            img_tensor = torch.tensor(color_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            pooled_tensor = max_pool(max_pool(max_pool(img_tensor)))
                # 4. 转换回 NumPy 数组，并将通道维度移到最后，得到形状 (88, 620, 3)
            pooled_tensor=F.interpolate(pooled_tensor, size=[size,size_2])

            pooled_image = pooled_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
            pred_array[:, ptr:ptr+size_2, :] = semseg
            gt_array[:, ptr:ptr+size_2, :] = target
            rgb_array[:, ptr:ptr+size_2, :] = rgb
            mask_array[:, ptr:ptr+size_2, :] = mask
            panpotic_array[:, ptr:ptr+size_2, :] = pooled_image
            ptr += size_2 + offset
            if j == max_size - 1:
                break
            
        if stack_images:
            self.write_images(np.vstack([rgb_array, gt_array, pred_array, mask_array, panpotic_array]), 'rgb_gt_pred_ae_train.jpg')
        else:
            self.write_images([pred_array, gt_array, rgb_array],
                              ['pred_ae_train.png', 'gt_ae_train.png', 'rgb_ae_train.jpg'])

    @torch.no_grad()
    def compute_miou(self, threshold_output=False, stack_images=True, save_images=False):
        """
        Computes the mIoU on the validation set
        Note that this is an approximation since we do not resize to the original image size
        """

        print(colored('Distributed evaluation on the validation set', 'blue'))
        if threshold_output:
            print('Thresholding output')
        self.vae_model.eval()
        semseg_meter = SemsegMeter(self.p['num_classes'],
                                   self.ds.get_class_names(),
                                   has_bg=False,
                                   ignore_index=self.ds.ignore_label,
                                   gpu_idx=self.args['gpu'])

        for batch_idx, data in tqdm(enumerate(self.dl_val)):
            images = data['image_semseg'].cuda(self.args['gpu'], non_blocking=True)
            
            targets = data['semseg'].cuda(self.args['gpu'], non_blocking=True)
            images = 2. * images - 1.
            rgbs = None
            if self.fuse_rgb:
                rgbs = data['image'].cuda(self.args['gpu'], non_blocking=True)
                rgbs = 2. * rgbs - 1.
            output = self.vae_model(images, sample_posterior=False, rgb_sample=rgbs)

            output.sample = F.interpolate(output.sample, size=targets.shape[-2:], mode='bilinear', align_corners=True)
            
            preds = torch.argmax(output.sample, dim=1)

            

            if threshold_output:
                probs = F.softmax(output.sample, dim=1)
                probs = probs.max(dim=1)[0]
                preds[probs < self.mask_th] = self.ds.ignore_label
                
           
            semseg_meter.update(preds, targets)

            # save images
            if save_images and batch_idx == 0 and is_main_process():
                predictions = preds.cpu().numpy().astype(np.uint8)
                

                predictions = self.encode_seg(predictions).astype(np.uint8)
                
                targets = targets.cpu().numpy()
                targets = self.encode_seg(targets).astype(np.uint8)
                rgbs = (data['image'].numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
                masks = (data['mask'].unsqueeze(1).repeat(1, 3, 1, 1).numpy().transpose(0, 2, 3, 1) * 255
                         ).astype(np.uint8)
                size = predictions.shape[1]
                size_0 = predictions.shape[0]
                size_2 = predictions.shape[2]
                
                offset = int(0.02 * size)
                max_size = 6
                bs = min(predictions.shape[0], max_size)
                pred_array = np.zeros((size, bs * (size_2 + offset), 3), dtype=np.uint8)
                gt_array = np.zeros((size, bs * (size_2 + offset), 3), dtype=np.uint8)
                rgb_array = np.zeros((size, bs * (size_2 + offset), 3), dtype=np.uint8)
                mask_array = np.zeros((size, bs * (size_2 + offset), 3), dtype=np.uint8)
                panpotic_array= np.zeros((size, bs * (size_2 + offset), 3), dtype=np.uint8)
                
                ptr = 0
                colormap = get_color_map(20)
                max_pool = nn.MaxPool2d(kernel_size=2, stride=1,padding=1)
                for j, (semseg, target, rgb, mask) in enumerate(zip(predictions, targets, rgbs, masks)):
                    seg = self.ds.decode_bitmap(data['image_semseg'][j,:,:,:],n = 16).cpu().numpy().astype(np.uint8)  # [3, H, W], 已经是 0～1 之间
                    
                    pop=seg
                    color_image = colorize_panoptic(pop, colormap)
                    img_tensor = torch.tensor(color_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                    pooled_tensor = max_pool(max_pool(max_pool(img_tensor)))
                        # 4. 转换回 NumPy 数组，并将通道维度移到最后，得到形状 (88, 620, 3)
                    pooled_tensor=F.interpolate(pooled_tensor, size=[size,size_2])

                    pooled_image = pooled_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    
                    mask_array[:, ptr:ptr+size_2, :] = mask
                    pred_array[:, ptr:ptr+size_2, :] = semseg
                    gt_array[:, ptr:ptr+size_2, :] = target
                    rgb_array[:, ptr:ptr+size_2, :] = rgb
                    panpotic_array[:, ptr:ptr+size_2, :] = pooled_image
                    ptr += size_2 + offset
                    if j == max_size - 1:
                        break
                if stack_images:
                    self.write_images(np.vstack([rgb_array, gt_array, pred_array, mask_array,panpotic_array]),
                                      'rgb_gt_pred_ae_val.jpg')
                else:
                    self.write_images([pred_array, gt_array, rgb_array],
                                      ['pred_ae_val.png', 'gt_ae_val.png', 'rgb_ae_val.jpg'])
                print('images saved')

        # accumulate results from all processes
        semseg_meter.synchronize_between_processes()
        results_val = semseg_meter.return_score(verbose=False, name='val set')

        self.vae_model.train()
        return results_val

    def encode_seg(self, semseg, cmap=None):
        # we will encode the semseg map with a color map
        if cmap is None:
            cmap = color_map()
        seg_t = semseg.astype(np.uint8)
        array_seg_t = np.empty((seg_t.shape[0], seg_t.shape[1], seg_t.shape[2], cmap.shape[1]), dtype=cmap.dtype)
        for class_i in np.unique(seg_t):
            array_seg_t[seg_t == class_i] = cmap[class_i]
        return array_seg_t

    def write_images(self, images: Union[np.ndarray, List[np.ndarray]], path_names: Union[str, List[str]]) -> None:
        """ Write images to disk
        """
        if isinstance(images, np.ndarray):
            images = [images]
        if isinstance(path_names, str):
            path_names = [path_names]
        for image, file_name in zip(images, path_names):
            image = Image.fromarray(image)
            image.save(os.path.join(self.visualization_dir, file_name))

    def overlay_predictions(self, file_names, processed_results, meta_data, identifier: str = ''):
        from ldmseg.utils import MyVisualizer
        import cv2

        bs = len(file_names)
        size = self.image_size
        offset = int(0.02 * size)
        panoptic_overlay_array = np.zeros((size, bs * (size + offset), 3), dtype=np.uint8)

        ptr = 0
        for file_name, processed_res in zip(file_names, processed_results):
            curr_image = Image.open(file_name).convert("RGB")
            curr_image = np.asarray(curr_image)
            vis_image = MyVisualizer(curr_image, metadata=meta_data, scale=1.0)
            res = vis_image.draw_panoptic_seg(
                torch.from_numpy(processed_res["panoptic_seg"][0]),
                processed_res["panoptic_seg"][1],
                suppress_thing_labels=True,
                random_colors=True,
                alpha=0.8,
            )
            ratio = size / max(curr_image.shape[:2])
            h_new, w_new = int(curr_image.shape[0] * ratio), int(curr_image.shape[1] * ratio)
            image_overlay = cv2.resize(res.get_image(), (w_new, h_new), interpolation=cv2.INTER_CUBIC)
            panoptic_overlay_array[:h_new, ptr:ptr+w_new] = image_overlay
            ptr += size + offset

        key = f'panoptic_overlay{identifier}.jpg'
        self.write_images(panoptic_overlay_array, key)
        return

    @torch.no_grad()
    def compute_pq_kitti(self, threshold_output=True, save_images=False):
        """
        针对 KITTI 数据集计算全景质量（PQ）
          1. 将模型置于评估模式
          2. 对验证集中的每个 batch：
               - 对图像输入（image_semseg）归一化处理后，执行模型预测，
                 并用 argmax 得到预测的单通道全景分割图（假设输出通道对应类别概率）。
               - 对于 batch 中的每张图像，提取 GT 的语义分割（'semseg'）和实例分割（'instance'），
                 转换为 numpy 数组（注意数据类型转换），
                 调用 evaluator.add_image(pred, gt_semseg, gt_instance) 累积评价统计。
          3. 遍历完所有 batch 后，通过 evaluator.evaluate() 输出 PQ、SQ、RQ 等指标。
        """
        self.vae_model.eval()
        evaluator = KITTIPanopticEvaluator(
            iou_thresh=0.5,
            thing_ids={10, 11, 12, 13, 14, 15, 16, 17},
            ignore_label=self.ds_val.ignore_label,
            max_ins=30
        )
        
        for batch_idx, data in tqdm(enumerate(self.dl_val)):
            # 获取经过预处理的输入图像（例如 image_semseg 经过归一化）
            images = data['image_semseg'].cuda(self.args['gpu'], non_blocking=True)
            # 假设输入已归一化为 [-1,1]，否则可按 2*x-1 处理
            images = 2. * images - 1.
            target_size = data['semseg'][0].shape
            
            # 添加RGB支持
            rgbs = None
            if self.fuse_rgb:
                rgbs = data['image'].cuda(self.args['gpu'], non_blocking=True)
                rgbs = 2. * rgbs - 1.
            
            # 模型推理：得到预测 logits（形状 (B, C, H, W)）
            output = self.vae_model(images, sample_posterior=False, rgb_sample=rgbs)
            output = F.interpolate(output.sample, size=target_size, mode="bilinear", align_corners=False)
            
            # 对输出使用 argmax 获得预测分割（形状 (B, H, W)）
            pred_seg = torch.argmax(output, dim=1).cpu().numpy()
            
            batch_size = pred_seg.shape[0]
            for i in range(batch_size):
                # 获取 GT 的语义与实例分割，转换为 numpy 数组（保证为 int32 类型）
                gt_semseg = data['semseg'][i].cpu().numpy().astype(np.int32)
               
                
                # 生成预测的实例分割图（这里简化处理，直接用语义预测作为实例ID）
                # 在实际应用中，您可能需要根据您的数据和任务来调整此部分
                pred_instance = np.zeros_like(pred_seg[i])
                
                # 更新评价器（内部会先合并 semseg 与 instance，再计算匹配 IoU）
                evaluator.add_image(pred_seg[i], pred_instance, gt_semseg, gt_instance)

        results = evaluator.evaluate()
        print("KITTI 全景评价结果：", results)
        self.vae_model.train()
        return results
