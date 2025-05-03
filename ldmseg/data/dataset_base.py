"""
Author: Wouter Van Gansbeke (Adapted for KITTI)
Dataset class to be used for training and evaluation.
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
from torch import nn
from torchvision import transforms as T
from typing import Callable, Dict, Tuple, Any, Optional

class DatasetBase(object):
    def __init__(self, data_dir: str) -> None:
        """Base class for datasets."""
        self.data_dir = data_dir

    def get_train_transforms(self, p: Dict[str, Any]) -> Callable:
        """Returns a composition of transformations for training images."""
        normalize = T.Normalize(**p['normalize_params']) if p['normalize'] else nn.Identity()
        if p['type'] == 'crop_resize_pil':
            from .util import pil_transforms as pil_tr
            size = p['size']
            size_2 = p['size_2']
            normalize = pil_tr.Normalize(**p['normalize_params']) if p['normalize'] else nn.Identity()
            transforms = T.Compose([
                pil_tr.RandomHorizontalFlip() if p['flip'] else nn.Identity(),
                pil_tr.CropResize((size, size_2), crop_mode=None),
                pil_tr.ToTensor(),
                normalize
            ])
        else:
            raise NotImplementedError(f'Unknown transformation type {p["type"]}')
        return transforms

    def get_val_transforms(self, p: Dict) -> Callable:
        """Returns a composition of transformations for validation images."""
        normalize = T.Normalize(**p['normalize_params']) if p['normalize'] else nn.Identity()
        if p['type'] in ['crop_resize_pil', 'random_crop_resize_pil']:
            from .util import pil_transforms as pil_tr
            size = p['size']
            size_2 = p['size_2']
            normalize = pil_tr.Normalize(**p['normalize_params']) if p['normalize'] else nn.Identity()
            transforms = T.Compose([
                pil_tr.CropResize((size, size_2), crop_mode=None),
                pil_tr.ToTensor(),
                normalize
            ])
        else:
            raise NotImplementedError(f'Unknown transformation type {p["type"]}')
        return transforms

    def get_dataset(
        self,
        db_name: str,
        *,
        split: Any,
        tokenizer: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        remap_labels: bool = False,
        caption_dropout: float = 0.0,
        download: bool = False,
        overfit: bool = False,
        encoding_mode: str = 'color',
        caption_type: Optional[str] = 'none',
        inpaint_mask_size: Optional[Tuple[int]] = None,
        num_classes: Optional[int] = None,
        fill_value: Optional[int] = None,
        ignore_label: Optional[int] = None,
        inpainting_strength: Optional[float] = None,
    ) -> Any:
        """Returns the dataset for training or evaluation."""
        if db_name in 'coco':
            from .coco import COCO
            dataset_cls = COCO
        elif db_name in ['kitti', 'simikitti-dvps']:
            from .kitti import KITTI
            dataset_cls = KITTI
        elif db_name in ['cityscapes', 'cityscapes-dvps']:
            from .cityscapes import Cityscapes
            dataset_cls = Cityscapes
        else:
            raise NotImplementedError(f'Unknown dataset {db_name}')

        if isinstance(split, list):
            datasets = [
                dataset_cls(
                    prefix=self.data_dir,
                    split=sp,
                    transform=transform,
                    download=download,
                    remap_labels=remap_labels,
                    tokenizer=tokenizer,
                    caption_dropout=caption_dropout,
                    overfit=overfit,
                    caption_type=caption_type,
                    encoding_mode=encoding_mode,
                    inpaint_mask_size=inpaint_mask_size,
                    num_classes=num_classes,
                    fill_value=fill_value,
                    ignore_label=ignore_label,
                    inpainting_strength=inpainting_strength,
                ) for sp in split
            ]
            return torch.utils.data.ConcatDataset(datasets)
        else:
            dataset = dataset_cls(
                prefix=self.data_dir,
                split=split,
                transform=transform,
                download=download,
                remap_labels=remap_labels,
                tokenizer=tokenizer,
                caption_dropout=caption_dropout,
                overfit=overfit,
                caption_type=caption_type,
                encoding_mode=encoding_mode,
                inpaint_mask_size=inpaint_mask_size,
                num_classes=num_classes,
                fill_value=fill_value,
                ignore_label=ignore_label,
                inpainting_strength=inpainting_strength,
            )
            return dataset

if __name__ == '__main__':
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    import os
    import torch
    from PIL import Image

    # 设置数据集路径
    dataset_root = '/root/autodl-tmp/cityscapes-dvps/video_sequence'
    
    # 创建 DatasetBase 实例
    dataset_base = DatasetBase(data_dir=dataset_root)
    
    # 定义 transform
    transform = T.Compose([
        T.Resize((192, 640)),  # 对 image 使用 bilinear resize
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    dataset = dataset_base.get_dataset(
        db_name='cityscapes',
        split='train',
        transform=transform,
        encoding_mode='bits',
        num_classes=30,
        fill_value=0.5,
        ignore_label=0
    )
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # 获取第一个样本
    sample = next(iter(dataloader))
    
    # 打印样本信息
    print("\nSample information:")
    print("Image shape:", sample['image'].shape)
    print("Semantic segmentation shape:", sample['semseg'].shape)
    print("Depth shape:", sample['depth'].shape)
    print("Mask shape:", sample['mask'].shape)
    print("Image semantic segmentation shape:", sample['image_semseg'].shape)
    print("Inpainting mask shape:", sample['inpainting_mask'].shape)
    print("Meta information:", sample['meta'])
    
    # 创建输出目录
    out_dir = os.path.join(os.getcwd(), 'sample_outputs')
    os.makedirs(out_dir, exist_ok=True)
    
    # 保存样本数据
    # 1. 保存RGB图像
    img = sample['image'][0]  # [3, H, W]
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    img = img * std + mean  # 反归一化
    img = img.clamp(0, 1) * 255  # [0,255]
    img = img.byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(os.path.join(out_dir, 'image.png'))
    
    # 2. 保存语义分割标签
    sem = sample['semseg'][0].cpu().numpy()
    Image.fromarray(sem.astype(np.uint8)).save(os.path.join(out_dir, 'semseg.png'))
    
    # 3. 保存掩码
    m = sample['mask'][0].cpu().numpy()
    Image.fromarray(m.astype(np.uint8)).save(os.path.join(out_dir, 'mask.png'))
    
    # 4. 保存深度图
    depth = sample['depth'][0].cpu().numpy()
    # 归一化深度图以便可视化
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    Image.fromarray(depth_normalized.astype(np.uint8)).save(os.path.join(out_dir, 'depth.png'))
    
    # 5. 保存编码后的语义分割
    seg_bit = sample['image_semseg'][0].cpu().numpy()
    # 保存每个bit通道
    for i in range(seg_bit.shape[0]):
        bit_channel = (seg_bit[i] * 255).astype(np.uint8)
        Image.fromarray(bit_channel).save(os.path.join(out_dir, f'bit_channel_{i}.png'))
    
    print(f"\nSaved sample outputs to {out_dir}")
