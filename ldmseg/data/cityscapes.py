"""
Author: (Adapted from Wouter Van Gansbeke)
Dataset class for Cityscapes Panoptic Segmentation
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from typing import Optional, Tuple, Any
import random
from collections import defaultdict
import torchvision.transforms as T
import torch.nn as nn
from ldmseg.data.util.mypath import MyPath
from ldmseg.utils.utils import color_map
from ldmseg.data.util.mask_generator import MaskingGenerator

max_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

class Cityscapes(data.Dataset):
    CITYSCAPES_CATEGORIES = [
    {"color": [128,  64, 128], "isthing": 0, "id":  0, "name": "road"},
    {"color": [244,  35, 232], "isthing": 0, "id":  1, "name": "sidewalk"},
    {"color": [ 70,  70,  70], "isthing": 0, "id":  2, "name": "building"},
    {"color": [102, 102, 156], "isthing": 0, "id":  3, "name": "wall"},
    {"color": [190, 153, 153], "isthing": 0, "id":  4, "name": "fence"},
    {"color": [153, 153, 153], "isthing": 0, "id":  5, "name": "pole"},
    {"color": [250, 170,  30], "isthing": 0, "id":  6, "name": "traffic light"},
    {"color": [220, 220,   0], "isthing": 0, "id":  7, "name": "traffic sign"},
    {"color": [107, 142,  35], "isthing": 0, "id":  8, "name": "vegetation"},
    {"color": [152, 251, 152], "isthing": 0, "id":  9, "name": "terrain"},
    {"color": [ 70, 130, 180], "isthing": 0, "id": 10, "name": "sky"},
    {"color": [220,  20,  60], "isthing": 1, "id": 11, "name": "person"},
    {"color": [255,   0,   0], "isthing": 1, "id": 12, "name": "rider"},
    {"color": [  0,   0, 142], "isthing": 1, "id": 13, "name": "car"},
    {"color": [  0,   0,  70], "isthing": 1, "id": 14, "name": "truck"},
    {"color": [  0,  60, 100], "isthing": 1, "id": 15, "name": "bus"},
    {"color": [  0,  80, 100], "isthing": 1, "id": 16, "name": "train"},
    {"color": [  0,   0, 230], "isthing": 1, "id": 17, "name": "motorcycle"},
    {"color": [119,  11,  32], "isthing": 1, "id": 18, "name": "bicycle"},
]

    CITYSCAPES_CATEGORY_NAMES = [k["name"] for k in CITYSCAPES_CATEGORIES]
    def __init__(
        self,
        prefix: str,
        split: str = 'train',
        tokenizer: Optional[Any] = None,
        transform: Optional[Any] = None,
        download: bool = False,
        remap_labels: bool = False,
        caption_dropout: float = 0.0,
        overfit: bool = False,
        encoding_mode: str = 'bits',   # 'color', 'random_color', 'bits', 'none'
        caption_type: str = 'none',     # 'none', 'caption', 'class_label', 'blip'
        inpaint_mask_size: Optional[Tuple[int]] = None,
        num_classes: int = 128,
        fill_value: int = 0.5,
        ignore_label: int = 0,
        inpainting_strength: float = 0.0,
    ):
        """
        Args:
          prefix: 数据集根目录，例如 '/root/autodl-tmp/video_sequence'
          split: 'train' 或 'val'
          tokenizer: 用于 caption 处理
          transform: 针对输入 image 的 transform
          download: 必须为 False
          remap_labels: 是否重新映射标签
          caption_dropout: caption 的 dropout 概率
          overfit: 是否过拟合到少量样本
          encoding_mode: 编码模式
          caption_type: caption 类型
          inpaint_mask_size: inpainting 掩码尺寸
          num_classes: 类别数量
          fill_value: bit encoding 时 ignore 区域的填充值
          ignore_label: ignore label 数值
          inpainting_strength: inpainting 掩码生成强度
        """
        transform = T.Compose([
            T.Resize((192, 640)),  # 对 image 使用 bilinear resize
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        
        self.root = prefix
        self.prefix = prefix
        valid_splits = ['train', 'val', 'test']
        print("Split:", split)
        assert split in valid_splits, f"Split must be one of {valid_splits}"
        assert not download, "download must be False"
        self.split = split
        self.tokenizer = tokenizer
        self.caption_dropout = caption_dropout
        assert caption_type in ['none', 'caption', 'class_label', 'blip']
        assert encoding_mode in ['color', 'random_color', 'bits', 'none']
        self.encoding_mode = encoding_mode
        self.caption_type = caption_type

        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.fill_value = fill_value
        self.inpainting_strength = inpainting_strength
        self.remap_labels = True
        if inpaint_mask_size is None:
            inpaint_mask_size = (64, 64)
        self.maskgenerator = MaskingGenerator(input_size=inpaint_mask_size, mode='random_local')
        self.meta_data = self.get_metadata()
        self.transform = transform
        self.cmap = color_map()

        print("Initializing dataloader for Cityscapes {} set".format(split))
        _image_dir = os.path.join(self.root, split)
        self.samples = []
        sample_dict = {}

        # 遍历目录中的所有文件
        for file in sorted(os.listdir(_image_dir)):
            base, ext = os.path.splitext(file)
            if ext.lower() != ".png":
                continue
                
            parts = base.split('_')
            if len(parts) >= 5:  # 确保文件名格式正确
                scene, frame = parts[0], parts[1]
                typ = parts[-1]  # 'depth', 'instanceTrainIds', 'leftImg8bit'
                
                if scene not in sample_dict:
                    sample_dict[scene] = {}
                if frame not in sample_dict[scene]:
                    sample_dict[scene][frame] = {}
                sample_dict[scene][frame][typ] = os.path.join(_image_dir, file)

        # 收集完整的样本
        for scene, frames in sample_dict.items():
            for frame, files in frames.items():
                # if scene >"000003":
                #     continue
                # if frame >"000003":
                #     continue
                if all(key in files for key in ['leftImg8bit', 'instanceTrainIds', 'depth']):
                    self.samples.append(files)

        print("Found {} samples in split {}".format(len(self.samples), split))

        if self.split == 'train':
            self.pixel_threshold = 10
            self.training = True
        else:
            self.pixel_threshold = 0
            self.training = False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = {}
        sample_paths = self.samples[idx]
        
        # 1. 读取并处理RGB图像
        image = Image.open(sample_paths['leftImg8bit']).convert('RGB')
        image = image.resize((640, 192), Image.BILINEAR)
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = T.ToTensor()(image)
        sample['image'] = image_tensor

        # 2. 读取全景分割标签
        panoptic_img = Image.open(sample_paths['instanceTrainIds'])
        panoptic_img = panoptic_img.resize((640, 192), Image.NEAREST)
        panoptic_np = np.array(panoptic_img, dtype=np.uint32)
        panoptic_np = panoptic_np.astype(np.int32)

        # 保存原始分割信息（如果需要）
        unique_classes = np.unique(panoptic_np)
        unique_classes = unique_classes[unique_classes != self.ignore_label]

        # 使用COCO风格的重映射
        if self.remap_labels:
            remapped_panoptic, mapping = self._remap_labels_fn(
                panoptic_np, 
                max_val=self.num_classes, 
                keep_background_fixed=True,
                min_pixels=10  # 设置小区域的像素阈值
            )
            # 更新分割元信息（如果有需要）
            # 这里可以添加segments_info的更新，类似COCO中的处理
        else:
            # 使用当前简单的重映射方法
            unique_ids = np.unique(panoptic_np)
            id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
            remapped_panoptic = np.zeros_like(panoptic_np)
            for old_id, new_id in id_mapping.items():
                remapped_panoptic[panoptic_np == old_id] = new_id

        # 验证重映射的最大值
        assert remapped_panoptic.max() < self.num_classes, f"Remapped values exceed num_classes: {remapped_panoptic.max()} vs {self.num_classes}"

        sample['semseg'] = torch.from_numpy(remapped_panoptic).long()

        # 3. 读取深度图
        depth_img = Image.open(sample_paths['depth'])
        depth_img = depth_img.resize((640, 192), Image.BILINEAR)
        depth_np = np.array(depth_img, dtype=np.float32)
        sample['depth'] = torch.from_numpy(depth_np)

        # 4. 创建掩码
        mask_np = np.ones_like(panoptic_np, dtype=np.uint8)
        sample['mask'] = torch.from_numpy(mask_np)
        sample['mask'][remapped_panoptic > 128] = 0
        sample['mask'][remapped_panoptic < 0] = 0
        # 5. 处理全景分割标签
        if self.encoding_mode == 'bits':
            seg_bit, _ = self.encode_bitmap(sample['semseg'], n=16, fill_value=self.fill_value)
            sample['image_semseg'] = seg_bit
        elif self.encoding_mode == 'none':
            sample['image_semseg'] = sample['semseg'].unsqueeze(0).repeat(3, 1, 1).float() / self.num_classes

        # 6. 创建元数据
        try:
            parts = os.path.basename(sample_paths['leftImg8bit']).split('_')
            scene = int(parts[0])
            frame = int(parts[1])
            image_id = scene * 10000 + frame
        except Exception:
            image_id = os.path.basename(sample_paths['leftImg8bit'])

        meta = {
            'im_size': (192, 640),
            'image_file': sample_paths['leftImg8bit'],
            'image_id': image_id,
            'segments_info': {}  # 暂不提供详细 segments_info
        }
        sample['meta'] = meta

        # 7. 添加文本和inpainting掩码
        sample['text'] = ""
        inpainting_mask = self.maskgenerator(t=self.inpainting_strength)
        sample['inpainting_mask'] = torch.from_numpy(inpainting_mask).bool()

        # 8. 如果提供tokenizer，则进行tokenization
        if self.tokenizer is not None:
            sample['tokens'] = self.tokenizer(sample['text'],
                                            padding='max_length',
                                            max_length=77,
                                            truncation=True,
                                            return_tensors='pt').input_ids.squeeze(0)

        return sample

    def encode_bitmap(self, x: torch.Tensor, n: int = 5, fill_value: float = 0.5):
        ignore_mask = x == self.ignore_label
        x = torch.bitwise_right_shift(x, torch.arange(n, device=x.device)[:, None, None])
        x = torch.remainder(x, 2).float()
        x[:, ignore_mask] = fill_value
        return x, ignore_mask

    def decode_bitmap(self, x: torch.Tensor, n: int = 5):
        x = (x > 0.).float()
        n = x.shape[0]
        x = x * 2 ** torch.arange(n, device=x.device)[:, None, None]
        x = torch.sum(x, dim=0)
        x = x.long()
        x[x == 31] = 0
        return x

    def get_inpainting_mask(self, strength=0.5):
        mask = self.maskgenerator(t=strength)
        mask = torch.from_numpy(mask).bool()
        return mask
    def get_class_names(self):
        return self.CITYSCAPES_CATEGORIES
    def get_metadata(self):
        meta = {}
        thing_dataset_id_to_contiguous_id = {}
        stuff_dataset_id_to_contiguous_id = {i: i for i in range(self.num_classes)}
        cat2name = {i: f"cls_{i}" for i in range(self.num_classes)}
        meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
        meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
        meta["cat2name"] = cat2name
        meta["panoptic_json"] = None
        meta["panoptic_root"] = self.root
        return meta

    def __str__(self):
        return 'Cityscapes(split=' + str(self.split) + ')'

    def _remap_labels_fn(self, labels, max_val=None, keep_background_fixed=True, min_pixels=10):
        """
        将标签重映射到较小范围内的随机ID，在一开始就将小区域标记为最大ID
        
        参数:
            labels: 输入标签数组
            max_val: 最大允许的标签值，默认为self.num_classes
            keep_background_fixed: 是否保持背景类别索引不变
            min_pixels: 像素点数量的最小阈值，少于此值的区域将被置为最大ID
            
        返回:
            remapped_labels: 重映射后的标签数组
            mapping: 原标签到新标签的映射字典
        """
        # 设置最大值，默认使用类别数
        max_val = max_val if max_val is not None else self.num_classes
        max_target_val = max_val - 1  # 最大标签值，用于小区域
        
        # 1. 先创建输出标签数组，默认填充忽略标签
        remapped_labels = np.full(labels.shape, self.ignore_label, dtype=labels.dtype)
        
        # 2. 统计每个标签的像素数量
        unique_values, counts = np.unique(labels, return_counts=True)
        
        # 3. 将忽略标签排除
        valid_mask = unique_values != self.ignore_label
        unique_values = unique_values[valid_mask]
        counts = counts[valid_mask]
        
        # 4. 在一开始就将小区域(像素<min_pixels)标记为最大ID
        mapping = {}  # 最终的ID映射字典
        
        # 先处理小区域
        for idx, (val, count) in enumerate(zip(unique_values, counts)):
            if count < min_pixels:
                # 直接将小区域映射到最大ID
                mapping[val] = max_target_val
                # 应用到输出标签
                remapped_labels[labels == val] = max_target_val
        
        # 5. 筛选出需要参与正常重映射的区域
        normal_regions = []
        for val, count in zip(unique_values, counts):
            if count >= min_pixels and val != self.ignore_label:
                normal_regions.append(val)
        
        # 6. 检查正常区域是否需要进一步筛选(如果数量超过可用ID范围)
        available_values = np.arange(1, max_target_val)  # 1到max_target_val-1的范围
        
        if len(normal_regions) > len(available_values):
            # 按像素数量排序，只保留像素最多的前N个区域
            region_counts = {val: np.sum(labels == val) for val in normal_regions}
            sorted_regions = sorted(region_counts.keys(), key=lambda x: region_counts[x], reverse=True)
            
            # 分为保留区域和剩余区域
            kept_regions = sorted_regions[:len(available_values)]
            remaining_regions = sorted_regions[len(available_values):]
            
            # 剩余区域也映射到最大ID
            for val in remaining_regions:
                mapping[val] = max_target_val
                remapped_labels[labels == val] = max_target_val
            
            # 更新正常区域列表
            normal_regions = kept_regions
        
        # 7. 为正常区域分配随机ID(范围1~max_target_val-1)
        if normal_regions:
            targets = np.random.choice(available_values, size=len(normal_regions), replace=False)
            for val, target in zip(normal_regions, targets):
                mapping[val] = target
                remapped_labels[labels == val] = target
        
        return remapped_labels, mapping

if __name__ == '__main__':
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    import numpy as np
    import os

    # 设置数据集路径
    dataset_root = '/root/autodl-tmp/cityscapes-dvps/video_sequence'
    
    # 创建数据集实例
    dataset = Cityscapes(
        prefix=dataset_root,
        split='train',
        transform=T.Compose([
            T.Resize((192, 640)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ]),
        remap_labels=False,
        encoding_mode="bits"
    )

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # 获取第一个样本
    sample = next(iter(dataloader))

    # 创建输出目录
    out_dir = os.path.join(os.getcwd(), 'sample_outputs')
    os.makedirs(out_dir, exist_ok=True)

    # 打印样本信息
    print("\nSample information:")
    print("Image shape:", sample['image'].shape)
    print("Semantic segmentation shape:", sample['semseg'].shape)
    print("Depth shape:", sample['depth'].shape)
    print("Mask shape:", sample['mask'].shape)
    print("Image semantic segmentation shape:", sample['image_semseg'].shape)
    print("Inpainting mask shape:", sample['inpainting_mask'].shape)
    print("Meta information:", sample['meta'])

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
    print(np.unique(sem,return_counts=True))
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