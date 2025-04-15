"""
Author: (Adapted from Wouter Van Gansbeke)
Dataset class for KITTI Panoptic Segmentation and Mask Inpainting
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

from ldmseg.data.util.mypath import MyPath
from ldmseg.utils.utils import color_map
from ldmseg.data.util.mask_generator import MaskingGenerator

# ----------------- 辅助函数 -----------------

def get_color_map(num_colors):
    np.random.seed(42)
    return np.random.randint(0, 256, (num_colors, 3), dtype=np.uint8)

def colorize_panoptic(panoptic_map, colormap):
    h, w = panoptic_map.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(panoptic_map)
    for uid in unique_ids:
        if uid == 255:
            color = np.array([0, 0, 0], dtype=np.uint8)
        else:
            color = colormap[uid % len(colormap)]
        color_image[panoptic_map == uid] = color
    return color_image

def encode_segmentation_mask(seg_img: np.ndarray, color_to_label: dict) -> np.ndarray:
    H, W = seg_img.shape
    label_map = np.zeros((H, W), dtype=np.int64)
    for color, label in color_to_label.items():
        mask = np.all(seg_img == np.array(color, dtype=np.uint8), axis=-1)
        label_map[mask] = label
    return label_map

def encode_bitmap(x: torch.Tensor, n: int = 7, fill_value: float = 0.5):
    """
    将二维标签图 x (H, W) 进行 bit 编码，返回形状 (n, H, W) 的 Tensor 以及 ignore mask。
    假设 ignore_label 为 0。
    """
    ignore_mask = x == 0
    x = torch.bitwise_right_shift(x, torch.arange(n, device=x.device)[:, None, None])
    x = torch.remainder(x, 2).float()
    x[:, ignore_mask] = fill_value
    return x, ignore_mask

# ----------------- KITTI 数据集类 -----------------

class KITTI(data.Dataset):
     
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
        encoding_mode: str = 'color',   # 'color', 'random_color', 'bits', 'none'
        caption_type: str = 'none',     # 'none', 'caption', 'class_label', 'blip'
        inpaint_mask_size: Optional[Tuple[int]] = None,
        num_classes: int = 6,
        fill_value: int = 0.5,
        ignore_label: int = 0,
        inpainting_strength: float = 0.0,
    ):
        KITTI_CATEGORIES = [
        {"color": [0, 0, 0], "isthing": 0, "id": 0,  "name": "unlabeled"},
        {"color": [0, 0, 0], "isthing": 0, "id": 1,  "name": "outlier"},
        {"color": [0, 0, 142], "isthing": 1, "id": 10, "name": "car"},
        {"color": [119, 11, 32], "isthing": 1, "id": 11, "name": "bicycle"},
        {"color": [0, 0, 230], "isthing": 1, "id": 12, "name": "motorcycle"},
        {"color": [106, 0, 228], "isthing": 1, "id": 13, "name": "truck"},
        {"color": [0, 60, 100], "isthing": 1, "id": 14, "name": "other-vehicle"},
        {"color": [0, 80, 100], "isthing": 1, "id": 15, "name": "person"},
        {"color": [0, 0, 70], "isthing": 1, "id": 16, "name": "bicyclist"},
        {"color": [0, 0, 192], "isthing": 1, "id": 17, "name": "motorcyclist"},
        {"color": [250, 170, 30], "isthing": 0, "id": 18, "name": "road"},
        {"color": [100, 170, 30], "isthing": 0, "id": 19, "name": "parking"},
        {"color": [220, 220, 0], "isthing": 0, "id": 20, "name": "sidewalk"},
        {"color": [175, 116, 175], "isthing": 0, "id": 21, "name": "other-ground"},
        {"color": [250, 0, 30], "isthing": 0, "id": 22, "name": "building"},
        {"color": [165, 42, 42], "isthing": 0, "id": 23, "name": "fence"},
        {"color": [255, 77, 255], "isthing": 0, "id": 24, "name": "pole"},
        {"color": [0, 226, 252], "isthing": 0, "id": 25, "name": "traffic sign"},
        {"color": [182, 182, 255], "isthing": 0, "id": 26, "name": "vegetation"},
        {"color": [0, 82, 0], "isthing": 0, "id": 27, "name": "trunk"},
        {"color": [120, 166, 157], "isthing": 0, "id": 28, "name": "terrain"},
        {"color": [110, 76, 0], "isthing": 0, "id": 29, "name": "sky"}]
        self.KITTI_CATEGORY_NAMES = [cat["name"] for cat in KITTI_CATEGORIES]
        self.KITTI_COLOR_TO_LABEL = {tuple(cat["color"]): idx for idx, cat in enumerate(KITTI_CATEGORIES)}
        """
        Args:
          prefix: 数据集根目录，例如 '/root/autodl-tmp/kitti'
          split: 'train' 或 'val'
          tokenizer: 用于 caption 处理
          transform: 针对输入 image 的 transform，注意 resize 用 bilinear 插值
          download: 必须为 False
          remap_labels, caption_dropout, overfit, encoding_mode, caption_type: 与 COCO 接口一致
          inpaint_mask_size: inpainting 掩码尺寸，默认为 (64, 64)
          num_classes: 数据集类别数，此处为6
          fill_value: bit encoding 时 ignore 区域的填充值
          ignore_label: ignore label 数值（这里按 KITTI 数据设置为0）
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
        self.remap_labels = remap_labels
        if inpaint_mask_size is None:
            inpaint_mask_size = (64, 64)
        self.maskgenerator = MaskingGenerator(input_size=inpaint_mask_size, mode='random_local')
        self.meta_data = self.get_metadata()
        self.transform = transform  # 仅适用于 image
        self.cmap = color_map()

        print("Initializing dataloader for KITTI {} set".format(split))
        # 假设 KITTI 数据存放于 prefix/split 目录下
        _image_dir = os.path.join(self.root, split)
        self.samples = []
        sample_dict = {}
        # 文件名格式示例：
        # 000000_000000_gtFine_class.png, 000000_000000_gtFine_instance.png,
        # 000000_000000_leftImg8bit.png, 000008_000000_depth_707.0911865234375.png
        for file in sorted(os.listdir(_image_dir)):
            base, ext = os.path.splitext(file)
            if ext.lower() != ".png":
                continue
            parts = base.split('_')
            if len(parts) >= 4 and parts[2] == "gtFine":
                scene, frame = parts[0], parts[1]
                typ = parts[3]  # 'class' 或 'instance'
            elif len(parts) == 3 and parts[2] == "leftImg8bit":
                scene, frame = parts[0], parts[1]
                typ = "leftImg8bit"
            elif len(parts) >= 4 and parts[2] == "depth":
                scene, frame = parts[0], parts[1]
                typ = "depth"
            else:
                continue
            if self.split=="train":
                if frame>="000005":
                    continue
            else:
                if frame>="000100":
                    continue
            if scene not in sample_dict:
                sample_dict[scene] = {}
            if frame not in sample_dict[scene]:
                sample_dict[scene][frame] = {}
            sample_dict[scene][frame][typ] = os.path.join(_image_dir, file)
        for scene, frames in sample_dict.items():
            for frame, files in frames.items():
                if all(key in files for key in ['leftImg8bit', 'class', 'instance', 'depth']):
                    self.samples.append(files)
        print("Found {} samples in split {}".format(len(self.samples), split))
        if overfit:
            self.samples = self.samples[:1000]
        if self.split == 'train':
            self.pixel_threshold = 10
            self.training = True
        else:
            self.pixel_threshold = 0
            self.training = False

    def __len__(self):
        return len(self.samples)
    def _remap_labels_fn(self, labels, max_val=None, keep_background_fixed=True):
        # keep the original background class index
        # max val is the maximum number of classes to remap to
        # ignore index is kept fixed

        # remapping only works if additional background classes are ordered from 0 to N.
        max_val = max_val if max_val is not None else self.num_classes
        unique_values = [x for x in np.unique(labels) if x != self.ignore_label]
        assert len(unique_values) < max_val, f"Number of unique values {len(unique_values)} is larger or equal than max_val {max_val}"  # noqa

        # np.random.seed(1)
        targets = np.random.choice(max_val - 1,
                                   size=len(unique_values),
                                   replace=False)  # sampling without replacement
        targets = targets + 1

        # mapping dict
        mapping = dict(zip(unique_values, targets))
        remapped_labels = np.full(labels.shape, self.ignore_label, dtype=labels.dtype)
        for val, remap_val in mapping.items():
            remapped_labels[labels == val] = remap_val
        mapping_np = np.full(self.num_classes, -1, dtype=int)
        for idx, (_, remap_val) in enumerate(mapping.items()):
            mapping_np[idx] = remap_val

        # sanity checks: make sure all target values are smaller than max_val and unique
        assert np.all(mapping_np[mapping_np != -1] < max_val)
        assert np.all(mapping_np[mapping_np != -1] >= 0)
        assert len(np.unique(mapping_np[mapping_np != -1])) == len(mapping_np[mapping_np != -1])
        assert len(np.unique(mapping_np[mapping_np != -1])) == len(unique_values)

        return remapped_labels, mapping

    def encode_semseg(self, semseg, cmap=None):
        # we will encode the semseg map with a color map
        if cmap is None:
            cmap = color_map()
        seg_t = semseg.astype(np.uint8)
        array_seg_t = np.full((seg_t.shape[0], seg_t.shape[1], cmap.shape[1]), self.ignore_label, dtype=cmap.dtype)
        for class_i in np.unique(seg_t):
            array_seg_t[seg_t == class_i] = cmap[class_i]
        return array_seg_t

    def encode_semseg_random(self, semseg, cmap=None):
        seg_t = semseg.astype(np.uint8)
        color_palette = set()
        array_seg_t = np.full((seg_t.shape[0], seg_t.shape[1], cmap.shape[1]), self.ignore_label, dtype=cmap.dtype)
        unique_classes = np.unique(seg_t)
        while len(color_palette) < len(unique_classes):
            color_palette.add(tuple(np.random.choice(range(256), size=3)))
        for class_i in unique_classes:
            if class_i == self.ignore_label:
                continue
            array_seg_t[seg_t == class_i] = color_palette.pop()
        assert array_seg_t.max() < 256
        return array_seg_t

    def encode_bitmap(self, x: torch.Tensor, n: int = 7, fill_value: float = 0.5):
        ignore_mask = x == self.ignore_label
        x = torch.bitwise_right_shift(x, torch.arange(n, device=x.device)[:, None, None])  # shift with n bits
        x = torch.remainder(x, 2).float()                                                  # take modulo 2 to get 0 or 1
        x[:, ignore_mask] = fill_value                                                     # set invalid pixels to 0.5
        return x, ignore_mask

    def decode_bitmap(self, x: torch.Tensor, n: int = 7):
        x = (x > 0.).float()                                          # output between -1 and 1
        n = x.shape[0]                                                # number of channels = number of bits
        x = x * 2 ** torch.arange(n, device=x.device)[:, None, None]  # get the value of each bit
        x = torch.sum(x, dim=0)                                       # sum over bits (no keepdim!)
        x = x.long()                                                  # cast to int64 (or long)
        return x

    def get_inpainting_mask(self, strength=0.5):
        mask = self.maskgenerator(t=strength)
        mask = torch.from_numpy(mask).bool()
        return mask

    def get_class_names(self):
        return self.KITTI_CATEGORY_NAMES

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

    def encode_bitmap(self, x: torch.Tensor, n: int = 7, fill_value: float = 0.5):
        ignore_mask = x == self.ignore_label
        x = torch.bitwise_right_shift(x, torch.arange(n, device=x.device)[:, None, None])
        x = torch.remainder(x, 2).float()
        x[:, ignore_mask] = fill_value
        return x, ignore_mask

    def __getitem__(self, idx):
        sample = {}
        sample_paths = self.samples[idx]
        
        # ---------- 1. 读取并 Resize 输入图像 ----------
        # 输入图像：leftImg8bit.png，采用 bilinear 插值 resize 至 (640,192)
        image = Image.open(sample_paths['leftImg8bit']).convert('RGB')
        image = image.resize((640, 192), Image.BILINEAR)
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = T.ToTensor()(image)
        sample['image'] = image_tensor

        # ---------- 2. 处理 Ground Truth 标签 ----------
        # 语义标签：class.png，转换为单通道 GT。用 nearest 插值 resize 至 (640,192)
        sem_img = Image.open(sample_paths['class']).convert('L')
        sem_img = sem_img.resize((640, 192), Image.NEAREST)
        sem_np_color = np.array(sem_img, dtype=np.uint8)
        sem_np = encode_segmentation_mask(sem_np_color, self.KITTI_COLOR_TO_LABEL)
        # 返回为整数 Tensor（不做归一化），直接转换为 tensor
        sample['semseg'] = torch.from_numpy(sem_np).long()

        # 实例标签：instance.png，灰度加载，用 nearest 插值 resize
        inst_img = Image.open(sample_paths['instance']).convert('L')
        inst_img = inst_img.resize((640, 192), Image.NEAREST)
        inst_np = np.array(inst_img, dtype=np.uint8)
        
        # 深度图：depth.png，用 nearest 插值 resize；保持原有数值
        depth_img = Image.open(sample_paths['depth'])
        depth_img = depth_img.resize((640, 192), Image.NEAREST)
        # 如果深度图本身为单通道且保存数值，则直接转换为 tensor
        depth_np = np.array(depth_img, dtype=np.float32)  # 假定深度为 float32 数值
       
        # ---------- 3. 构造 mask ----------
        mask_np = np.ones_like(sem_np, dtype=np.uint8) * 255
        sample['mask'] = torch.from_numpy(mask_np)  # 返回 shape (H, W)

        # ---------- 4. 读取预生成的全景彩色图像 ----------
        # 从固定目录 "/root/autodl-tmp/pop_gt" 读取，文件名格式为 "{idx}_output.png"
        if self.split=="train":
            pop_gt_path = os.path.join("/root/autodl-tmp/pop_gt", f"{idx}_output.png")
        else:
            pop_gt_path = os.path.join("/root/autodl-tmp/pop_gt_val", f"{idx}_output.png")
        color_img = Image.open(pop_gt_path).convert('RGB')
        color_img = color_img.resize((640, 192), Image.BILINEAR)
        sample['image_semseg'] = T.ToTensor()(color_img)

        # ---------- 5. 处理 depth 与 instance ----------
        sample['depth'] = torch.from_numpy(depth_np)
        sample['instance'] = torch.from_numpy(inst_np).long()

        # ---------- 6. 构造 meta 信息 ----------
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
        meta['gt_cat'] = torch.from_numpy(sem_np).long()
        meta['gt_ins'] = torch.from_numpy(inst_np).long()
        sample['meta'] = meta

        # ---------- 7. caption 与 inpainting_mask ----------
        sample['text'] = ""
        inpainting_mask = self.maskgenerator(t=self.inpainting_strength)
        sample['inpainting_mask'] = torch.from_numpy(inpainting_mask).bool()

        # ---------- 8. 根据 encoding_mode 后处理 image_semseg ----------
        if self.encoding_mode == 'bits':
            sample_sem = torch.from_numpy(sem_np).long()
            image_semseg_bits, _ = self.encode_bitmap(sample_sem, n=7, fill_value=self.fill_value)
            sample['image_semseg'] = image_semseg_bits
        elif self.encoding_mode == 'none':
            sample_sem = torch.from_numpy(sem_np).float()
            sample['image_semseg'] = sample_sem.unsqueeze(0).repeat(3, 1, 1) / self.num_classes

        # ---------- 9. 如果提供 tokenizer，则 tokenization ----------
        if self.tokenizer is not None:
            sample['tokens'] = self.tokenizer(sample['text'],
                                              padding='max_length',
                                              max_length=77,
                                              truncation=True,
                                              return_tensors='pt').input_ids.squeeze(0)
        assert 'instance' in sample, "Missing instance segmentation in sample"
        return sample

    def _load_img(self, index, mode='array'):
        _img = Image.open(self.samples[index]['leftImg8bit']).convert('RGB')
        if mode == 'pil':
            return _img
        return np.array(_img)

    def _load_semseg(self, index, mode='array'):
        sample_paths = self.samples[index]
        sem_img = Image.open(sample_paths['class']).convert('L')
        sem_img = sem_img.resize((640, 192), Image.NEAREST)
        sem_np_color = np.array(sem_img, dtype=np.uint8)
        sem_np = encode_segmentation_mask(sem_np_color, self.KITTI_COLOR_TO_LABEL)
        segments_info = {}
        captions_info = []
        key_id = os.path.basename(sample_paths['leftImg8bit'])
        if mode == 'pil':
            return Image.fromarray(sem_np.astype(np.uint8))
        return sem_np, segments_info, captions_info, key_id

    def _validate_annotations_simple(self):
        from tqdm import tqdm
        for i in tqdm(range(len(self))):
            semseg, _, _, _ = self._load_semseg(i)
            unique_labels = np.unique(semseg)
            unique_labels = unique_labels[unique_labels != self.ignore_label]
            assert len(unique_labels) > 0
        return

    def __str__(self):
        return 'KITTI(split=' + str(self.split) + ')'
        
if __name__ == '__main__':
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transforms = T.Compose([
        T.Resize((192, 640)),  # 对 image 使用 bilinear resize
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    dataset_root = '/root/autodl-tmp/video_sequence'
    dataset = KITTI(
        prefix=dataset_root,
        split='val',
        transform=transforms,
        remap_labels=False
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    for sample in dataloader:
        print("image:", sample['image'].shape)
        print("semseg:", sample['semseg'].shape if isinstance(sample['semseg'], torch.Tensor) else sample['semseg'].size())
        print("mask:", sample['mask'].shape if isinstance(sample['mask'], torch.Tensor) else sample['mask'].size())
        print("image_semseg:", sample['image_semseg'].shape)
        print("depth:", sample['depth'].shape)
        print("instance:", sample['instance'].shape)
        print("meta:", sample['meta'])
        print("text:", sample['text'])
        print("inpainting_mask:", sample['inpainting_mask'].shape)
        break
