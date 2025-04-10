import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import random
from collections import defaultdict

from ldmseg.data.util.mask_generator import MaskingGenerator
from ldmseg.utils.utils import color_map
from ldmseg.data.util.mypath import MyPath

# 以下两个函数为生成颜色映射和对全景 mask 进行着色
def get_color_map(num_colors):
    np.random.seed(42)  # 固定种子，保证颜色一致
    return np.random.randint(0, 256, (num_colors, 3), dtype=np.uint8)

def colorize_panoptic(panoptic_map, colormap):
    """
    根据 panoptic_map 中每个像素的 panoptic_id，
    从 colormap 中取对应颜色，生成彩色图像。
    """
    h, w = panoptic_map.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(panoptic_map)
    for uid in unique_ids:
        # 若 uid 为 ignore_label（这里假设为 255），设为黑色
        if uid == 255:
            color = np.array([0, 0, 0], dtype=np.uint8)
        else:
            color = colormap[uid % len(colormap)]
        color_image[panoptic_map == uid] = color
    return color_image

def encode_segmentation_mask(seg_img: np.ndarray, color_to_label: dict) -> np.ndarray:
    """
    将彩色分割图像 seg_img (H, W, 3) 转换为单通道的类别标签 (H, W)，
    利用 color_to_label 将每个 RGB 颜色转换为类别索引。
    """
    H, W, _ = seg_img.shape
    label_map = np.zeros((H, W), dtype=np.int64)
    for color, label in color_to_label.items():
        # 对每个像素判断是否匹配给定的颜色
        mask = np.all(seg_img == np.array(color, dtype=np.uint8), axis=-1)
        label_map[mask] = label
    return label_map

class SemKITTI_DVPS_Dataset(Dataset):
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
        {"color": [110, 76, 0], "isthing": 0, "id": 29, "name": "sky"}
    ]
    KITTI_CATEGORY_NAMES = [cat["name"] for cat in KITTI_CATEGORIES]
    # 创建颜色到标签的映射字典，连续索引从 0 开始
    KITTI_COLOR_TO_LABEL = {tuple(cat["color"]): idx for idx, cat in enumerate(KITTI_CATEGORIES)}

    def __init__(self, root, image_transform, GT_transform, KITTI_COLOR_TO_LABEL=None, split='train'):
        """
        Args:
            root (str): 数据集根目录，例如 '/path/to/dataset'
            image_transform: 对 RGB 图像的 transform
            GT_transform: 对标签图（class、instance等）的 transform（通常采用最近邻插值）
            KITTI_COLOR_TO_LABEL: 用于将彩色标签转换为类别索引的字典，默认为 None，会使用类变量中的映射
            split (str): 数据集划分，'train' 或 'val'
        """
        self.root = root
        self.split = split
        self.image_transform = image_transform
        self.GT_transform = GT_transform

        if KITTI_COLOR_TO_LABEL is None:
            KITTI_COLOR_TO_LABEL = SemKITTI_DVPS_Dataset.KITTI_COLOR_TO_LABEL
        self.KITTI_COLOR_TO_LABEL = KITTI_COLOR_TO_LABEL

        # 构造样本列表，假定所有样本存放在 root/split 目录下
        self.samples = []
        split_dir = os.path.join(root, split)
        all_files = sorted(os.listdir(split_dir))
        sample_dict = {}
        for file in all_files:
            file_elems = file.split("_")
            # 示例中以 scene 和 frame 分组，如 "000000_00000_class.png" 等
            scene = file_elems[0]
            frame = file_elems[1]
            if scene != "000003":  # 只加载第一组
                continue

            if scene not in sample_dict:
                sample_dict[scene] = {}
            if frame not in sample_dict[scene]:
                sample_dict[scene][frame] = {}
            
            # 根据文件名判断图片类型
            if 'depth' in file_elems:
                sample_dict[scene][frame]['depth'] = os.path.join(split_dir, file)
                sample_dict[scene][frame]['focal'] = file_elems[3].split(".")[0]
            if 'class.png' in file:
                sample_dict[scene][frame]['class'] = os.path.join(split_dir, file)
            if 'instance.png' in file:
                sample_dict[scene][frame]['instance'] = os.path.join(split_dir, file)
            if 'leftImg8bit.png' in file:
                sample_dict[scene][frame]['Img'] = os.path.join(split_dir, file)
        for scene, frames in sample_dict.items():
            for frame, files in frames.items():
                if all(key in files for key in ['depth', 'Img', 'class', 'instance']):
                    self.samples.append(files)
        print("Found {} samples in split {}".format(len(self.samples), split))
        # 设置 ignore_label 与类别数（根据需求自行调整）
        self.ignore_label = 255
        self.num_classes = 6
        # 初始化颜色映射（这里用自定义或者默认值）
        self.cmap = get_color_map(256)
        # 初始化 MaskingGenerator
        self.maskgenerator = MaskingGenerator(input_size=(64, 64), mode='random_local')
        self.meta_data = {}

        self.remap_labels = False
        self.caption_type = 'none'
        self.caption_dropout = 0.0
        if self.split == 'train':
            self.pixel_threshold = 10
        else:
            self.pixel_threshold = 0

    def __len__(self):
        return len(self.samples)

    def get_class_names(self):
        return self.KITTI_CATEGORY_NAMES

    def encode_bitmap(self, x: torch.Tensor, n: int = 11, fill_value: float = 0.5):
        """
        将二维标签图 x (H, W) 转换为 bit 编码表示。
        假设 KITTI 需要对语义和 instance 分别编码后拼接成 11 个通道（语义5位 + instance6位）。
        实现简单示例：将 x 的整数值按二进制位拆分。
        """
        ignore_mask = x == self.ignore_label
        H, W = x.shape
        bits = []
        for i in range(n):
            bit = ((x >> i) & 1).float()
            bits.append(bit)
        bits = torch.stack(bits, dim=0)  # [n, H, W]
        bits[:, ignore_mask] = fill_value
        return bits, ignore_mask

    def __getitem__(self, idx):
        sample_paths = self.samples[idx]
        # 加载 RGB 图像
        image = Image.open(sample_paths['Img']).convert('RGB')
        # 加载语义分割彩色标签
        sem_img_color = Image.open(sample_paths['class']).convert('RGB')
        # 加载实例分割标签（灰度）
        inst_img = Image.open(sample_paths['instance']).convert('L')
        # 加载深度图
        depth = Image.open(sample_paths['depth'])

        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = T.ToTensor()(image)

        # 将彩色语义标签转换为 numpy 数组
        sem_np_color = np.array(sem_img_color, dtype=np.uint8)
        # 使用颜色到标签映射将彩色标签转换为单通道标签
        sem_np = encode_segmentation_mask(sem_np_color, self.KITTI_COLOR_TO_LABEL)

        # 加载实例标签为 numpy 数组
        inst_np = np.array(inst_img, dtype=np.uint8)

        # 将语义标签转换为 tensor
        semseg_tensor = torch.from_numpy(sem_np).long()

        # 生成全景分割图（简单将语义标签与实例标签相加，可根据需要修改合并规则）
        panoptic_map = sem_np + inst_np
        color_image = colorize_panoptic(panoptic_map, self.cmap)
        semseg_color = T.ToTensor()(Image.fromarray(color_image))

        # 对语义与实例标签分别进行 bit 编码，再拼接为多通道 tensor
        sem_bits, _ = self.encode_bitmap(torch.from_numpy(sem_np), n=5, fill_value=0.5)
        inst_bits, _ = self.encode_bitmap(torch.from_numpy(inst_np), n=6, fill_value=0.5)
        image_semseg = torch.cat([sem_bits, inst_bits], dim=0)

        # 构造全1的 mask
        mask_np = np.ones_like(sem_np, dtype=np.uint8) * 255
        mask = T.ToTensor()(Image.fromarray(mask_np)).squeeze(0)

        base_name = os.path.basename(sample_paths['Img'])
        parts = base_name.split('_')
        try:
            scene = int(parts[0])
            frame = int(parts[1])
            image_id = scene * 10000 + frame
        except Exception:
            image_id = base_name

        meta = {
            'im_size': (image.shape[1], image.shape[2]) if isinstance(image, torch.Tensor) else image.size,
            'image_file': sample_paths['Img'],
            'image_id': image_id,
            'segments_info': {}
        }
        meta['gt_cat'] = torch.from_numpy(sem_np).long()
        meta['gt_ins'] = torch.from_numpy(inst_np).long()

        text = ""
        inpainting_mask = self.maskgenerator(t=0.0)
        inpainting_mask = torch.from_numpy(inpainting_mask).bool()

        sample = {
            'image': image,                   # RGB 图像 tensor
            'semseg': semseg_tensor,            # 单通道标签 tensor, 形状 (H, W)
            'semseg_color': semseg_color,       # 彩色全景分割 tensor, 形状 (3, H, W)
            'mask': mask,                     # mask tensor
            'image_semseg': image_semseg,       # bit 编码后的标签 tensor, 形状 (11, H, W)
            'meta': meta,
            'text': text,
            'inpainting_mask': inpainting_mask,
        }
        return sample


if __name__ == '__main__':
    dataset_root = '/root/autodl-tmp/video_sequence'
    image_transforms = T.Compose([
        T.Resize((192, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    GT_transforms = T.Compose([
        T.Resize((192, 640), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
    ])
    dataset = SemKITTI_DVPS_Dataset(root=dataset_root,
                                    split='train',
                                    image_transform=image_transforms,
                                    GT_transform=GT_transforms)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    for sample in dataloader:
        print("image:", sample['image'].shape)
        print("semseg:", sample['semseg'].shape)
        print("mask:", sample['mask'].shape)
        print("image_semseg:", sample['image_semseg'].shape)
        print("meta:", sample['meta'])
        print("text:", sample['text'])
        print("inpainting_mask:", sample['inpainting_mask'].shape)
        break