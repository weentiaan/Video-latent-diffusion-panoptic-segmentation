"""
Author: Wouter Van Gansbeke

Dataset class for COCO Panoptic Segmentation
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import json
import torch

import numpy as np
import torch.utils.data as data
from PIL import Image
from typing import Optional, Any, Tuple
import random
from collections import defaultdict

from ldmseg.data.util.mypath import MyPath
from ldmseg.utils.utils import color_map
from ldmseg.data.util.mask_generator import MaskingGenerator


class COCO(data.Dataset):
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
    KITTI_CATEGORY_NAMES = [k["name"] for k in KITTI_CATEGORIES]

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
        encoding_mode: str = 'bits',  # 使用 bit 编码
        caption_type: str = 'none',
        inpaint_mask_size: Optional[Tuple[int, int]] = None,
        num_classes: int = 6,  # KITTI 此处类别总数为：我们保留背景(0)和 5 个实例目标，即6类标签
        fill_value: int = 0.5,
        ignore_label: int = 255,
        inpainting_strength: float = 0.0,
    ):
        """
        Args:
            prefix (str): KITTI 数据集根目录
            split (str): 'train' 或 'val'（根据你的数据集划分）
            tokenizer: (可选) 文本 tokenizer
            transform: (可选) 图像和 mask 的 transform
            download (bool): 是否下载数据（通常为 False）
            remap_labels (bool): 是否重新映射标签
            caption_dropout (float): caption dropout 概率
            overfit (bool): 是否只使用一小部分图像用于 overfit 调试
            encoding_mode (str): 这里设为 'bits' 表示使用 bit 编码
            caption_type (str): 'none' 或其他文本描述模式
            inpaint_mask_size (Tuple[int, int], optional): inpainting mask 大小
            num_classes (int): 输出类别数（包括背景），此处设为 6（背景 + 5个目标）
            fill_value (int): bit 编码中无效像素填充值
            ignore_label (int): 无效标签的值
            inpainting_strength (float): inpainting 强度
        """
        # 设置路径，假设 KITTI 的图像位于 training/image_2，mask 位于 training/instance_2
        root = MyPath.db_root_dir('kitti', prefix=prefix)
        self.root = root
        self.prefix = prefix
        valid_splits = ['train', 'val']
        assert split in valid_splits, f"split should be one of {valid_splits}"
        self.split = split
        self.tokenizer = tokenizer
        self.caption_dropout = caption_dropout

        self.num_classes = num_classes  # 例如 6
        self.ignore_label = ignore_label
        self.fill_value = fill_value
        self.inpainting_strength = inpainting_strength
        self.remap_labels = remap_labels
        if inpaint_mask_size is None:
            inpaint_mask_size = (64, 64)
        self.maskgenerator = MaskingGenerator(input_size=inpaint_mask_size, mode='random_local')

        self.transform = transform
        self.cmap = color_map()  # 可以自行定制或使用默认颜色

        print("Initializing dataloader for KITTI {} set".format(self.split))
        if self.split == 'train':
            image_dir = os.path.join(self.root, 'training', 'image_2')
            semseg_dir = os.path.join(self.root, 'training', 'instance_2')
            self.training = True
        elif self.split == 'val':
            image_dir = os.path.join(self.root, 'testing', 'image_2')  # 假设 val 使用 testing 目录
            semseg_dir = os.path.join(self.root, 'testing', 'instance_2')
            self.training = False

        self.image_dir = image_dir
        self.semseg_dir = semseg_dir

        # 获取所有图像文件名（假设文件后缀为 .png 或 .jpg）
        self.images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])
        self.semsegs = sorted([os.path.join(semseg_dir, f) for f in os.listdir(semseg_dir) if f.endswith('.png')])
        print('Found {} images and {} segmentation maps.'.format(len(self.images), len(self.semsegs)))

        # 根据文件名关联图片与分割标注（假设文件名相同）
        self.images, self.semsegs = zip(*[
            (img, sem) for img, sem in zip(self.images, self.semsegs)
            if os.path.basename(img).split('.')[0] == os.path.basename(sem).split('.')[0]
        ])
        self.images = list(self.images)
        self.semsegs = list(self.semsegs)
        print("After matching, found {} samples".format(len(self.images)))

        if overfit:
            n_of = 1000
            self.images = self.images[:n_of]
            self.semsegs = self.semsegs[:n_of]

        # 对于 KITTI，一般类别较少，这里直接构造映射字典
        self.cat_info = {cat['id']: {'name': cat['name'], 'isthing': cat['isthing']} for cat in self.KITTI_CATEGORIES}

        # 对于 remap_labels 部分，根据需要实现相应逻辑（这里简化为保持原标签）
        self.meta_data = {}  # 可添加 remap 信息

        # Display dataset stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}
        # 加载图像
        img_path = self.images[index]
        _img = Image.open(img_path).convert('RGB')
        sample['image'] = _img

        # 加载分割 mask
        semseg_path = self.semsegs[index]
        _semseg = np.array(Image.open(semseg_path).convert('L'))  # 假设分割 mask 为灰度图，数值表示类别标签
        # 注意：KITTI 数据集中的分割 mask 数据格式可能需要根据具体格式做处理，
        # 例如：可能需要对像素值进行 bit 拆分，如果原始标签为整数，使用 bit 编码将其转换为多通道 binary 表示
        # 这里假设 _semseg 中的数值范围 [0, 2^n-1]，我们需要将其转换为一个多通道 bit 图像。
        # 此处设定 bit 数：对于 6 个类别（含背景），可能需要 3 个 bit（但本问中语义类别为20，instance 50，所以 bit 数应为 5 和6），
        # 但这里 KITTI 使用的标签就较少，我们假设 _semseg 的标签已经在 0~(num_classes-1) 范围内。
        sample['semseg'] = _semseg.astype(np.uint8)

        # 这里你需要根据你使用 bit 编码的方案将整型 mask 转换为 bit 编码表示，
        # 示例：调用一个 encode_bitmap 函数（可以参考COCO模板中的 encode_bitmap），下面我们假设使用 7 位编码，
        # 但请注意对于 KITTI，你可能希望按实际需要使用更少或更多位数。这里假设我们对 KITTI 语义 mask 使用固定 bit 数进行编码，
        # 注意：如果你的 KITTI 语义标签实际数值较低（例如 0~5），则 bit 编码可能只需 3 位，但如果你同时编码 instance，
        # 则假设语义部分使用 5 位，instance 部分使用 6 位，合计11位。如你所问，上述 KITTI 应使用 11 个通道作为输入。
        # 下面只给出语义部分的例子，实例部分需类似处理，并最终拼接。
        # 例如，我们调用一个 encode_bitmap： (注意，这里仅作为参考)
        from ldmseg.data.util.mypath import MyPath  # 假设你的工具中有该函数
        # 这里假设你的 encoding_mode 为 'bits'
        if hasattr(self, 'encoding_mode') and self.encoding_mode == 'bits':
            # 此处 n 可设置为合计 bit 数，例如语义部分 5, instance 部分 6，共 11。
            n_bits = 11
            # 这里我们利用一个简单的实现，将 _semseg (二维) 转换为 n_bits 通道二值表示
            # 我们用 np.unpackbits 需要 uint8 数组且默认拆成 8 位，这里可以自定义实现
            def int_to_bits(arr, bits):
                # arr: numpy array (H, W)
                H, W = arr.shape
                res = np.zeros((bits, H, W), dtype=np.uint8)
                for i in range(bits):
                    res[i] = ((arr >> i) & 1)
                return res
            encoded_semseg = int_to_bits(_semseg, n_bits)  # shape: [n_bits, H, W]
            # 注意：实际 KITTI 数据可能需要分别对语义和 instance 分开编码，然后拼接，
            # 这里为示例直接编码整幅 mask
            sample['image_semseg'] = encoded_semseg
            # 将该编码转换为 tensor
            sample['image_semseg'] = torch.from_numpy(sample['image_semseg']).float()
        else:
            # 其他编码方式，直接转换为 tensor
            sample['image_semseg'] = torch.from_numpy(_semseg).unsqueeze(0).float()

        # 设置 mask（这里简单赋值全1，实际可根据需要修改）
        sample['mask'] = torch.ones_like(sample['image_semseg'][0]).float()

        # meta 信息
        sample['meta'] = {
            'im_size': (_img.size[1], _img.size[0]),
            'image_file': img_path,
            'image_id': int(os.path.basename(img_path).split('.')[0]),
        }

        # 处理 transform
        if self.transform is not None:
            sample = self.transform(sample)

        # 如果使用 tokenizer（例如用于 captioning 之类），这里设为空
        sample['text'] = ""

        # inpainting mask（可选）
        sample['inpainting_mask'] = self.maskgenerator(t=self.inpainting_strength)

        return sample

    def __len__(self):
        return len(self.images)

    def get_class_names(self):
        return self.KITTI_CATEGORY_NAMES

    def __str__(self):
        return 'KITTI(split=' + str(self.split) + ')'

    # 其他函数例如 _remap_labels_fn、encode_semseg 等可以按需移植或简化
    # 这里简单保留其中的 encode_bitmap 用于 bit 编码转换

    def encode_bitmap(self, x: torch.Tensor, n: int = 7, fill_value: float = 0.5):
        # x: tensor (H, W), ignore pixels对应 ignore_label 的位置设置为 fill_value
        ignore_mask = x == self.ignore_label
        # 对每个像素的整数进行 bit 操作，得到 n 个通道
        H, W = x.shape
        bits = []
        for i in range(n):
            bit = ((x >> i) & 1).float()
            bits.append(bit)
        bits = torch.stack(bits, dim=0)  # shape: [n, H, W]
        bits[:, ignore_mask] = fill_value
        return bits, ignore_mask

if __name__ == '__main__':
    import torchvision.transforms as T
    from ldmseg.data.util.mypath import MyPath

    size = 256
    transforms = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
    ])

    dataset = KITTI(
        prefix='/path/to/KITTI',  # 根据实际路径修改
        split='train',
        transform=None,
        remap_labels=False,
        encoding_mode='bits',
        caption_type='none',
        overfit=False,
        num_classes=6,  # 背景 + 5个实例目标
    )
    print("Number of KITTI samples:", len(dataset))
    sample = dataset[0]
    print("Sample image_semseg shape:", sample['image_semseg'].shape)
