import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
class SemKITTI_DVPS_Dataset(Dataset):
    def __init__(self, root,
                 image_transform,
                 GT_transform,
                 split='train',
                 ):
        """
        Args:
            root (str): 数据集根目录，例如 '/path/to/dataset'
            split (str): 数据集的划分，'train' 或 'val'
            image_transform: 对 RGB 图像进行预处理的 transform
            depth_transform: 对深度图进行预处理的 transform
            seg_transform: 对语义分割标签进行预处理的 transform
            inst_transform: 对实例分割标签进行预处理的 transform
        """
        self.root = root
        self.split = split
        self.image_transform = image_transform
        self.GT_transform = GT_transform
        
        self.samples = []  # 每个元素为一个字典，包含该样本的各个图片路径
        split_dir = os.path.join(root, split)
        all_files = sorted(os.listdir(split_dir))
        
        # 按照样本前缀分组，假设前缀为前4个字符（例如 "0001"）
        sample_dict = {}
        for file in all_files:
            # 去除空格、统一小写
            file_name_element = file.split("_")#[00000_00000_depth_718]
            scene=file_name_element[0]#标记场景
            frame=file_name_element[1]#标记帧
            
            if scene != "000000":#只加载第一组
                continue
                
            if scene not in sample_dict:
                sample_dict[scene] = {}
                if frame not in sample_dict[scene]:
                    sample_dict[scene][frame]={}
                else:
                    pass
            else:
                if frame not in sample_dict[scene]:
                    sample_dict[scene][frame]={}
                else:
                    pass
            
            # 根据文件名中包含的关键字确定图片类型
            if 'depth' in file_name_element:
                sample_dict[scene][frame]['depth'] = os.path.join(split_dir, file)
                sample_dict[scene][frame]['focal'] = file_name_element[3].split(".")[0]
            else:
                pass
            
            if 'class.png' in file_name_element:
                sample_dict[scene][frame]['class'] = os.path.join(split_dir, file)
                
            else:
                pass
            
            if 'instance.png' in file_name_element:
                sample_dict[scene][frame]['instance'] = os.path.join(split_dir, file)
                
            else:
                pass
            
            if 'leftImg8bit.png' in file_name_element:
                sample_dict[scene][frame]['Img'] = os.path.join(split_dir, file)
                
            else:
                pass
        
        # 过滤出包含所有四种图片的样本frame
        for scene, frames in sample_dict.items():
            for frame, files in frames.items():
                if all(key in files for key in ['depth', 'Img', 'class', 'instance']):
                    self.samples.append(files)
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 加载 RGB 图像，并转换为 RGB 格式
        image = Image.open(sample['Img']).convert('RGB')
        # 加载深度图（深度图可能为单通道图像）
        depth = Image.open(sample['depth'])
        # 加载语义分割标签，通常为单通道标签图
        seg = Image.open(sample['class'])
        # 加载实例分割标签
        inst = Image.open(sample['instance'])
        
        # 对图像应用预处理 transform
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        
        depth = self.GT_transform(depth)
        
        seg = self.GT_transform(seg)
            
        inst = self.GT_transform(inst)
        
        return image, depth, seg, inst
    
    
if __name__ == '__main__':
    dataset_root = '/root/autodl-tmp/video_sequence'  # 修改为你的数据集根目录

    # 定义图像预处理
    image_transforms = transforms.Compose([
        transforms.Resize((192, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    GT_transforms = transforms.Compose([
        transforms.Resize((192, 640),interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    # 可根据需要为深度图单独定义 transform，例如仅做 resize 和 ToTensor
    # 对于 segment 和 instance，由于它们是标签，通常不希望有归一化操作，可以直接转换为 Tensor
    # 这里我们在 __getitem__ 中已处理

    # 构造训练集
    train_dataset = SemKITTI_DVPS_Dataset(root=dataset_root,
                                          split='train',
                                          image_transform=image_transforms,
                                             GT_transform=GT_transforms)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=4)

    # 测试加载数据
    for images, depths, segments, instances in train_loader:
        print("RGB 图像 batch 尺寸:", images.shape)       # 例如 [4, 3, 256, 512]
        print("深度图 batch 尺寸:", depths.shape)          # 例如 [4, 1, 256, 512]
        print("语义标签 batch 尺寸:", segments.shape)       # 例如 [4, 256, 512]
        print("实例标签 batch 尺寸:", instances.shape)      # 例如 [4, 256, 512]
        break
