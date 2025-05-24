import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

class Caltech101Dataset:    
    def __init__(self, data_dir, batch_size=32):
        # 训练集数据增强
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 测试集转换
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
          # 加载训练集和验证集
        self.train_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, "train"),
            transform=self.train_transform
        )
        
        self.val_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, "test"),
            transform=self.val_transform
        )
        
        # 打印数据集信息
        print(f"Number of classes: {len(self.train_dataset.classes)}")
        print(f"Class names: {self.train_dataset.classes}")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
