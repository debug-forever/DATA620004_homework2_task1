import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

def prepare_caltech_data(root_dir=os.path.join('caltech-101', '101_ObjectCategories'), train_ratio=0.8):
    """
    准备 Caltech-101 数据集，将其分割为训练集和测试集
    """
    # 使用 os.path.join 处理所有路径
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 遍历原始数据集
    categories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) 
                 and d not in ['train', 'test']]
    
    for category in categories:
        # 创建类别对应的文件夹
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)
        
        # 获取该类别下的所有图片
        image_files = [f for f in os.listdir(os.path.join(root_dir, category)) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # 划分训练集和测试集
        train_files, test_files = train_test_split(
            image_files, 
            train_size=train_ratio,
            random_state=42
        )
        
        # 移动文件到对应目录
        for f in train_files:
            src = os.path.join(root_dir, category, f)
            dst = os.path.join(train_dir, category, f)
            shutil.copy2(src, dst)
            
        for f in test_files:
            src = os.path.join(root_dir, category, f)
            dst = os.path.join(test_dir, category, f)
            shutil.copy2(src, dst)

if __name__ == '__main__':
    prepare_caltech_data()
    print("数据集准备完成！")
