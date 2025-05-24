from train import train
import os

def main():    # 预训练模型配置
    pretrained_config = {
        'data_dir': os.path.join('data'),
        'model_name': 'resnet18',
        'pretrained': True,
        'batch_size': 32,
        'lr': 1e-4,
        'lr_fc': 1e-2,
        'epochs': 20
    }
    
    # 训练预训练模型
    print("Training with pretrained weights...")
    train(pretrained_config)
    
if __name__ == '__main__':
    main()
