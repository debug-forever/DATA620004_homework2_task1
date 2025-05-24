import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataset import Caltech101Dataset
from model import Caltech101Model
import os

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = f'caltech101_{config["model_name"]}{"_pretrained" if config["pretrained"] else "_scratch"}'
    writer = SummaryWriter(os.path.join('runs', run_name))
    
    # 加载数据
    dataset = Caltech101Dataset(config['data_dir'], config['batch_size'])
      # 创建模型
    model_wrapper = Caltech101Model(
        model_name=config['model_name'],
        pretrained=config['pretrained']
    )
    model = model_wrapper.model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    if config['pretrained']:
        params = model_wrapper.get_params_for_finetune(config['lr'], config['lr_fc'])
    else:
        params = model.parameters()
    
    optimizer = torch.optim.Adam(params, lr=config['lr'])
      # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.1, 
        patience=5
    )
    
    # 打印当前学习率
    def log_lr(optimizer):
        for param_group in optimizer.param_groups:
            print(f"Current learning rate: {param_group['lr']}")
    
    best_acc = 0
    # 训练循环
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(dataset.train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{config["epochs"]} | '
                      f'Batch: {batch_idx+1}/{len(dataset.train_loader)} | '
                      f'Loss: {loss.item():.4f}')
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in dataset.val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # 计算指标
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(dataset.train_loader)
        avg_val_loss = val_loss / len(dataset.val_loader)

        # 记录训练指标到Tensorboard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # 打印训练信息
        print(f'Epoch: {epoch+1}/{config["epochs"]} | '
              f'Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        # 更新学习率
        scheduler.step(val_acc)
        log_lr(optimizer)  # 打印更新后的学习率
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            # 修改模型保存路径
            checkpoint_path = os.path.join('checkpoints', f'{config["model_name"]}_{"pretrained" if config["pretrained"] else "scratch"}_best.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Best accuracy: {best_acc:.2f}%')
            print(f'Model saved to {checkpoint_path}')

    # 关闭Tensorboard writer
    writer.close()
    return best_acc

if __name__ == '__main__':
    config = {
        'data_dir': 'path/to/caltech101',
        'model_name': 'resnet18',
        'pretrained': True,
        'batch_size': 32,
        'lr': 1e-4,
        'lr_fc': 1e-3,
        'epochs': 50
    }
    train(config)
