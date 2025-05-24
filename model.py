import torch
import torch.nn as nn
from torchvision import models

class Caltech101Model:
    def __init__(self, model_name='resnet18', pretrained=True):
        if model_name == 'resnet18':
            if pretrained:
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet18(weights=None)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 102)  # 101类 + BACKGROUND_Google
        
        elif model_name == 'alexnet':
            if pretrained:
                self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
            else:
                self.model = models.alexnet(weights=None)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, 102)  # 101类 + BACKGROUND_Google
            
    def get_params_for_finetune(self, lr, lr_fc):
        if isinstance(self.model, models.ResNet):            return [
                {'params': self.model.conv1.parameters(), 'lr': lr},
                {'params': self.model.bn1.parameters(), 'lr': lr},
                {'params': self.model.layer1.parameters(), 'lr': lr},
                {'params': self.model.layer2.parameters(), 'lr': lr},
                {'params': self.model.layer3.parameters(), 'lr': lr},
                {'params': self.model.layer4.parameters(), 'lr': lr},
                {'params': self.model.fc.parameters(), 'lr': lr_fc}
            ]
