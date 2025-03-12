import torch 
import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        
        self.features = self._make_layers([
            64, 64, 'M', 
            128, 128, 'M', 
            256, 256, 256, 'M', 
            512, 512, 512, 'M', 
            512, 512, 512, 'M'
        ])
        
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)  # Extract features
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)  # Linear layer
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)