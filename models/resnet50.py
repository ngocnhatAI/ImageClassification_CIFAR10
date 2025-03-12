import torch 
import torch.nn as nn
import torch.nn.functional as F

class IdentityBlock(nn.Module):
    def __init__(self, f, filters):
        super(IdentityBlock, self).__init__()
        F0,F1,F2,F3 = filters
        
        self.conv1 = nn.Conv2d(in_channels=F0, out_channels=F1, kernel_size=1, stride=1, padding=0)
        self.bn1   = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=f, stride=1, padding='same')
        self.bn2   = nn.BatchNorm2d(F2)
        self.conv3 = nn.Conv2d(in_channels=F2, out_channels=F3, kernel_size=1, stride=1, padding=0)
        self.bn3   = nn.BatchNorm2d(F3)
        
    def forward(self, x):
        x_shortcut = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.relu(x+x_shortcut)
        
        return x
    
class ConvolutionalBlock(nn.Module):
    def __init__(self, f, s, filters):
        super(ConvolutionalBlock, self).__init__()
        F0,F1,F2,F3 = filters
        
        self.conv1 = nn.Conv2d(in_channels=F0, out_channels=F1, kernel_size=1, stride=s, padding=0)
        self.bn1   = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=f, stride=1, padding='same')
        self.bn2   = nn.BatchNorm2d(F2)
        self.conv3 = nn.Conv2d(in_channels=F2, out_channels=F3, kernel_size=1, stride=1, padding=0)
        self.bn3   = nn.BatchNorm2d(F3)
        
        self.shortcut_conv = nn.Conv2d(in_channels=F0, out_channels=F3, kernel_size=1, stride=s, padding=0)
        self.shortcut_bn  = nn.BatchNorm2d(F3)
        
    def forward(self, x):
        x_shortcut = x
        x_shortcut = self.shortcut_bn(self.shortcut_conv(x_shortcut))

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.relu(x + x_shortcut)

        return x
    
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)

        self.conv_block2 = ConvolutionalBlock(3,1, (64,64,64,256))
        self.id_block2   = IdentityBlock(3, (256,64,64,256))

        self.conv_block3 = ConvolutionalBlock(3,2, (256,128,128,512))
        self.id_block3   = IdentityBlock(3, (512,128,128,512))

        self.conv_block4 = ConvolutionalBlock(3,2, (512,256,256,1024))
        self.id_block4   = IdentityBlock(3, (1024,256,256,1024))

        self.conv_block5 = ConvolutionalBlock(3,2, (1024,512,512,2048))
        self.id_block5   = IdentityBlock(3, (2048,512,512,2048))

        self.avg_pooling = nn.AvgPool2d(kernel_size = (4,4))
        self.linear      = nn.Linear(2048,10)
        
        
    def forward(self, x):
        # Stage 1
        x = torch.relu(self.bn1(self.conv1(x)))

        # Stage 2
        x = self.id_block2(self.id_block2(self.conv_block2(x)))

        # Stage 3
        x = self.id_block3(self.id_block3(self.id_block3(self.conv_block3(x))))

        # Stage 4
        x = self.id_block4(self.id_block4(self.id_block4(self.id_block4(self.id_block4(self.conv_block4(x))))))

        # Stage 5
        x = self.id_block5(self.id_block5(self.conv_block5(x)))

        # Stage 6
        x = self.avg_pooling(x)
        x = x.reshape(x.shape[0],-1)
        x = self.linear(x)

        return x