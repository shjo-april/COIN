import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import activation
import torch.utils.model_zoo as model_zoo

urls_dic = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',   # IN1K
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',   # IN1K
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',   # IN1K
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', # IN1K
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth', # IN1K
}

# refer to https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
urls_dic_v2 = {
    # resnet18, resnet34 same
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',   # IN1K
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',   # IN1K
    'resnet50': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth',   # IN21K
    'resnet101': 'https://download.pytorch.org/models/resnet101-cd907fc2.pth', # IN21K
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth'  # IN21K
}

layers_dic = {
    'resnet18' : [2, 2, 2, 2],
    'resnet34' : [3, 4, 6, 3],
    'resnet50' : [3, 4, 6, 3],
    'resnet101' : [3, 4, 23, 3],
    'resnet152' : [3, 8, 36, 3]
}

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, batch_norm_fn=nn.BatchNorm2d, activation_fn=nn.ReLU):
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = batch_norm_fn(planes)
        self.relu1 = activation_fn()

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = batch_norm_fn(planes)
        self.relu2 = activation_fn()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, batch_norm_fn=nn.BatchNorm2d, activation_fn=nn.ReLU):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = batch_norm_fn(planes)
        self.relu1 = activation_fn()

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = batch_norm_fn(planes)
        self.relu2 = activation_fn()

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = batch_norm_fn(planes * 4)
        self.relu3 = activation_fn()

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out

class ResNet(nn.Module):
    
    def __init__(self, block, layers, strides=(2, 2, 2, 2), dilations=(1, 1, 1, 1), batch_norm_fn=nn.BatchNorm2d, activation_fn=nn.ReLU):
        self.batch_norm_fn = batch_norm_fn
        self.activation_fn = activation_fn

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=strides[0], padding=3,
                               bias=False)
        self.bn1 = self.batch_norm_fn(64)
        self.relu = activation_fn()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3])
        self.inplanes = 1024

        self.out_features = 512 * block.expansion

        #self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, 1000)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.batch_norm_fn(planes * block.expansion),
            )
        
        layers = [block(self.inplanes, planes, stride, downsample, dilation=1, batch_norm_fn=self.batch_norm_fn, activation_fn=self.activation_fn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, batch_norm_fn=self.batch_norm_fn, activation_fn=self.activation_fn))
        
        return nn.Sequential(*layers)

    def forward(self, x, detach=False):
        xs = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x); xs.append(x)

        x = self.layer1(x); xs.append(x)
        x = self.layer2(x); xs.append(x)

        if detach:
            x = x.detach()
        
        x = self.layer3(x); xs.append(x)
        x = self.layer4(x); xs.append(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        
        return xs

