# Copyright (C) 2024 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import math
import torch
from torch import nn
from torch.nn import functional as F

from .backbones import resnet
from .backbones import resnest
from .backbones import wide_resnet

from .backbones import swinv1
from .backbones import swinv2

class Backbone(nn.Module):
    def __init__(self, backbone, norm_fn='bn', act_fn='relu', pretrained=True, last_stride=2, output_stride=16):
        super().__init__()

        if norm_fn == 'bn': self.norm_fn = nn.BatchNorm2d
        if act_fn == 'relu': self.act_fn = lambda:nn.ReLU(inplace=True)
        
        if 'wide' in backbone:
            self.model = wide_resnet.build_wide_resnet(backbone, last_stride, pretrained, freeze=True)
            self.in_channels = [128, 256, 512, 1024, 4096]

        elif 'resnet' in backbone:
            v2 = 'v2' in backbone
            if v2: backbone = backbone.replace('v2', '')
            self.model = resnet.build_resnet(backbone, self.norm_fn, self.act_fn, last_stride, pretrained, output_stride, v2=v2)
            
            self.in_channels = [64, 256, 512, 1024, 2048]
            if backbone in ['resnet18', 'resnet34']: self.in_channels = [64, 64, 128, 256, 512]
        
        elif 'resnest' in backbone:
            self.model = resnest.build_resnest(backbone, pretrained, output_stride)
            self.in_channels = [128, 256, 512, 1024, 2048]

        elif 'swin' in backbone:
            if 'v2' in backbone: self.model = swinv2.build_swinv2_transformer(backbone, pretrained=pretrained)
            else: self.model = swinv1.build_swin_transformer(backbone, pretrained=pretrained)

            embed_dims = {'tiny': 96, 'small': 96, 'base': 128, 'large': 192, 'huge': 352, 'giant': 448}
            embed_dim = embed_dims[backbone.split('_')[-1]]
            self.in_channels = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
            self.in_strides = [4, 8, 16, 32]

    def get_parameters(self):
        groups = ([], [], [], [])
        for name, value in self.named_parameters():
            # pretrained weights
            if 'model' in name:
                if 'weight' in name: groups[0].append(value)
                else: groups[1].append(value)
            # scracthed weights
            else:
                if 'weight' in name: groups[2].append(value)
                else:groups[3].append(value)
        return groups

class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding=0, dilation=1, bias=False, dropout=0., norm=nn.BatchNorm2d, act=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)
        self.norm = norm(planes) if norm is not None else nn.Identity()
        self.act = act() if act is not None else nn.Identity()    
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class DeepLabv3_ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride):
        super().__init__()
        
        if output_stride == 16: dilations = [1, 6, 12, 18]
        elif output_stride == 8: dilations = [1, 12, 24, 36]
        
        self.aspp1 = ConvBlock(in_channels, out_channels, 1, padding=0, dilation=dilations[0])
        self.aspp2 = ConvBlock(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ConvBlock(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ConvBlock(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3])

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBlock(in_channels, out_channels, 1)
        )
        self.block = ConvBlock(out_channels * 5, out_channels, 1, dropout=0.5)
    
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x5 = self.gap(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.block(x)

        return x
    
class DeepLabv3plus_Head(nn.Module):
    def __init__(self, in_channels, out_channels, low_in_channels, low_out_channels, output_stride, num_classes):
        super().__init__()

        self.aspp = DeepLabv3_ASPP(in_channels, out_channels, output_stride)
        
        self.low_block = ConvBlock(low_in_channels, low_out_channels, 1)
        self.mid_block = nn.Sequential(
            ConvBlock(out_channels + low_out_channels, out_channels, kernel_size=3, padding=1, dropout=0.5),
            ConvBlock(out_channels, out_channels, kernel_size=3, padding=1, dropout=0.1),
        )
        self.classifier = nn.Conv2d(out_channels, num_classes, kernel_size=1, bias=False)
    
    def forward(self, x, x_low):
        x = self.aspp(x)
        x_low = self.low_block(x_low)

        x_seg = F.interpolate(x, size=x_low.size()[2:], mode='bilinear', align_corners=False)
        x_seg = torch.cat((x_seg, x_low), dim=1)

        x_dec = self.mid_block(x_seg)
        return self.classifier(x_dec)

class DeepLabv2_Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        dilations = [6, 12, 18, 24]
        
        self.aspp1 = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, 
            padding=dilations[0], dilation=dilations[0], bias=True
        )
        self.aspp2 = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, 
            padding=dilations[1], dilation=dilations[1], bias=True
        )
        self.aspp3 = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, 
            padding=dilations[2], dilation=dilations[2], bias=True
        )
        self.aspp4 = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, 
            padding=dilations[3], dilation=dilations[3], bias=True
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        return x1 + x2 + x3 + x4
    
class DeepLabv1_Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, 512, 
            kernel_size=3, stride=1, 
            padding=12, dilation=12, bias=False
        )
        self.bn1 = nn.BatchNorm2d(512, momentum=0.0003)

        self.conv2 = nn.Conv2d(
            512, 512, 
            kernel_size=1, stride=1, 
            padding=0, dilation=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(512, momentum=0.0003)
        
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Conv2d(512, num_classes, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.dropout(x)
        x = self.classifier(x)

        return x
    
class AuxHead(nn.Module):
    def __init__(self, num_classes=20+1, num_convs=1, kernel_size=3, concat_input=True, in_channels=1024, channels=256, dropout_ratio=0.1):
        assert num_convs >= 0
        self.concat_input = concat_input

        if num_convs == 0: assert in_channels == channels

        convs = []
        convs.append(ConvBlock(in_channels, channels, kernel_size, kernel_size//2))
        for i in range(num_convs - 1):
            convs.append(ConvBlock(channels, channels, kernel_size, kernel_size//2))

        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)

        if self.concat_input:
            self.conv_cat = ConvBlock(in_channels+channels, channels, kernel_size, kernel_size//2)

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x):
        """Forward function."""
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        
        x = self.dropout(x)
        output = self.classifier(output)
        return output

class UperNet_Head(nn.Module):
    def __init__(self, num_classes=20+1, fc_dim=2048, pool_scales=(1, 2, 3, 6), fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=512, dropout_ratio=0.1):
        super().__init__()

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d((scale, scale))) 
            self.ppm_conv.append(ConvBlock(fc_dim, fpn_dim, 1))
        
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)

        self.ppm_last_conv = ConvBlock(fc_dim + len(pool_scales)*fpn_dim, fpn_dim, 3, padding=1)

        # FPN Module
        self.fpn_in = []
        self.fpn_out = []

        for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
            self.fpn_in.append(ConvBlock(fpn_inplane, fpn_dim, kernel_size=1))
            self.fpn_out.append(ConvBlock(fpn_dim, fpn_dim, kernel_size=3, padding=1))

        self.fpn_in = nn.ModuleList(self.fpn_in)
        self.fpn_out = nn.ModuleList(self.fpn_out)
        
        self.conv_fusion = ConvBlock(len(fpn_inplanes) * fpn_dim, fpn_dim, kernel_size=3, padding=1)

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.classifier = nn.Conv2d(fpn_dim, num_classes, kernel_size=1)

    def forward(self, conv_out):
        # for PPM Head
        C5 = conv_out[-1]
        
        ppm_out = [C5]
        ppm_size = C5.shape[2:]

        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(F.interpolate(pool_scale(C5), ppm_size, mode='bilinear', align_corners=False)))
        f = self.ppm_last_conv(torch.cat(ppm_out, 1))
        
        # for Fuse (Top-Down)
        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            x = self.fpn_in[i](conv_out[i])
            f = x + F.interpolate(f, size=x.size()[2:], mode='bilinear', align_corners=False)
            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse() # [P2, P2, P3, P4, P5]

        output_size = fpn_feature_list[0].shape[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(fpn_feature_list[i], output_size, mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)

        x = self.conv_fusion(fusion_out)
        x = self.dropout(x)
        x = self.classifier(x)

        return x
