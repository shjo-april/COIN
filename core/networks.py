# Copyright (C) 2023 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import torch

from torch import nn
from torch.nn import functional as F

from .layers import Backbone, ConvBlock, DeepLabv1_Head, DeepLabv2_Head, DeepLabv3plus_Head, UperNet_Head, AuxHead
from tools import torch_utils

class DeepLabv3plus(Backbone):
    def __init__(self, backbone, num_classes=20+1, feature_size=256, low_channels=48, output_stride=8):
        super().__init__(backbone, last_stride=1, output_stride=output_stride)
        self.seg_decoder = DeepLabv3plus_Head(
            self.in_channels[-1], feature_size, 
            self.in_channels[1] if len(self.in_channels) == 5 else self.in_channels[0], 
            low_channels, output_stride, num_classes
        )
        
    def forward(self, x):
        f_list = self.model(x)
        if len(f_list) == 4: C2, C3, C4, C5 = f_list
        else: C1, C2, C3, C4, C5 = f_list
        return self.seg_decoder(C5, C2)
    
    @torch.no_grad()
    def apply_ms(self, image, scales=[1.0, 0.5, 1.5, 2.0], hflip=True, interpolation='bilinear'):
        size = None
        pred_masks = []

        for scale in scales:
            # rescale
            images = torch_utils.resize(image, image.shape[2:], scale=scale)
            if hflip: images = torch.cat([images, images.flip(-1)], dim=0)
            
            # inference
            logits = self.forward(images)

            if size is None: size = logits.shape[2:]
            
            # segmentation masks
            masks = F.softmax(logits, dim=1)
            masks = torch_utils.resize(masks, size, mode=interpolation)
            
            pred_masks.append(masks[0])
            if hflip: pred_masks.append(masks[1].flip(-1))

        return torch_utils.resize(torch.mean(torch.stack(pred_masks), dim=0), image.shape[2:])

class DeepLabv3plus_Edge(Backbone):
    def __init__(self, backbone, num_classes=20+1, feature_size=256, low_channels=48, output_stride=8):
        super().__init__(backbone, last_stride=1, output_stride=output_stride)
        self.seg_decoder = DeepLabv3plus_Head(
            self.in_channels[-1], feature_size, 
            self.in_channels[1] if len(self.in_channels) == 5 else self.in_channels[0], 
            low_channels, output_stride, num_classes
        )
        self.edge_decoder = DeepLabv3plus_Head(
            self.in_channels[-1], feature_size, 
            self.in_channels[1] if len(self.in_channels) == 5 else self.in_channels[0], 
            low_channels, output_stride, 2 # for edge
        )
        
    def forward(self, x):
        f_list = self.model(x)
        if len(f_list) == 4: C2, C3, C4, C5 = f_list
        else: C1, C2, C3, C4, C5 = f_list
        return self.seg_decoder(C5, C2), F.interpolate(self.edge_decoder(C5, C2), x.shape[2:], mode='bilinear')
    
    @torch.no_grad()
    def apply_ms(self, image, scales=[1.0, 0.5, 1.5, 2.0], hflip=True, interpolation='bilinear'):
        size = None
        pred_masks = []
        pred_edges = []

        for scale in scales:
            # rescale
            images = torch_utils.resize(image, image.shape[2:], scale=scale)
            if hflip: images = torch.cat([images, images.flip(-1)], dim=0)
            
            # inference
            logits_seg, logits_edge = self.forward(images)

            if size is None: size = logits_seg.shape[2:]
            
            # semantic outputs
            masks = F.softmax(logits_seg, dim=1)
            masks = torch_utils.resize(masks, size, mode=interpolation)
            
            pred_masks.append(masks[0])
            if hflip: pred_masks.append(masks[1].flip(-1))

            # instance outputs
            edges = F.softmax(logits_edge, dim=1)
            edges = torch_utils.resize(edges, size, mode=interpolation)

            pred_edges.append(edges[0])
            if hflip: pred_edges.append(edges[1].flip(-1))

        return torch_utils.resize(torch.mean(torch.stack(pred_masks), dim=0), image.shape[2:]), torch_utils.resize(torch.mean(torch.stack(pred_edges), dim=0), image.shape[2:])
