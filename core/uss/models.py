# Copyright (C) 2024 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import timm # pip install timm==0.9.10 for FiT3D
import torch
import types
from torch import nn

from .dinov1 import vit_small, vit_base
from .utils import get_feature_size

class DINOv1(nn.Module):
    def __init__(self, arch='ViT-B', patch_size=8, device=torch.device('cuda:0')):
        super().__init__()
        if arch == 'ViT-B':
            # e.g., https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
            self.tag = 'vitbase'
            self.backbone = vit_base(patch_size, num_classes=0)
        else:
            # e.g., https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth
            self.tag = 'deitsmall'
            self.backbone = vit_small(patch_size, num_classes=0)
        
        state_dict = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dino/" + f"dino_{self.tag}{patch_size}_pretrain/dino_{self.tag}{patch_size}_pretrain.pth")
        self.backbone.load_state_dict(state_dict, strict=True)

        self.device = device
        self.patch_size = patch_size
        self.embed_dim = self.output_dim = self.backbone.embed_dim
        
        self.eval()
        self.to(device)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def to(self, device):
        super().to(device)
        self.device = device
    
    @torch.no_grad()
    def forward(self, image):
        if len(image.shape) == 3: image = image[None]

        fh, fw = get_feature_size(image.shape[2:], self.patch_size)
        image_feat, _, _ = self.backbone.get_intermediate_feat(image)
        
        image_feat = image_feat[0][:, 1:, :]
        image_feat = image_feat.reshape(image_feat.shape[0], fh, fw, -1).permute(0, 3, 1, 2)
        
        return image_feat
    
    @torch.no_grad()
    def forward_features(self, x):
        image_feat, _, _ = self.backbone.get_intermediate_feat(x)
        return image_feat[0]

class DINOv2(nn.Module):
    def __init__(self, arch='ViT-S', patch_size=14, device=torch.device('cuda:0'), reg=False):
        super().__init__()

        if arch == 'ViT-S': tag = f'dinov2_vits{patch_size}'
        elif arch == 'ViT-B': tag = f'dinov2_vitb{patch_size}'
        elif arch == 'ViT-L': tag = f'dinov2_vitl{patch_size}'
        elif arch == 'ViT-G': tag = f'dinov2_vitg{patch_size}'
        else: raise ValueError("Unknown arch")
        
        self.backbone = torch.hub.load('facebookresearch/dinov2', tag + ('_reg' if reg else ''), force_reload=False) 
        
        self.device = device
        self.patch_size = patch_size
        self.embed_dim = self.output_dim = self.backbone.embed_dim

        self.eval()
        self.to(device)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def to(self, device):
        super().to(device)
        self.device = device
    
    @torch.no_grad()
    def forward(self, image):
        if len(image.shape) == 3: image = image[None]
        return self.backbone.get_intermediate_layers(image, n=1, reshape=True)[0]
    
    @torch.no_grad()
    def forward_features(self, x, masks=None):
        if len(x.shape) == 3: x = x[None]

        if isinstance(x, list):
            return self.backbone.forward_features_list(x, masks)

        x = self.backbone.prepare_tokens_with_masks(x, masks)
        for blk in self.backbone.blocks:
            x = blk(x)

        return self.backbone.norm(x)

class FiT3D(nn.Module):
    def __init__(self, arch='DINOv2-reg', backbone='ViT-S', fit3d=True, device=torch.device('cuda:0')):
        super().__init__()

        def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n=1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            return_class_token: bool = False,
            norm: bool = True,
        ):
            outputs = self._intermediate_layers(x, n)
            if norm:
                outputs = [self.norm(out) for out in outputs]
            if return_class_token:
                prefix_tokens = [out[:, 0] for out in outputs]
            else:
                prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
            outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

            if reshape:
                B, C, H, W = x.shape
                grid_size = (
                    (H - self.patch_embed.patch_size[0])
                    // self.patch_embed.proj.stride[0]
                    + 1,
                    (W - self.patch_embed.patch_size[1])
                    // self.patch_embed.proj.stride[1]
                    + 1,
                )
                outputs = [
                    out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    for out in outputs
                ]

            if return_prefix_tokens or return_class_token:
                return tuple(zip(outputs, prefix_tokens))
            return tuple(outputs)

        self.device = device
        
        backbone = {
            'ViT-S': 'small',
            'ViT-B': 'base'
        }[backbone]
        
        timm_model_card = {
            "DINOv2": f"vit_{backbone}_patch14_dinov2.lvd142m",
            "DINOv2-reg": f"vit_{backbone}_patch14_reg4_dinov2.lvd142m",
            "CLIP": f"vit_{backbone}_patch16_clip_384.laion2b_ft_in12k_in1k",
            "MAE": f"vit_{backbone}_patch16_224.mae",
            "DeiT-III": f"deit3_{backbone}_patch16_224.fb_in1k"
        }
        timm_model = timm.create_model(timm_model_card[arch], pretrained=True, num_classes=0, dynamic_img_size=True, dynamic_img_pad=False)

        config = timm.data.resolve_data_config(timm_model.pretrained_cfg, model=timm_model)
        self.mean = config['mean']
        self.std = config['std']

        p = timm_model.patch_embed.patch_size
        self.patch_size = p if isinstance(p, int) else p[0]

        if fit3d:
            del timm_model

            fit3d_model_card = {
                "DINOv2": f"dinov2_{backbone}_fine",
                "DINOv2-reg": f"dinov2_reg_{backbone}_fine",
                "CLIP": f"clip_{backbone}_fine",
                "MAE": f"mae_{backbone}_fine",
                "DeiT-III": f"deit3_{backbone}_fine"
            }
            self.model = torch.hub.load("ywyue/FiT3D", fit3d_model_card[arch])
        else:
            self.model = timm_model

        self.model.eval()
        self.model.to(self.device)

        self.model.get_intermediate_layers = types.MethodType(get_intermediate_layers, self.model)
    
    @property
    def embed_dim(self):
        return self.model.embed_dim

    def forward(self, images):
        params = {
            'x': images,
            'n': [8,9,10,11],
            'reshape': True,
            'return_prefix_tokens': False,
            'return_class_token': False,
            'norm': True
        }
        features = self.model.get_intermediate_layers(**params)

        return features[-1]
