# Copyright (C) 2024 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import torch

from .models import *
from .utils import *

@torch.no_grad()
def inference(model: FiT3D, image: torch.Tensor, scales: list=[1.0], hflip: bool=False, image_size: int=0, reduce_vram: bool=False):
    if scales is None: scales = [1.]
    
    oh, ow = image.shape[1:]
    ms_image_feat = None
    
    if image_size > 0:
        image, image_size = resize_for_aspect_ratio(image, image_size)
    
    for scale in scales:
        # preprocess
        sh, sw = int(oh*scale), int(ow*scale)
        while sw % model.patch_size != 0: sw += 1
        while sh % model.patch_size != 0: sh += 1
        images = resize(image[None], (sh, sw), mode='bicubic').to(model.device)

        if not reduce_vram:
            if hflip: 
                images = torch.cat(
                    [
                        images, 
                        images.flip(-1)
                    ], dim=0
                )
            image_feat = model(images)
        else:
            image_feat = torch.cat(
                [
                    model(images),
                    model(images.flip(-1)),
                ], dim=0
            )
        
        # postprocess
        if hflip: 
            image_feat[1] = image_feat[1].flip(-1)
            image_feat = image_feat.mean(dim=0)
        else:
            image_feat = image_feat[0]
        
        # accumulate
        if ms_image_feat is None: ms_image_feat = image_feat
        else: ms_image_feat += resize(image_feat, ms_image_feat.shape[1:], mode='nearest')

    return ms_image_feat / len(scales)
